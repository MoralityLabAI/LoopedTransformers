"""Recursive self-improvement engine: sample -> verify -> frontier -> advantages -> update ->
evaluate vs predecessor -> frozen acceptance -> persist/rollback -> export.

Backends implement:
    efforts() -> tuple[str, ...]                    efforts the route can control
    sample(task, effort, n, seed) -> list[dict]     {"text","completion_tokens","finish_reason",...}
    train(arm, items) -> dict                       items: [{"task","sample","reward","advantage"}]
    save(tag) -> str                                checkpoint identity
    restore(checkpoint) -> None                     make `checkpoint` the active policy
    checkpoint_meta(tag) -> dict / adopt(tag, meta)  persist and re-learn checkpoint identities
    evaluate(tasks, checkpoint) -> list[dict]       per-case outcomes (paired)
Everything is streamed to JSONL under the run directory; resumable from the ledger.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import acceptance as acc
from . import rewards as rw
from .curriculum import Task, verify
from . import exports
from .ledger import Ledger, append_jsonl, file_sha256

SAMPLE_PLAN = (("low", 2), ("medium", 2), ("high", 2))


@dataclass(frozen=True)
class ArmConfig:
    name: str
    method: str                 # frozen | sft | grpo | gspo_effort | gspo_shuffled | adaptive | cost | analytic
    loss_fn: str = "importance_sampling"
    reward: rw.RewardConfig = field(default_factory=rw.RewardConfig)
    lr: float = 1e-5
    rank: int = 8
    problems_per_generation: int = 16
    adaptive_curriculum: bool = False


ARMS = {
    "A": ArmConfig("A", "frozen"),
    "B": ArmConfig("B", "sft", loss_fn="cross_entropy"),
    "C": ArmConfig("C", "grpo", reward=rw.RewardConfig(rescue_bonus=0.0)),
    "D": ArmConfig("D", "gspo_effort", loss_fn="gspo"),
    "E": ArmConfig("E", "gspo_shuffled", loss_fn="gspo", reward=rw.RewardConfig(shuffle_rescue=True)),
    "F": ArmConfig("F", "adaptive", loss_fn="gspo", adaptive_curriculum=True),
    "G": ArmConfig("G", "cost", loss_fn="gspo", reward=rw.RewardConfig(token_cost_per_1k=0.25)),
    "H": ArmConfig("H", "analytic"),
}
UNSUPPORTED_NATIVE = {"H": "Tinker exposes no activations, gradients or adapter tensors; analytic and "
                           "projected updates are evaluated in the synthetic CPU proxy only"}


def sample_group(backend, task: Task, seed: int) -> list[dict]:
    efforts = backend.efforts()
    plan = [(e, n) for e, n in SAMPLE_PLAN if e in efforts] or [(efforts[0], 6)]
    total = sum(n for _, n in plan)
    if total < 6:  # route with fewer efforts: keep six samples per problem
        plan = [(plan[0][0], plan[0][1] + 6 - total)] + plan[1:]
    out = []
    for i, (effort, n) in enumerate(plan):
        for s in backend.sample(task, effort, n, seed * 101 + i):
            v = verify(task, s["text"])
            out.append({**s, **v, "effort": effort, "task_id": task.task_id})
    return out


def select_batch(pool: list[Task], arm: ArmConfig, generation: int, frontier_rate: dict) -> list[Task]:
    rng = random.Random(9_100_000 + generation)
    if not arm.adaptive_curriculum or not frontier_rate:
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        return [pool[i] for i in idx[:arm.problems_per_generation]]
    # Adaptive: weight families by frontier rate p(1-p) estimate, floor for exploration.
    fams = sorted({t.family for t in pool})
    w = {f: 0.1 + frontier_rate.get(f, 0.25) for f in fams}
    chosen, by_fam = [], {f: [t for t in pool if t.family == f] for f in fams}
    for f in by_fam:
        rng.shuffle(by_fam[f])
    while len(chosen) < arm.problems_per_generation and any(by_fam.values()):
        live = [f for f in fams if by_fam[f]]
        f = rng.choices(live, weights=[w[x] for x in live])[0]
        chosen.append(by_fam[f].pop())
    return chosen


def run_arm(arm: ArmConfig, backend, train_pool: list[Task], accept_tasks: list[Task],
            run_dir: Path, generations: int, monitor=None, criteria: acc.Criteria = acc.Criteria(),
            base_checkpoint: str = "base") -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir / "ledger.jsonl")
    if arm.name in UNSUPPORTED_NATIVE and arm.method == "analytic":
        if ledger.last("unsupported") is None:
            ledger.append("unsupported", {"arm": arm.name, "reason": UNSUPPORTED_NATIVE[arm.name]})
        return {"arm": arm.name, "status": "unsupported", "reason": UNSUPPORTED_NATIVE[arm.name]}
    if acc.accepted_head(ledger, arm.name) is None:
        acc.register_base(ledger, arm.name, base_checkpoint)
        ledger.append("arm_config", {"arm": arm.name, "config": _jsonable(asdict(arm))})
    done = {b["generation"] for b in ledger.kinds("generation_done") if b["arm"] == arm.name}
    eval_cache: dict[str, list[dict]] = {b["checkpoint"]: b["cases"] for b in ledger.kinds("eval")}
    for b in ledger.kinds("checkpoint"):  # resume: re-learn saved checkpoint identities
        if b["arm"] == arm.name:
            backend.adopt(b["tag"], b["meta"])
    frontier_rate: dict[str, float] = {}
    for b in ledger.kinds("generation_done"):
        frontier_rate = b.get("frontier_rate", frontier_rate)

    def evaluate(ckpt):
        if ckpt not in eval_cache:
            cases = backend.evaluate(accept_tasks, ckpt)
            ledger.append("eval", {"checkpoint": ckpt, "cases": cases})
            eval_cache[ckpt] = cases
        return eval_cache[ckpt]

    for g in range(generations):
        if g in done:
            continue
        if monitor is not None:
            monitor.gate()
        head = acc.accepted_head(ledger, arm.name)
        backend.restore(head)
        rollouts = run_dir / f"rollouts_g{g}.jsonl"
        if rollouts.exists():
            rollouts.unlink()  # partial generation from an interrupted run: redo it whole
        batch = select_batch(train_pool, arm, g, frontier_rate)
        items, fam_stats = [], {}
        rng = random.Random(7_000 + g)
        for t in batch:
            if monitor is not None:
                monitor.gate()
            group = sample_group(backend, t, seed=g * 1000 + hash_int(t.task_id))
            rs = rw.rewards(group, arm.reward, rng)
            adv = rw.group_advantages(rs)
            front = rw.is_frontier(group)
            st = fam_stats.setdefault(t.family, [0, 0])
            st[0] += front
            st[1] += 1
            for s, r, a in zip(group, rs, adv):
                rec = {"generation": g, "arm": arm.name, "task": t.receipt(), "effort": s["effort"],
                       "text": s["text"], "correct": s["correct"], "completed": s["completed"],
                       "repetition": s["repetition"], "completion_tokens": s.get("completion_tokens", 0),
                       "finish_reason": s.get("finish_reason"), "reward": r, "advantage": a,
                       "frontier": front, "policy_checkpoint": head,
                       "simulated": bool(getattr(backend, "simulated", False))}
                append_jsonl(rollouts, rec)
                if arm.method == "sft":
                    if s["correct"]:
                        items.append({"task": t, "sample": s, "reward": r, "advantage": 1.0})
                elif front and a != 0.0:
                    items.append({"task": t, "sample": s, "reward": r, "advantage": a})
        frontier_rate = {f: v[0] / v[1] for f, v in fam_stats.items()}
        if arm.method == "frozen" or not items:
            train_receipt = {"skipped": True, "reason": "frozen arm" if arm.method == "frozen" else "no frontier"}
            cand = head
        else:
            train_receipt = backend.train(arm, items)
            cand = backend.save(f"{arm.name}-g{g}")
            ledger.append("checkpoint", {"arm": arm.name, "generation": g, "tag": cand,
                                         "meta": backend.checkpoint_meta(cand)})
        if cand != head:
            pred_cases, cand_cases = evaluate(head), evaluate(cand)
            decision = acc.record_candidate(
                ledger, criteria, generation=g, arm=arm.name, predecessor_checkpoint=head,
                candidate_checkpoint=cand, predecessor_cases=pred_cases, candidate_cases=cand_cases,
                hyperparameters={"lr": arm.lr, "rank": arm.rank, "loss_fn": arm.loss_fn,
                                 "reward": asdict(arm.reward)},
                scales_examined=[1.0])["decision"]
            if not decision["accepted"]:
                backend.restore(head)
        else:
            decision = None
        ledger.append("generation_done", {
            "arm": arm.name, "generation": g, "policy_checkpoint": head, "candidate": cand,
            "accepted": None if decision is None else decision["accepted"],
            "frontier_rate": frontier_rate, "n_train_items": len(items), "train": train_receipt,
            "rollouts_sha256": file_sha256(rollouts),
            "resources": monitor.receipt() if monitor is not None else None})
    paths = sorted(run_dir.glob("rollouts_g*.jsonl"), key=lambda p: int(p.stem.split("_g")[1]))
    manifest = exports.build(paths, run_dir / "exports", {t.task_id: t.prompt for t in train_pool})
    return {"arm": arm.name, "status": "complete", "head": acc.accepted_head(ledger, arm.name),
            "replay": acc.replay(ledger), "exports": {k: v["unique_examples"] for k, v in manifest["streams"].items()}}


def hash_int(s: str) -> int:
    import hashlib
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 997


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


# ---- simulated backend (engineering only) ------------------------------------------------
class SimulatedBackend:
    """Deterministic toy policy for exercising the engine. Evidence class: engineering_simulation.

    Per-family logit skill; effort adds a fixed bonus; training nudges skills of trained
    families and adds seeded interference on others so acceptance and rollback are exercised.
    Its outputs are answer strings; it says nothing about any language model.
    """

    EFFORT_BONUS = {"low": -0.6, "medium": 0.0, "high": 0.6}
    simulated = True

    def __init__(self, families, seed=0, interference=0.15, update_noise=0.35):
        self.rng = random.Random(seed)
        self.skill = {f: self.rng.uniform(-1.5, 0.5) for f in families}
        self.store = {"base": dict(self.skill)}
        self.interference = interference
        self.update_noise = update_noise
        self.calls = 0

    def efforts(self):
        return ("low", "medium", "high")

    def _p(self, family, effort, skill=None):
        z = (skill or self.skill)[family] + self.EFFORT_BONUS.get(effort, 0.0)
        return 1 / (1 + math.exp(-z))

    def sample(self, task, effort, n, seed):
        r = random.Random(seed ^ hash_int(task.task_id) ^ hash_int(effort))
        out = []
        for _ in range(n):
            self.calls += 1
            ok = r.random() < self._p(task.family, effort)
            ans = task.answer if ok else task.wrong_answer(r)
            out.append({"text": f"work\nANSWER: {ans}", "completion_tokens": 40 + 40 * self.EFFORT_BONUS.get(effort, 0) + 60,
                        "finish_reason": "stop"})
        return out

    def train(self, arm, items):
        fam_signal = {}
        for it in items:
            fam_signal.setdefault(it["task"].family, []).append(it["advantage"] * (1 if it["sample"]["correct"] else -1))
        for f, v in fam_signal.items():
            # Signal plus seeded update noise, so some candidates regress and get rejected.
            self.skill[f] += 0.25 * max(-1.0, min(1.0, sum(v) / len(v))) + self.rng.gauss(0.0, self.update_noise)
        for f in self.skill:
            if f not in fam_signal:
                self.skill[f] -= self.interference * self.rng.random()
        return {"families": sorted(fam_signal), "items": len(items), "simulated": True}

    def save(self, tag):
        self.store[tag] = dict(self.skill)
        return tag

    def restore(self, ckpt):
        self.skill = dict(self.store[ckpt])

    def checkpoint_meta(self, tag):
        return {"skill": self.store[tag]}

    def adopt(self, tag, meta):
        self.store[tag] = dict(meta["skill"])

    def evaluate(self, tasks, ckpt):
        skill = self.store[ckpt]
        out = []
        for t in tasks:
            r = random.Random(hash_int(t.task_id) * 7919 + 1)
            out.append({"task_id": t.task_id, "family": t.family,
                        "correct": r.random() < self._p(t.family, "medium", skill),
                        "completed": True, "repetition": False})
        return out
