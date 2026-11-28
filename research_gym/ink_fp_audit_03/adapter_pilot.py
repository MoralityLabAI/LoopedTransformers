"""Bounded recursive adapter pilot on free Inkling-2-Mini (native Tinker; run in a Linux venv).

Arms (matched per generation: 8 train problems x 4 on-policy samples, one AdamW step, LoRA rank 8,
same learning rate):
    B  verified rejection-sampled SFT on the policy's own verified samples   (loss: cross_entropy)
    E  cost-sensitive GSPO, R = verified - lam * tokens/cap - 0.5 rep - 0.25 incomplete   (loss: gspo)
    F  adaptive-curriculum GSPO (same reward as E; families weighted by observed frontier rate)
Behaviour policy for GSPO: the native sampler at render-time effort 0.2, temperature 1.0; the
sampled token ids and their logprobs are used directly (no re-tokenisation).
Acceptance: the frozen audit-02 rule on paired greedy evaluation of the acceptance partition, in a
hash-chained ledger; a rejected candidate is persisted and the next generation restarts from the
accepted head. Resumable from the ledgers. The credential comes from TINKER_API_KEY only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research_gym.ink_fp_audit_02 import acceptance as acc  # noqa: E402
from research_gym.ink_fp_audit_02 import gyms  # noqa: E402,F401
from research_gym.ink_fp_audit_02 import resources as res  # noqa: E402
from research_gym.ink_fp_audit_02 import rewards as rw  # noqa: E402
from research_gym.ink_fp_audit_02.curriculum import Task, verify  # noqa: E402
from research_gym.ink_fp_audit_02.ledger import Ledger, append_jsonl, iter_jsonl  # noqa: E402
from research_gym.ink_fp_audit_03 import frontier  # noqa: E402,F401
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_qual import MODEL, message_text  # noqa: E402

EFFORT = 0.2
CAP = 4096
LR = 5e-5
RANK = 8
ARMS = {"B": {"method": "sft", "loss": "cross_entropy"},
        "E": {"method": "cost_gspo", "loss": "gspo", "lam": 0.5},
        "F": {"method": "adaptive_gspo", "loss": "gspo", "lam": 0.5, "adaptive": True}}


def host_free_bytes() -> int:
    """Windows host available memory via WSL interop (the WSL VM's MemAvailable is not the host's)."""
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out) * 1024
    except Exception:
        return 8 * res.GIB  # interop unavailable: fall back to the cgroup cap alone (recorded)


def load_tasks(path: Path) -> list[Task]:
    f = Task.__dataclass_fields__
    return [Task(**{k: v for k, v in r.items() if k in f}) for r in iter_jsonl(path)]


class Native:
    def __init__(self):
        import tinker
        from tinker import types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        self.tinker, self.types = tinker, types
        self.r = renderers.get_renderer(model_info.get_recommended_renderer_name(MODEL), get_tokenizer(MODEL), model_name=MODEL)
        self.svc = tinker.ServiceClient()
        self.requests = 0
        self.sample_tokens = 0
        self.train_tokens = 0

    @staticmethod
    def _res(x):
        return x.result() if hasattr(x, "result") and not hasattr(x, "sample") and not hasattr(x, "forward_backward") else x

    def sampler(self, path=None):
        sc = self._res(self.svc.create_sampling_client(base_model=MODEL) if path is None
                       else self.svc.create_sampling_client(model_path=path))
        if sc.get_base_model() != MODEL:
            raise RuntimeError("sampler is not on the free model")
        return sc

    def trainer(self, state_path=None):
        tc = self._res(self.svc.create_lora_training_client(base_model=MODEL, rank=RANK, seed=0) if state_path is None
                       else self.svc.create_training_client_from_state_with_optimizer(state_path))
        if tc.get_info().model_data.model_name != MODEL:
            raise RuntimeError("training client is not on the free model")
        return tc

    def prompt(self, task: Task):
        p = self.r.build_generation_prompt([{"role": "user", "content": task.prompt}], effort=EFFORT)
        if p.length + CAP > 16_384:
            raise ValueError("context ceiling")
        return p

    def sample(self, sc, task, n, temperature, seed):
        p = self.prompt(task)
        res_ = sc.sample(prompt=p, num_samples=n, sampling_params=self.types.SamplingParams(
            max_tokens=CAP, temperature=temperature, seed=seed, stop=self.r.get_stop_sequences())).result()
        self.requests += 1
        out = []
        for seq in res_.sequences:
            toks = list(seq.tokens)
            self.sample_tokens += len(toks)
            text = message_text(self.r.parse_response(toks)[0])
            out.append({"tokens": toks, "logprobs": list(seq.logprobs or []), "text": text,
                        "stop_reason": str(getattr(seq, "stop_reason", "")), **verify(task, text)})
        return p, out

    def td(self, v, dt):
        return self.types.TensorData(data=list(v), dtype=dt, shape=[len(v)])

    def datum_rl(self, p, toks, lps, adv):
        pi = p.to_ints()
        full, n = pi + toks, len(pi)
        T = self.types
        return T.Datum(model_input=T.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
            "target_tokens": self.td(full[1:], "int64"),
            "logprobs": self.td([0.0] * (n - 1) + [float(x) for x in lps], "float32"),
            "advantages": self.td([0.0] * (n - 1) + [float(adv)] * len(toks), "float32"),
            "sequence_ratio_mask": self.td([0.0] * (n - 1) + [1.0] * len(toks), "float32")})

    def datum_sft(self, p, toks):
        pi = p.to_ints()
        full, n = pi + toks, len(pi)
        T = self.types
        return T.Datum(model_input=T.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
            "target_tokens": self.td(full[1:], "int64"),
            "weights": self.td([0.0] * (n - 1) + [1.0] * len(toks), "float32")})


def evaluate(nat, sampler_path, tasks, cache_path: Path) -> list[dict]:
    cached = {r["task_id"]: r for r in iter_jsonl(cache_path)} if cache_path.exists() else {}
    sc = nat.sampler(sampler_path)
    out = []
    for t in tasks:
        if t.task_id not in cached:
            _, s = nat.sample(sc, t, 1, 0.0, 0)
            s = s[0]
            rec = {"task_id": t.task_id, "family": t.family, "correct": s["correct"], "completed": s["completed"],
                   "repetition": s["repetition"], "completion_tokens": len(s["tokens"]), "stop_reason": s["stop_reason"]}
            append_jsonl(cache_path, rec)
            cached[t.task_id] = rec
        out.append({k: cached[t.task_id][k] for k in ("task_id", "family", "correct", "completed", "repetition")})
    return out


def reward(s, lam):
    r = 0.0
    if s["correct"]:
        r += 1.0
    r -= lam * len(s["tokens"]) / CAP
    if s["repetition"]:
        r -= 0.5
    if not s["completed"]:
        r -= 0.25
    return r


def pick(train, arm, g, frontier_rate):
    rng = random.Random(9_300_000 + g)
    if not ARMS[arm].get("adaptive") or not frontier_rate:
        idx = list(range(len(train)))
        rng.shuffle(idx)
        return [train[i] for i in idx[:8]]
    fams = sorted({t.family for t in train})
    w = {f: 0.1 + frontier_rate.get(f, 0.25) for f in fams}
    by = {f: [t for t in train if t.family == f] for f in fams}
    for f in by:
        rng.shuffle(by[f])
    out = []
    while len(out) < 8:
        live = [f for f in fams if by[f]]
        f = rng.choices(live, weights=[w[x] for x in live])[0]
        out.append(by[f].pop())
    return out


def run_arm(nat, arm, train, accept, out: Path, gens: int, monitor) -> dict:
    d = out / f"arm_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    led = Ledger(d / "ledger.jsonl")
    crit = acc.Criteria()
    if acc.accepted_head(led, arm) is None:
        acc.register_base(led, arm, "base")
        led.append("arm_config", {"arm": arm, **ARMS[arm], "effort": EFFORT, "cap": CAP, "lr": LR, "rank": RANK})
    paths = {"base": {"state_path": None, "sampler_path": None}}
    for b in led.kinds("checkpoint"):
        paths[b["tag"]] = b["meta"]
    done = {b["generation"] for b in led.kinds("generation_done")}
    frontier_rate = {}
    for b in led.kinds("generation_done"):
        frontier_rate = b.get("frontier_rate", frontier_rate)
    base_cache = out / "eval_base.jsonl"
    for g in range(gens):
        if g in done:
            continue
        monitor.gate()
        head = acc.accepted_head(led, arm)
        sc = nat.sampler(paths[head]["sampler_path"])
        batch = pick(train, arm, g, frontier_rate)
        items, fam = [], defaultdict(lambda: [0, 0])
        for i, t in enumerate(batch):
            # Unseeded on purpose: a fixed seed makes all num_samples identical (receipts/sampling_diversity_20261128.json),
            # which zeroes group-relative advantages.
            p, samples = nat.sample(sc, t, 4, 1.0, None)
            for s in samples:
                append_jsonl(d / f"rollouts_g{g}.jsonl", {"generation": g, "task_id": t.task_id, "family": t.family,
                             "correct": s["correct"], "completed": s["completed"], "repetition": s["repetition"],
                             "completion_tokens": len(s["tokens"]), "stop_reason": s["stop_reason"],
                             "logprob_sum": sum(s["logprobs"]), "text_tail": s["text"][-400:], "policy": head})
            front = any(s["correct"] for s in samples) and not all(s["correct"] for s in samples)
            fam[t.family][0] += front
            fam[t.family][1] += 1
            if ARMS[arm]["method"] == "sft":
                items += [nat.datum_sft(p, s["tokens"]) for s in samples if s["correct"]]
            else:
                rs = [reward(s, ARMS[arm]["lam"]) for s in samples]
                adv = rw.group_advantages(rs)
                items += [nat.datum_rl(p, s["tokens"], s["logprobs"], a) for s, a in zip(samples, adv)
                          if a != 0.0 and s["tokens"] and len(s["logprobs"]) == len(s["tokens"])]
        frontier_rate = {f: v[0] / v[1] for f, v in fam.items()}
        if not items:
            led.append("generation_done", {"arm": arm, "generation": g, "trained": False, "reason": "no training signal",
                                           "frontier_rate": frontier_rate, "requests": nat.requests})
            continue
        tc = nat.trainer(paths[head]["state_path"])
        fb = tc.forward_backward(items, ARMS[arm]["loss"], None).result()
        tc.optim_step(nat.types.AdamParams(learning_rate=LR)).result()
        ntok = sum(x.model_input.length for x in items)
        nat.train_tokens += ntok
        tag = f"ink03-{arm}-g{g}-{int(time.time())}"
        st = tc.save_state(tag).result()
        sw = tc.save_weights_for_sampler(tag + "-s").result()
        paths[tag] = {"state_path": st.path, "sampler_path": sw.path}
        led.append("checkpoint", {"arm": arm, "generation": g, "tag": tag, "meta": paths[tag]})
        monitor.gate()
        pred_cache = base_cache if head == "base" else d / f"eval_{head}.jsonl"
        pred_cases = evaluate(nat, paths[head]["sampler_path"], accept, pred_cache)
        cand_cases = evaluate(nat, paths[tag]["sampler_path"], accept, d / f"eval_{tag}.jsonl")
        dec = acc.record_candidate(led, crit, generation=g, arm=arm, predecessor_checkpoint=head, candidate_checkpoint=tag,
                                   predecessor_cases=pred_cases, candidate_cases=cand_cases,
                                   hyperparameters={**ARMS[arm], "lr": LR, "rank": RANK, "effort": EFFORT, "cap": CAP},
                                   scales_examined=[LR])["decision"]
        led.append("generation_done", {"arm": arm, "generation": g, "trained": True, "datums": len(items),
                                       "train_tokens": ntok, "accepted": dec["accepted"], "checks": dec["checks"],
                                       "primary_gain": dec["primary_gain"], "frontier_rate": frontier_rate,
                                       "metrics": dict(getattr(fb, "metrics", {}) or {}), "requests": nat.requests})
    return {"arm": arm, "head": acc.accepted_head(led, arm), "replay_mismatches": len(acc.replay(led)["mismatches"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--accept", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["B", "E", "F"])
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--accept-per-family", type=int, default=3)
    a = ap.parse_args()
    if a.generations > 4:
        raise SystemExit("at most four improvement generations")
    a.out.mkdir(parents=True, exist_ok=True)
    os.nice(5)
    train = load_tasks(a.train)
    acc_all = load_tasks(a.accept)
    accept, per = [], defaultdict(int)
    for t in acc_all:
        if per[t.family] < a.accept_per_family:
            accept.append(t)
            per[t.family] += 1
    monitor = res.ResourceMonitor(reader=host_free_bytes, poll_seconds=30)
    nat = Native()
    summary = {"model": MODEL, "effort": EFFORT, "cap": CAP, "lr": LR, "rank": RANK, "n_train": len(train),
               "n_accept": len(accept), "arms": {}}
    for arm in a.arms:
        summary["arms"][arm] = run_arm(nat, arm, train, accept, a.out, a.generations, monitor)
        summary["usage"] = {"requests": nat.requests, "sample_tokens": nat.sample_tokens, "train_tokens": nat.train_tokens}
        summary["governor"] = {"min_host_free_bytes": monitor.governor.min_available,
                               "transitions": monitor.governor.transitions}
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, default=str)[:2000])


if __name__ == "__main__":
    main()
