"""RAM-01: CPU training of concrete Lean tactic outcome scorers.

Only compiler-visible state, candidate text and family enter the network.
The source artifact carries scratch outcomes; this module claims zero proofs.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import gc
import hashlib
import json
import platform
from pathlib import Path
import re
import resource
import time
import torch
from torch import nn
import torch.nn.functional as F

FAMILIES = ("omega", "rfl", "simp_all", "assumption", "intro", "constructor",
            "exact_local", "apply_local", "rw_local", "rcases_local",
            "induction_list", "simpa_local")
STATE_SHA = "403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0"
EFFECT_SHA = "bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec"
DEV_MODULES = {"MathCan/DeletionMemory.lean", "MathCan/TOECosmology.lean"}
OUTCOMES = ("failed_or_unchanged", "single_goal_progress", "multi_goal_progress", "scratch_closed")
SEEDS = (17, 29)
BUCKETS = 2048
CONTEXT_TOKENS = GOAL_TOKENS = 96
ACTION_TOKENS = 24
SEMANTIC_KEYS = ("stage_id", "family", "script", "accepted", "progress",
                 "closed_in_scratch", "remaining_goal_count", "compiler_returncode",
                 "candidate_source_sha256")

@dataclass(frozen=True)
class Config:
    name: str
    width: int
    loops: int

CONFIGS = (Config("flat64", 64, 0), Config("loop32", 32, 4),
           Config("loop64", 64, 4), Config("loop128", 128, 4))

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]

def label(row: dict) -> int:
    if not (row["accepted"] and row["progress"]):
        return 0
    if row["closed_in_scratch"]:
        if row["remaining_goal_count"] != 0:
            raise ValueError("inconsistent scratch closure")
        return 3
    return 1 if row["remaining_goal_count"] == 1 else 2

def load_data(root: Path) -> tuple[list[dict], list[dict]]:
    state_file = root / "train_source/actual_goal_states.jsonl"
    if digest(state_file) != STATE_SHA:
        raise ValueError("frozen state hash changed")
    states = read_rows(state_file)
    probes = read_rows(root / "train_probes/train_candidate_outcomes.jsonl")
    # FC-12's historical semantic serialization uses literal backslash-n separators.
    canonical = "".join(json.dumps({k: p.get(k) for k in SEMANTIC_KEYS},
                                   sort_keys=True, ensure_ascii=False) + "\\n" for p in probes)
    if hashlib.sha256(canonical.encode()).hexdigest() != EFFECT_SHA:
        raise ValueError("frozen action-effect semantics changed")
    if len(states) != 51 or len(probes) != 961:
        raise ValueError("unexpected frozen corpus size")
    by_id = {s["stage_id"]: s for s in states}
    if len(by_id) != 51:
        raise ValueError("duplicate state identifiers")
    for p in probes:
        s = by_id[p["stage_id"]]
        if p["state_sha256"] != s["actual_state_sha256"]:
            raise ValueError("candidate paired with the wrong state")
        if p["source_file"] != s["source_file"] or p["theorem"] != s["theorem"]:
            raise ValueError("source metadata mismatch")
        p["goal_text"] = s["actual_lean_goal_state"]
        p["label"] = label(p)
    train = {s["actual_state_sha256"] for s in states if s["source_file"] not in DEV_MODULES}
    dev = {s["actual_state_sha256"] for s in states if s["source_file"] in DEV_MODULES}
    if train & dev:
        raise ValueError("exact duplicate state crosses source-module split")
    return states, probes

def token_ids(text: str, length: int, tail: bool = False) -> list[int]:
    tokens = re.findall(r"[\w']+|[^\w\s]", text, flags=re.UNICODE)
    tokens = tokens[-length:] if tail else tokens[:length]
    ids = [1 + int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
           % (BUCKETS - 1) for t in tokens]
    return ids + [0] * (length - len(ids))

def encode(rows: list[dict]) -> dict[str, torch.Tensor]:
    contexts, goals, actions, families = [], [], [], []
    for row in rows:
        text = row["goal_text"]
        if text.count("⊢") != 1:
            raise ValueError("expected one actual compiler goal")
        context, goal = text.split("⊢", 1)
        contexts.append(token_ids(context, CONTEXT_TOKENS, tail=True))
        goals.append(token_ids(goal, GOAL_TOKENS))
        actions.append(token_ids(row["script"], ACTION_TOKENS))
        families.append(FAMILIES.index(row["family"]))
    return {"context": torch.tensor(contexts, dtype=torch.long),
            "goal": torch.tensor(goals, dtype=torch.long),
            "action": torch.tensor(actions, dtype=torch.long),
            "family": torch.tensor(families, dtype=torch.long)}

class CandidatePolicy(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        d = config.width
        self.embedding = nn.Embedding(BUCKETS, d, padding_idx=0)
        self.family = nn.Embedding(len(FAMILIES), d)
        self.fuse = nn.Sequential(nn.Linear(4*d, 2*d), nn.GELU(), nn.Linear(2*d, d), nn.GELU())
        if config.loops:
            self.cell = nn.Sequential(nn.Linear(2*d, 2*d), nn.GELU(), nn.Linear(2*d, d))
            self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 4)

    def mean(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).unsqueeze(-1)
        return (self.embedding(ids)*mask).sum(1)/mask.sum(1).clamp_min(1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fixed = self.fuse(torch.cat([self.mean(batch["context"]), self.mean(batch["goal"]),
                                    self.mean(batch["action"]), self.family(batch["family"])], -1))
        z = fixed
        for _ in range(self.config.loops):
            z = self.norm(z + self.cell(torch.cat([z, fixed], -1)))
        return self.head(z)

    @torch.no_grad()
    def rank(self, goal: str, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if not candidates:
            return []
        rows = [{"goal_text": goal, "family": f, "script": s} for f, s in candidates]
        probs = self(encode(rows)).softmax(-1)
        utility = 2*probs[:, 3] + probs[:, 1]
        order = sorted(range(len(rows)), key=lambda i: (-float(utility[i]), i))
        return [candidates[i] for i in order]

@torch.no_grad()
def cached_metrics(model: CandidatePolicy | None, rows: list[dict],
                   frequency: dict | None = None) -> dict:
    if model is not None:
        model.eval()
        outputs = model(encode(rows)).softmax(-1)
        scores = (2*outputs[:, 3] + outputs[:, 1]).tolist()
    else:
        scores = [frequency.get(p["family"], 0.) if frequency is not None else 0. for p in rows]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["stage_id"]].append(i)
    counts = Counter()
    selected = []
    for stage, ids in groups.items():
        ranked = sorted(ids, key=lambda i: (-scores[i], i))
        labels = [rows[i]["label"] for i in ranked]
        for k in (1, 3):
            counts[f"top{k}_progress"] += any(v > 0 for v in labels[:k])
            counts[f"top{k}_executable"] += any(v in (1, 3) for v in labels[:k])
            counts[f"top{k}_closed"] += 3 in labels[:k]
        counts["oracle_progress"] += any(v > 0 for v in labels)
        counts["oracle_executable"] += any(v in (1, 3) for v in labels)
        counts["oracle_closed"] += 3 in labels
        selected.append({"stage_id": stage, "family": rows[ranked[0]]["family"],
                         "script": rows[ranked[0]]["script"], "observed_outcome": labels[0]})
    return {"states": len(groups), "candidate_rows": len(rows), **dict(counts),
            "selected": selected, "complete_theorem_proofs_claimed": 0}

def fit(rows: list[dict], config: Config, seed: int, dest: Path, steps: int) -> tuple[CandidatePolicy, dict]:
    torch.manual_seed(seed)
    model = CandidatePolicy(config)
    tensors = encode(rows)
    y = torch.tensor([r["label"] for r in rows])
    counts = torch.bincount(y, minlength=4).float()
    weights = (len(rows)/(4*counts.clamp_min(1))).sqrt()
    weights /= weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed+1000)
    began = time.perf_counter()
    loss_log = []
    for step in range(steps):
        ix = torch.randint(len(rows), (64,), generator=generator)
        logits = model({k: v[ix] for k, v in tensors.items()})
        loss = F.cross_entropy(logits, y[ix], weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step in {0, steps//2, steps-1}:
            loss_log.append({"step": step+1, "loss": float(loss.detach())})
    elapsed = time.perf_counter()-began
    model.eval()
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": model.state_dict(), "seed": seed,
                "state_sha256": STATE_SHA, "effect_sha256": EFFECT_SHA}, dest)
    restored = load_checkpoint(dest)
    sample = {k: v[:8] for k, v in tensors.items()}
    if not torch.equal(model(sample), restored(sample)):
        raise AssertionError("saved checkpoint fails exact CPU reload test")
    receipt = {"name": config.name, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
               "loops": config.loops, "width": config.width, "updates": steps, "batch": 64,
               "candidate_examples": len(rows), "states": len({r['stage_id'] for r in rows}),
               "outcome_counts": {OUTCOMES[i]: int(counts[i]) for i in range(4)},
               "loss_log": loss_log, "training_seconds": round(elapsed, 4),
               "process_peak_rss_mib_so_far": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
               "checkpoint": str(dest.name), "checkpoint_bytes": dest.stat().st_size,
               "checkpoint_sha256": digest(dest), "reload_exact": True}
    print("RAM01_FIT", json.dumps(receipt, sort_keys=True), flush=True)
    return model, receipt

def load_checkpoint(path: Path) -> CandidatePolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["state_sha256"] != STATE_SHA or data["effect_sha256"] != EFFECT_SHA:
        raise ValueError("unrecognized training artifact")
    model = CandidatePolicy(Config(**data["config"]))
    model.load_state_dict(data["state_dict"])
    return model.eval()

def run(root: Path, out: Path, steps: int = 400) -> dict:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    states, probes = load_data(root)
    train = [p for p in probes if p["source_file"] not in DEV_MODULES]
    dev = [p for p in probes if p["source_file"] in DEV_MODULES]
    positive_support = Counter(p["family"] for p in train if p["label"] > 0)
    if any(positive_support[f] < 1 for f in ("exact_local", "rw_local", "rcases_local", "induction_list")):
        raise AssertionError("mandatory positive tactic family absent from development training")
    frequencies = {}
    for f in FAMILIES:
        q = [p for p in train if p["family"] == f]
        frequencies[f] = sum(2*(p["label"] == 3)+(p["label"] == 1) for p in q)/max(1, len(q))
    report = {"experiment": "RAM01_CPU_concrete_Lean_candidate_training", "device": "cpu",
              "torch_version": torch.__version__, "python_version": platform.python_version(),
              "threads": torch.get_num_threads(), "state_sha256": STATE_SHA, "effect_sha256": EFFECT_SHA,
              "configurations": [asdict(c) for c in CONFIGS], "development_modules": sorted(DEV_MODULES),
              "development_train_states": len({p['stage_id'] for p in train}),
              "development_test_states": len({p['stage_id'] for p in dev}),
              "development_positive_support": dict(positive_support),
              "native_development": cached_metrics(None, dev),
              "frequency_development": cached_metrics(None, dev, frequencies),
              "development": [], "final": [], "complete_theorem_proofs_claimed_here": 0}
    for phase, rows in (("development", train), ("final", probes)):
        for config in CONFIGS:
            for seed in SEEDS:
                name = f"{config.name}_s{seed}.pt"
                model, receipt = fit(rows, config, seed, out/phase/name, steps)
                receipt["metrics"] = cached_metrics(model, dev if phase == "development" else probes)
                receipt["metric_scope"] = "source-module-held-out cached compiler outcomes" if phase == "development" else "training resubstitution only"
                report[phase].append(receipt)
                out.mkdir(parents=True, exist_ok=True)
                (out/"training_report.json").write_text(json.dumps(report, indent=2)+"\n")
                del model
                gc.collect()
    report["total_wall_seconds"] = round(time.perf_counter()-start, 3)
    report["peak_process_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)
    report["total_optimizer_steps"] = steps*len(CONFIGS)*len(SEEDS)*2
    (out/"training_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("RAM01_COMPLETE", json.dumps({k: report[k] for k in ("total_wall_seconds", "peak_process_rss_mib", "total_optimizer_steps")}), flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
