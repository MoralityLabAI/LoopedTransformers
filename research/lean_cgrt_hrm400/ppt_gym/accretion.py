"""Replayable online fast-weight accretion from independent Lean candidate receipts.

Only the train split is accepted for persistent learning. A simulated fixture
may exercise this pipeline but is always exported as non-training evidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .core import FastHead, PlasticProofTransformer
from .data import Case, canonical, digest, sha256_file


def state_hash(fast: FastHead) -> str:
    return hashlib.sha256(fast.weights.detach().cpu().numpy().tobytes()).hexdigest()


def _ordered(cases: list[Case], maximum: int):
    if maximum < 1 or maximum > 5000:
        raise ValueError("accretion must have a bounded 1..5000 case budget")
    rows = sorted(cases, key=lambda c: (c.lineage, c.root))
    chosen = []
    seen = set()
    dropped = 0
    for case in rows:
        if case.split != "train":
            raise ValueError("persistent online learning is restricted to train")
        if case.lineage in seen:
            dropped += 1
            continue
        seen.add(case.lineage)
        chosen.append(case)
        if len(chosen) >= maximum:
            break
    return chosen, dropped


@torch.no_grad()
def build_ledger(
    model: PlasticProofTransformer, cases: list[Case], *,
    maximum: int, depth: int, learning_rate: float, score_scale: float,
):
    """Deterministic per-session greedy policy; one selected receipt per root."""
    if not (1 <= depth <= 16 and 0 < learning_rate <= 20 and 0 <= score_scale <= 20):
        raise ValueError("invalid online budget or scale")
    selected, dropped = _ordered(cases, maximum)
    fast = FastHead(model.cfg.width, learning_rate)
    ledger = []
    outcomes = []
    sft = []
    for case in selected:
        before = state_hash(fast)
        out = model(case.goal, case.candidates, depth=depth)[-1]
        feature = out["features"]
        selected_logits = out["logits"] + score_scale * fast.predict(feature)
        chosen = int(selected_logits.argmax())
        result = case.labels[chosen]
        if result >= 0:
            fast.update(feature[chosen], result)
        after = state_hash(fast)
        event = {
            "sequence": len(ledger), "root": case.root, "lineage": case.lineage,
            "family": case.family, "split": case.split, "candidate_index": chosen,
            "tactic_sha256": digest(case.candidates[chosen]),
            "verdict": result, "definitive": result in (0, 1),
            "fast_weights_before": before, "fast_weights_after": after,
            "online_updates": fast.updates,
        }
        ledger.append(event)
        eligible = case.evidence == "lean_kernel_receipt_replay"
        row = {
            "root": case.lineage, "source_root": case.root, "family": "lean_proof",
            "subfamily": case.family, "split": "train",
            "evidence_class": case.evidence, "visit": 1,
            "input": {"goal": case.goal, "candidate_menu": list(case.candidates),
                      "selected_index": chosen, "budget": 8},
            "target": {"chosen_tactic": case.candidates[chosen],
                       "verified_closed": result == 1 if result >= 0 else None},
            "loss": {"boundary": "response_only", "field": "target",
                     "eligible": eligible and result >= 0},
            "receipt": {"original_tactic_sha256": event["tactic_sha256"],
                        "result": "kernel_certified_replay" if result == 1 else
                                  "known_nonclosure" if result == 0 else "censored",
                        "source_case_id": case.root},
            "producer": f"ppt-{model.cfg.plan}-online",
            "fast_weights_before": before,
        }
        outcomes.append(row)
        if result == 1:
            sft.append({
                **row, "target": {"chosen_tactic": case.candidates[chosen]},
                "loss": {"boundary": "response_only", "field": "target",
                         "eligible": eligible},
            })
    return {
        "ledger": ledger, "outcomes": outcomes, "sft": sft,
        "fast_weights": fast.weights.detach().cpu().tolist(),
        "weight_hash": state_hash(fast),
        "updates": fast.updates, "processed": len(selected),
        "duplicate_lineages_skipped": dropped,
    }


def _write_rows(path: Path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(canonical(row) + "\n")


def collect(model: PlasticProofTransformer, cases: list[Case], source: dict, checkpoint: Path,
            out: Path, *, maximum: int, depth: int, learning_rate: float, score_scale: float):
    if source.get("split") != "train":
        raise ValueError("only train sources are eligible for accretion")
    result = build_ledger(model, cases, maximum=maximum, depth=depth,
                          learning_rate=learning_rate, score_scale=score_scale)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name in ("ledger", "outcomes", "sft"):
        path = out / f"{name}.jsonl"
        _write_rows(path, result[name])
        artifacts[path.name] = {"sha256": sha256_file(path), "rows": len(result[name])}
    snapshot = {
        "schema": "ppt_gym.fast_memory.v1", "weights_fp32": result["fast_weights"],
        "weights_sha256": result["weight_hash"], "updates": result["updates"],
        "model_checkpoint_sha256": sha256_file(checkpoint),
        "source_manifest_sha256": source["manifest_sha256"],
        "mutable_inference_memory": True, "training_eligible": source["scientific_eligible"],
    }
    save = out / "fast_memory.json"
    save.write_text(canonical(snapshot) + "\n", encoding="utf-8")
    artifacts[save.name] = {"sha256": sha256_file(save), "rows": 1}
    report = {
        "schema": "ppt_gym.accretion.v1", "model": model.architecture(),
        "depth": depth, "learning_rate": learning_rate, "score_scale": score_scale,
        "maximum": maximum, "source": source, "checkpoint_sha256": sha256_file(checkpoint),
        "processed_roots": result["processed"], "definitive_updates": result["updates"],
        "duplicate_lineages_skipped": result["duplicate_lineages_skipped"],
        "final_weight_sha256": result["weight_hash"],
        "artifacts": artifacts, "new_lean_kernel_calls": 0,
        "scientific_eligible": source["scientific_eligible"],
        "metric": "retrospective online fast-weight updates on train-only Lean tactic receipts",
    }
    (out / "manifest.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def audit(model: PlasticProofTransformer, cases: list[Case], source: dict, checkpoint: Path, folder: Path):
    report = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if report.get("schema") != "ppt_gym.accretion.v1":
        raise ValueError("unknown accretion schema")
    if report["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("checkpoint changed since online session")
    if report["source"]["manifest_sha256"] != source["manifest_sha256"]:
        raise ValueError("source manifest changed since online session")
    for name, info in report["artifacts"].items():
        if name != Path(name).name or sha256_file(folder / name) != info["sha256"]:
            raise ValueError(f"accretion artifact differs: {name}")
    reproduced = build_ledger(
        model, cases, maximum=report["maximum"], depth=report["depth"],
        learning_rate=report["learning_rate"], score_scale=report["score_scale"],
    )
    for name in ("ledger", "outcomes", "sft"):
        expected = [json.loads(s) for s in (folder / f"{name}.jsonl").read_text().splitlines()]
        if expected != reproduced[name]:
            raise ValueError(f"online history differs: {name}")
    snapshot = json.loads((folder / "fast_memory.json").read_text(encoding="utf-8"))
    if snapshot.get("schema") != "ppt_gym.fast_memory.v1" or snapshot["weights_fp32"] != reproduced["fast_weights"]:
        raise ValueError("online memory snapshot differs")
    if snapshot["weights_sha256"] != reproduced["weight_hash"]:
        raise ValueError("online weight hash differs")
    if report["final_weight_sha256"] != reproduced["weight_hash"]:
        raise ValueError("accretion receipt final hash differs")
    return {"replay_passed": True, "roots": reproduced["processed"],
            "definitive_updates": reproduced["updates"], "weight_sha256": reproduced["weight_hash"],
            "source_manifest_sha256": source["manifest_sha256"],
            "new_lean_kernel_calls": 0}
