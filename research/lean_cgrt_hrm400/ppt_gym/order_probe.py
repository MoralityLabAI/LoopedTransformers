"""LSA-02B: paired AB/BA block order intervention on frozen PPT-B weights.

This imports Lean-verified *retrospective* checked-candidate labels or explicitly
marked synthetic fixtures. It never runs the Lean kernel, creates a new proof,
updates model parameters or imports correctness into a model-facing packet.
Scientific claims require the separate checkpoint-competence and registration
gates in docs/lsa02b_block_order_protocol.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from .cli import checkpoint_load
from .data import (
    Case, canonical, ensure_no_overlap, load_checked_pack, load_fixture,
    sha256_file,
)

SCHEMA = "lsa02b.block_order_probe.v1"
BOOTSTRAP_SEED = 20261123
BOOTSTRAP_DRAWS = 2000


def _source(directory: Path, synthetic: bool, split: str, max_cases: int):
    loader = load_fixture if synthetic else load_checked_pack
    return loader(directory, split, max_cases)


def _ranks(values: list[float]) -> list[float]:
    positions = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(positions):
        end = start + 1
        while end < len(positions) and values[positions[end]] == values[positions[start]]:
            end += 1
        average = (start + end + 1) / 2.0
        for item in positions[start:end]:
            ranks[item] = average
        start = end
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    avg_a, avg_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - avg_a) * (y - avg_b) for x, y in zip(a, b))
    den_a = sum((x - avg_a) ** 2 for x in a)
    den_b = sum((y - avg_b) ** 2 for y in b)
    if den_a <= 0 or den_b <= 0:
        return None
    return numerator / math.sqrt(den_a * den_b)


def _interval(differences: list[int]) -> tuple[float, float] | None:
    if len(differences) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        means.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    means.sort()
    return means[int(0.025 * BOOTSTRAP_DRAWS)], means[int(0.975 * BOOTSTRAP_DRAWS) - 1]


@torch.no_grad()
def _prelude_state(model, case: Case) -> torch.Tensor:
    """Reconstruct exactly the Plan-B state at the first middle block input."""
    goal = model.text([case.goal])[0]
    candidates = model.text(list(case.candidates))
    context = model.context(torch.cat([
        goal, candidates.mean(0), goal.new_tensor([4.0 / 8.0]),
    ]))
    state = model.roles.unsqueeze(0) + context[None, None, :]
    state = state.clone()
    state[:, 0] = state[:, 0] + goal
    state[:, 4] = state[:, 4] + candidates.mean(0)
    return model.prelude(model.work_norm(state))


@torch.no_grad()
def _commutator(model, case: Case) -> float:
    """Label-blind normalized block-order difference before message gating."""
    z = _prelude_state(model, case)
    ab = model.middle_b(model.middle_a(z))
    ba = model.middle_a(model.middle_b(z))
    return float((ab - ba).norm() / z.norm().clamp_min(1e-8))


@torch.no_grad()
def _case_probe(model, case: Case, depth: int) -> dict:
    logits = {}
    for order in ("ab", "ba"):
        for message in ("normal", "zero"):
            logits[(order, message)] = model(
                case.goal, case.candidates, depth=depth,
                middle_order=order, message_mode=message,
            )[-1]["logits"].float().detach().cpu()
    normal_ab, normal_ba = logits[("ab", "normal")], logits[("ba", "normal")]
    zero_ab, zero_ba = logits[("ab", "zero")], logits[("ba", "zero")]
    index_ab, index_ba = int(normal_ab.argmax()), int(normal_ba.argmax())
    all_known = all(label in (0, 1) for label in case.labels)
    witness = all_known and any(label == 1 for label in case.labels)
    return {
        "root": case.root, "lineage": case.lineage, "family": case.family,
        "evidence_class": case.evidence,
        "candidate_count": len(case.candidates),
        "all_labels_known": all_known,
        "oracle_has_one_call_witness": witness,
        "ab_index": index_ab, "ba_index": index_ba,
        "ab_label": case.labels[index_ab] if all_known else None,
        "ba_label": case.labels[index_ba] if all_known else None,
        "ab_zero_index": int(zero_ab.argmax()),
        "ba_zero_index": int(zero_ba.argmax()),
        "commutator_norm": _commutator(model, case),
        "normal_logit_gap": float((normal_ab - normal_ba).norm()),
        "zero_logit_gap": float((zero_ab - zero_ba).norm()),
        "ab_message_gap": float((normal_ab - zero_ab).norm()),
        "ba_message_gap": float((normal_ba - zero_ba).norm()),
        "random_one_call_baseline": (
            sum(label == 1 for label in case.labels) / len(case.labels)
            if witness else None
        ),
    }


def run_probe(
    directory: Path,
    checkpoint_path: Path,
    out: Path,
    *,
    split: str = "calibration",
    synthetic: bool = False,
    depth: int = 3,
    max_cases: int = 256,
    gate_manifest: Path | None = None,
) -> dict:
    """Frozen checkpoint, roots, source and message schedule in one paired study."""
    if out.exists():
        raise ValueError("output must be a fresh directory")
    if split not in ("calibration", "test"):
        raise ValueError("only calibration and locked test splits are supported")
    if not 1 <= depth <= 16 or not 1 <= max_cases <= 20000:
        raise ValueError("invalid depth or case bound")
    if split == "test" and gate_manifest is None:
        raise ValueError("locked test requires a successful calibration gate")

    model, checkpoint = checkpoint_load(checkpoint_path)
    if model.cfg.plan != "b":
        raise ValueError("LSA-02B requires PPT Plan B with two middle blocks")
    if checkpoint.get("train_order", "ab") not in ("ab", "ba"):
        raise ValueError("unrecognized checkpoint training order")
    torch.set_num_threads(1)
    model.eval()
    checkpoint_hash = sha256_file(checkpoint_path)
    cases, source = _source(directory, synthetic, split, max_cases)
    if not cases:
        raise ValueError("empty scoring cohort")
    if checkpoint.get("source_manifest_sha256") != source["manifest_sha256"]:
        raise ValueError("checkpoint not bound to the selected source manifest")
    if checkpoint.get("source_evidence_class") != source["evidence_class"]:
        raise ValueError("checkpoint evidence class mismatch")
    support, _ = _source(directory, synthetic, "train", 20000)
    ensure_no_overlap(support, cases)
    if split == "test":
        calibration, _ = _source(directory, synthetic, "calibration", 20000)
        ensure_no_overlap(calibration, cases)
        gate = json.loads(gate_manifest.read_text(encoding="utf-8"))
        if (
            gate.get("schema") != SCHEMA
            or gate.get("split") != "calibration"
            or gate.get("checkpoint_sha256") != checkpoint_hash
            or gate.get("source", {}).get("manifest_sha256") != source["manifest_sha256"]
            or not gate.get("competence_gate_passed")
        ):
            raise ValueError("pre-scored calibration gate missing or incompatible")
    if len({case.root for case in cases}) != len(cases):
        raise ValueError("duplicate goal root")
    if len({case.lineage for case in cases}) != len(cases):
        raise ValueError("nonindependent repeated lineage")

    rows = [_case_probe(model, case, depth) for case in cases]
    if not all(
        math.isfinite(row[k])
        for row in rows
        for k in ("commutator_norm", "normal_logit_gap",
                  "zero_logit_gap", "ab_message_gap", "ba_message_gap")
    ):
        raise FloatingPointError("nonfinite order-geometry measurement")
    scored = [row for row in rows if row["oracle_has_one_call_witness"]]
    n = len(scored)
    pairs = [
        int(row["ab_label"] == 1) - int(row["ba_label"] == 1)
        for row in scored
    ]
    ab_accuracy = sum(row["ab_label"] == 1 for row in scored) / n if n else None
    ba_accuracy = sum(row["ba_label"] == 1 for row in scored) / n if n else None
    random_baseline = (
        sum(row["random_one_call_baseline"] for row in scored) / n if n else None
    )
    trained_order = checkpoint.get("train_order", "ab")
    trained_acc = ab_accuracy if trained_order == "ab" else ba_accuracy
    competence = bool(
        source["scientific_eligible"]
        and checkpoint.get("train_steps", 0) > 0
        and n >= 32
        and trained_acc is not None
        and trained_acc >= 0.40
        and trained_acc >= random_baseline + 0.10
    )
    commutators = [row["commutator_norm"] for row in rows]
    logit_gaps = [row["normal_logit_gap"] for row in rows]
    summary = {
        "schema": SCHEMA,
        "split": split,
        "checkpoint_sha256": checkpoint_hash,
        "source": source,
        "train_order": trained_order,
        "depth": depth,
        "cases": len(cases),
        "eligible_witness_roots": n,
        "unknown_or_no_witness_roots": len(cases) - n,
        "ab_one_call_close_rate": ab_accuracy,
        "ba_one_call_close_rate": ba_accuracy,
        "random_menu_baseline": random_baseline,
        "paired_delta_ab_minus_ba": sum(pairs) / n if n else None,
        "paired_root_bootstrap_95": _interval(pairs),
        "ab_wins": pairs.count(1),
        "ba_wins": pairs.count(-1),
        "ties": pairs.count(0),
        "mean_normal_logit_gap": sum(logit_gaps) / len(rows),
        "mean_commutator_norm": sum(commutators) / len(rows),
        "commutator_vs_logit_gap_spearman": _pearson(
            _ranks(commutators), _ranks(logit_gaps)
        ),
        "score_uses_independent_lineages": True,
        "updates_from_query_labels": 0,
        "new_lean_kernel_calls": 0,
        "competence_gate_passed": competence,
        "scientific_eligible": (
            bool(source["scientific_eligible"] and competence)
            if split == "calibration" else bool(source["scientific_eligible"])
        ),
        "interpretation": (
            "Frozen-checkpoint dispatch intervention; conditional one-checkpoint estimate. "
            "A trained AB-vs-trained BA comparison requires separately fitted matched seeds. "
            "No claim that commutator predicts task success."
        ),
    }
    out.mkdir(parents=True)
    rows_path = out / "paired_roots.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    summary["rows_sha256"] = sha256_file(rows_path)
    (out / "manifest.json").write_text(canonical(summary) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pack", type=Path)
    src.add_argument("--fixture", type=Path)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "test"), default="calibration")
    parser.add_argument("--gate-manifest", type=Path)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=256)
    args = parser.parse_args(argv)
    if args.fixture and not args.allow_synthetic:
        raise ValueError("synthetic fixtures require --allow-synthetic")
    return run_probe(
        args.fixture or args.pack, args.checkpoint, args.out,
        synthetic=bool(args.fixture), split=args.split,
        gate_manifest=args.gate_manifest, depth=args.depth,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    print(canonical(main()))
