"""LSA-02C matched raw/typed CPU study: read-only native-root aggregation.

The independent unit is the theorem lineage. Seeds repeat the *same* roots.
This is a development-only paired effect estimate, not a significance test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch

from .cli import checkpoint_load
from .data import canonical, sha256_file
from .native_train import _load_teacher
from .typed_state import encode_public_packet


SCHEMA = "lsa02c.matched_public_view.v1"
SEEDS = (107, 109)
VIEWS = ("raw", "typed")
DRAWS = 2000


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check_artifacts(directory: Path, claimed: dict[str, str]) -> None:
    for filename, digest in claimed.items():
        if Path(filename).name != filename or sha256_file(directory / filename) != digest:
            raise ValueError("frozen native proof receipt hash differs: " + filename)


def _interval(values: list[float], seed: int = 103) -> list[float]:
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(DRAWS)
    )
    return [means[int(DRAWS * 0.025)], means[int(DRAWS * 0.975) - 1]]


@torch.no_grad()
def training_fork_check(model, frames: list[dict], view: str, depth: int) -> dict:
    """Root-weighted strictly discriminating training-fork accuracy."""
    groups: dict[str, list[int]] = defaultdict(list)
    baselines: dict[str, list[float]] = defaultdict(list)
    for row in frames:
        labels = row["labels"]
        if len(labels) < 2 or not (0 < sum(labels) < len(labels)):
            continue
        text = encode_public_packet(
            row["goal"], row["candidates"], view=view,
            max_bytes=model.cfg.max_bytes,
        )
        out = model(
            text, row["candidates"], depth=depth,
            checks_remaining=row["checks_remaining"],
            middle_order="ab",
        )[-1]["logits"]
        chosen = int(out.argmax())
        groups[row["lineage"]].append(int(labels[chosen] == 1))
        baselines[row["lineage"]].append(sum(labels) / len(labels))
    roots = sorted(groups)
    if not roots:
        raise ValueError("no discriminating training forks in native source")
    return {
        "independent_fork_roots": len(roots),
        "train_fork_frames": sum(map(len, groups.values())),
        "trained_fork_accuracy_root_mean": sum(
            sum(groups[root]) / len(groups[root]) for root in roots
        ) / len(roots),
        "random_menu_baseline_root_mean": sum(
            sum(baselines[root]) / len(baselines[root]) for root in roots
        ) / len(roots),
    }


def aggregate(teacher: Path, fits: Path, probes: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("report output must be new")
    frames, source = _load_teacher(teacher)
    teacher_sha = sha256_file(teacher / "manifest.json")
    checked = {}
    root_sets = []
    for seed in SEEDS:
        initial = {}
        for view in VIEWS:
            key = f"{seed}-{view}"
            cell = fits / key
            fit_manifest = _json(cell / "manifest.json")
            model, checkpoint = checkpoint_load(cell / "checkpoint.pt")
            source_check = (
                fit_manifest.get("native_teacher_manifest_sha256") == teacher_sha
                and checkpoint.get("native_teacher_sha256") == teacher_sha
                and fit_manifest.get("checkpoint_sha256") == sha256_file(cell / "checkpoint.pt")
                and fit_manifest.get("source_manifest_sha256")
                == source["source"]["suite_manifest_sha256"]
                and checkpoint.get("source_manifest_sha256") == source["source"]["suite_manifest_sha256"]
                and fit_manifest.get("training_source_native") is True
                and fit_manifest.get("optimizer_steps") == 720
                and fit_manifest.get("input_view") == view
                and checkpoint.get("input_view") == view
                and checkpoint.get("seed") == seed
                and fit_manifest.get("seed") == seed
                and checkpoint.get("train_order") == "ab"
                and fit_manifest.get("train_order") == "ab"
                and model.cfg.max_bytes == 384
            )
            if not source_check:
                raise ValueError("source and fitted checkpoint provenance mismatch")
            initial[view] = fit_manifest["initial_weights_sha256"]
            base = probes / key
            probe_dir = base / "lsa02c-probe"
            probe = _json(probe_dir / "manifest.json")
            replay = _json(base / "lsa02c-replay" / "replay.json")
            _check_artifacts(probe_dir, probe["artifacts"])
            if (
                probe["checkpoint_sha256"] != fit_manifest["checkpoint_sha256"]
                or probe.get("input_view") != view
                or probe.get("split") != "calibration"
                or probe.get("paired_roots") != 32
                or probe.get("standalone_reference_certificates") != 32
                or probe.get("backend", {}).get("actual_lean_process") is not True
                or probe.get("training_provenance", {}).get("native_teacher_manifest_sha256") != teacher_sha
                or probe.get("training_provenance", {}).get("source_lineages_disjoint") is not True
                or probe.get("query_label_training_updates") != 0
                or not replay.get("replay_passed")
                or replay.get("independent_replay_episodes") != 32 * 3
            ):
                raise ValueError("native Lean evaluation/replay custody or split mismatch")
            rows = _records(probe_dir / "paired_roots.jsonl")
            if len(rows) != 32 or len({row["lineage"] for row in rows}) != 32:
                raise ValueError("native calibration contains repeated theorem lineages")
            root_sets.append({row["root"] for row in rows})
            model.eval()
            checked[key] = {
                "seed": seed, "input_view": view,
                "fit": {
                    "checkpoint_sha256": fit_manifest["checkpoint_sha256"],
                    "initial_weights_sha256": initial[view],
                    "mean_raw_public_bytes": fit_manifest["mean_raw_public_bytes"],
                    "mean_view_public_bytes": fit_manifest["mean_view_public_bytes"],
                    "raw_input_over_budget": fit_manifest["raw_input_over_budget"],
                    "view_input_over_budget": fit_manifest["view_input_over_budget"],
                    "conflicting_training_view_collisions": fit_manifest["conflicting_training_view_collisions"],
                    "last_loss": fit_manifest["last_training_loss"],
                },
                "training_forks": training_fork_check(
                    model, frames, view, fit_manifest["depth"],
                ),
                "native": {
                    "closure": probe["scores"]["ab"]["standalone_verified_closures"],
                    "oracle_ceiling": probe["reference_reachable_within_budget"],
                    "auto8": probe["scores"]["auto8"]["standalone_verified_closures"],
                    "tactic_checks": probe["scores"]["ab"]["tactic_checks"],
                    "first_tactic_disagreements_with_ba": probe["first_tactic_disagreements"],
                },
                "roots": {row["root"]: row for row in rows},
            }
        if initial["raw"] != initial["typed"]:
            raise ValueError("same-seed raw and typed models have different starting tensors")
    if any(roots != root_sets[0] for roots in root_sets[1:]):
        raise ValueError("representation arms used different calibration theorem sets")

    root_changes: list[dict] = []
    for root in sorted(root_sets[0]):
        results = []
        for seed in SEEDS:
            a = checked[f"{seed}-raw"]["roots"][root]
            b = checked[f"{seed}-typed"]["roots"][root]
            if (
                a["lineage"] != b["lineage"]
                or a["reference_budget_attainable"] != b["reference_budget_attainable"]
                or a["auto8_certified"] != b["auto8_certified"]
            ):
                raise ValueError("paired root oracle/control evidence differs across representations")
            results.append(int(b["ab_certified"]) - int(a["ab_certified"]))
        root_changes.append({"root": root, "mean_seed_delta": sum(results) / len(results),
                             "seed_deltas": results})
    units = [r["mean_seed_delta"] for r in root_changes]
    details = []
    for seed in SEEDS:
        for view in VIEWS:
            value = checked[f"{seed}-{view}"]
            details.append({k: v for k, v in value.items() if k != "roots"})
    report = {
        "schema": SCHEMA,
        "source_evidence": "native_lean_shortest_path_teacher_with_standalone_proofs",
        "teacher_manifest_sha256": teacher_sha,
        "train_lineages": source["independent_lineages"],
        "train_frames": len(frames),
        "calibration_unique_roots": len(units),
        "matched_training_seeds": list(SEEDS),
        "same_initial_weights_per_seed": True,
        "source_bytes_budget": 384,
        "trained_order": "ab",
        "input_views": list(VIEWS),
        "cells": details,
        "root_paired_effects": root_changes,
        "typed_minus_raw_proof_closure_rate": sum(units) / len(units),
        "descriptive_root_bootstrap_95": _interval(units),
        "scientific_eligible": False,
        "locked_evaluation_opened": False,
        "trained_competence_gate": "DEVELOPMENT_ONLY; formal registered 32-root/random-menu gate pending",
        "conclusion_boundary": (
            "Paired fresh development roots and two common-seed CPU fits; "
            "descriptive proof effects only, no architecture promotion or Mathlib transfer"
        ),
    }
    out.mkdir(parents=True)
    (out / "matched_views.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = aggregate(args.teacher, args.fits, args.probes, args.out)
    print(canonical({
        "source": report["teacher_manifest_sha256"],
        "train_frames": report["train_frames"],
        "calibration_unique_roots": report["calibration_unique_roots"],
        "typed_minus_raw_closure_rate": report["typed_minus_raw_proof_closure_rate"],
        "root_bootstrap_95": report["descriptive_root_bootstrap_95"],
        "cells": report["cells"],
    }))
    return report


if __name__ == "__main__":
    main()
