"""LSA-02E: hash-custodied root-paired native Lean comparison of 3 public views.

Only the training teacher and 32 predeclared development roots are accessible.
A theorem lineage is the unit; repeated seeds reuse the same theorem, and
descriptive root bootstraps sample unique roots, never per-episode rows.
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
from .typed_compare import training_fork_check
from .typed_state import PRETTY_SCHEMA, _public, encode_public_packet, schema_for_view
from .pretty_train import SCHEMA as TRAIN_SCHEMA, SEEDS, VIEWS_PAIRED


SCHEMA = "lsa02e.paired_native_public_views.v1"
DRAWS = 2000


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _bootstrap(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("no independent theorem roots for bootstrap")
    rng = random.Random(113127)
    n = len(values)
    sorted_means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(DRAWS)
    )
    return [sorted_means[int(.025 * DRAWS)], sorted_means[int(.975 * DRAWS) - 1]]


def _auth_files(directory: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        if name != Path(name).name or sha256_file(directory / name) != expected:
            raise ValueError("native proof artifact SHA-256 differs from manifest: " + name)


def _source_view_coverage(frames: list[dict]) -> dict:
    count = {"frames": len(frames), "source_pretty_nonempty": 0,
             "hybrid_pretty_excerpt_nonempty": 0, "hybrid_candidate_types_present": 0,
             "public_candidate_types_total": 0, "hybrid_within_384": 0}
    for frame in frames:
        text = encode_public_packet(
            frame["goal"], frame["candidates"], view="typed_pp", max_bytes=384,
        )
        goal = _public(frame["goal"])[0][0]
        pp = goal.get("pp", "")
        count["source_pretty_nonempty"] += int(bool(pp and pp.strip()))
        # The typed+pretty encoder's mandatory C:name:<public type>; section
        # precedes the bounded P: text. This checks every candidate relation.
        front = text.split("P:", 1)[0]
        candidate_names = [
            a[6:] for a in frame["candidates"] if a.startswith("apply ")
        ]
        count["public_candidate_types_total"] += len(candidate_names)
        count["hybrid_candidate_types_present"] += sum(
            ("C:" + name + ":") in front for name in candidate_names
        )
        if "P:" not in text:
            raise ValueError("typed+pretty view lost public-text separator")
        pp_excerpt = text.split("P:", 1)[1].split("|", 1)[0]
        count["hybrid_pretty_excerpt_nonempty"] += int(bool(pp_excerpt and pp_excerpt != "_"))
        count["hybrid_within_384"] += int(len(text.encode("utf-8")) <= 384)
    if count["hybrid_candidate_types_present"] != count["public_candidate_types_total"]:
        raise ValueError("typed+pretty view dropped a mandatory public candidate type")
    if count["hybrid_within_384"] != len(frames):
        raise ValueError("typed+pretty view exceeded byte budget")
    return count


def aggregate(teacher: Path, fits: Path, probes: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("output directory must be fresh")
    torch.set_num_threads(1)
    frames, source = _load_teacher(teacher)
    trainer = _read(fits / "manifest.json")
    teacher_sha = sha256_file(teacher / "manifest.json")
    if (
        trainer.get("schema") != TRAIN_SCHEMA
        or trainer.get("teacher_manifest_sha256") != teacher_sha
        or trainer.get("source_manifest_sha256") != source["source"]["suite_manifest_sha256"]
        or trainer.get("view_roster") != list(VIEWS_PAIRED)
        or trainer.get("seed_roster") != list(SEEDS)
        or trainer.get("same_worker") is not True
        or trainer.get("all_duplicate_fits_reproduced") is not True
        or trainer.get("fully_matched_optimizer_samples") is not True
        or trainer.get("same_seed_initial_weights") is not True
        or trainer.get("optimizer_steps") != 720
        or trainer.get("single_torch_thread") is not True
        or trainer.get("torch_deterministic") is not True
        or trainer.get("mkldnn_enabled") is not False
    ):
        raise ValueError("training artifact lacks matched source or same-host deterministic proof")
    cells = {
        (cell["seed"], cell["view"]): cell
        for cell in trainer["cells"]
    }
    if len(cells) != len(SEEDS) * len(VIEWS_PAIRED):
        raise ValueError("training receipt has missing or duplicate model cells")
    samples = {}
    cohort_sets = []
    report_cells = []
    for seed in SEEDS:
        initials = set()
        for view in VIEWS_PAIRED:
            cell = cells[(seed, view)]
            key = f"{seed}-{view}"
            folder = fits / "fits" / key
            fit_meta = _read(folder / "manifest.json")
            checkpoint_path = folder / "checkpoint.pt"
            model, checkpoint = checkpoint_load(checkpoint_path)
            checkpoint_sha = sha256_file(checkpoint_path)
            initials.add(cell["initial_weights_sha256"])
            if not (
                checkpoint_sha == cell["checkpoint_sha256"] == fit_meta["checkpoint_sha256"]
                and checkpoint.get("input_view") == view
                and checkpoint.get("input_view_schema") == schema_for_view(view)
                and checkpoint.get("native_teacher_sha256") == teacher_sha
                and checkpoint.get("source_manifest_sha256") == source["source"]["suite_manifest_sha256"]
                and checkpoint.get("seed") == seed
                and checkpoint.get("train_order") == "ab"
                and fit_meta.get("optimizer_steps") == 720
                and model.cfg.max_bytes == 384
                and model.parameter_count() == 57475
            ):
                raise ValueError("native checkpoint source/view/seed/architecture mismatch")
            root = probes / key
            probedir = root / "lsa02e-probe"
            probe = _read(probedir / "manifest.json")
            replay = _read(root / "lsa02e-replay" / "replay.json")
            _auth_files(probedir, probe["artifacts"])
            if not (
                probe.get("schema") == "lsa02b.native_block_order.v1"
                and probe.get("checkpoint_sha256") == checkpoint_sha
                and probe.get("input_view") == view
                and probe.get("split") == "calibration"
                and probe.get("paired_roots") == 32
                and probe.get("standalone_reference_certificates") == 32
                and probe.get("training_provenance", {}).get("native_teacher_manifest_sha256") == teacher_sha
                and probe.get("training_provenance", {}).get("source_lineages_disjoint") is True
                and probe.get("backend", {}).get("actual_lean_process") is True
                and probe.get("query_label_training_updates") == 0
                and probe.get("scientific_eligible") is False
                and replay.get("replay_passed") is True
                and replay.get("independent_replay_episodes") == 96
                and replay.get("session_manifest_sha256") == sha256_file(probedir / "manifest.json")
            ):
                raise ValueError("native proof or model-free replay custody mismatch")
            rows = _rows(probedir / "paired_roots.jsonl")
            if len(rows) != 32 or len({r["lineage"] for r in rows}) != 32:
                raise ValueError("native calibration has repeated/missing theorem roots")
            cohort_sets.append({row["root"] for row in rows})
            samples[key] = {row["root"]: row for row in rows}
            model.eval()
            stats = training_fork_check(model, frames, view, int(fit_meta["depth"]))
            report_cells.append({
                "seed": seed, "view": view,
                "checkpoint_sha256": checkpoint_sha,
                "fit_start_sha256": cell["initial_weights_sha256"],
                "fit_final_sha256": cell["final_weights_sha256"],
                "same_host_duplicate_passed": cell["same_worker_replicated"],
                "training_mean_final_32_loss": cell["mean_final_32_loss"],
                "train_forks": stats,
                "mean_input_bytes": fit_meta["mean_view_public_bytes"],
                "input_over_budget": fit_meta["view_input_over_budget"],
                "source_collisions": fit_meta["conflicting_training_view_collisions"],
                "calibration_AB_proofs": probe["scores"]["ab"]["standalone_verified_closures"],
                "calibration_BA_proofs": probe["scores"]["ba"]["standalone_verified_closures"],
                "calibration_auto8_proofs": probe["scores"]["auto8"]["standalone_verified_closures"],
                "attainable_references": probe["reference_reachable_within_budget"],
                "calibration_tactic_checks": probe["scores"]["ab"]["tactic_checks"],
                "full_native_replay_passed": True,
            })
        if len(initials) != 1:
            raise ValueError("three views had different seeded initial weights")
    if any(roots != cohort_sets[0] for roots in cohort_sets[1:]):
        raise ValueError("the compared models saw nonidentical calibration roots")

    root_effects = []
    for root in sorted(cohort_sets[0]):
        family = []
        for seed in SEEDS:
            rows = {view: samples[f"{seed}-{view}"][root] for view in VIEWS_PAIRED}
            assert len(rows) == len(VIEWS_PAIRED)
            ref = rows["raw"]
            if any(
                row["lineage"] != ref["lineage"]
                or row["reference_budget_attainable"] != ref["reference_budget_attainable"]
                or row["auto8_certified"] != ref["auto8_certified"]
                for row in rows.values()
            ):
                raise ValueError("native oracle/control mismatches among same-root views")
            family.append({
                "seed": seed,
                "raw": int(rows["raw"]["ab_certified"]),
                "typed": int(rows["typed"]["ab_certified"]),
                "typed_pp": int(rows["typed_pp"]["ab_certified"]),
                "oracle_attainable": bool(ref["reference_budget_attainable"]),
            })
        root_effects.append({
            "root": root,
            "lineage": ref["lineage"],
            "seed_outcomes": family,
            "typed_pp_minus_raw": sum(
                a["typed_pp"] - a["raw"] for a in family
            ) / len(family),
            "typed_pp_minus_typed": sum(
                a["typed_pp"] - a["typed"] for a in family
            ) / len(family),
        })
    main = [row["typed_pp_minus_raw"] for row in root_effects]
    secondary = [row["typed_pp_minus_typed"] for row in root_effects]
    def avg(values):
        return sum(values) / len(values)

    relative = {
        view: sum(
            cell["calibration_AB_proofs"] for cell in report_cells
            if cell["view"] == view
        ) / (len(SEEDS) * len(root_effects))
        for view in VIEWS_PAIRED
    }
    effect_report = {
        "typed_pp_minus_raw": avg(main),
        "typed_pp_minus_raw_root_bootstrap95": _bootstrap(main),
        "typed_pp_minus_typed": avg(secondary),
        "typed_pp_minus_typed_root_bootstrap95": _bootstrap(secondary),
        "minimum_useful_development_effect": 0.05,
        "registered_screen_passed": (
            avg(main) >= .05 and avg(secondary) >= .05
            and all(
                next(c["calibration_AB_proofs"] for c in report_cells if c["view"]=="typed_pp" and c["seed"]==seed)
                >= next(c["calibration_AB_proofs"] for c in report_cells if c["view"]=="raw" and c["seed"]==seed)
                for seed in SEEDS
            )
        ),
    }
    report = {
        "schema": SCHEMA,
        "teacher_manifest_sha256": teacher_sha,
        "trainer_manifest_sha256": sha256_file(fits / "manifest.json"),
        "source_lineages": source["independent_lineages"],
        "source_frames": len(frames),
        "calibration_unique_lineages": len(root_effects),
        "seed_roster": list(SEEDS),
        "view_roster": list(VIEWS_PAIRED),
        "training_same_worker": True,
        "training_repeated_exactly": True,
        "byte_budget": 384,
        "source_pretty_coverage": _source_view_coverage(frames),
        "cells": report_cells,
        "native_closure_rate_by_view": relative,
        "paired_effect": effect_report,
        "root_effects": root_effects,
        "formal_competence_gate": "BLOCKED_UNTIL_REGISTERED_RANDOM_MENU_BASELINE_AND_COMPETENT_PROOF_CLOSURE",
        "architecture_promotion_eligible": False,
        "scientific_eligible": False,
        "heldout_test_opened": False,
        "new_model_training_from_query_labels": 0,
        "independent_root_unit": "one frozen theorem lineage; seeds paired within lineage",
        "interpretation": "Exploratory native-Lean result, no causal-mechanism or model-scale claim",
    }
    out.mkdir(parents=True)
    (out / "matched_views.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    report = aggregate(args.teacher, args.fits, args.probes, args.out)
    print(canonical({
        "native_closure_rate_by_view": report["native_closure_rate_by_view"],
        "paired_effect": report["paired_effect"],
        "calibration_unique_lineages": report["calibration_unique_lineages"],
        "cells": [
            {"seed": cell["seed"], "view": cell["view"],
             "closed": cell["calibration_AB_proofs"],
             "attainable": cell["attainable_references"],
             "train_fork_accuracy": cell["train_forks"]["trained_fork_accuracy_root_mean"]}
            for cell in report["cells"]
        ],
    }))
    return report


if __name__ == "__main__":
    main()
