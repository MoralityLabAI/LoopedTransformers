"""LSA-02F: root-paired 384-byte pretty-dose and uniform-menu comparison.

Scores exactly 32 independent new Lean calibration theorem roots. Source
lineages and user-specified seed rosters are immutable across arms. Each
checkpoint was trained on one CPU worker with exact duplicate fits.
Uniform-menu baseline uses separately sampled root-specific public-only seeds
and independent real Lean kernel plus standalone script compilation.
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
from .typed_state import PRETTY_DOSES, _public, _type_entries, encode_public_packet, schema_for_view
from .dose_train import SCHEMA as TRAIN_SCHEMA, SEEDS, VIEWS
from .uniform_baseline import SCHEMA as RANDOM_SCHEMA, RANDOM_SEEDS


SCHEMA = "lsa02f.native_dosage_comparison.v1"
BOOTSTRAP_DRAWS = 2000


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _auth_files(directory: Path, refs: dict[str, str]):
    for filename, sha in refs.items():
        if Path(filename).name != filename or sha256_file(directory / filename) != sha:
            raise ValueError("native result artifact hash changed: " + filename)


def _bootstrap(values: list[float]) -> list[float]:
    n = len(values)
    if n < 2:
        raise ValueError("two or more independent theorem roots required")
    rng = random.Random(20261127)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return [means[int(.025 * BOOTSTRAP_DRAWS)],
            means[int(.975 * BOOTSTRAP_DRAWS) - 1]]


def _coverage(frames: list[dict]) -> dict:
    measures = {}
    for view in VIEWS:
        lens, pretty, candidate_seen, candidate_count = [], [], 0, 0
        for frame in frames:
            value = encode_public_packet(
                frame["goal"], frame["candidates"], view=view, max_bytes=384,
            )
            lens.append(len(value.encode("utf-8")))
            if view in PRETTY_DOSES:
                prefix, suffix = value.split("P:", 1)
                excerpt = suffix.split("|", 1)[0]
                pretty.append(len(excerpt.encode("utf-8")))
                goal = _public(frame["goal"])[0][0]
                names = {
                    name: (pre, head) for name, pre, head in _type_entries(goal)
                }
                for action in frame["candidates"]:
                    if not action.startswith("apply "):
                        continue
                    name = action[6:]
                    pre, head = names[name]
                    exact = (
                        name + ":" + ("+".join(pre) + ">" if pre else "")
                        + head + ";"
                    )
                    candidate_count += 1
                    candidate_seen += int(exact in prefix)
        measures[view] = {
            "frames": len(frames),
            "mean_public_bytes": sum(lens) / len(lens),
            "raw_over_384": sum(n > 384 for n in lens),
            "mean_effective_pretty_bytes": (
                sum(pretty) / len(pretty) if pretty else None
            ),
            "candidate_types_visible": candidate_seen if pretty else None,
            "candidate_types_total": candidate_count if pretty else None,
            "dosage_cap": PRETTY_DOSES.get(view),
        }
        if pretty and (
            any(n > 384 for n in lens)
            or any(n > PRETTY_DOSES[view] for n in pretty)
            or candidate_seen != candidate_count
        ):
            raise ValueError("public pretty-dose encoder lost candidate relations or byte cap")
    return measures


def analyze(teacher: Path, fits: Path, probes: Path, uniform: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("paired result destination must be fresh")
    torch.set_num_threads(1)
    frames, source = _load_teacher(teacher)
    teacher_sha = sha256_file(teacher / "manifest.json")
    trained = _json(fits / "manifest.json")
    if not (
        trained.get("schema") == TRAIN_SCHEMA
        and trained.get("teacher_manifest_sha256") == teacher_sha
        and trained.get("source_manifest_sha256") == source["source"]["suite_manifest_sha256"]
        and trained.get("view_roster") == list(VIEWS)
        and trained.get("seed_roster") == list(SEEDS)
        and trained.get("same_worker") is True
        and trained.get("fully_matched_optimizer_samples") is True
        and trained.get("same_seed_initial_weights") is True
        and trained.get("all_duplicate_fits_reproduced") is True
        and trained.get("single_torch_thread") is True
        and trained.get("torch_deterministic") is True
        and trained.get("mkldnn_enabled") is False
        and trained.get("bytes_budget") == 384
        and trained.get("optimizer_steps") == 720
    ):
        raise ValueError("registered one-worker dosage training custody mismatch")
    trainer_cells = {(c["seed"], c["view"]): c for c in trained["cells"]}
    if len(trainer_cells) != len(SEEDS) * len(VIEWS):
        raise ValueError("missing or duplicated same-worker cell")
    all_roots = []
    by_key = {}
    cells = []
    for seed in SEEDS:
        init_roster = set()
        for view in VIEWS:
            key = f"{seed}-{view}"
            cell = trainer_cells[(seed, view)]
            folder = fits / "fits" / key
            checkpoint_path = folder / "checkpoint.pt"
            model, checkpoint = checkpoint_load(checkpoint_path)
            fit_receipt = _json(folder / "manifest.json")
            checkpoint_sha = sha256_file(checkpoint_path)
            if not (
                checkpoint_sha == cell["checkpoint_sha256"] == fit_receipt["checkpoint_sha256"]
                and checkpoint["native_teacher_sha256"] == teacher_sha
                and checkpoint["source_manifest_sha256"] == source["source"]["suite_manifest_sha256"]
                and checkpoint.get("input_view") == view
                and checkpoint.get("input_view_schema") == schema_for_view(view)
                and checkpoint.get("train_order") == "ab"
                and checkpoint.get("seed") == seed
                and fit_receipt.get("optimizer_steps") == 720
                and cell.get("same_worker_replicated") is True
                and model.cfg.max_bytes == 384
                and model.parameter_count() == 57475
            ):
                raise ValueError("trained dosage model provenance mismatches checkpoint")
            init_roster.add(cell["initial_weights_sha256"])
            base = probes / key
            pdir = base / "lsa02f-probe"
            probe = _json(pdir / "manifest.json")
            replay = _json(base / "lsa02f-replay" / "replay.json")
            _auth_files(pdir, probe["artifacts"])
            if not (
                probe.get("schema") == "lsa02b.native_block_order.v1"
                and probe.get("checkpoint_sha256") == checkpoint_sha
                and probe.get("input_view") == view
                and probe.get("paired_roots") == 32
                and probe.get("split") == "calibration"
                and probe.get("standalone_reference_certificates") == 32
                and probe.get("backend", {}).get("actual_lean_process") is True
                and probe.get("training_provenance", {}).get("training_source_verified") is True
                and probe.get("training_provenance", {}).get("source_lineages_disjoint") is True
                and probe.get("training_provenance", {}).get("native_teacher_manifest_sha256") == teacher_sha
                and probe.get("query_label_training_updates") == 0
                and probe.get("architecture_promotion_eligible") is False
                and replay.get("replay_passed") is True
                and replay.get("independent_replay_episodes") == 96
                and replay.get("session_manifest_sha256") == sha256_file(pdir / "manifest.json")
            ):
                raise ValueError("Lean native proof replay or source receipt invalid")
            rows = _rows(pdir / "paired_roots.jsonl")
            if len(rows) != 32 or len({r["lineage"] for r in rows}) != 32:
                raise ValueError("calibration roots repeated inside a cell")
            all_roots.append({r["root"] for r in rows})
            by_key[key] = {r["root"]: r for r in rows}
            model.eval()
            score = training_fork_check(
                model, frames, view, int(fit_receipt["depth"]),
            )
            cells.append({
                "seed": seed, "view": view,
                "checkpoint_sha256": checkpoint_sha,
                "trained_AB_closures": probe["scores"]["ab"]["standalone_verified_closures"],
                "frozen_BA_closures": probe["scores"]["ba"]["standalone_verified_closures"],
                "auto8_closures": probe["scores"]["auto8"]["standalone_verified_closures"],
                "native_attainable": probe["reference_reachable_within_budget"],
                "train_fork_accuracy_root_mean": score["trained_fork_accuracy_root_mean"],
                "train_random_menu_optimal_baseline_root_mean": score["random_menu_baseline_root_mean"],
                "tactic_checks": probe["scores"]["ab"]["tactic_checks"],
                "same_worker_fit_verified": True,
            })
        if len(init_roster) != 1:
            raise ValueError("different initial Plan B tensors within a training seed")
    if any(v != all_roots[0] for v in all_roots[1:]):
        raise ValueError("calibration roots differ between dosages or seeds")

    random_manifest = _json(uniform / "manifest.json")
    random_replay = _json(uniform / "replay" / "replay.json")
    _auth_files(uniform, random_manifest["artifacts"])
    if not (
        random_manifest.get("schema") == RANDOM_SCHEMA
        and random_manifest.get("source_teacher_sha256") == teacher_sha
        and random_manifest.get("seed_roster") == list(RANDOM_SEEDS)
        and random_manifest.get("independent_calibration_roots") == 32
        and random_manifest.get("backend", {}).get("actual_lean_process") is True
        and random_manifest.get("all_replays_passed") is True
        and random_manifest.get("native_model_calls") == 0
        and random_manifest.get("query_updates") == 0
        and random_manifest.get("heldout_test_opened") is False
        and random_replay.get("all_replays_passed") is True
        and random_replay.get("verified_episodes") == 32 * len(RANDOM_SEEDS)
        and random_replay.get("session_manifest_sha256") == sha256_file(uniform / "manifest.json")
    ):
        raise ValueError("native uniform-menu control or independent replay ineligible")
    baseline = _rows(uniform / "root_scores.jsonl")
    if len(baseline) != 32 or {r["root"] for r in baseline} != all_roots[0]:
        raise ValueError("uniform-menu baseline used a different theorem roster")
    baseline_by_root = {row["root"]: row for row in baseline}
    root_deltas = []
    for root in sorted(all_roots[0]):
        matched = []
        for seed in SEEDS:
            original = by_key[f"{seed}-raw"][root]
            observed = {}
            for view in VIEWS:
                row = by_key[f"{seed}-{view}"][root]
                if (
                    row["lineage"] != original["lineage"]
                    or row["reference_budget_attainable"] != original["reference_budget_attainable"]
                    or row["auto8_certified"] != original["auto8_certified"]
                ):
                    raise ValueError("native same-root oracle or public-order control diverged")
                observed[view] = int(row["ab_certified"])
            matched.append({"seed": seed, "view_closures": observed})
        root_deltas.append({
            "root": root, "lineage": original["lineage"],
            "seed_views": matched,
            "reference_reachable": bool(original["reference_budget_attainable"]),
            "random_mean_closure": baseline_by_root[root]["mean_seed_closure"],
            "pp96_minus_pp0": sum(
                r["view_closures"]["pp96"] - r["view_closures"]["pp0"] for r in matched
            ) / len(matched),
            "pp96_minus_raw": sum(
                r["view_closures"]["pp96"] - r["view_closures"]["raw"] for r in matched
            ) / len(matched),
            "pp48_minus_pp0": sum(
                r["view_closures"]["pp48"] - r["view_closures"]["pp0"] for r in matched
            ) / len(matched),
            "pp160_minus_pp0": sum(
                r["view_closures"]["pp160"] - r["view_closures"]["pp0"] for r in matched
            ) / len(matched),
        })
    per_view = {
        view: sum(c["trained_AB_closures"] for c in cells if c["view"] == view)
        / (len(SEEDS) * len(root_deltas))
        for view in VIEWS
    }
    contrasts = {}
    for name in ("pp96_minus_pp0", "pp96_minus_raw", "pp48_minus_pp0", "pp160_minus_pp0"):
        values = [row[name] for row in root_deltas]
        contrasts[name] = {
            "mean": sum(values) / len(values),
            "root_cluster_bootstrap95_descriptive": _bootstrap(values),
        }
    random_close = random_manifest["uniform_mean_closure_rate"]
    if abs(random_close - sum(r["random_mean_closure"] for r in root_deltas)/32) > 1e-12:
        raise ValueError("uniform-menu root-mean and manifest closures disagree")
    primary = contrasts["pp96_minus_pp0"]["mean"]
    screen = (
        primary >= .05
        and all(
            next(c["trained_AB_closures"] for c in cells if c["seed"] == seed and c["view"] == "pp96")
            >= next(c["trained_AB_closures"] for c in cells if c["seed"] == seed and c["view"] == "pp0")
            for seed in SEEDS
        )
    )
    best = max(per_view.values())
    competence = bool(
        best >= .40 and best - random_close >= .10
        and source["independent_lineages"] >= 32
    )
    report = {
        "schema": SCHEMA,
        "teacher_manifest_sha256": teacher_sha,
        "trainer_manifest_sha256": sha256_file(fits / "manifest.json"),
        "uniform_manifest_sha256": sha256_file(uniform / "manifest.json"),
        "native_training_roots": source["independent_lineages"],
        "native_training_frames": len(frames),
        "calibration_unique_roots": len(root_deltas),
        "view_roster": list(VIEWS),
        "seed_roster": list(SEEDS),
        "random_roster": list(RANDOM_SEEDS),
        "raw_and_dose_input_coverage": _coverage(frames),
        "cells": cells,
        "native_close_rate_by_view": per_view,
        "random_native_close_rate": random_close,
        "random_native_per_seed_closures": random_manifest["per_seed_closures"],
        "native_attainable_roots": sum(r["reference_reachable"] for r in root_deltas),
        "contrasts": contrasts,
        "primary_predeclared_screen_passed": screen,
        "formal_close_and_random_competence_floor_met": competence,
        "formal_competence_gate": (
            "CANDIDATE_DESERVES_NEW_SEALED_CONFIRMATION" if competence
            else "BLOCKED_INCOMPETENT_CHECKPOINT"
        ),
        "root_paired_effects": root_deltas,
        "independent_root_unit": "one theorem lineage with paired seeds",
        "no_test_or_ood_opened": True,
        "no_query_label_fit_updates": True,
        "scientific_eligible": False,
        "architecture_promotion_eligible": False,
        "caveat": "Exploratory developmental dosage curve and three uniform-menu seeds; no FSI-01 confirmatory holdout",
    }
    out.mkdir(parents=True)
    (out / "matched_dose.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--uniform", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    result = analyze(args.teacher, args.fits, args.probes, args.uniform, args.out)
    print(canonical({
        "roots": result["calibration_unique_roots"],
        "views": result["native_close_rate_by_view"],
        "uniform": result["random_native_close_rate"],
        "contrasts": result["contrasts"],
        "screen": result["primary_predeclared_screen_passed"],
        "competence": result["formal_competence_gate"],
    }))
    return result


if __name__ == "__main__":
    main()
