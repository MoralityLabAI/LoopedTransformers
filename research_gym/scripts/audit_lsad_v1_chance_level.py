"""Audit sealed LSAD v1 cells against uniform and best-constant baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_gym.analysis.lsad_chance_level import (
    ANSWER_RANGES,
    binomial_item_sd,
    family_baselines,
    prediction_profile,
    target_entropy,
)
from research_gym.integrity import canonical_file_sha256, verify_file_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "experiments" / "loop_schedule_architecture_discovery_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _train_entropy(experiment: Path) -> dict[str, float]:
    by_family: dict[str, list[int]] = {family: [] for family in ANSWER_RANGES}
    for row in _rows(experiment / "datasets" / "train.jsonl"):
        if row["family"] in by_family:
            by_family[row["family"]].append(int(row["target_token"]))
    entropies = {family: target_entropy(targets) for family, targets in by_family.items()}
    entropies["mean"] = sum(entropies.values()) / len(entropies)
    return entropies


def build_audit(experiment: Path) -> dict[str, Any]:
    manifest_path = experiment / "datasets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = experiment / "datasets" / manifest["files"]["train"]["path"]
    if not verify_file_sha256(train_path, manifest["files"]["train"]["sha256"]):
        raise ValueError("train shard hash mismatch")
    entropy = _train_entropy(experiment)
    sequence_length = int(manifest["sequence_length"])

    baselines: dict[str, dict[str, Any]] | None = None
    scored_ids: tuple[str, ...] | None = None
    cells = []
    for result_path in sorted((experiment / "runs").glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            continue
        artifact = result["prediction_artifacts"][str(result["train_visits"])]
        prediction_path = ROOT / artifact["path"]
        if not verify_file_sha256(prediction_path, artifact["sha256"]):
            raise ValueError(f"prediction hash mismatch: {result['cell_id']}")
        rows = _rows(prediction_path)
        ids = tuple(row["example_id"] for row in rows)
        if scored_ids is None:
            scored_ids = ids
            baselines = family_baselines(rows)
        elif ids != scored_ids:
            raise ValueError("cells were scored on different examples")
        profile = prediction_profile(rows)
        total = sum(value["examples"] for value in profile.values())
        cells.append(
            {
                "cell_id": result["cell_id"],
                "stage": result["stage"],
                "seed": int(result["seed"]),
                "optimizer_steps": int(result["optimizer_steps"]),
                "examples_seen": int(result["token_visit_exposures"])
                // (sequence_length * int(result["train_visits"])),
                "final_training_loss": float(result["final_loss"]),
                "loss_minus_target_entropy": float(result["final_loss"]) - entropy["mean"],
                "macro_exact": float(result["macro_exact"]),
                "correct": sum(value["correct"] for value in profile.values()),
                "total": total,
                "in_answer_range_share": sum(
                    value["in_answer_range"] for value in profile.values()
                )
                / total,
                "by_family": profile,
                "prediction_sha256": artifact["sha256"],
            }
        )
    if baselines is None or not cells:
        raise ValueError("no completed cells with prediction artifacts were found")

    families = len(baselines)
    uniform_macro = sum(value["uniform_chance"] for value in baselines.values()) / families
    constant_macro = sum(value["best_constant"] for value in baselines.values()) / families
    uniform_sd = binomial_item_sd(baselines, rate="uniform_chance")
    uniform_correct = sum(
        value["examples"] * value["uniform_chance"] for value in baselines.values()
    )
    for cell in cells:
        cell["z_versus_uniform"] = (cell["correct"] - uniform_correct) / uniform_sd
    return {
        "schema_version": "lsad-v1-chance-level-audit-v1",
        "sources": {
            "manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": canonical_file_sha256(manifest_path),
            },
            "train": {
                "path": train_path.relative_to(ROOT).as_posix(),
                "sha256": manifest["files"]["train"]["sha256"],
            },
        },
        "train_target_entropy_nats": entropy,
        "scored_subset": baselines,
        "baselines": {
            "uniform_macro_exact": uniform_macro,
            "best_constant_macro_exact": constant_macro,
            "uniform_expected_correct": uniform_correct,
            "uniform_item_sd": uniform_sd,
        },
        "cells": cells,
        "summary": {
            "cells": len(cells),
            "cells_below_best_constant": sum(
                cell["macro_exact"] < constant_macro for cell in cells
            ),
            "cells_with_loss_below_target_entropy": sum(
                cell["loss_minus_target_entropy"] < 0.0 for cell in cells
            ),
            "max_abs_z_versus_uniform": max(abs(cell["z_versus_uniform"]) for cell in cells),
            "max_examples_seen": max(cell["examples_seen"] for cell in cells),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    output = args.output or experiment / "chance_level_audit.json"
    audit = build_audit(experiment)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({**audit["baselines"], **audit["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
