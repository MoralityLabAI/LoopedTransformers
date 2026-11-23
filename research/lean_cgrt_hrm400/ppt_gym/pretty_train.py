"""LSA-02E: paired, same-worker CPU raw/typed/typed+pretty training.

Exactly one physical CPU worker trains the view matrix, then independently
repeats each cell to verify tensor and optimizer-trace reproducibility.
The sampler seed, model initialization, native Lean shortest-path source,
optimizer count, width, recurrent pass count, and text budget are fixed.
No Lean model scoring or test-set queries occur in this process.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import torch

from .data import canonical, sha256_file
from .determinism_audit import _compare_checkpoints
from .native_train import _load_teacher, fit
from .typed_state import VIEWS, schema_for_view


SCHEMA = "lsa02e.same_worker_three_view_cpu.v1"
SEEDS = (113, 127)
VIEWS_PAIRED = ("raw", "typed", "typed_pp")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_keys(path: Path) -> list[tuple]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            result.append((item["step"], item["root"], item["teacher_frame_index"]))
    return result


def run(
    teacher: Path, out: Path, *,
    seeds: tuple[int, ...] = SEEDS, steps: int = 720,
    max_seconds_per_fit: float = 300.0,
) -> dict:
    if out.exists():
        raise ValueError("output directory must be fresh")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("duplicate or absent seed")
    if not 1 <= steps <= 5000 or not 0 < max_seconds_per_fit <= 1200:
        raise ValueError("bounded CPU fitting contract violated")
    frames, source = _load_teacher(teacher)
    if not source["source_compiler_passed"] or not source["backend"]["actual_lean_process"]:
        raise ValueError("native independently compiled Lean teacher is required")
    if source["independent_lineages"] < 32 or len(frames) < 32:
        raise ValueError("insufficient native source roots")
    teacher_sha = sha256_file(teacher / "manifest.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    if torch.get_num_threads() != 1:
        raise ValueError("single-thread PyTorch contract did not hold")

    out.mkdir(parents=True)
    cells = []
    for seed in seeds:
        starts = {}
        samplers = {}
        for view in VIEWS_PAIRED:
            if view not in VIEWS:
                raise ValueError("unsupported view roster")
            key = f"{seed}-{view}"
            destination = out / "fits" / key
            duplicate = out / "replicas" / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            receipts = [
                fit(
                    teacher, directory, order="ab", seed=seed,
                    steps=steps, depth=2, width=32, max_bytes=384,
                    threads=1, max_seconds=max_seconds_per_fit,
                    input_view=view,
                )
                for directory in (destination, duplicate)
            ]
            first, repeated = receipts
            if any(
                item.get("optimizer_steps") != steps
                or item.get("source_manifest_sha256") != source["source"]["suite_manifest_sha256"]
                or item.get("native_teacher_manifest_sha256") != teacher_sha
                or item.get("input_view_schema") != schema_for_view(view)
                or item.get("train_order") != "ab"
                or item.get("unique_parameter_count") != 57475
                or item.get("optimizer_calls_on_calibration_or_test") != 0
                for item in receipts
            ):
                raise ValueError("CPU fit violated registered source or parameter parity")
            parity = _compare_checkpoints(
                destination / "checkpoint.pt", duplicate / "checkpoint.pt",
            )
            trace_sha_equal = (
                first["optimizer_steps_sha256"] == repeated["optimizer_steps_sha256"]
            )
            if not (
                parity["all_tensors_bitwise_equal"]
                and parity["full_checkpoint_bytes_equal"]
                and trace_sha_equal
                and first["initial_weights_sha256"] == repeated["initial_weights_sha256"]
                and first["final_weights_sha256"] == repeated["final_weights_sha256"]
            ):
                raise ValueError(f"same-worker duplicate training did not reproduce: {key}")
            keys = _sample_keys(destination / "optimizer_steps.jsonl")
            if keys != _sample_keys(duplicate / "optimizer_steps.jsonl"):
                raise ValueError("repeated CPU trainer sampled different roots")
            starts[view] = first["initial_weights_sha256"]
            samplers[view] = keys
            cells.append({
                "seed": seed, "view": view,
                "checkpoint_sha256": first["checkpoint_sha256"],
                "initial_weights_sha256": first["initial_weights_sha256"],
                "final_weights_sha256": first["final_weights_sha256"],
                "optimizer_step_trace_sha256": first["optimizer_steps_sha256"],
                "optimizer_steps": first["optimizer_steps"],
                "parameters": first["unique_parameter_count"],
                "mean_public_bytes": first["mean_view_public_bytes"],
                "view_over_budget": first["view_input_over_budget"],
                "conflicting_input_label_collisions": first["conflicting_training_view_collisions"],
                "last_loss": first["last_training_loss"],
                "mean_final_32_loss": first["mean_final_32_loss"],
                "same_worker_replicated": True,
                "duplicate_check": parity,
            })
        if len(set(starts.values())) != 1:
            raise ValueError("three views started from different model tensors in the same seed")
        if samplers["raw"] != samplers["typed"] or samplers["raw"] != samplers["typed_pp"]:
            raise ValueError("three views had different source-root and frame exposure")
    result = {
        "schema": SCHEMA,
        "source_manifest_sha256": source["source"]["suite_manifest_sha256"],
        "teacher_manifest_sha256": teacher_sha,
        "native_training_roots": source["independent_lineages"],
        "native_training_frames": len(frames),
        "view_roster": list(VIEWS_PAIRED),
        "seed_roster": list(seeds),
        "trained_order": "ab",
        "same_worker": True,
        "fully_matched_optimizer_samples": True,
        "same_seed_initial_weights": True,
        "all_duplicate_fits_reproduced": True,
        "model_width": 32,
        "model_depth": 2,
        "bytes_budget": 384,
        "optimizer_steps": steps,
        "single_torch_thread": torch.get_num_threads() == 1,
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_version": torch.__version__,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "python_version": sys.version,
        "host_platform": platform.platform(),
        "host_processor": platform.processor(),
        "host_machine": platform.machine(),
        "thread_environment": {
            k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "cells": cells,
        "new_lean_calls": 0,
        "heldout_test_opened": False,
        "scientific_eligible": False,
        "interpretation": "same-host CPU training reproducibility, independent native Lean evaluation is a downstream job",
    }
    (out / "manifest.json").write_text(canonical(result) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--max-seconds-per-fit", type=float, default=300)
    args = parser.parse_args(argv)
    report = run(
        args.teacher, args.out, seeds=tuple(args.seeds),
        steps=args.steps, max_seconds_per_fit=args.max_seconds_per_fit,
    )
    print(canonical({
        "schema": report["schema"],
        "same_worker": report["same_worker"],
        "duplicated": report["all_duplicate_fits_reproduced"],
        "source_roots": report["native_training_roots"],
        "cells": [
            {"seed": c["seed"], "view": c["view"],
             "checkpoint_sha256": c["checkpoint_sha256"],
             "mean_public_bytes": c["mean_public_bytes"],
             "last_loss": c["last_loss"]}
            for c in report["cells"]
        ],
    }))
    return report


if __name__ == "__main__":
    main()
