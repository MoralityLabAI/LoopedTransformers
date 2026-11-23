"""LSA-02G: same-worker CPU training for reference-path fork emphasis.

Same native Lean training teacher, public 96-byte-pretty input and source
sample stream across plain, fork4 and shuffled-sham4 loss modes. All six
model cells receive separate duplicate training on one CPU host.
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
from .typed_state import schema_for_view


SCHEMA = "lsa02g.matched_fork_emphasis_cpu.v1"
SEEDS = (149, 151)
MODES = ("plain", "fork4", "sham4")
VIEW = "pp96"


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
    modes: tuple[str, ...] = MODES,
    schema: str = SCHEMA,
) -> dict:
    if out.exists():
        raise ValueError("output directory must be fresh")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("duplicate or absent seed")
    if not modes or len(modes) != len(set(modes)) or any(v not in MODES for v in modes):
        raise ValueError("invalid matched CPU emphasis roster")
    if not isinstance(schema, str) or not schema:
        raise ValueError("missing registered training schema")
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
        for mode in modes:
            if mode not in MODES:
                raise ValueError("unsupported emphasis roster")
            key = f"{seed}-{mode}"
            destination = out / "fits" / key
            duplicate = out / "replicas" / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            receipts = [
                fit(
                    teacher, directory, order="ab", seed=seed,
                    steps=steps, depth=2, width=32, max_bytes=384,
                    threads=1, max_seconds=max_seconds_per_fit,
                    input_view=VIEW, weight_mode=mode,
                )
                for directory in (destination, duplicate)
            ]
            first, repeated = receipts
            if any(
                item.get("optimizer_steps") != steps
                or item.get("source_manifest_sha256") != source["source"]["suite_manifest_sha256"]
                or item.get("native_teacher_manifest_sha256") != teacher_sha
                or item.get("input_view_schema") != schema_for_view(VIEW)
                or item.get("weight_mode") != mode
                or item.get("weight_mode_schema") != "lsa02g.source_fork_weighting.v1"
                or item.get("native_train_fork_frames", 0) < 1
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
            starts[mode] = first["initial_weights_sha256"]
            samplers[mode] = keys
            cells.append({
                "seed": seed, "mode": mode, "view": VIEW,
                "train_discriminating_forks": first["native_train_fork_frames"],
                "boosted_source_frames": first["source_boosted_frames"],
                "boosted_overlap_with_forks": first["source_boosted_true_fork_overlap"],
                "optimizer_draws_on_forks": first["optimizer_draws_on_forks"],
                "optimizer_draws_on_boosted_frames": first["optimizer_draws_on_boosted_frames"],
                "source_loss_mass_normalizer": first["source_loss_mass_normalizer"],
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
            raise ValueError("loss-emphasis roster started from different model tensors in the same seed")
        first_mode = modes[0]
        if any(samplers[first_mode] != samplers[mode] for mode in modes[1:]):
            raise ValueError("loss-emphasis arms had different source-root and frame exposure")
    result = {
        "schema": schema,
        "source_manifest_sha256": source["source"]["suite_manifest_sha256"],
        "teacher_manifest_sha256": teacher_sha,
        "native_training_roots": source["independent_lineages"],
        "native_training_frames": len(frames),
        "mode_roster": list(modes),
        "input_view": VIEW,
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
        "interpretation": "matched same-host fork-vs-sham train-only weighting, with native independent proof evaluation pending",
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
            {"seed": c["seed"], "mode": c["mode"], "view": c["view"],
             "checkpoint_sha256": c["checkpoint_sha256"],
             "mean_public_bytes": c["mean_public_bytes"],
             "last_loss": c["last_loss"]}
            for c in report["cells"]
        ],
    }))
    return report


if __name__ == "__main__":
    main()
