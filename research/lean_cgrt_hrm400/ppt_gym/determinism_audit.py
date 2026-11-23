"""LSA-02D: same-runner, one-thread, deterministic duplicate CPU fitting.

This is a source-only engineering diagnostic. It performs no Lean scoring,
reads no held-out roots and cannot promote an architecture. The paired raw and
typed fits start from the *same seeded tensors* on the same CPU worker; each is
repeated to test exact saved parameter and optimizer-trace reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import torch

from .data import canonical, sha256_file
from .native_train import _load_teacher, fit


SCHEMA = "lsa02d.same_worker_determinism.v1"
VIEWS = ("raw", "typed")
SEEDS = (107, 109)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_checkpoints(first: Path, second: Path) -> dict:
    a = torch.load(first, map_location="cpu", weights_only=True)
    b = torch.load(second, map_location="cpu", weights_only=True)
    if a.get("schema") != "ppt_gym.checkpoint.v1" or b.get("schema") != a.get("schema"):
        raise ValueError("unknown checkpoint provenance")
    left = a["weights"]
    right = b["weights"]
    if set(left) != set(right):
        raise ValueError("repeated model has different parameter keys")
    mismatched = []
    max_abs = 0.0
    for key in sorted(left):
        if left[key].shape != right[key].shape or left[key].dtype != right[key].dtype:
            raise ValueError("model parameter shape or dtype changed across duplicate fits")
        if not torch.equal(left[key], right[key]):
            mismatched.append(key)
            if left[key].is_floating_point():
                delta = float((left[key] - right[key]).abs().max())
                max_abs = max(max_abs, delta)
    return {
        "all_tensors_bitwise_equal": len(mismatched) == 0,
        "different_tensor_count": len(mismatched),
        "different_tensor_keys": mismatched[:10],
        "largest_tensor_abs_displacement": max_abs,
        "weight_key_count": len(left),
        "full_checkpoint_bytes_equal": sha256_file(first) == sha256_file(second),
    }


def run(teacher: Path, out: Path, *, seeds: tuple[int, ...] = SEEDS,
        steps: int = 720, width: int = 32, max_bytes: int = 384,
        max_seconds_per_fit: float = 300) -> dict:
    if out.exists():
        raise ValueError("same-worker output directory must be fresh")
    if not seeds or len(set(seeds)) != len(seeds) or steps not in range(1, 5001):
        raise ValueError("invalid duplicate fit seed roster or optimizer budget")
    frames, source = _load_teacher(teacher)
    if not source.get("source_compiler_passed") or source.get("independent_lineages") != 48:
        raise ValueError("expected the frozen LSA-02C 48-root native Lean source")
    if source.get("train_examples") != 259:
        raise ValueError("unexpected frozen LSA-02C source frame count")
    # One-thread CPU, deterministic PyTorch operator selection.
    # Disable MKLDNN dispatch so a changed oneDNN kernel cannot masquerade as
    # a block-order or public-state effect within this diagnostic.
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise ValueError("one-thread CPU contract failed")
    out.mkdir(parents=True)
    teacher_sha = sha256_file(teacher / "manifest.json")
    rows = []
    for seed in seeds:
        start_hashes = {}
        for view in VIEWS:
            receipts = []
            for repetition in ("a", "b"):
                path = out / f"{seed}-{view}-{repetition}"
                fit(
                    teacher, path,
                    order="ab", seed=seed, steps=steps,
                    depth=2, width=width, max_bytes=max_bytes,
                    threads=1, max_seconds=max_seconds_per_fit,
                    input_view=view,
                )
                report = _json(path / "manifest.json")
                if (
                    report.get("optimizer_steps") != steps
                    or report.get("train_roots") != 48
                    or report.get("teacher_frames") != 259
                    or report.get("training_source_native") is not True
                    or report.get("native_teacher_manifest_sha256") != teacher_sha
                    or report.get("input_view") != view
                    or report.get("seed") != seed
                    or report.get("train_order") != "ab"
                    or report.get("optimizer_calls_on_calibration_or_test") != 0
                ):
                    raise ValueError("one of the duplicate fits broke source or training contract")
                receipts.append(report)
            a_dir = out / f"{seed}-{view}-a"
            b_dir = out / f"{seed}-{view}-b"
            parity = _compare_checkpoints(a_dir / "checkpoint.pt", b_dir / "checkpoint.pt")
            exact_trace = sha256_file(a_dir / "optimizer_steps.jsonl") == sha256_file(
                b_dir / "optimizer_steps.jsonl"
            )
            same_init = receipts[0]["initial_weights_sha256"] == receipts[1]["initial_weights_sha256"]
            start_hashes[view] = receipts[0]["initial_weights_sha256"]
            rows.append({
                "seed": seed, "input_view": view,
                "repeat_initial_fingerprint_match": same_init,
                "repeat_trace_sha_match": exact_trace,
                "repeat_frozen_tensor_check": parity,
                "repeat_final_weights_sha_match": (
                    receipts[0]["final_weights_sha256"] == receipts[1]["final_weights_sha256"]
                ),
                "repeat_checkpoint_file_sha_match": (
                    receipts[0]["checkpoint_sha256"] == receipts[1]["checkpoint_sha256"]
                ),
                "initial_weights_sha256": receipts[0]["initial_weights_sha256"],
                "model_parameter_count": receipts[0]["unique_parameter_count"],
                "optimizer_steps": steps,
                "fitted_checkpoint_sha256": receipts[0]["checkpoint_sha256"],
                "training_loss_first": receipts[0]["initial_training_loss"],
                "training_loss_last": receipts[0]["last_training_loss"],
                "mean_final_32_loss": receipts[0]["mean_final_32_loss"],
            })
        if start_hashes["raw"] != start_hashes["typed"]:
            raise ValueError("same-seed raw/typed starting parameters differed on same worker")
    success = all(
        r["repeat_initial_fingerprint_match"]
        and r["repeat_trace_sha_match"]
        and r["repeat_final_weights_sha_match"]
        and r["repeat_frozen_tensor_check"]["all_tensors_bitwise_equal"]
        for r in rows
    )
    report = {
        "schema": SCHEMA,
        "source_teacher_manifest_sha256": teacher_sha,
        "source_train_roots": source["independent_lineages"],
        "source_train_frames": len(frames),
        "seeds": list(seeds),
        "input_views": list(VIEWS),
        "cpu_only": True,
        "same_worker": True,
        "one_torch_thread": torch.get_num_threads() == 1,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_version": torch.__version__,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "python_version": sys.version,
        "host_platform": platform.platform(),
        "cpu_machine": platform.machine(),
        "cpu_processor": platform.processor(),
        "environment_threads": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "fitness_and_reproducibility": rows,
        "matched_raw_typed_initial_parameters": True,
        "all_duplicate_fits_bitwise_reproduced": success,
        "live_lean_calls": 0,
        "query_label_training_updates": 0,
        "heldout_or_ood_opened": False,
        "scientific_eligible": False,
        "claim": "single-worker source-only deterministic duplicate fit contract",
    }
    (out / "determinism_report.json").write_text(canonical(report) + "\n", encoding="utf-8")
    if not success:
        raise RuntimeError("a repeated same-worker CPU fit failed exact tensor/trace parity")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = p.parse_args(argv)
    report = run(args.teacher, args.out, seeds=tuple(args.seeds), steps=args.steps)
    print(canonical({
        "schema": report["schema"],
        "same_worker": report["same_worker"],
        "bitwise_reproduced": report["all_duplicate_fits_bitwise_reproduced"],
        "torch_version": report["torch_version"],
        "roster": report["fitness_and_reproducibility"],
    }))
    return report


if __name__ == "__main__":
    main()
