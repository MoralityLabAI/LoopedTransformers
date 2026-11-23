"""LSA-02H: one-worker matched CPU fit on verified off-reference Lean states.

Same frozen Plan B, pp96 public input, original 64 theorem roots, the exact
seeded source-root and in-root sampler, and 720 AdamW steps per arm. The sole
intervention is the type of a prespecified additional source record on an
identical root/switch draw:

 reference: the original verified shortest reference-path frame
 offpath:   verified novel correction state after a legal recoverable deviation
 sham:      another native shortest reference-path frame of the same root

The offpath and sham arms have identical opportunities to replace the base
frame; the total number of optimizer steps, physical parameters and initial
tensors match. A state from a calibration or locked-test lineage is refused.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import torch

from .core import Config, PlasticProofTransformer, supervised_loss
from .data import canonical, sha256_file
from .determinism_audit import _compare_checkpoints
from .native_train import _load_teacher, _model_fingerprint
from .offpath_source import SCHEMA as OFFPATH_SCHEMA, ROW_SCHEMA
from .typed_state import encode_public_packet, schema_for_view


SCHEMA = "lsa02h.matched_recoverable_offpath_cpu.v1"
SEEDS = (157, 163)
ARMS = ("reference", "offpath", "sham")
VIEW = "pp96"
STEPCOUNT = 720
SWITCH_MODULUS = 2


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _write(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _load_offpath(teacher: Path, pack: Path, native_frames: list[dict],
                  source_meta: dict) -> tuple[dict[str, list[dict]], dict]:
    manifest = _json(pack / "manifest.json")
    if not (
        manifest.get("schema") == OFFPATH_SCHEMA
        and manifest.get("source_teacher_sha256") == sha256_file(teacher / "manifest.json")
        and manifest.get("suite_manifest_sha256") == source_meta["source"]["suite_manifest_sha256"]
        and manifest.get("backend", {}).get("actual_lean_process") is True
        and manifest.get("independent_compiler_passed") is True
        and manifest.get("train_split_only") is True
        and manifest.get("model_calls_during_collection") == 0
        and manifest.get("locked_test_opened") is False
        and manifest.get("training_lineages") == source_meta["independent_lineages"]
        and manifest.get("certified_alternate_full_proofs", 0) > 0
        and manifest.get("novel_offreference_training_states", 0) > 0
        and manifest.get("roots_with_recoverable_deviations", 0) > 0
    ):
        raise ValueError("native certified offreference training source is missing or empty")
    for name, digest in manifest.get("artifacts", {}).items():
        if name != Path(name).name or sha256_file(pack / name) != digest:
            raise ValueError("offreference artifact is missing or tampered")
    frames = _rows(pack / "offreference_frames.jsonl")
    if len(frames) != manifest["novel_offreference_training_states"]:
        raise ValueError("novel correction frame count differs from source manifest")
    authorized_roots = {r["root"]: r["lineage"] for r in native_frames}
    original_by_root: dict[str, set] = defaultdict(set)
    for row in native_frames:
        original_by_root[row["root"]].add(row["public_state_sha256"])
    by_root: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in frames:
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("source_kind") != "off_reference"
            or row.get("source_proof_compiled") is not True
            or row.get("root") not in authorized_roots
            or authorized_roots[row["root"]] != row.get("lineage")
            or row.get("public_state_sha256") in original_by_root[row["root"]]
            or row.get("remaining_proof_steps", 1000) > row.get("checks_remaining", 0)
        ):
            raise ValueError("offreference correction violates root ancestry, novelty or budget")
        if row["step"] in seen:
            raise ValueError("offreference per-run state ordinal duplicated")
        seen.add(row["step"])
        if not (
            1 <= len(row["candidates"]) <= 64
            and len(row["labels"]) == len(row["candidates"])
            and all(type(x) is int and x in (0,1) for x in row["labels"])
            and any(row["labels"])
        ):
            raise ValueError("offreference tactic candidate labels are malformed")
        by_root[row["root"]].append(row)
    if len(by_root) != manifest["roots_with_recoverable_deviations"]:
        raise ValueError("offreference occupied root counts differ")
    return by_root, manifest


def _sample_roster(reference: list[dict], supplement: dict[str, list[dict]],
                   seed: int, steps: int) -> list[dict]:
    by_root: dict[str, list[dict]] = defaultdict(list)
    for row in reference:
        by_root[row["root"]].append(row)
    roots = sorted(by_root)
    rng = random.Random(seed)
    result = []
    for i in range(steps):
        root = roots[rng.randrange(len(roots))]
        index = rng.randrange(len(by_root[root]))
        switched = rng.randrange(SWITCH_MODULUS) == 0 and root in supplement
        result.append({
            "step": i + 1,
            "root": root,
            "ref_index": index,
            "switched": switched,
            "supplement_index": (
                rng.randrange(len(supplement[root])) if switched else -1
            ),
        })
    return result


def _fit(teacher: Path, source: dict[str, list[dict]], refs: list[dict],
         pack_meta: dict, destination: Path, *, seed: int, arm: str, steps: int,
         max_seconds: float) -> dict:
    if destination.exists() or arm not in ARMS or not 1 <= steps <= 5000:
        raise ValueError("fresh matched offreference destination and registered arm required")
    ref_by_root: dict[str, list[dict]] = defaultdict(list)
    for row in refs:
        ref_by_root[row["root"]].append(row)
    schedule = _sample_roster(refs, source, seed, steps)
    torch.manual_seed(seed)
    cfg = Config(plan="b", width=32, heads=4, max_bytes=384).validate()
    model = PlasticProofTransformer(cfg).cpu().float()
    init_sha = _model_fingerprint(model)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    destination.mkdir(parents=True)
    trace = []
    begun = time.perf_counter()
    for draw in schedule:
        if time.perf_counter() - begun > max_seconds:
            raise TimeoutError("same-worker offreference fit exceeded registered CPU budget")
        root = draw["root"]
        base = ref_by_root[root][draw["ref_index"]]
        chosen = base
        kind = "reference"
        if draw["switched"] and arm == "offpath":
            chosen = source[root][draw["supplement_index"]]
            kind = "off_reference"
        elif draw["switched"] and arm == "sham":
            # Matched root, distinct reference-frame when possible. A
            # single-frame root necessarily duplicates the only reference.
            base_index = (draw["ref_index"] + 1) % len(ref_by_root[root])
            chosen = ref_by_root[root][base_index]
            kind = "reference_sham"
        payload = encode_public_packet(
            chosen["goal"], chosen["candidates"],
            view=VIEW, max_bytes=cfg.max_bytes,
        )
        out = model(
            payload, chosen["candidates"], depth=2,
            checks_remaining=chosen["checks_remaining"],
            middle_order="ab", return_all=True,
        )
        loss = supervised_loss(out, tuple(chosen["labels"]))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite offreference source loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(clip)):
            raise FloatingPointError("nonfinite source gradients")
        optim.step()
        trace.append({
            **draw, "kind": kind, "chosen_step": chosen["step"],
            "chosen_state_sha256": chosen["public_state_sha256"],
            "loss": float(loss.detach()),
        })
    model.eval()
    path = destination / "checkpoint.pt"
    torch.save({
        "schema": "ppt_gym.checkpoint.v1",
        "config": asdict(cfg),
        "weights": model.state_dict(),
        "seed": seed, "train_steps": steps, "train_order": "ab",
        "input_view": VIEW, "input_view_schema": schema_for_view(VIEW),
        "native_teacher_sha256": sha256_file(teacher / "manifest.json"),
        "source_manifest_sha256": pack_meta["suite_manifest_sha256"],
        "source_evidence_class": "lean_kernel_verified_reference_path",
        "offreference_training_mode": arm,
        "offreference_source_manifest_sha256": sha256_file(
            teacher.parent / "offreference" / "manifest.json"
        ),
        "offreference_source_schema": OFFPATH_SCHEMA,
    }, path)
    optimizer_steps = destination / "optimizer_steps.jsonl"
    optimizer_steps.write_text(
        "".join(canonical(x) + "\n" for x in trace), encoding="utf-8",
    )
    return {
        "schema": "lsa02h.offreference_cpu_fit.v1",
        "seed": seed, "arm": arm, "view": VIEW, "train_order": "ab",
        "native_teacher_sha256": sha256_file(teacher / "manifest.json"),
        "offreference_source_manifest_sha256": sha256_file(
            teacher.parent / "offreference" / "manifest.json"
        ),
        "source_manifest_sha256": pack_meta["suite_manifest_sha256"],
        "optimizer_steps": len(trace),
        "initial_weights_sha256": init_sha,
        "final_weights_sha256": _model_fingerprint(model),
        "checkpoint_sha256": sha256_file(path),
        "optimizer_steps_sha256": sha256_file(optimizer_steps),
        "parameter_count": model.parameter_count(),
        "offreference_source_roots": len(source),
        "offreference_source_frames": sum(len(x) for x in source.values()),
        "matched_switch_opportunities": sum(r["switched"] for r in trace),
        "actual_offreference_optimizer_draws": sum(r["kind"] == "off_reference" for r in trace),
        "actual_sham_optimizer_draws": sum(r["kind"] == "reference_sham" for r in trace),
        "mean_final_32_loss": sum(r["loss"] for r in trace[-32:])/min(len(trace),32),
        "query_label_optimizer_updates": 0,
        "scientific_eligible": False,
    }


def run(teacher: Path, supplement: Path, out: Path,
        *, seeds: tuple[int, ...] = SEEDS, steps: int = STEPCOUNT,
        max_seconds_per_fit: float = 300) -> dict:
    if out.exists() or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("fresh same-worker output and unique model seeds required")
    if steps != STEPCOUNT or max_seconds_per_fit <= 0:
        raise ValueError("registered 720 CPU steps required")
    original, teacher_meta = _load_teacher(teacher)
    supplement_rows, pack_meta = _load_offpath(
        teacher, supplement, original, teacher_meta,
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    if torch.get_num_threads() != 1:
        raise ValueError("one Torch worker-thread contract failed")
    # Checkpoint embeds the explicit supplement location under the frozen
    # teacher's sibling path; source selection is fixed before fitting.
    if supplement.resolve() != (teacher.parent / "offreference").resolve():
        raise ValueError("supplement must be the manifest-attested teacher sibling")
    out.mkdir(parents=True)
    all_cells = []
    for seed in seeds:
        init = set()
        samplers = {}
        for arm in ARMS:
            base = out / "fits" / f"{seed}-{arm}"
            duplicate = out / "replicas" / f"{seed}-{arm}"
            base.parent.mkdir(parents=True, exist_ok=True)
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            first = _fit(
                teacher, supplement_rows, original, pack_meta, base,
                seed=seed, arm=arm, steps=steps,
                max_seconds=max_seconds_per_fit,
            )
            repeated = _fit(
                teacher, supplement_rows, original, pack_meta, duplicate,
                seed=seed, arm=arm, steps=steps,
                max_seconds=max_seconds_per_fit,
            )
            identity = _compare_checkpoints(base / "checkpoint.pt",
                                             duplicate / "checkpoint.pt")
            if not (
                first["optimizer_steps"] == repeated["optimizer_steps"] == steps
                and identity["all_tensors_bitwise_equal"]
                and identity["full_checkpoint_bytes_equal"]
                and first["optimizer_steps_sha256"] == repeated["optimizer_steps_sha256"]
                and first["initial_weights_sha256"] == repeated["initial_weights_sha256"]
                and first["final_weights_sha256"] == repeated["final_weights_sha256"]
            ):
                raise ValueError("offreference same-host fit duplicate failed exact equality")
            records = _records(base / "optimizer_steps.jsonl")
            samplers[arm] = [
                (r["step"],r["root"],r["ref_index"],r["switched"],r["supplement_index"])
                for r in records
            ]
            init.add(first["initial_weights_sha256"])
            all_cells.append(first)
        if len(init) != 1 or any(
            samplers[arm] != samplers[ARMS[0]] for arm in ARMS[1:]
        ):
            raise ValueError("matched offreference arms had different initial weights or sample schedule")
    report = {
        "schema": SCHEMA,
        "teacher_manifest_sha256": sha256_file(teacher / "manifest.json"),
        "offreference_manifest_sha256": sha256_file(supplement / "manifest.json"),
        "suite_manifest_sha256": pack_meta["suite_manifest_sha256"],
        "training_lineages": teacher_meta["independent_lineages"],
        "train_reference_frames": len(original),
        "train_novel_offreference_states": sum(len(r) for r in supplement_rows.values()),
        "train_offreference_roots": len(supplement_rows),
        "seed_roster": list(seeds), "arm_roster": list(ARMS),
        "view": VIEW, "optimizer_steps_per_cell": steps,
        "all_exact_duplicate_fits_passed": True,
        "same_seed_identical_source_draws": True,
        "same_seed_bitwise_initial_weights": True,
        "one_cpu_worker": True,
        "torch_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_version": torch.__version__,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "python": sys.version, "platform": platform.platform(),
        "process_env": {k:os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "PYTHONHASHSEED",
        )},
        "cells": all_cells,
        "native_calibration_queried_during_fit": False,
        "test_or_ood_opened": False,
        "scientific_eligible": False,
    }
    _json(out / "manifest.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--supplement", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=STEPCOUNT)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--max-seconds-per-fit", type=float, default=300)
    a = p.parse_args(argv)
    result = run(a.teacher, a.supplement, a.out,
                 seeds=tuple(a.seeds), steps=a.steps,
                 max_seconds_per_fit=a.max_seconds_per_fit)
    print(canonical({
        "schema": result["schema"],
        "offreference_roots": result["train_offreference_roots"],
        "offreference_states": result["train_novel_offreference_states"],
        "train_schedules_matched": result["same_seed_identical_source_draws"],
        "six_duplicate_fits_reproduced": result["all_exact_duplicate_fits_passed"],
        "cells": [{
            "seed": x["seed"], "arm": x["arm"],
            "offreference_draws": x["actual_offreference_optimizer_draws"],
            "sham_draws": x["actual_sham_optimizer_draws"],
            "checkpoint_sha256": x["checkpoint_sha256"],
        } for x in result["cells"]],
    }))
    return result


if __name__ == "__main__":
    main()
