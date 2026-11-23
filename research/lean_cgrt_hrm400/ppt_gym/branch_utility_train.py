"""LSA-02J: deterministic CPU training of a tiny branch-utility readout.

Plan B 57,475-param backbones are trained once per seed on the frozen native
shortest-proof teacher, duplicated on one physical CPU worker, and then
**frozen**. Viability and recovery cost probes train only on the new Lean
TRAIN-only branch-label corpus. The sham permutes each frame's paired labels
and costs without changing candidate menus, feature vectors, source roots,
head architecture, optimizer draws, initial weights or training step count.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import sys

import torch
import torch.nn.functional as F

from .branch_utility import BranchUtility, action_features, SCHEMA as HEAD_SCHEMA, SOURCE_MAX_COST
from .branch_utility_source import SCHEMA as SOURCE_SCHEMA
from .cli import checkpoint_load
from .data import canonical, sha256_file
from .native_order import FrozenOrder, _weight_fingerprint
from .native_train import _load_teacher
from .pretty_train import run as train_backbone
from .typed_state import encode_public_packet


SCHEMA = "lsa02j.same_worker_utility_training.v1"
SEEDS = (183, 191)
MODES = ("viability", "shuffled_sham")
VIEW = "pp96"
PROBE_STEPS = 480


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def _write(path, data):
    Path(path).write_text(canonical(data) + "\n", encoding="utf-8")


def _load_source(teacher: Path, labels: Path):
    teacher_rows, teacher_manifest = _load_teacher(teacher)
    source = _json(labels / "manifest.json")
    if not (
        source.get("schema") == SOURCE_SCHEMA
        and source.get("teacher_sha256") == sha256_file(teacher / "manifest.json")
        and source.get("suite_manifest_sha256") == teacher_manifest["source"]["suite_manifest_sha256"]
        and source.get("split") == "train"
        and source.get("backend", {}).get("actual_lean_process") is True
        and source.get("source_reference_independently_compiled") is True
        and source.get("training_lineages") == teacher_manifest["independent_lineages"]
        and source.get("no_ordinary_or_ood_opened") is True
        and source.get("calibration_model_updates") == 0
    ):
        raise ValueError("native training-only viability/cost corpus failed provenance")
    for name, digest in source.get("artifacts", {}).items():
        if name != Path(name).name or sha256_file(labels / name) != digest:
            raise ValueError("native viability labels artifact changed")
    rows = _rows(labels / "branch_labels.jsonl")
    if len(rows) != source["public_frames"] or len(rows) != len(teacher_rows):
        raise ValueError("native viability frame count differs from teacher")
    for row, source_frame in zip(rows, teacher_rows):
        if (
            row.get("schema") != SOURCE_SCHEMA
            or row.get("source_only") is not True
            or (row["root"], row["lineage"], row["goal"], row["candidates"])
            != (
                source_frame["root"], source_frame["lineage"],
                source_frame["goal"], source_frame["candidates"],
            )
            or len(row["minimum_whole_state_tactic_cost"]) != len(row["candidates"])
            or len(row["finite_viability"]) != len(row["candidates"])
        ):
            raise ValueError("training native state labels or public candidate menu misaligned")
    return rows, teacher_manifest, source


def _cached_features(checkpoint, frames):
    model, stored = checkpoint_load(checkpoint)
    if (
        model.parameter_count() != 57475
        or model.cfg.max_bytes != 384 or model.cfg.width != 32
        or stored.get("input_view") != VIEW or stored.get("train_order") != "ab"
        or stored.get("train_steps") != 720
    ):
        raise ValueError("branch readout requires frozen registered Plan B pp96 checkpoint")
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    before = _weight_fingerprint(model)
    adapter = FrozenOrder(model, "ab", view=VIEW)
    entries = []
    with torch.no_grad():
        for row in frames:
            packet = row["goal"]
            menu = row["candidates"]
            value = adapter(
                packet, menu, depth=2,
                checks_remaining=row["checks_remaining"],
            )[-1]
            entries.append({
                "root": row["root"], "step": row["step"],
                "features": value["features"].clone().detach(),
                "public": action_features(packet, menu),
                "viability": torch.tensor(row["finite_viability"], dtype=torch.float32),
                "cost": torch.tensor([
                    min(c, SOURCE_MAX_COST) / SOURCE_MAX_COST if c is not None else 1.0
                    for c in row["minimum_whole_state_tactic_cost"]
                ], dtype=torch.float32),
                "cost_known": torch.tensor(row["finite_viability"], dtype=torch.bool),
            })
    if before != _weight_fingerprint(model):
        raise ValueError("frozen backbone parameter tensors changed during source feature caching")
    return entries, before


def _shuffled_targets(row: dict, seed: int):
    values = list(zip(row["viability"].tolist(), row["cost"].tolist()))
    if len(values) < 2:
        return row["viability"], row["cost"], row["cost_known"]
    raw = canonical([seed, row["root"], row["step"], "sham"])
    rng = random.Random(int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big"))
    rng.shuffle(values)
    viable = torch.tensor([x[0] for x in values], dtype=torch.float32)
    costs = torch.tensor([x[1] for x in values], dtype=torch.float32)
    return viable, costs, viable > 0.5


def _head_fit(rows, teacher_sha: str, label_sha: str,
              backbone_sha: str, output: Path, *,
              mode: str, seed: int, steps: int):
    if mode not in MODES or output.exists() or steps != PROBE_STEPS:
        raise ValueError("registered frozen-probe fit needs a fresh destination and 480 updates")
    torch.manual_seed(seed + 1024)
    head = BranchUtility(width=32)
    initial = sha256_file  # avoid hashing tensor repr, save the initialized state.
    output.mkdir(parents=True)
    torch.save(head.state_dict(), output / "initial.pt")
    initial_sha = sha256_file(output / "initial.pt")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003)
    grouped: dict[str, list[int]] = defaultdict(list)
    mixed: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row["root"]].append(idx)
        if 0 < float(row["viability"].sum()) < len(row["viability"]):
            mixed[row["root"]].append(idx)
    roots = sorted(grouped)
    rng = random.Random(seed ^ 0x02A02026)
    trace = []
    for step in range(1, steps + 1):
        root = roots[rng.randrange(len(roots))]
        # Train on the source's actual viable-vs-dead decision forks while
        # retaining forced rows as negative/positive grounding.
        use_mixed = bool(mixed[root]) and rng.randrange(2) == 0
        pool = mixed[root] if use_mixed else grouped[root]
        row = rows[pool[rng.randrange(len(pool))]]
        viability, cost, known = (
            (row["viability"], row["cost"], row["cost_known"])
            if mode == "viability" else _shuffled_targets(row, seed)
        )
        predictions = head(row["features"], row["public"])
        viability_loss = F.binary_cross_entropy_with_logits(predictions[:, 0], viability)
        cost_loss = (
            F.smooth_l1_loss(predictions[:, 1].sigmoid()[known], cost[known])
            if bool(known.any()) else predictions[:, 1].sum() * 0.0
        )
        loss = viability_loss + 0.25 * cost_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite CPU head viability/cost training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        trace.append({
            "step": step, "root": root, "source_step": row["step"],
            "mixed_fork": use_mixed,
            "loss": float(loss.detach()),
        })
    output_path = output / "head.pt"
    torch.save({
        "schema": HEAD_SCHEMA,
        "mode": mode, "seed": seed,
        "architecture": {"width": 32, "public_features": 8, "hidden": 24},
        "weights": head.state_dict(),
        "backbone_checkpoint_sha256": backbone_sha,
        "native_teacher_sha256": teacher_sha,
        "viability_source_sha256": label_sha,
        "optimizer_steps": steps,
        "sham": mode == "shuffled_sham",
    }, output_path)
    trace_path = output / "optimizer_steps.jsonl"
    trace_path.write_text(
        "".join(canonical(x) + "\n" for x in trace), encoding="utf-8",
    )
    receipt = {
        "schema": "lsa02j.utility_probe_fit.v1",
        "mode": mode, "seed": seed,
        "head_sha256": sha256_file(output_path),
        "initial_head_sha256": initial_sha,
        "optimizer_trace_sha256": sha256_file(trace_path),
        "backbone_checkpoint_sha256": backbone_sha,
        "teacher_sha256": teacher_sha,
        "branch_source_sha256": label_sha,
        "optimizer_steps": steps,
        "source_root_count": len(roots),
        "source_rows": len(rows),
        "source_mixed_viability_fork_rows": sum(len(x) for x in mixed.values()),
        "mean_last32_loss": sum(r["loss"] for r in trace[-32:]) / 32,
        "native_calibration_updates": 0,
    }
    _write(output / "manifest.json", receipt)
    return receipt


def run(teacher: Path, labels: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("new source/cpu training output must be fresh")
    rows, native_teacher, branch_labels = _load_source(teacher, labels)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    out.mkdir(parents=True)
    # Two checkpoints are trained twice each, sequentially on this CPU worker.
    backbones = train_backbone(
        teacher, out / "backbones", seeds=SEEDS, views=(VIEW,), steps=720,
        max_seconds_per_fit=300,
        schema="lsa02j.native_source_planb_backbones.v1",
    )
    if not backbones["all_duplicate_fits_reproduced"]:
        raise ValueError("native model training did not reproduce on one CPU worker")
    teacher_sha = sha256_file(teacher / "manifest.json")
    label_sha = sha256_file(labels / "manifest.json")
    records = []
    for seed in SEEDS:
        checkpoint = out / "backbones" / "fits" / f"{seed}-{VIEW}" / "checkpoint.pt"
        backbone_sha = sha256_file(checkpoint)
        cache, fingerprint = _cached_features(checkpoint, rows)
        sample_streams = {}
        initials = set()
        for mode in MODES:
            path = out / "heads" / f"{seed}-{mode}"
            replica = out / "replicas" / f"{seed}-{mode}"
            path.parent.mkdir(parents=True, exist_ok=True)
            replica.parent.mkdir(parents=True, exist_ok=True)
            first = _head_fit(cache, teacher_sha, label_sha, backbone_sha,
                              path, mode=mode, seed=seed, steps=PROBE_STEPS)
            second = _head_fit(cache, teacher_sha, label_sha, backbone_sha,
                               replica, mode=mode, seed=seed, steps=PROBE_STEPS)
            if (
                first["head_sha256"] != second["head_sha256"]
                or first["optimizer_trace_sha256"] != second["optimizer_trace_sha256"]
                or first["initial_head_sha256"] != second["initial_head_sha256"]
            ):
                raise ValueError("same-worker utility head duplicate was not bitwise reproducible")
            stream = [
                (row["step"], row["root"], row["source_step"], row["mixed_fork"])
                for row in _rows(path / "optimizer_steps.jsonl")
            ]
            sample_streams[mode] = stream
            initials.add(first["initial_head_sha256"])
            records.append({**first, "duplicate_bitwise": True})
        if len(initials) != 1 or sample_streams[MODES[0]] != sample_streams[MODES[1]]:
            raise ValueError("matched utility and sham source draws or initial head tensors differ")
    report = {
        "schema": SCHEMA,
        "native_teacher_sha256": teacher_sha,
        "branch_labels_sha256": label_sha,
        "suite_manifest_sha256": native_teacher["source"]["suite_manifest_sha256"],
        "training_root_count": native_teacher["independent_lineages"],
        "training_public_frames": len(rows),
        "training_mixed_fork_frames": sum(row["has_mixed_finite_actions"] for row in _rows(labels / "branch_labels.jsonl")),
        "backbone_manifest_sha256": sha256_file(out / "backbones" / "manifest.json"),
        "backbone_seed_roster": list(SEEDS),
        "head_mode_roster": list(MODES),
        "input_view": VIEW,
        "frozen_backbone": True,
        "head_optimizer_steps": PROBE_STEPS,
        "backbone_optimizer_steps": 720,
        "same_worker": True,
        "same_seed_initial_head_weights": True,
        "same_seed_source_draws": True,
        "all_duplicate_fits_bitwise": True,
        "single_torch_thread": torch.get_num_threads() == 1,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_version": torch.__version__,
        "python": sys.version, "platform": platform.platform(),
        "cells": records,
        "native_calibration_updates": 0,
        "test_or_ood_opened": False,
        "scientific_eligible": False,
    }
    _write(out / "manifest.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    report = run(a.teacher, a.labels, a.out)
    print(canonical({
        "source_roots": report["training_root_count"],
        "label_frames": report["training_public_frames"],
        "forks": report["training_mixed_fork_frames"],
        "seeds": report["backbone_seed_roster"],
        "heads": [(r["seed"],r["mode"],r["mean_last32_loss"])
                  for r in report["cells"]],
        "bitwise_repeatability": report["all_duplicate_fits_bitwise"],
    }))
    return report


if __name__ == "__main__":
    main()
