"""Phase 1, E1: arrival matrices and the ridge test of docs/rmp_phase1_plan.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.analysis import load_analysis_model, with_injection
from research_gym.interp.arrival import (
    MIN_SPREAD,
    arrival_matrix,
    arrival_visits,
    edge_rewire_pairs,
    paired_difference_interval,
    patch_recovery,
    permutation_p,
    ridge_rho,
)
from research_gym.interp.bundle import read_shard


FAMILIES = ("pointer_chase", "gated_pointer_chase", "reachability")
SWEEP_PAIRS = 200


def model_specs(matched: dict | None) -> list[dict]:
    specs = [
        {"name": f"LOOP-T s{seed}", "cell": f"LOOP-T-core4-ds-s{seed}", "label": "C3",
         "control": (f"LOOP-T-core4-s{seed}", "C0"), "primary": True}
        for seed in (1, 2, 3)
    ]
    specs.append({"name": "FF-U s1 (below gate)", "cell": "FF-U-core4-ds-inj-s1", "label": "latest",
                  "control": ("FF-U-core4-s1", "C0"), "primary": False})
    # Added 2026-10-12: the first untied seed to pass G0 (plan amendment).
    specs.append({"name": "FF-U s3", "cell": "FF-U-core4-ds-inj-s3", "label": "C3",
                  "control": ("FF-U-core4-s3", "C0"), "primary": False})
    if matched is not None:
        seed = matched["seed"]
        specs.append({"name": f"LOOP-T s{seed} matched", "cell": matched["cell_id"], "label": matched["label"],
                      "control": (f"LOOP-T-core4-s{seed}", "C0"), "primary": False})
    return specs


def summarize(step, depth, arrival, control_arrival) -> dict:
    rho = ridge_rho(step, arrival, depth)
    control_rho = ridge_rho(step, control_arrival, depth)
    low, high = paired_difference_interval(step, depth, arrival, control_arrival)
    p_value = permutation_p(step, arrival, depth)
    return {
        "rho": rho, "p": p_value, "control_rho": control_rho, "difference_interval": [low, high],
        "ridge": bool(rho > 0 and p_value < 0.01 and low > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--matched", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/matched_checkpoint.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/e1_arrival.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse", action="store_true", help="keep models already in --out; measure only the rest")
    args = parser.parse_args()

    matched = json.loads(args.matched.read_text(encoding="utf-8"))["chosen"] if args.matched.exists() else None
    pairs_by_family = {
        family: edge_rewire_pairs(read_shard(args.monitor_bundle, "monitor_calibration", family))
        for family in FAMILIES
    }
    report = {"matched": matched, "min_spread": MIN_SPREAD, "models": {}}
    if args.reuse and args.out.exists():
        report["models"] = json.loads(args.out.read_text(encoding="utf-8"))["models"]
    for spec in model_specs(matched):
        if spec["name"] in report["models"]:
            print("reused:", spec["name"], flush=True)
            continue
        model = load_analysis_model(args.runs, args.artifact_dir, spec["cell"], spec["label"], device=args.device)
        # The random-init control runs the same variant as the model it controls for.
        control = with_injection(
            load_analysis_model(args.runs, args.artifact_dir, *spec["control"], device=args.device),
            model.config.input_injection,
        )
        entry = {"cell_id": spec["cell"], "label": spec["label"], "step": model.step, "primary": spec["primary"],
                 "control": "@".join(spec["control"]), "families": {}}
        for family, pairs in pairs_by_family.items():
            result = patch_recovery(model, pairs)
            spread = result["clean_ld"] - result["corrupt_ld"]
            keep = result["clean_ok"] & result["corrupt_ok"] & (np.abs(spread) >= MIN_SPREAD)
            kept = pairs.subset(np.flatnonzero(keep))
            recovery = result["recovery"][keep, :, 0]
            # The control is scored on the model's own pairs; its logit gaps are often tiny, so no pair is
            # dropped for them and the statistics use every kept pair.
            control_result = patch_recovery(control, kept)
            control_spread = np.abs(control_result["clean_ld"] - control_result["corrupt_ld"])
            arrival = arrival_visits(recovery)
            control_arrival = arrival_visits(control_result["recovery"][:, :, 0])
            stats = summarize(kept.step, kept.depth, arrival, control_arrival)

            sweep_index = np.random.default_rng(0).permutation(len(kept))[:SWEEP_PAIRS]
            positions = tuple(range(kept.clean.shape[1]))
            sweep = patch_recovery(model, kept.subset(sweep_index), positions=positions)["recovery"]
            entry["families"][family] = {
                "pairs": len(pairs), "kept": int(keep.sum()),
                "control_spread_median": float(np.median(control_spread)),
                "control_spread_below_min": int((control_spread < MIN_SPREAD).sum()),
                "dropped_wrong": int((~(result["clean_ok"] & result["corrupt_ok"])).sum()),
                **stats,
                "arrival_matrix": arrival_matrix(recovery, kept.step),
                "arrival_matrix_by_depth": {
                    str(value): arrival_matrix(recovery[kept.depth == value], kept.step[kept.depth == value])
                    for value in np.unique(kept.depth)
                },
                "control_arrival_matrix": arrival_matrix(control_result["recovery"][:, :, 0], kept.step),
                "mean_arrival_by_step": {
                    str(value): float(arrival[kept.step == value].mean()) for value in np.unique(kept.step)
                },
                # Mean recovery by visit (rows) and position (columns) over the sweep pairs.
                "position_sweep": sweep.mean(axis=0).round(4).tolist(),
                "position_sweep_steps": kept.step[sweep_index].tolist(),
            }
            print(spec["name"], family, {key: entry["families"][family][key] for key in
                  ("kept", "rho", "p", "control_rho", "difference_interval", "ridge")}, flush=True)
        report["models"][spec["name"]] = entry

    gate = {
        family: all(report["models"][f"LOOP-T s{seed}"]["families"][family]["ridge"] for seed in (1, 2, 3))
        for family in FAMILIES
    }
    report["phase1_gate"] = {"by_family": gate, "met": any(gate.values())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n",
                        encoding="utf-8", newline="\n")
    print("phase 1 gate:", report["phase1_gate"])


if __name__ == "__main__":
    main()
