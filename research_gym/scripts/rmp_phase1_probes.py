"""Phase 1, E2: trace probes with their baselines, transfer, MLP probes, and online code length."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.analysis import load_analysis_model, with_injection
from research_gym.interp.bundle import read_shard
from research_gym.interp.probes import (
    answer_states,
    control_labels,
    given_values,
    mlp_sites,
    online_code_ratio,
    probe_sites,
    rms_normalize,
    shuffled_within,
    trace_labels,
)


FAMILIES = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")
MARGIN = 0.05


def model_specs(matched: dict | None) -> list[dict]:
    specs = [{"name": f"LOOP-T s{seed}", "cell": f"LOOP-T-core4-ds-s{seed}", "label": "C3", "arm": "LOOP-T",
              "c0": f"LOOP-T s{seed} C0"} for seed in (1, 2, 3)]
    specs.append({"name": "FF-U s1 (below gate)", "cell": "FF-U-core4-ds-inj-s1", "label": "latest", "arm": "FF-U",
                  "c0": "FF-U s1 C0"})
    if matched is not None:
        specs.append({"name": f"LOOP-T s{matched['seed']} matched", "cell": matched["cell_id"],
                      "label": matched["label"], "arm": "LOOP-T", "c0": f"LOOP-T s{matched['seed']} C0"})
    baselines = [{"name": f"LOOP-T s{seed} C0", "cell": f"LOOP-T-core4-s{seed}", "label": "C0"} for seed in (1, 2, 3)]
    # The untied comparator runs with input re-injection, so its random-init control does too.
    baselines.append({"name": "FF-U s1 C0", "cell": "FF-U-core4-s1", "label": "C0", "injection": True})
    baselines += [{"name": f"{arm} permuted", "cell": f"{arm}-core4-perm-s1", "label": "latest"} for arm in ("LOOP-T", "FF-U")]
    return specs + baselines


def halves(rows) -> np.ndarray:
    """True for the selection half of monitor_calibration, split by group."""

    return np.array([hashlib.sha256(row.group_id.encode()).digest()[0] % 2 == 0 for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--matched", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/matched_checkpoint.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/e2_probes.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse", action="store_true", help="keep models already in --out; probe only the rest")
    args = parser.parse_args()
    device = torch.device(args.device)

    matched = json.loads(args.matched.read_text(encoding="utf-8"))["chosen"] if args.matched.exists() else None
    data = {}
    for family in FAMILIES:
        train = [row for row in read_shard(args.monitor_bundle, "monitor_train", family) if row.sibling is None]
        calibration = [row for row in read_shard(args.monitor_bundle, "monitor_calibration", family) if row.sibling is None]
        select = halves(calibration)
        data[family] = {
            "train": train, "calibration": calibration, "select": select,
            "labels": {"train": trace_labels(family, train), "calibration": trace_labels(family, calibration)},
            "given": {"train": given_values(family, train), "calibration": given_values(family, calibration)},
            "depth": np.array([row.difficulty for row in train]),
        }

    report: dict = {"margin": MARGIN, "matched": matched, "models": {}}
    if args.reuse and args.out.exists():
        report["models"] = json.loads(args.out.read_text(encoding="utf-8"))["models"]
    for spec in model_specs(matched):
        if spec["name"] in report["models"]:
            print("reused:", spec["name"], flush=True)
            continue
        cell_dir = args.artifact_dir / spec["cell"] / "checkpoints"
        result_path = args.runs / spec["cell"] / "result.json"
        finished = result_path.exists() and json.loads(result_path.read_text(encoding="utf-8"))["status"] != "paused"
        if not (cell_dir / f"{spec['label']}.pt").exists() or not finished:
            # A control still training has a latest.pt that is not its final model.
            print("not available yet, skipped:", spec["name"], flush=True)
            continue
        model = load_analysis_model(args.runs, args.artifact_dir, spec["cell"], spec["label"], device=device)
        if spec.get("injection"):
            model = with_injection(model, True)
        full = "c0" in spec
        entry = {"cell_id": spec["cell"], "label": spec["label"], "step": model.step, "families": {}}
        for family, block in data.items():
            tokens = {split: torch.tensor([row.tokens for row in block[split]]) for split in ("train", "calibration")}
            states, pooled = {}, {}
            for split, values in tokens.items():
                raw, bag = answer_states(model, values)
                states[split], pooled[split] = rms_normalize(raw), rms_normalize(bag).unsqueeze(1)
            select = block["select"]
            family_entry = {}
            for variable, steps in block["labels"]["train"].items():
                for step, train_labels in steps.items():
                    calibration_labels = block["labels"]["calibration"][variable][step]
                    split = {"select": np.where(select, calibration_labels, -1),
                             "report": np.where(~select, calibration_labels, -1)}
                    if min((split["select"] >= 0).sum(), (split["report"] >= 0).sum(), (train_labels >= 0).sum()) < 50:
                        continue
                    args_probe = (states["train"], train_labels, states["calibration"], split["select"],
                                  states["calibration"], split["report"])
                    main = probe_sites(*args_probe, device=device, transfer=full)
                    result = {"accuracy": main["accuracy"].tolist(), "weight_decay": main["weight_decay"].tolist(),
                              "examples": {key: int((value >= 0).sum()) for key, value in split.items()}}
                    if full:
                        classes = int(max(train_labels.max(), calibration_labels.max())) + 1
                        floor = probe_sites(states["train"], shuffled_within(train_labels, block["depth"], 0),
                                            *args_probe[2:], device=device)
                        control_train = np.where(train_labels >= 0, control_labels(block["given"]["train"], classes, 0), -1)
                        control_cal = control_labels(block["given"]["calibration"], classes, 0)
                        control = probe_sites(
                            states["train"], control_train, states["calibration"],
                            np.where(split["select"] >= 0, control_cal, -1), states["calibration"],
                            np.where(split["report"] >= 0, control_cal, -1), device=device,
                        )
                        bag = probe_sites(pooled["train"], train_labels, pooled["calibration"], split["select"],
                                          pooled["calibration"], split["report"], device=device)
                        result.update({
                            "transfer": main["transfer"].round(4).tolist(),
                            "floor": floor["accuracy"].tolist(),
                            "control_task": control["accuracy"].tolist(),
                            "input_bag": float(bag["accuracy"][0]),
                            "mlp": mlp_sites(states["train"], train_labels, states["calibration"], split["report"],
                                             device=device).tolist(),
                            "code_ratio": online_code_ratio(states["train"], train_labels, device=device).tolist(),
                        })
                    family_entry[f"{variable}@{step}"] = result
            entry["families"][family] = family_entry
            print(spec["name"], family, {key: [round(v, 2) for v in value["accuracy"]]
                                         for key, value in list(family_entry.items())[:3]}, flush=True)
        report["models"][spec["name"]] = {**entry, "spec": {k: v for k, v in spec.items() if k != "cell"}}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, entry in report["models"].items():
        spec = entry["spec"]
        if "c0" not in spec:
            continue
        c0 = report["models"].get(spec["c0"])
        permuted = report["models"].get(f"{spec['arm']} permuted")
        for family, probes in entry["families"].items():
            for key, result in probes.items():
                baselines = [np.array(result["floor"]), np.array(result["control_task"]),
                             np.full(len(result["accuracy"]), result["input_bag"])]
                missing = []
                for label, other in (("C0", c0), ("permuted", permuted)):
                    if other is None or key not in other["families"].get(family, {}):
                        missing.append(label)
                    else:
                        baselines.append(np.array(other["families"][family][key]["accuracy"]))
                best = np.max(np.stack(baselines), axis=0)
                result["best_baseline"] = best.tolist()
                result["computed"] = (np.array(result["accuracy"]) >= best + MARGIN).tolist()
                result["missing_baselines"] = missing

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8", newline="\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
