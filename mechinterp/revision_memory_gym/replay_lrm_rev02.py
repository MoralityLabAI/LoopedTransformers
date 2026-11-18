"""Independent CPU checkpoint replay from archived LRM-REV-02 inputs.

This module imports no runner, fixture or scorer. Each arm executes one full
24-token stream from zero state. The input receipt directory is read-only.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from torch import nn


ARMS = ("corrected", "random_0", "random_1", "random_2", "random_3")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checked_path(folder, relative):
    path = (folder / relative).resolve()
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_relative_to(folder):
        raise ValueError("artifact path leaves the input receipt directory")
    return path


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite numeric evidence required")
    return float(value)


def probability(values):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("three-class probability vector required")
    result = [number(value) for value in values]
    if min(result) < 0 or max(result) > 1 or abs(math.fsum(result) - 1) > 1e-6:
        raise ValueError("invalid probability vector")
    return result


def encode(row, slots):
    old, revised, replacements = row["old"], row["correction_slots"], row["replacements"]
    if (len(old) != 12 or len(replacements) != 6
            or any(type(bit) is not int or bit not in (0, 1) for bit in old + replacements)):
        raise ValueError("invalid archived report bits")
    for positions in (revised, slots):
        if (len(positions) != 6 or positions != sorted(positions) or len(set(positions)) != 6
                or any(type(i) is not int or not 0 <= i < 12 for i in positions)):
            raise ValueError("six sorted distinct positions required")
    events = [(0, i, bit) for i, bit in enumerate(old)]
    events += [(1, i, bit) for i, bit in zip(revised, replacements)]
    events += [(2, i, old[i]) for i in slots]
    inputs = torch.zeros((1, 24, 17), dtype=torch.float32, device="cpu")
    for step, (kind, position, bit) in enumerate(events):
        inputs[0, step, kind] = inputs[0, step, 3 + position] = inputs[0, step, 15 + bit] = 1
    return inputs


class ReplayGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(17, 32, num_layers=1, batch_first=True, device="cpu", dtype=torch.float32)
        self.head = nn.Linear(32, 3, device="cpu", dtype=torch.float32)

    def forward(self, inputs):
        states, _ = self.gru(inputs)
        return self.head(states), states


def replay_root(folder, summary, root):
    root_id = root["root"]
    result = {"root": root_id, "collection_status": root["status"], "rows_checked": 0,
              "probability_abs_max": 0.0, "hidden_displacement_abs_max": 0.0,
              "replayed_paired_effect": None, "paired_effect_abs_difference": None}
    count = root["development_count"]
    if type(count) is not int or not 0 <= count <= 512:
        raise ValueError("invalid development row count")
    if "checkpoint_file" not in root:
        if count or root["status"] == "COMPLETE":
            raise ValueError("scored or complete root is missing its checkpoint")
        result["checkpoint_loaded"] = False
        return result
    checkpoint = checked_path(folder, root["checkpoint_file"])
    if digest(checkpoint) != root["checkpoint_sha256"] or summary["artifact_sha256"].get(root["checkpoint_file"]) != root["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["root"] != root_id or payload["source_commit"] != summary["source_commit"]
            or payload["source_hashes"] != summary["source_hashes"]
            or payload["steps"] != root["train_steps_completed"]):
        raise ValueError("checkpoint provenance differs from the collection summary")
    model = ReplayGRU()
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all().item()
           for value in payload["model_state_dict"].values()):
        raise ValueError("checkpoint contains invalid FP32 weights")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    result["checkpoint_loaded"] = True
    if not count and "development_outcomes_file" not in root:
        return result
    relative = root["development_outcomes_file"]
    archive = checked_path(folder, relative)
    if digest(archive) != summary["artifact_sha256"].get(relative):
        raise ValueError("development archive SHA-256 mismatch")
    effects = []
    with gzip.open(archive, "rt", encoding="utf-8") as handle, torch.no_grad():
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row["root"] != root_id or row["episode_index"] != index or set(row["arms"]) != set(ARMS):
                raise ValueError("archived root, episode order or control family differs")
            target = probability(row["target_full"])
            initial = probability(row["initial_probability"])
            losses = {}
            for arm in ARMS:
                data = row["arms"][arm]
                if arm == "corrected" and data["slots"] != row["correction_slots"]:
                    raise ValueError("candidate reminder schedule differs from corrected slots")
                if len(data["curve"]) != 7 or [p["r"] for p in data["curve"]] != list(range(7)):
                    raise ValueError("incomplete reminder curve")
                logits, hidden = model(encode(row, data["slots"]))
                forecasts = logits[0].double().softmax(-1).tolist()
                result["probability_abs_max"] = max(result["probability_abs_max"],
                    max(abs(a - b) for a, b in zip(probability(forecasts[11]), initial)))
                for r, point in enumerate(data["curve"]):
                    forecast, archived = probability(forecasts[17 + r]), probability(point["probability"])
                    result["probability_abs_max"] = max(result["probability_abs_max"], max(abs(a - b) for a, b in zip(forecast, archived)))
                    displacement = float(torch.linalg.vector_norm(hidden[0, 17 + r].double() - hidden[0, 17].double()))
                    result["hidden_displacement_abs_max"] = max(result["hidden_displacement_abs_max"], abs(displacement - number(point["hidden_displacement_l2"])))
                losses[arm] = math.fsum((a - b) ** 2 for a, b in zip(forecasts[23], target))
            effects.append(math.fsum(losses[arm] for arm in ARMS[1:]) / 4 - losses["corrected"])
    if len(effects) != count:
        raise ValueError("archive length differs from declared development_count")
    result["rows_checked"] = len(effects)
    if effects:
        result["replayed_paired_effect"] = math.fsum(effects) / len(effects)
        result["paired_effect_abs_difference"] = abs(result["replayed_paired_effect"] - number(root["means"]["paired_effect"]))
    result["hidden_displacement_diagnostic_tolerance"] = 1e-5
    result["hidden_displacement_within_tolerance"] = result["hidden_displacement_abs_max"] <= 1e-5
    return result


def replay_run(directory):
    folder = Path(directory).resolve()
    report = {"schema_version": 1, "experiment_id": "LRM-REV-02", "passed": False,
              "complete": False, "run_status": None, "roots": [], "failures": [],
              "device": "cpu", "recurrence_dtype": "float32", "inference_batch_size": 1}
    try:
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        summary = json.loads((folder / "summary.json").read_text())
        report["run_status"] = summary["run_status"]
        if summary["run_status"] not in ("COMPLETE", "FAILED", "BUDGET_STOP"):
            raise ValueError("collection must have a final preserved status")
        registration = summary["registration"]
        for field in ("checkpoint_replay_probability_abs_max", "checkpoint_replay_effect_abs_max"):
            if number(registration["diagnostics"][field]) != 1e-6:
                raise ValueError("replay tolerance differs from the frozen contract")
        if (summary["planned_roots"] != [0, 1, 2, 3] or len(summary["roots"]) > 4
                or any(type(r["root"]) is not int for r in summary["roots"])
                or [r["root"] for r in summary["roots"]] != list(range(len(summary["roots"])))):
            raise ValueError("planned or attempted model roots differ from the sequential contract")
        if any(root["status"] != "COMPLETE" for root in summary["roots"][:-1]):
            raise ValueError("a later root follows an interrupted collection")
        manifest = folder / "source_manifest.json"
        source = json.loads(manifest.read_text())
        if (digest(manifest) != summary["artifact_sha256"].get("source_manifest.json")
                or source["source_commit"] != summary["source_commit"]
                or source["source_hashes"] != summary["source_hashes"]):
            raise ValueError("frozen source provenance mismatch")
        report["source_commit"] = summary["source_commit"]
        completed = [r["root"] for r in summary["roots"] if r["status"] == "COMPLETE"]
        if completed != summary["completed_roots"] or len(completed) != summary["n_units_completed"]:
            raise ValueError("independent-root completion accounting differs")
        for root in summary["roots"]:
            if root["status"] not in ("COMPLETE", "FAILED", "BUDGET_STOP"):
                raise ValueError("root collection status is unfinished")
            if root["status"] == "COMPLETE" and (root["development_count"] != 512 or root["train_steps_completed"] != 800):
                raise ValueError("complete root is missing registered training or evaluation")
            result = replay_root(folder, summary, root)
            report["roots"].append(result)
            if result["probability_abs_max"] > 1e-6 or (result["paired_effect_abs_difference"] or 0) > 1e-6:
                report["failures"].append(f"root {root['root']}: checkpoint behavior exceeds registered replay tolerance")
        report["complete"] = summary["run_status"] == "COMPLETE" and completed == [0, 1, 2, 3]
        if summary["run_status"] == "COMPLETE" and not report["complete"]:
            report["failures"].append("complete collection lacks all four complete model roots")
        report["passed"] = not report["failures"]
    except Exception as error:
        report["failures"].append(f"{type(error).__name__}: {error}")
    report["development_rows_checked"] = sum(r["rows_checked"] for r in report["roots"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = replay_run(args.out)
    if args.report:
        path = args.report.resolve()
        if path.is_relative_to(args.out.resolve()):
            raise ValueError("replay reports must be outside the raw receipt directory")
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
