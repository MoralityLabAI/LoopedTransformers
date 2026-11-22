"""Verify immutable native receipts and summarize paired development effects."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from zipfile import ZipFile

import numpy as np

from ..inkling_routing_01.checkpoint import FILE_HASHES, WEIGHTS, file_sha256
from ..inkling_routing_01.local import PROMPTS, RECONSTRUCTION_LIMIT, write_json
from .native import CONTROLS, NULL_LOGP, PASSES, PLACEBOS, REFERENCES, REPLAYS, digest, development_roots

CHANNELS = ("Q", "K", "V", "MoE")
SCIENTIFIC_MINIMUM = 0.10
RESOLUTION = 0.001
PREFLIGHT_SHA256 = "c2621a7900930f81ed53799ce439ee1c8f2d7006918a04ef57b5381630d15494"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verified_run(path, *, tensor_test=False):
    hashes = read_json(path / "sha256.json")
    names = {p.name for p in path.iterdir() if p.is_file()} - {"sha256.json"}
    if names != set(hashes):
        raise ValueError(f"{path.name}: unexpected or missing receipt files")
    for name, expected in hashes.items():
        artifact = path / name
        if file_sha256(artifact) != expected:
            raise ValueError(f"{artifact}: SHA-256 mismatch")
        if tensor_test and artifact.suffix == ".npz":
            with ZipFile(artifact) as archive:
                if archive.testzip() is not None or "writer_last_token.npy" not in archive.namelist():
                    raise ValueError(f"{artifact}: corrupt or incomplete tensor archive")
    manifest = read_json(path / "manifest.json")
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError(f"{path.name}: manifest mismatch")
    rows = read_rows(path / "records.jsonl")
    failures = read_rows(path / "failures.jsonl")
    identity = read_json(path / "checkpoint.json")
    if rows and (failures or identity["weight_sha256"] != FILE_HASHES[WEIGHTS]):
        raise ValueError(f"{path.name}: failed run or checkpoint mismatch")
    return manifest, rows, failures


def audit(history):
    index = []
    primary = None
    for path in sorted(history.glob("native-*")) + sorted(history.glob("ink-route-pass3-k-native60")):
        manifest, rows, failures = verified_run(path, tensor_test=True)
        record = {
            "run": path.name, "rows": len(rows), "roots": len({r["root_id"] for r in rows}),
            "file_manifest_sha256": file_sha256(path / "sha256.json"),
            "plan_sha256": manifest["manifest_sha256"],
            "tensor_archives": sum(p.suffix == ".npz" for p in path.iterdir()),
            "failure_stages": [row["stage"] for row in failures],
        }
        index.append(record)
        if path.name == "native-cuda-dev8-01":
            expected = set(product(PROMPTS, PASSES, manifest["arms"]))
            by_key = {(r["root_id"], r["reader_pass"], r["arm"]): r for r in rows}
            if len(rows) != len(expected) or set(by_key) != expected or record["tensor_archives"] != 8:
                raise ValueError("historical eight-root grid incomplete")
            recon = max(max(r["reconstruction_relative_error"]) for r in rows)
            if recon >= RECONSTRUCTION_LIMIT or any(not r["restored_bank_exact"] for r in rows):
                raise ValueError("historical reconstruction/restoration failed")
            primary = {
                "rows": len(rows), "roots": len(PROMPTS), "max_reconstruction_error": recon,
                "mean_pass3_k_zero_minus_sham": mean(
                    by_key[root, 3, "zero_2027"]["patched_beta3_qkvm"][1]
                    - by_key[root, 3, "l2_sham"]["patched_beta3_qkvm"][1] for root in PROMPTS
                ),
                "rescues": dict(Counter(r["rescue_status"] for r in rows if r["arm"] == "reader_rescue")),
                "top_token_flips": sum(r["top1_flip"] for r in rows),
                "checkpoint": read_json(path / "checkpoint.json"),
                "timing": read_json(path / "timing.json"),
                "resources_after": read_json(path / "resources_after.json"),
            }
    if primary is None:
        raise ValueError("historical eight-root CUDA receipts missing")
    return {"merge_commit": "f511affb0aa2346e8d72b8b5f8af9dded241d5da", "runs": index, "pilot": primary}


def labels():
    return ("clean", *CONTROLS, *(f"reference_{i:02d}" for i in range(REFERENCES)),
            *(f"placebo_{i:02d}" for i in range(PLACEBOS)))


def root_grid(rows, roots):
    by_key = {(r["root_id"], r["reader_pass"], r["replay"], r["label"]): r for r in rows}
    expected = set(product(roots, PASSES, range(REPLAYS), labels()))
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("missing or duplicated native root/pass/replay/arm records")
    for row in rows:
        if (row["validation_issues"] or row["restored_calls"] != list(range(7, 12) if row["reader_pass"] == 2 else range(19, 24))
                or max(row["reader_scores"]["reconstruction_relative_error"]) >= RECONSTRUCTION_LIMIT):
            raise ValueError(f"failed native instrumentation at {row['root_id']} pass {row['reader_pass']} {row['label']}")
    return by_key


def per_endpoint(grid, roots, pass_number, channel_index):
    def measurement(root, replay):
        return {label: grid[root, pass_number, replay, label]["reader_scores"]["beta3_qkvm"][channel_index]
                for label in labels()}

    result = []
    for root in roots:
        def paired(beta):
            refs = [beta[f"reference_{i:02d}"] for i in range(REFERENCES)]
            reference = mean(refs)
            shams = [reference - beta[f"placebo_{i:02d}"] for i in range(PLACEBOS)]
            return refs, reference, shams, reference - beta["zero_2027"]

        beta = measurement(root, 0)
        refs, reference, shams, effect = paired(beta)
        _, _, replay_shams, replay_effect = paired(measurement(root, 1))
        result.append({
            "root_id": root, "paired_effect": effect, "reference_minus_placebo": shams,
            "worst_sham": max(map(abs, shams)),
            "max_replay_discrepancy": max(abs(effect - replay_effect),
                                          *(abs(a - b) for a, b in zip(shams, replay_shams))),
            "zero_minus_each_coordinate_sham": [beta["zero_2027"] - value for value in refs],
            "max_leave_one_out_change": max(abs(value - reference) / (REFERENCES - 1) for value in refs),
            "control_minus_clean": {label: beta[label] - beta["clean"] for label in CONTROLS},
        })
    return result


def describe_endpoint(roots):
    contrasts = [r["paired_effect"] for r in roots]
    n = len(roots)
    worst = max(r["worst_sham"] for r in roots)
    floor = worst if 1 - 0.95**n >= 0.95 else None
    max_replay = max(r["max_replay_discrepancy"] for r in roots)
    resolution = max(RESOLUTION, max_replay)
    sham_differences = [x for r in roots for x in r["reference_minus_placebo"]]
    return {
        "roots": n, "mean_zero_minus_sham": -mean(contrasts),
        "root_sd": stdev(contrasts) if n > 1 else None, "target_decrease_roots": sum(x > 0 for x in contrasts),
        "worst_observed_sham": worst, "sham_floor_95_95": floor, "resolution": resolution,
        "mechanistic_floor": max(SCIENTIFIC_MINIMUM, resolution, floor) if floor is not None else None,
        "sham_contrast_quantiles_05_50_95": np.quantile(sham_differences, [0.05, 0.5, 0.95]).tolist(),
        "max_leave_one_out_change": max(r["max_leave_one_out_change"] for r in roots),
        "roots_with_coordinate_dependent_sign": sum(
            min(r["zero_minus_each_coordinate_sham"]) < 0 < max(r["zero_minus_each_coordinate_sham"]) for r in roots
        ),
        "mean_control_minus_clean": {label: mean(r["control_minus_clean"][label] for r in roots) for label in CONTROLS},
    }


def preflight(path, out, primary, summary):
    if file_sha256(path) != PREFLIGHT_SHA256:
        raise ValueError("use the audited EvalMath preflight script")
    plan = {
        "schema_version": 1, "experiment_id": "INK-ROUTE-02-development",
        "independent_unit": "one public FineWeb document per domain",
        "pilot_from_development_only": True, "units_independent": True,
        "pilot": [{"unit_id": r["root_id"], "paired_effect": str(r["paired_effect"]),
                   "sham_effects": list(map(str, r["reference_minus_placebo"]))} for r in primary],
        "scientific_min_effect": str(SCIENTIFIC_MINIMUM), "measurement_resolution": str(summary["resolution"]),
        "target_effect": "0.20", "confirm_units_planned": 1800, "planned_comparisons": 1,
        "alpha": "0.05", "heldout_sealed": False, "sham_confidence": "0.95", "sham_quantile": "0.95",
    }
    write_json(out / "preflight-input.json", plan)
    process = subprocess.run(
        [sys.executable, str(path), "--pilot", str(out / "preflight-input.json"), "--strict"],
        check=False, capture_output=True, text=True,
    )
    (out / "preflight-stderr.log").write_text(process.stderr, encoding="utf-8")
    report = json.loads(process.stdout)
    if process.returncode not in (0, 2):
        raise RuntimeError(f"EvalMath preflight failed with exit code {process.returncode}")
    write_json(out / "preflight-strict.json", report)
    return {"exit_code": process.returncode, "decision": report["decision"], "reasons": report["reasons"]}


def analyze(paths, roster, out, preflight_script):
    roots = {r["root_id"]: r for r in development_roots(roster)}
    source = None
    rows, inputs = [], []
    expected_roster = file_sha256(roster) if roster else None
    for path in paths:
        manifest, data, failures = verified_run(path)
        identity = (manifest["source_sha256"], manifest["launch"])
        if (failures or manifest["development_only"] is not True or manifest["sealed_evaluations_accessed"] is not False
                or manifest["roster_sha256"] != expected_roster or (source is not None and source != identity)):
            raise ValueError(f"incompatible development shard: {path}")
        source = identity
        rows.extend(data)
        inputs.append({"run": path.name, "sha256_manifest": file_sha256(path / "sha256.json")})

    analyzed = sorted({r["root_id"] for r in rows})
    if not analyzed or not set(analyzed) <= set(roots):
        raise ValueError("expected development-only roots")
    grid = root_grid(rows, analyzed)
    for row in rows:
        root = roots[row["root_id"]]
        for group in ("reference", "placebo"):
            if row["label"].startswith(group) and row["sham_coordinate"] != root[f"{group}_coordinates"][int(row["label"][-2:])]:
                raise ValueError("sham coordinate drift")
    endpoints = {f"pass{p}_{channel}": per_endpoint(grid, analyzed, p, c)
                 for p in PASSES for c, channel in enumerate(CHANNELS)}
    effects = {name: describe_endpoint(results) for name, results in endpoints.items()}
    first = [row for row in rows if row["replay"] == 0 and row["label"] != "clean"]
    summary = {
        "development_only": True, "sealed_evaluations_accessed": False, "roots": len(analyzed),
        "cohort": "public FineWeb development" if roster else "fresh smoke prompts",
        "rows": len(rows), "effects": effects, "receipt_inputs": inputs,
        "max_reconstruction_error": max(max(r["reader_scores"]["reconstruction_relative_error"]) for r in rows),
        "max_dose_error": max(r["relative_sham_dose_error"] or 0 for r in rows),
        "max_rescue_logprob_error": max(abs(r["reference_logprob_delta"]) for r in rows if r["label"] == "reader_rescue"),
        "first_replay_counts": dict(Counter(row["label"] for row in first if row["top_token_flip"])),
        "zero_logprob_deltas": [r["reference_logprob_delta"] for r in first if r["label"] == "zero_2027"],
        "zero_logprob_nulls": sum(abs(r["reference_logprob_delta"]) <= NULL_LOGP for r in first if r["label"] == "zero_2027"),
        "forward_seconds": sum(r["paired_forward_seconds"] for r in rows if r["lane"] == 0),
        "confirmation": "BLOCKED_DEVELOPMENT_ONLY",
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "root-effects.json", endpoints)
    if preflight_script is not None:
        summary["evalmath"] = preflight(preflight_script, out, endpoints["pass3_K"], effects["pass3_K"])
    write_json(out / "summary.json", summary)
    write_json(out / "sha256.json", {p.name: file_sha256(p) for p in out.iterdir()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    historical = sub.add_parser("audit")
    historical.add_argument("--history", type=Path, required=True)
    historical.add_argument("--out", type=Path, required=True)
    development = sub.add_parser("analyze")
    development.add_argument("--receipts", type=Path, nargs="+", required=True)
    development.add_argument("--public-roster", type=Path)
    development.add_argument("--preflight", type=Path)
    development.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        if args.out.exists():
            raise FileExistsError(args.out)
        write_json(args.out, audit(args.history))
    else:
        analyze(args.receipts, args.public_roster, args.out, args.preflight)


if __name__ == "__main__":
    main()
