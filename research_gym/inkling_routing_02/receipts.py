"""Verify immutable native receipts and summarize paired development effects."""

import argparse
from collections import Counter
import gzip
from itertools import product
import json
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
from zipfile import ZipFile

import numpy as np

from ..inkling_routing_01.checkpoint import FILE_HASHES, WEIGHTS, file_sha256
from ..inkling_routing_01.core import vector_metrics
from ..inkling_routing_01.local import PROMPTS, RECONSTRUCTION_LIMIT, score_reader, write_json
from .native import CONTROLS, NULL_LOGP, PASSES, PLACEBOS, REFERENCES, REPLAYS, digest, development_roots

CHANNELS = ("Q", "K", "V", "MoE")
SCIENTIFIC_MINIMUM = 0.10
RESOLUTION = 0.001
PREFLIGHT_SHA256 = "c2621a7900930f81ed53799ce439ee1c8f2d7006918a04ef57b5381630d15494"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20271122


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
            tensor_error = audit_pilot_vectors(path, by_key)
            primary = {
                "rows": len(rows), "roots": len(PROMPTS), "max_reconstruction_error": recon,
                "max_archive_beta_disagreement": tensor_error,
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


def audit_pilot_vectors(path, by_key):
    beta_error = 0.0
    for root in PROMPTS:
        with np.load(path / f"vectors_{root}.npz", allow_pickle=False) as archive:
            for p in PASSES:
                reader = {key: archive[f"pass{p}_{key}"] for key in
                          ("bank2", "bank3", "original", "restored", "message", "delivered", "query")}
                if (not np.array_equal(reader["original"], reader["restored"])
                        or not np.array_equal(reader["original"][0], archive["writer_last_token"][p - 2])):
                    raise ValueError(f"writer or restoration mismatch in {root} pass {p}")
                for (root_id, reader_pass, arm), row in by_key.items():
                    if root_id != root or reader_pass != p:
                        continue
                    lane = row["lane"]
                    fit = score_reader(reader["bank2"][lane], reader["bank3"][lane],
                                       reader["message"][lane], reader["query"])
                    error = max(abs(a - b) for a, b in zip(fit["beta3_qkvm"], row["patched_beta3_qkvm"]))
                    beta_error = max(beta_error, error)
                    if arm == "reader_rescue" and not np.array_equal(reader["delivered"][lane], reader["message"][0]):
                        raise ValueError(f"archived rescue mismatch in {root} pass {p}")
    if beta_error > 1e-10:
        raise ValueError(f"archived beta does not match native rows: {beta_error}")
    return beta_error


def labels(controls=CONTROLS):
    return ("clean", *controls, *(f"reference_{i:02d}" for i in range(REFERENCES)),
            *(f"placebo_{i:02d}" for i in range(PLACEBOS)))


def root_grid(rows, roots, *, reader_only=False, passes=PASSES):
    by_key = {(r["root_id"], r["reader_pass"], r["replay"], r["label"]): r for r in rows}
    expected = set(product(roots, passes, range(REPLAYS), labels()))
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("missing or duplicated native root/pass/replay/arm records")
    for row in rows:
        if reader_only:
            valid_path = (row["execution"] == "native_reader_only"
                          and row["clean_capture_relative_error"] < RECONSTRUCTION_LIMIT)
        else:
            calls = range(7, 12) if row["reader_pass"] == 2 else range(19, 24)
            valid_path = row["restored_calls"] == list(calls)
        if (row["validation_issues"] or not valid_path
                or max(row["reader_scores"]["reconstruction_relative_error"]) >= RECONSTRUCTION_LIMIT):
            raise ValueError(f"failed native instrumentation at {row['root_id']} pass {row['reader_pass']} {row['label']}")
    return by_key


def per_endpoint(grid, roots, pass_number, channel_index, controls=CONTROLS):
    arm_labels = labels(controls)

    def measurement(root, replay):
        return {label: grid[root, pass_number, replay, label]["reader_scores"]["beta3_qkvm"][channel_index]
                for label in arm_labels}

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
            "control_minus_clean": {label: beta[label] - beta["clean"] for label in controls},
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
    interval = None
    if floor is not None:
        samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, (BOOTSTRAP_DRAWS, n))
        means = -np.asarray(contrasts)[samples].mean(axis=1)
        interval = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "roots": n, "mean_zero_minus_sham": -mean(contrasts),
        "root_sd": stdev(contrasts) if n > 1 else None, "target_decrease_roots": sum(x > 0 for x in contrasts),
        "worst_observed_sham": worst, "sham_floor_95_95": floor, "resolution": resolution,
        "mechanistic_floor": max(SCIENTIFIC_MINIMUM, resolution, floor) if floor is not None else None,
        "descriptive_95_root_bootstrap_interval": interval,
        "sham_contrast_quantiles_05_50_95": np.quantile(sham_differences, [0.05, 0.5, 0.95]).tolist(),
        "max_leave_one_out_change": max(r["max_leave_one_out_change"] for r in roots),
        "roots_with_coordinate_dependent_sign": sum(
            min(r["zero_minus_each_coordinate_sham"]) < 0 < max(r["zero_minus_each_coordinate_sham"]) for r in roots
        ),
        "mean_control_minus_clean": {
            label: mean(r["control_minus_clean"][label] for r in roots)
            for label in roots[0]["control_minus_clean"]
        },
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
        identity = (manifest["source_sha256"], manifest.get("launch"), manifest["schema"], tuple(manifest["reader_passes"]))
        if (failures or manifest["development_only"] is not True or manifest["sealed_evaluations_accessed"] is not False
                or manifest["roster_sha256"] != expected_roster or (source is not None and source != identity)):
            raise ValueError(f"incompatible development shard: {path}")
        source = identity
        rows.extend(data)
        inputs.append({"run": path.name, "sha256_manifest": file_sha256(path / "sha256.json")})

    analyzed = sorted({r["root_id"] for r in rows})
    if not analyzed or not set(analyzed) <= set(roots):
        raise ValueError("expected development-only roots")
    reader_only = source[2] in ("ink-route-02.native-reader-development.v1", "ink-route-02.archived-pass3-native-reader.v1")
    passes = source[3]
    grid = root_grid(rows, analyzed, reader_only=reader_only, passes=passes)
    for row in rows:
        root = roots[row["root_id"]]
        for group in ("reference", "placebo"):
            if row["label"].startswith(group) and row["sham_coordinate"] != root[f"{group}_coordinates"][int(row["label"][-2:])]:
                raise ValueError("sham coordinate drift")
    endpoints = {f"pass{p}_{channel}": per_endpoint(grid, analyzed, p, c)
                 for p in passes for c, channel in enumerate(CHANNELS)}
    effects = {name: describe_endpoint(results) for name, results in endpoints.items()}
    first = [row for row in rows if row["replay"] == 0 and row["label"] != "clean"]
    summary = {
        "development_only": True, "sealed_evaluations_accessed": False, "roots": len(analyzed),
        "execution": "native_reader_only" if reader_only else "full_native_prefill",
        "cohort": "public FineWeb development" if roster else "fresh smoke prompts",
        "rows": len(rows), "effects": effects, "receipt_inputs": inputs,
        "max_reconstruction_error": max(max(r["reader_scores"]["reconstruction_relative_error"]) for r in rows),
        "max_dose_error": max(r["relative_sham_dose_error"] or 0 for r in rows),
        "rescues": sum(row["label"] == "reader_rescue" for row in rows),
        "confirmation": "BLOCKED_DEVELOPMENT_ONLY",
    }
    if reader_only:
        summary["max_clean_capture_error"] = max(r["clean_capture_relative_error"] for r in rows)
    else:
        summary.update(
            max_rescue_logprob_error=max(abs(r["reference_logprob_delta"]) for r in rows if r["label"] == "reader_rescue"),
            first_replay_interventions=len(first),
            first_replay_top_token_flips=sum(row["top_token_flip"] for row in first),
            first_replay_top_token_flips_by_arm=dict(Counter(row["label"] for row in first if row["top_token_flip"])),
            zero_logprob_deltas=[r["reference_logprob_delta"] for r in first if r["label"] == "zero_2027"],
            zero_logprob_nulls=sum(abs(r["reference_logprob_delta"]) <= NULL_LOGP for r in first if r["label"] == "zero_2027"),
            forward_seconds=sum(r["paired_forward_seconds"] for r in rows if r["lane"] == 0),
        )
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "root-effects.json", endpoints)
    if preflight_script is not None:
        summary["evalmath"] = preflight(preflight_script, out, endpoints["pass3_K"], effects["pass3_K"])
    write_json(out / "summary.json", summary)
    write_json(out / "sha256.json", {p.name: file_sha256(p) for p in out.iterdir()})


def bundle(path, out):
    """Export byte-preserved text receipts while keeping the local tensor manifest."""
    verified_run(path, tensor_test=True)
    out.mkdir(parents=True, exist_ok=False)
    for artifact in path.iterdir():
        if artifact.is_file() and artifact.suffix in (".json", ".jsonl"):
            if artifact.name == "records.jsonl":
                with artifact.open("rb") as original, (out / "records.jsonl.gz").open("wb") as compressed:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
                        shutil.copyfileobj(original, stream)
            else:
                shutil.copyfile(artifact, out / artifact.name)
    write_json(out / "bundled-sha256.json", {p.name: file_sha256(p) for p in out.iterdir()})


def reanalyze_pass3(path, roster, out):
    """Recover all four channel distributions from the earlier native K calibration."""
    manifest, rows, failures = verified_run(path)
    roots = {r["root_id"]: r for r in development_roots(roster)}
    roster_identity = read_json(roster)["roster_sha256"]
    if (manifest["schema"] != "ink-route-01.pass3-k.calibration.v1" or manifest["roster_sha256"] != roster_identity
            or manifest["sealed_evaluations_accessed"] is not False or failures):
        raise ValueError("expected the verified earlier public pass-3 calibration")
    controls = ("zero_2027", "norm_preserving_zero", "reader_rescue")
    expected = set(product(roots, (3,), range(REPLAYS), labels(controls)))
    grid = {(r["root_id"], r["reader_pass"], r["replay"], r["label"]): r for r in rows}
    if len(rows) != len(expected) or set(grid) != expected:
        raise ValueError("earlier pass-3 paired grid is incomplete")
    for row in rows:
        if row["validation_issues"] or not row["restored_bank_exact"] or row["reconstruction_relative_error_max"] >= RECONSTRUCTION_LIMIT:
            raise ValueError("earlier pass-3 native validation failed")
    endpoints = {channel: per_endpoint(grid, roots, 3, i, controls) for i, channel in enumerate(CHANNELS)}
    writers = []
    for root in roots:
        with np.load(path / f"vectors_{root}_replay0.npz", allow_pickle=False) as archive:
            writers.append([vector_metrics(vector, vector) for vector in archive["writer_last_token"]])
    writer_metrics = ("raw_coordinate", "rms_normalized_coordinate", "coordinate_squared_norm_fraction")
    writer_means = {
        str(p + 1): {metric: mean(root[p][metric] for root in writers) for metric in writer_metrics}
        for p in range(3)
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "pass3-root-effects.json", endpoints)
    write_json(out / "pass3-channel-summary.json", {
        "source_run_sha256_manifest": file_sha256(path / "sha256.json"),
        "development_only": True, "sealed_evaluations_accessed": False, "rows": len(rows),
        "effects": {channel: describe_endpoint(result) for channel, result in endpoints.items()},
        "exploratory_writer_means_by_pass": writer_means,
    })
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
    export = sub.add_parser("bundle")
    export.add_argument("--receipts", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    prior = sub.add_parser("reanalyze-pass3")
    prior.add_argument("--receipts", type=Path, required=True)
    prior.add_argument("--public-roster", type=Path, required=True)
    prior.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        if args.out.exists():
            raise FileExistsError(args.out)
        write_json(args.out, audit(args.history))
    elif args.command == "analyze":
        analyze(args.receipts, args.public_roster, args.out, args.preflight)
    elif args.command == "bundle":
        bundle(args.receipts, args.out)
    else:
        reanalyze_pass3(args.receipts, args.public_roster, args.out)


if __name__ == "__main__":
    main()
