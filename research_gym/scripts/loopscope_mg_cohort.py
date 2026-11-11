"""Hash-gated RMP-MG-01 evaluation on the original fixed-depth Q4/Q5 models.

Preflight reads the committed training receipts. The gitignored .final.pt files
must be supplied on the machine holding the trained weights. No actor training,
checkpoint downloads, or replacement of missing weights occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("experiments/recurrence_monitorability_v0")
COHORTS = {"q4": ("loopscope_toy", 12), "q5": ("loopscope_toy_q5", 12)}


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(args):
    records = []
    for cohort in args.cohorts:
        folder, expected_count = COHORTS[cohort]
        receipt_paths = sorted((args.receipt_root / folder / "runs").glob("*/receipt.json"))
        if len(receipt_paths) != expected_count:
            raise ValueError(f"{cohort}: expected {expected_count} immutable training receipts, found {len(receipt_paths)}")
        for path in receipt_paths:
            receipt = json.loads(path.read_text())
            arm = path.parent.name
            filename = arm + ".final.pt"
            config = receipt.get("config") or {}
            sha = receipt.get("checkpoint_sha256", "")
            if receipt.get("arm") != arm or receipt.get("checkpoint") != filename:
                raise ValueError(f"{path}: arm or checkpoint name conflicts with receipt")
            if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
                raise ValueError(f"{path}: invalid SHA256")
            if config.get("fixed_depth") is not True or config.get("seed") != receipt.get("seed") or config.get("lam") != receipt.get("lam"):
                raise ValueError(f"{path}: inconsistent fixed-depth training configuration")
            weights = args.q4_dir if cohort == "q4" else args.q5_dir
            checkpoint = weights / filename
            found = checksum(checkpoint) if checkpoint.is_file() else None
            records.append({
                "cohort": cohort, "arm": arm, "seed": receipt["seed"], "lambda": receipt["lam"],
                "receipt": str(path), "receipt_sha256": checksum(path),
                "checkpoint": str(checkpoint), "expected_sha256": sha, "observed_sha256": found,
                "status": "MISSING" if found is None else "READY" if found == sha else "HASH_MISMATCH",
            })
    return records


def selected_models(records, count):
    if count == 0:
        return records[:]
    picked = []
    for cohort in COHORTS:
        grouped = {}
        for row in records:
            if row["cohort"] == cohort:
                grouped.setdefault(row["lambda"], []).append(row)
        for group in grouped.values():
            group.sort(key=lambda row: row["seed"])
        for index in range(max(map(len, grouped.values()), default=0)):
            for value in sorted(grouped):
                if index < len(grouped[value]) and sum(x["cohort"] == cohort for x in picked) < count:
                    picked.append(grouped[value][index])
    return picked


def jobs(row, args):
    root = args.out / row["cohort"] / row["arm"]
    prefix = [sys.executable, "-m"]
    path = row["checkpoint"]
    tasks = []
    if args.assay in ("both", "probes"):
        name = "probes_k" + "_".join(map(str, args.targets))
        command = prefix + ["research_gym.scripts.loopscope_bounded_probes",
            "--checkpoint", path, "--out", str(root / name),
            "--targets", *map(str, args.targets), "--group-counts", *map(str, args.group_counts),
            "--items-per-group", str(args.items_per_group), "--steps", str(args.probe_steps),
            "--batch-size", str(args.batch_size), "--hidden", str(args.hidden),
            "--threads", str(args.threads), "--device", args.device,
            "--dtype", args.dtype, "--seed", str(args.seed)]
        tasks.append((name, root / name, command))
    if args.assay in ("both", "causal"):
        for observation in args.observations:
            name = "causal_" + observation
            command = prefix + ["research_gym.scripts.loopscope_monitorability_gap",
                "--checkpoint", path, "--out", str(root / name),
                "--items", str(args.causal_items), "--visits", "1", "4", "8",
                "--horizon", str(args.horizon), "--observation", observation,
                "--rank-rtol", "1e-5" if args.dtype == "float32" else "1e-9",
                "--rank-atol", "1e-7" if args.dtype == "float32" else "1e-11",
                "--device", args.device, "--dtype", args.dtype, "--seed", str(args.seed)]
            tasks.append((name, root / name, command))
    return tasks


def execute(name, output, command, expected_sha, resume):
    result = output / "results.json"
    log = output.parent / (name + ".process.json")
    if resume and result.is_file():
        previous = json.loads(result.read_text())
        process = json.loads(log.read_text()) if log.is_file() else {}
        if (previous.get("checkpoint_sha256") == expected_sha and
            previous.get("status") in ("CHECKPOINT_ASSAY", "FROZEN_CHECKPOINT_DEVELOPMENT") and
            process.get("argv") == command and process.get("returncode") == 0):
            return {"name": name, "status": "REUSED_VERIFIED", "result": str(result)}
        raise ValueError(f"{result}: receipt or command provenance changed")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output}: previous output is nonempty; choose another --out")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    save(log, {"argv": command, "returncode": process.returncode,
               "stdout": process.stdout[-12000:], "stderr": process.stderr[-12000:]})
    if process.returncode:
        return {"name": name, "status": "FAILED", "returncode": process.returncode, "log": str(log)}
    if not result.is_file():
        return {"name": name, "status": "MISSING_RESULT", "log": str(log)}
    measured = json.loads(result.read_text())
    if measured.get("checkpoint_sha256") != expected_sha:
        raise ValueError(f"{result}: result checkpoint hash differs from training receipt")
    return {"name": name, "status": "SUCCEEDED", "result": str(result)}


def summarize(runs, blockers, selected_count):
    summaries = []
    for run in runs:
        model = run["model"]
        measurements = []
        for job in run["jobs"]:
            if job["status"] not in ("SUCCEEDED", "REUSED_VERIFIED"):
                continue
            data = json.loads(Path(job["result"]).read_text())
            if job["name"].startswith("probes_"):
                for study in data.get("results", []):
                    if study.get("family") != "mlp" or study.get("mode") != "separate" or study.get("shuffled_training_labels"):
                        continue
                    final = [row for row in study["rows"] if row["visit"] == study["target"]]
                    if len(final) != 1:
                        raise ValueError(f"{job['result']}: target-visit probe score missing")
                    g = final[0]["gain_from_input_only"]
                    measurements.append({"assay": job["name"], "target": study["target"],
                        "mlp_separate_gain_nats": g["mean_gain_nats"],
                        "bootstrap_percentile_95": g["bootstrap_percentile_95"],
                        "independent_test_groups": g["independent_groups"]})
            else:
                rows = data.get("rows", [])
                sensitivities = [float(row["local_direction"]["sensitivity"]) for row in rows]
                flips = [e["future_prediction_changed"] for row in rows
                         for e in row.get("finite_interventions", [])
                         if e["kind"] == "optimized_observation_null"]
                measurements.append({"assay": job["name"],
                    "assayed_visit_items": len(rows),
                    "mean_local_sensitivity": sum(sensitivities)/len(sensitivities) if sensitivities else None,
                    "finite_argmax_flip_fraction": sum(flips)/len(flips) if flips else None})
        summaries.append({"cohort": model["cohort"], "arm": model["arm"],
                          "seed": model["seed"], "lambda": model["lambda"],
                          "measurements": measurements})
    return {"schema_version": "rmp_mg01.cohort.v1",
        "status": "PARTIAL" if blockers else "TRAINED_COHORT_DEVELOPMENT",
        "selected_count": selected_count, "completed_count": len(runs),
        "missing_or_mismatched_count": len(blockers), "models": summaries,
        "statistical_unit": "checkpoint training seed; visits and interventions are repeated measures",
        "natural_language_monitor_evaluated": False,
        "prospective_model_seed_test": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt-root", type=Path, default=ROOT)
    p.add_argument("--q4-dir", type=Path, default=ROOT / "loopscope_toy" / "ckpt")
    p.add_argument("--q5-dir", type=Path, default=ROOT / "loopscope_toy_q5" / "ckpt")
    p.add_argument("--cohorts", choices=list(COHORTS), nargs="+", default=list(COHORTS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pilot-per-cohort", type=int, default=0)
    p.add_argument("--assay", choices=("probes", "causal", "both"), default="both")
    p.add_argument("--targets", choices=(8,16), type=int, nargs="+", default=[8,16])
    p.add_argument("--observations", choices=("cursor_logits", "all_token_logits"), nargs="+", default=["cursor_logits"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32","float64"), default="float32")
    p.add_argument("--group-counts", type=int, nargs=3, default=[12,4,4])
    p.add_argument("--items-per-group", type=int, default=8)
    p.add_argument("--probe-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--causal-items", type=int, default=2)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--seed", type=int, default=941118)
    args = p.parse_args(argv)
    if (args.pilot_per_cohort < 0 or min(args.group_counts) < 1 or
        min(args.items_per_group, args.probe_steps, args.batch_size, args.hidden,
            args.threads, args.causal_items, args.horizon) < 1):
        p.error("invalid nonpositive experiment budget")
    records = inventory(args)
    selected = selected_models(records, args.pilot_per_cohort)
    ids = {(r["cohort"], r["arm"]) for r in selected}
    for row in records:
        row["selected"] = (row["cohort"], row["arm"]) in ids
    blockers = [r for r in selected if r["status"] != "READY"]
    save(args.out / "inventory.json", {"schema_version": "rmp_mg01.cohort.v1",
        "action": "RUN" if args.run else "PREFLIGHT_ONLY",
        "records": records, "selected_count": len(selected),
        "ready_count": len(selected)-len(blockers),
        "blocked_count": len(blockers)})
    print(f"Q4/Q5 preflight: {len(records)} receipts, {len(selected)} selected, {len(blockers)} unavailable")
    for row in blockers[:12]:
        print(f"  {row['cohort']}/{row['arm']}: {row['status']} {row['checkpoint']}")
    if not args.run:
        return 2 if blockers else 0
    if blockers and not args.allow_partial:
        print("Strict checkpoint preflight failed; original weights required.", file=sys.stderr)
        return 2
    runs = []
    for row in selected:
        if row["status"] != "READY":
            continue
        outcomes = []
        for name, out, command in jobs(row, args):
            result = execute(name, out, command, row["expected_sha256"], args.resume)
            outcomes.append(result)
            print(f"{row['cohort']}/{row['arm']} {name}: {result['status']}", flush=True)
        runs.append({"model": row, "jobs": outcomes})
        save(args.out / "execution_receipts.json", {"runs": runs})
    save(args.out / "summary.json", summarize(runs, blockers, len(selected)))
    failures = [j for x in runs for j in x["jobs"] if j["status"] not in ("SUCCEEDED","REUSED_VERIFIED")]
    return 1 if failures else (2 if not runs else 0)


if __name__ == "__main__":
    raise SystemExit(main())
