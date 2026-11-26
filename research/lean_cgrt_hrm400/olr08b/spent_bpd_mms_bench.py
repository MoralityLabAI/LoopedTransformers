"""Exploratory spent-cohort diagnostic benchmark for OLR-08B (BPD predictiveness & MMS scaling).

Strictly runs ONLY on SPENT OLR-08A cohorts (106 spent TRAIN hard roots -> 18 spent OLR-08A CAL
hard roots). NEVER opens any uncontaminated calibration file, protected evaluation root, or SInt.

Produces:
  1. Replay-certified TRAIN action frames (>= 64 frames from OLR-08A's 33 solved TRAIN roots)
  2. Trained 3-seed OLR-08B models across all matched arms (recurrent, recurrent_bpd, direct_mlp,
     matched_compute_sham, one_way, linear) + explicit_memory + adaptive_ucb
  3. BPD decomposition (svd_seed_proxy vs learned_bpd) + RMS-matched Haar rotation null
     evaluation predicting: (a) branch utility, (b) tactic applicability, (c) failure recovery
  4. Configurable MMS memory budget benchmark across 0, 64, 256, 1024, 4096 MiB with exact
     build/lookup/maintenance timing, peak bytes, hit/eviction counts, and immutable receipt chain.

Explicitly tagged ``exploratory_spent_only: True`` — ineligible for gate decisions or promotion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import zipfile
import json

import numpy as np
import torch

OLR08B_DIR = Path(__file__).resolve().parent
OLR08_DIR = OLR08B_DIR.parent / "olr08"
for p in (str(OLR08B_DIR), str(OLR08_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run08  # noqa: E402
from allocator import search_olr08b  # noqa: E402
from bpd_frontier import BPDFrontierDecomposer, evaluate_bpd_predictiveness  # noqa: E402
from mms_cache import VALID_MMS_BUDGETS_MIB  # noqa: E402
from models import (  # noqa: E402
    MATCHED_PARAM_ARMS,
    NEURAL_ARMS,
    SEEDS,
    ExplicitMemoryPolicy,
    UtilityPredictor,
    fit_utility_model,
    hash_weights,
)
from run08b import extract_certified_frames, paired_file_ci, rate_summary, run_mms_budget_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--audit-zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.audit_zip) as z:
        reg = json.loads(z.read("registry.json"))
        prior = json.loads(z.read("prior.json"))
        by_id = {r["id"]: r for r in reg}
        teacher = [
            json.loads(z.read(n))
            for n in sorted(z.namelist())
            if n.startswith("source_s2/") and n.endswith(".json")
        ]
        spent_cal_ids = sorted(
            Path(n).stem for n in z.namelist() if n.startswith("cal_s2/") and n.endswith(".json")
        )

    tr_by_id = {r["id"]: by_id[r["id"]] for r in teacher}
    cal_rows = [by_id[i] for i in spent_cal_ids]
    for r in (*tr_by_id.values(), *cal_rows):
        if r["role"] == "evaluation" or r["family"] == "SInt" or "/SInt" in r["file"]:
            raise RuntimeError("Contamination guard: non-spent root in spent BPD/MMS benchmark")

    # 1. Extract replay-certified TRAIN frames (69 frames from 33 solved TRAIN paths)
    fp = args.out / "source_frames.pt"
    if fp.exists():
        frames = torch.load(fp, map_location="cpu", weights_only=False)
    else:
        frames, f_meters = extract_certified_frames(args.exe, args.source, teacher, tr_by_id, prior)
        run08.write(args.out / "frame_meter.json", f_meters)
        torch.save(frames, fp)
    print(f"[spent_bench] Replay-certified TRAIN frames: {len(frames)}", flush=True)

    # 2. Fit explicit_memory control on TRAIN teacher rows
    explicit_pol = ExplicitMemoryPolicy()
    explicit_meta = explicit_pol.fit_from_source_rows(teacher)

    # 3. Baselines on 18 spent CAL roots: s1, s2, s3, s3c, portfolios, adaptive_ucb
    base_runs = {}
    for arm in ("s1", "s2", "s3", "s3c", "portfolio_rr2", "portfolio_weighted", "portfolio_rr4", "adaptive_ucb"):
        base_runs[arm] = run08.stage(
            args.out,
            f"spent_cal_{arm}",
            cal_rows,
            lambda r, a=arm: search_olr08b(
                args.exe, args.source, r, a, prior=prior if a not in ("s1", "s2") else None,
                limit=run08.STEP_CAP, mms_budget_mib=0.0,
            ),
            args.workers,
        )

    # 4. Configurable MMS memory budget sweep (0, 64, 256, 1024, 4096 MiB) on all 18 spent CAL roots
    mms_sweep = run_mms_budget_sweep(
        args.exe, args.source, cal_rows, prior, explicit_pol, args.out,
        budgets_mib=VALID_MMS_BUDGETS_MIB, workers=args.workers, limit=run08.STEP_CAP,
    )

    # 5. Train 3-seed matched neural arms on TRAIN frames and score on 18 spent CAL roots
    fd = args.out / "fits"
    fd.mkdir(exist_ok=True)
    model_results = {}
    bpd_cal_recs = []
    bpd_train_recs = []

    for seed in SEEDS:
        sigs = []
        for arm in NEURAL_ARMS:
            key = f"{seed}_{arm}"
            pt_p = fd / f"{key}.pt"
            meta_p = fd / f"{key}.json"
            if pt_p.exists() and meta_p.exists():
                stats = run08.read(meta_p)
                m = UtilityPredictor(arm)
                m.load_state_dict(torch.load(pt_p, map_location="cpu", weights_only=True))
                m.eval()
                bpd_dec = BPDFrontierDecomposer.from_model(m, rank_c=8, learned=False, seed=seed) if arm in ("recurrent", "recurrent_bpd") else None
            else:
                m, bpd_dec, stats = fit_utility_model(
                    frames, arm, seed, steps=args.steps, bpd_rank=8, bpd_learned=(arm == "recurrent_bpd")
                )
                torch.save(m.state_dict(), pt_p)
                run08.write(meta_p, stats)
            if arm in MATCHED_PARAM_ARMS:
                sigs.append((stats["parameters"], stats["initial_sha256"], stats["sampler_sha256"]))

            collect_probe = (seed == SEEDS[0] and arm == "recurrent_bpd")
            rr = run08.stage(
                args.out,
                f"spent_cal_model_{key}",
                cal_rows,
                lambda row, m=m, bpd_dec=bpd_dec, arm=arm, cp=collect_probe: search_olr08b(
                    args.exe, args.source, row, arm, prior=prior, model=m, bpd_decomposer=bpd_dec,
                    alpha=1.0, limit=run08.STEP_CAP, mms_budget_mib=64.0, collect_bpd_probe=cp,
                ),
                1,
            )
            if collect_probe:
                for x in rr:
                    bpd_cal_recs.extend(x.get("bpd_probe_records", []))
                # Collect TRAIN probe records on 24 TRAIN roots for train->cal BPD predictiveness fit
                tr_sub = sorted(tr_by_id.values(), key=lambda x: run08.digest([run08.SEED, "bpd_spent", x["id"]]))[:24]
                tr_rr = run08.stage(
                    args.out,
                    f"spent_train_bpd_probe_{key}",
                    tr_sub,
                    lambda row, m=m, bpd_dec=bpd_dec: search_olr08b(
                        args.exe, args.source, row, "recurrent_bpd", prior=prior, model=m, bpd_decomposer=bpd_dec,
                        alpha=1.0, limit=150, mms_budget_mib=64.0, collect_bpd_probe=True,
                    ),
                    1,
                )
                for x in tr_rr:
                    bpd_train_recs.extend(x.get("bpd_probe_records", []))

            model_results[key] = {
                "arm": arm,
                "seed": seed,
                "fit": stats,
                "calibration": rate_summary(rr),
                "paired_vs_s2": paired_file_ci(rr, base_runs["s2"]),
                "paired_vs_s3": paired_file_ci(rr, base_runs["s3"]),
            }
        if len(set(sigs)) != 1:
            raise ValueError(f"matched param/init/sampler drift on seed {seed}")

    bpd_pred = evaluate_bpd_predictiveness(bpd_train_recs, bpd_cal_recs)
    seed_var = {}
    for arm in NEURAL_ARMS:
        rs = [model_results[f"{s}_{arm}"]["calibration"]["rate"] for s in SEEDS]
        seed_var[arm] = {
            "seeds": list(SEEDS),
            "rates": [round(v, 4) for v in rs],
            "mean_rate": round(float(np.mean(rs)), 4),
            "std_rate": round(float(np.std(rs)), 4),
        }

    report = {
        "schema": "olr08b.spent_bpd_mms_bench.v1",
        "exploratory_spent_only": True,
        "sealed_SInt_opened": False,
        "protected_eval_opened": False,
        "train_frames": len(frames),
        "spent_cal_roots": len(cal_rows),
        "explicit_memory_fit": explicit_meta,
        "controls": {k: rate_summary(v) for k, v in base_runs.items()},
        "mms_budget_sweep": mms_sweep,
        "models": model_results,
        "seed_variation": seed_var,
        "bpd_probe_counts": {"train_events": len(bpd_train_recs), "eval_events": len(bpd_cal_recs)},
        "bpd_predictiveness": bpd_pred,
    }
    run08.write(args.out / "spent_bpd_mms_report.json", report)
    print("[spent_bench] DONE", json.dumps(seed_var, indent=2), flush=True)
    print("[spent_bench] BPD predictiveness:", json.dumps(bpd_pred, indent=2), flush=True)


if __name__ == "__main__":
    main()
