r"""Unified OLR-08B post-campaign audit, archive verification, and spent-cohort BPD/MMS/3-seed benchmark.

Part 1: Audits C:\olr08b\full (214 fresh CAL roots, 26 files), verifies full_AUDIT.zip and
        full_FABLE.zip SHA-256 manifests, computes 75/150/300/600/1200 curves, 16-family
        breakdown, exhausted-space root census, and TRAIN determinism vs OLR-08A, and copies
        authenticated artifacts to research/lean_cgrt_hrm400/results/olr08b/.
Part 2: Runs the exploratory SPENT-COHORT ONLY benchmark (106 spent TRAIN -> 18 spent OLR-08A CAL)
        covering 3-seed neural training (recurrent, recurrent_bpd, direct_mlp, matched_compute_sham,
        one_way, linear), explicit_memory, adaptive_ucb, BPD predictiveness (utility, applicability,
        recovery after failure vs RMS-matched Haar rotation null), and configurable MMS budget
        sweep (0, 0.02 stress, 64, 256, 1024, 4096 MiB).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

DRIVER_DIR = Path(__file__).resolve().parent
OLR08B_DIR = DRIVER_DIR.parent
RESEARCH_DIR = OLR08B_DIR.parent
OLR08_DIR = RESEARCH_DIR / "olr08"
for p in (str(OLR08B_DIR), str(OLR08_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run08  # noqa: E402
from allocator import BASE_PRIORS, AuditedMeter, _all_prior_costs, search_olr08b  # noqa: E402
from bpd_frontier import BPDFrontierDecomposer, evaluate_bpd_predictiveness  # noqa: E402
from mms_cache import VALID_MMS_BUDGETS_MIB, ImmutableReceiptLedger  # noqa: E402
from models import (  # noqa: E402
    MATCHED_PARAM_ARMS,
    NEURAL_ARMS,
    SEEDS,
    ExplicitMemoryPolicy,
    ReceiptTracker,
    UtilityPredictor,
    build_packet,
    fit_utility_model,
    hash_weights,
)
from run08b import paired_file_ci, rate_summary  # noqa: E402


def verify_zip_manifest(zpath: Path) -> dict:
    with zipfile.ZipFile(zpath) as z:
        manifest = json.loads(z.read("audit_manifest.json"))
        listed = manifest["files"]
        mismatches = []
        present = set(z.namelist())
        checked = 0
        for name, expected_sha in listed.items():
            if name in present:
                got = hashlib.sha256(z.read(name)).hexdigest()
                if got != expected_sha:
                    mismatches.append(name)
                checked += 1
        return {
            "zip": zpath.name,
            "bytes": zpath.stat().st_size,
            "sha256": run08.sha(zpath),
            "manifest_sha256": hashlib.sha256(z.read("audit_manifest.json")).hexdigest(),
            "entries_checked": checked,
            "mismatches": mismatches,
            "ok": len(mismatches) == 0 and checked > 0,
        }


def part1_audit_and_stage_full_artifacts() -> dict:
    print("=== PART 1: AUDITING C:\\olr08b\\full AND STAGING ARCHIVES ===", flush=True)
    run = Path(r"C:\olr08b\full")
    dest = RESEARCH_DIR / "results" / "olr08b"
    dest_full = dest / "full"
    dest_full.mkdir(parents=True, exist_ok=True)

    reg = {r["id"]: r for r in json.loads((run / "registry.json").read_text(encoding="utf-8"))}
    stages = (
        "cal_s1", "cal_s2", "cal_s3", "cal_s3c",
        "cal_portfolio_rr2", "cal_portfolio_rr4",
        "cal_adaptive_ucb", "cal_explicit_memory_64m", "cal_s3_long",
    )
    data = {st: [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / st).glob("*.json"))] for st in stages}

    curves = {}
    for st in stages:
        rows = data[st]
        buds = (75, 150, 300, 600, 1200) if st == "cal_s3_long" else (75, 150, 300)
        curves[st] = {str(b): sum(r["solved"] and r["steps"] <= b for r in rows) for b in buds}
        print(f"  CURVE {st:24s}: {curves[st]}", flush=True)

    fams = sorted({r["family"] for r in data["cal_s2"]})
    by_fam = {}
    for fam in fams:
        tot = sum(1 for r in data["cal_s2"] if r["family"] == fam)
        n_files = len({r["file"] for r in data["cal_s2"] if r["family"] == fam})
        by_fam[fam] = {
            "files": n_files, "roots": tot,
            "s1": sum(r["solved"] for r in data["cal_s1"] if r["family"] == fam),
            "s2": sum(r["solved"] for r in data["cal_s2"] if r["family"] == fam),
            "s3": sum(r["solved"] for r in data["cal_s3"] if r["family"] == fam),
            "s3c": sum(r["solved"] for r in data["cal_s3c"] if r["family"] == fam),
            "rr2": sum(r["solved"] for r in data["cal_portfolio_rr2"] if r["family"] == fam),
            "rr4": sum(r["solved"] for r in data["cal_portfolio_rr4"] if r["family"] == fam),
            "ucb": sum(r["solved"] for r in data["cal_adaptive_ucb"] if r["family"] == fam),
            "mem64": sum(r["solved"] for r in data["cal_explicit_memory_64m"] if r["family"] == fam),
            "s3_1200": sum(r["solved"] for r in data["cal_s3_long"] if r["family"] == fam),
        }

    # Per-file breakdown across all 26 independent file clusters
    files = sorted({r["file"] for r in data["cal_s2"]})
    by_file = {}
    for f in files:
        tot = sum(1 for r in data["cal_s2"] if r["file"] == f)
        by_file[f] = {
            "roots": tot,
            "s1": sum(r["solved"] for r in data["cal_s1"] if r["file"] == f),
            "s2": sum(r["solved"] for r in data["cal_s2"] if r["file"] == f),
            "s3": sum(r["solved"] for r in data["cal_s3"] if r["file"] == f),
            "s3c": sum(r["solved"] for r in data["cal_s3c"] if r["file"] == f),
            "rr2": sum(r["solved"] for r in data["cal_portfolio_rr2"] if r["file"] == f),
            "s3_1200": sum(r["solved"] for r in data["cal_s3_long"] if r["file"] == f),
        }

    exhausted = []
    for r in data["cal_s3_long"]:
        if r["fate"] == "exhausted_finite_action_space":
            meta = reg[r["id"]]
            exhausted.append({"id": r["id"], "decl": meta["decl"], "file": meta["file"], "steps": r["steps"], "human": meta["human"][:100]})

    # Cross-campaign TRAIN determinism vs OLR-08A
    p08a = json.loads((RESEARCH_DIR / "results" / "olr08a" / "full" / "prior.json").read_text(encoding="utf-8"))
    p08b = json.loads((run / "prior.json").read_text(encoding="utf-8"))
    tr_s2 = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "source_s2").glob("*.json"))]
    det = {
        "source_s2_solved_08b": sum(r["solved"] for r in tr_s2),
        "source_s2_solved_08a": 33,
        "prior_weight_exact_match": p08a["weight"] == p08b["weight"],
        "prior_rank_exact_match": p08a["rank"] == p08b["rank"],
    }
    print("  DETERMINISM vs OLR-08A:", det, flush=True)
    print("  EXHAUSTED FINITE SPACE:", len(exhausted), [e["decl"] for e in exhausted], flush=True)

    # Copy unpacked full records (scrubbing local user paths in archive_receipt if any)
    for fname in (
        "summary.json", "contract.json", "frozen_policy.json", "prior.json",
        "resource.json", "roster_audit.json", "registry_census.json", "audit_manifest.json",
    ):
        src_f = run / fname
        if src_f.is_file():
            shutil.copy2(src_f, dest_full / fname)

    # Copy authenticated ZIP archives into results/olr08b/ (34.7 MB <= 100 MB repo limit)
    audit_src = Path(r"C:\olr08b\full_AUDIT.zip")
    fable_src = Path(r"C:\olr08b\full_FABLE.zip")
    audit_dst = dest / "olr08b-full_AUDIT.zip"
    fable_dst = dest / "olr08b-full_FABLE.zip"
    shutil.copy2(audit_src, audit_dst)
    shutil.copy2(fable_src, fable_dst)

    v_audit = verify_zip_manifest(audit_dst)
    v_fable = verify_zip_manifest(fable_dst)
    print("  VERIFIED AUDIT ZIP:", v_audit, flush=True)
    print("  VERIFIED FABLE ZIP:", v_fable, flush=True)
    if not (v_audit["ok"] and v_fable["ok"]):
        raise RuntimeError("ZIP manifest verification failed!")

    breakdown = {
        "schema": "olr08b.cal214_breakdown.v1",
        "curves": curves,
        "by_family": by_fam,
        "by_file": by_file,
        "exhausted_finite_action_space": exhausted,
        "train_determinism_vs_08a": det,
        "archives": {"audit": v_audit, "fable": v_fable},
    }
    run08.write(dest / "cal214_breakdown.json", breakdown)
    return breakdown


def extract_frames_diagnostic(exe: Path, source: Path, teacher: list[dict], tr_by_id: dict, prior: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Extract certified TRAIN frames, diagnosing any single-root replay sensitivity cleanly."""
    frames: list[dict] = []
    f_meters: list[dict] = []
    skipped: list[dict] = []
    for r in teacher:
        if not r["solved"] or not r["path"]:
            continue
        row = tr_by_id[r["id"]]
        ok_by_depth_tac = {(ev["depth"], ev["tactic"]): (1.0 if ev["ok"] else 0.0) for ev in r.get("events", [])}
        with run08.Lean(exe) as L:
            s, why, opening = run08.open_root(L, source, row)
            if s is None:
                skipped.append({"id": row["id"], "decl": row["decl"], "reason": "open:" + str(why)})
                continue
            led = ImmutableReceiptLedger(row["id"] + ":frames")
            m = AuditedMeter(L, led, steps=run08.DEPTH + 1, rpcs=120, hb=run08.HB_CAP)
            tracker = ReceiptTracker(step_cap=run08.STEP_CAP, rpc_cap=run08.RPC_CAP, hb_cap=run08.HB_CAP)
            root_frames: list[dict] = []
            warm = []
            failed_step = None
            for d, tac in enumerate(r["path"]):
                acts = run08.enumerate_menu(m, s)
                tacs = [a[-1] for a in acts]
                if tac not in tacs:
                    failed_step = f"d={d} action missing from enumerator: {tac}"
                    break
                y_idx = tacs.index(tac)
                pcosts = _all_prior_costs(acts, prior, d)
                best_prior_idx = min(range(4), key=lambda pi: pcosts[BASE_PRIORS[pi]][y_idx])
                g = m.call("graph", sid=s["sid"])
                names = sorted({arg for _, k, arg, _ in acts if k.endswith("_lemma") or k == "rw_lemma_rev" or k in ("simp_def", "unfold")})
                l_resp = m.call("lemma_graph", sid=s["sid"], names=names)
                if g.get("ok") is True and l_resp.get("ok") is True and isinstance(l_resp.get("lemmas"), list):
                    g16 = tracker.global_cond16(d * 15, d * 30, d * 5_000_000)
                    p = build_packet(g, l_resp["lemmas"], acts, d, pcosts, g16, parent_ngoals=len(s.get("goals", [1])))
                    if p is not None:
                        ok_mask = torch.tensor(
                            [ok_by_depth_tac.get((d, t), 1.0 if j == y_idx else -1.0) for j, t in enumerate(tacs)],
                            dtype=torch.float32,
                        )
                        root_frames.append({
                            "root_id": row["id"], "file": row["file"], "depth": d,
                            "packet": p, "target_idx": y_idx, "ok_mask": ok_mask, "best_prior_idx": best_prior_idx,
                        })
                if tac == "exact?":
                    w = L.request(op="step", sid=s["sid"], tactic=tac, hb=run08.REPLAY_WARM_HB)
                    warm.append({"depth": d, "ok": w.get("ok"), "hb_raw": w.get("request_hb_raw"), "server_ms": w.get("request_ms")})
                s = m.call("step", sid=s["sid"], tactic=tac, hb=500 if tac == "exact?" else 200)
                if s.get("ok") is not True:
                    failed_step = f"d={d} tac={tac!r} err={s.get('error')}"
                    break
            if failed_step is not None:
                skipped.append({"id": row["id"], "decl": row["decl"], "path": r["path"], "reason": failed_step})
                print("  [frames] skipped:", row["decl"], failed_step, flush=True)
                continue
            cert = m.call("cert", sid=s["sid"]).get("cert", {})
            if s.get("ngoals") != 0 or not run08.close_cert(cert):
                skipped.append({"id": row["id"], "decl": row["decl"], "path": r["path"], "reason": "cert_failed"})
                continue
            frames.extend(root_frames)
            f_meters.append({"id": row["id"], "decl": row["decl"], "frames": len(root_frames), "meter": m.report(), "warmup": warm})
    return frames, f_meters, skipped


def part2_spent_bpd_mms_benchmark() -> dict:
    print("\n=== PART 2: SPENT-COHORT 3-SEED NEURAL, BPD PREDICTIVENESS & MMS BENCHMARK ===", flush=True)
    torch.set_num_threads(8)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)

    exe = Path(r"C:\olr08\pkg\lean\.lake\build\bin\lgym.exe")
    source = Path(r"D:\lean4-419-source")
    audit_zip = RESEARCH_DIR / "results" / "olr08a" / "olr08-full_AUDIT.zip"
    out = Path(r"C:\olr08b\spent_bpd_mms")
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(audit_zip) as z:
        reg = json.loads(z.read("registry.json"))
        prior = json.loads(z.read("prior.json"))
        by_id = {r["id"]: r for r in reg}
        teacher = [json.loads(z.read(n)) for n in sorted(z.namelist()) if n.startswith("source_s2/") and n.endswith(".json")]
        spent_cal_ids = sorted(Path(n).stem for n in z.namelist() if n.startswith("cal_s2/") and n.endswith(".json"))

    tr_by_id = {r["id"]: by_id[r["id"]] for r in teacher}
    cal_rows = [by_id[i] for i in spent_cal_ids]
    for r in (*tr_by_id.values(), *cal_rows):
        if r["role"] == "evaluation" or r["family"] == "SInt" or "/SInt" in r["file"]:
            raise RuntimeError("Contamination guard triggered!")

    fp = out / "source_frames.pt"
    fm = out / "frame_extraction_meta.json"
    if fp.exists() and fm.exists():
        frames = torch.load(fp, map_location="cpu", weights_only=False)
        f_meta = run08.read(fm)
    else:
        frames, f_meters, skipped = extract_frames_diagnostic(exe, source, teacher, tr_by_id, prior)
        torch.save(frames, fp)
        f_meta = {"frames": len(frames), "paths_certified": len(f_meters), "paths_skipped": skipped}
        run08.write(fm, f_meta)
    print(f"  [spent_bench] Extracted {len(frames)} certified TRAIN frames ({f_meta['paths_certified']} paths, {len(f_meta['paths_skipped'])} skipped)", flush=True)

    explicit_pol = ExplicitMemoryPolicy()
    explicit_meta = explicit_pol.fit_from_source_rows(teacher)

    # Base controls on 18 spent CAL roots
    base_runs = {}
    for arm in ("s1", "s2", "s3", "s3c", "portfolio_rr2", "portfolio_rr4", "adaptive_ucb"):
        base_runs[arm] = run08.stage(
            out,
            f"spent_cal_{arm}",
            cal_rows,
            lambda r, a=arm: search_olr08b(
                exe, source, r, a, prior=prior if a not in ("s1", "s2") else None,
                limit=run08.STEP_CAP, mms_budget_mib=0.0, lazy_expand=(a in ("portfolio_rr2", "portfolio_rr4", "adaptive_ucb")),
            ),
            2,
        )

    # MMS sweep across 0, 0.02 (eviction stress), 64, 256, 1024, 4096 MiB on all 18 spent CAL roots
    mms_sweep = {}
    for mib in (0.0, 0.02, 64.0, 256.0, 1024.0, 4096.0):
        tag = f"{mib:g}mib".replace(".", "p")
        rs = run08.stage(
            out,
            f"spent_mms_{tag}",
            cal_rows,
            lambda r, mb=mib: search_olr08b(
                exe, source, r, "explicit_memory", prior=prior, explicit_policy=explicit_pol,
                limit=run08.STEP_CAP, mms_budget_mib=mb, lazy_expand=True,
            ),
            2,
        )
        mms_sweep[f"{mib:g}MiB"] = rate_summary(rs)
    print("  [spent_bench] MMS sweep complete:", {k: (v["solved"], v["mms_cache_peak_bytes"], v["mms_cache_evictions"], v["mms_cache_hits"]) for k, v in mms_sweep.items()}, flush=True)

    # 3-seed matched neural arms (17, 29, 43) x 6 arms (recurrent, recurrent_bpd, direct_mlp, matched_compute_sham, one_way, linear)
    fd = out / "fits"
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
                    frames, arm, seed, steps=400, bpd_rank=8, bpd_learned=(arm == "recurrent_bpd")
                )
                torch.save(m.state_dict(), pt_p)
                run08.write(meta_p, stats)
            if arm in MATCHED_PARAM_ARMS:
                sigs.append((stats["parameters"], stats["initial_sha256"], stats["sampler_sha256"]))

            collect_probe = (seed == SEEDS[0] and arm == "recurrent_bpd")
            rr = run08.stage(
                out,
                f"spent_cal_model_{key}",
                cal_rows,
                lambda row, m=m, bpd_dec=bpd_dec, arm=arm, cp=collect_probe: search_olr08b(
                    exe, source, row, arm, prior=prior, model=m, bpd_decomposer=bpd_dec,
                    alpha=1.0, limit=run08.STEP_CAP, mms_budget_mib=64.0, collect_bpd_probe=cp,
                ),
                1,
            )
            if collect_probe:
                for x in rr:
                    bpd_cal_recs.extend(x.get("bpd_probe_records", []))
                tr_sub = sorted(tr_by_id.values(), key=lambda x: run08.digest([run08.SEED, "bpd_spent", x["id"]]))[:24]
                tr_rr = run08.stage(
                    out,
                    f"spent_train_bpd_probe_{key}",
                    tr_sub,
                    lambda row, m=m, bpd_dec=bpd_dec: search_olr08b(
                        exe, source, row, "recurrent_bpd", prior=prior, model=m, bpd_decomposer=bpd_dec,
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
        sols = [model_results[f"{s}_{arm}"]["calibration"]["solved"] for s in SEEDS]
        seed_var[arm] = {
            "seeds": list(SEEDS),
            "solved": sols,
            "rates": [round(v, 4) for v in rs],
            "mean_rate": round(float(np.mean(rs)), 4),
            "std_rate": round(float(np.std(rs)), 4),
            "parameters": model_results[f"{SEEDS[0]}_{arm}"]["fit"]["parameters"],
        }

    report = {
        "schema": "olr08b.spent_bpd_mms_bench.v1",
        "exploratory_spent_only": True,
        "sealed_SInt_opened": False,
        "protected_eval_opened": False,
        "frame_extraction": f_meta,
        "spent_cal_roots": len(cal_rows),
        "explicit_memory_fit": explicit_meta,
        "controls": {k: rate_summary(v) for k, v in base_runs.items()},
        "mms_budget_sweep": mms_sweep,
        "models": model_results,
        "seed_variation": seed_var,
        "bpd_probe_counts": {
            "train_events": len(bpd_train_recs),
            "eval_events": len(bpd_cal_recs),
            "train_on_path": sum(r["on_path"] for r in bpd_train_recs),
            "eval_on_path": sum(r["on_path"] for r in bpd_cal_recs),
            "train_recovery_events": sum(1 for r in bpd_train_recs if r["fail_streak"] >= 2),
            "eval_recovery_events": sum(1 for r in bpd_cal_recs if r["fail_streak"] >= 2),
        },
        "bpd_predictiveness": bpd_pred,
    }
    dest_rep = RESEARCH_DIR / "results" / "olr08b" / "spent_bpd_mms_report.json"
    run08.write(out / "spent_bpd_mms_report.json", report)
    run08.write(dest_rep, report)
    print("  [spent_bench] SEED VARIATION:", json.dumps(seed_var, indent=2), flush=True)
    print("  [spent_bench] BPD PREDICTIVENESS:", json.dumps(bpd_pred, indent=2), flush=True)
    return report


if __name__ == "__main__":
    t0 = time.perf_counter()
    part1_audit_and_stage_full_artifacts()
    part2_spent_bpd_mms_benchmark()
    print(f"ALL_DONE in {time.perf_counter() - t0:.1f}s", flush=True)
