"""OLR-08B staged CPU native search gym: Adaptive Frontier Allocation & Memory-Augmented Lean Reasoning.

Preserves OLR-08A's failed headroom verdict and all repaired infrastructure invariants:
  - Pre-command scope checks (pre_command_session_v1, env=before, decl_in_env=False)
  - Fresh pre-command path replay with library-search cache warmup (repair 0001)
  - Premise enumerator heartbeat timeout -> empty proposals, still charged (repair 0002)
  - Append-only SHA-256-chained immutable verifier receipts
  - Configurable MMS memory budgets (0, 64, 256, 1024, 4096 MiB)
  - Optional BPD decomposition of proof-state representations & recurrent frontier updates
  - Stronger uncontaminated calibration campaign target: >= 120 menu-hard roots across >= 12 file clusters
  - Minimum 10 pp offline headroom gate vs strongest fixed-budget control (including portfolios)
  - Original 275-root evaluation registry protected (smoke-observed root excluded -> 274 pristine); SInt sealed.
  - No 500K/5M model promotion; PR stays in draft.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

OLR08B_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = OLR08B_DIR.parent
OLR08_DIR = RESEARCH_DIR / "olr08"
for p in (str(OLR08B_DIR), str(OLR08_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import probe  # noqa: E402
import run08  # noqa: E402
from search08 import FREE, reprice  # noqa: E402

from allocator import (  # noqa: E402
    BASE_PRIORS,
    PORTFOLIO_SCHEDULES,
    AuditedMeter,
    _all_prior_costs,
    search_olr08b,
)
from bpd_frontier import BPDFrontierDecomposer, evaluate_bpd_predictiveness  # noqa: E402
from mms_cache import VALID_MMS_BUDGETS_MIB, ImmutableReceiptLedger, ProofMemoryCache  # noqa: E402
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
from roster import (  # noqa: E402
    FROZEN_CAL_FILES_08B,
    SEED_08B,
    SMOKE_EVAL_ROOT_ID,
    freeze_olr08b_registry,
    verify_rule_matches_source,
)

MIN_TRAIN_HARD = 40
MIN_CAL_HARD = 120
MIN_CAL_FILE_CLUSTERS = 12
MAX_CONTROL_CLOSURE_RATE = 0.80
MIN_HEADROOM_PP = 10.0
MIN_SOURCE_FRAMES = 64


def rate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    fates = Counter(r.get("fate", "solved" if r["solved"] else "censored") for r in rows)
    return {
        "roots": len(rows),
        "solved": sum(1 for r in rows if r["solved"]),
        "rate": sum(1 for r in rows if r["solved"]) / n,
        "fates": dict(fates),
        "replayed_certified": sum(1 for r in rows if r["solved"] and "replay" in r),
        "mean_steps": sum(r["steps"] for r in rows) / n,
        "total_steps": sum(r["steps"] for r in rows),
        "total_rpcs": sum(r["meter"]["rpcs"] for r in rows),
        "total_hb_raw": sum(r["meter"]["hb_raw"] for r in rows),
        "mean_hb_raw": sum(r["meter"]["hb_raw"] for r in rows) / n,
        "total_server_ms": sum(r["meter"]["server_ms"] for r in rows),
        "total_native_seconds": round(sum(r["meter"]["native_seconds"] for r in rows), 4),
        "total_open_seconds": round(sum((r.get("open") or {}).get("wall_seconds") or 0.0 for r in rows), 4),
        "total_replay_seconds": round(
            sum(((r.get("replay") or {}).get("open") or {}).get("wall_seconds", 0.0) + ((r.get("replay") or {}).get("meter") or {}).get("native_seconds", 0.0) for r in rows),
            4,
        ),
        "total_feature_seconds": round(sum(r["meter"].get("feature_seconds", 0.0) for r in rows), 4),
        "total_inference_seconds": round(sum(r["meter"].get("inference_seconds", 0.0) for r in rows), 4),
        "total_elapsed_seconds": round(sum(r.get("elapsed", 0.0) for r in rows), 4),
        "mean_elapsed": sum(r.get("elapsed", 0.0) for r in rows) / n,
        "mms_cache_peak_bytes": max((r.get("mms_cache", {}).get("peak_allocated_bytes", 0) for r in rows), default=0),
        "mms_cache_hits": sum(r.get("mms_cache", {}).get("hits", 0) for r in rows),
        "mms_cache_misses": sum(r.get("mms_cache", {}).get("misses", 0) for r in rows),
        "mms_cache_evictions": sum(r.get("mms_cache", {}).get("evictions", 0) for r in rows),
        "mms_cache_seconds": round(sum(r.get("mms_cache", {}).get("total_cache_seconds", 0.0) for r in rows), 6),
        "all_receipt_chains_valid": all(r.get("mms_cache", {}).get("receipt_chain_valid", True) for r in rows),
    }


def paired_file_ci(a: list[dict[str, Any]], b: list[dict[str, Any]], seed: int = SEED_08B) -> dict[str, Any] | None:
    ai = {r["id"]: r for r in a}
    bi = {r["id"]: r for r in b}
    if ai.keys() != bi.keys() or not ai:
        return None
    diffs: dict[str, list[int]] = defaultdict(list)
    for k, r in ai.items():
        diffs[r["file"]].append(int(r["solved"]) - int(bi[k]["solved"]))
    fs = sorted(diffs)
    rng = random.Random(seed)
    boot = []
    for _ in range(2000):
        xs = [d for f in rng.choices(fs, k=len(fs)) for d in diffs[f]]
        boot.append(sum(xs) / len(xs))
    boot.sort()
    n = len(ai)
    return {
        "a_minus_b": (sum(r["solved"] for r in a) - sum(r["solved"] for r in b)) / n,
        "a_minus_b_pp": 100.0 * (sum(r["solved"] for r in a) - sum(r["solved"] for r in b)) / n,
        "ci95_file_bootstrap": [boot[50], boot[1949]],
        "ci95_pp": [round(100.0 * boot[50], 3), round(100.0 * boot[1949], 3)],
        "files": len(fs),
        "roots": n,
        "a_only": sum(ai[k]["solved"] and not bi[k]["solved"] for k in ai),
        "b_only": sum(bi[k]["solved"] and not ai[k]["solved"] for k in ai),
    }


def extract_certified_frames(
    exe: Path,
    source: Path,
    teacher_rows: list[dict[str, Any]],
    tr_by_id: dict[str, dict[str, Any]],
    prior: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay certified TRAIN paths in fresh pre-command sessions to build OLR-08B utility frames."""
    frames: list[dict[str, Any]] = []
    frame_meters: list[dict[str, Any]] = []
    for r in teacher_rows:
        if not r["solved"] or not r["path"]:
            continue
        row = tr_by_id[r["id"]]
        # Collect empirical applicability from source search trace for this root
        ok_by_depth_tac: dict[tuple[int, str], float] = {}
        for ev in r.get("events", []):
            ok_by_depth_tac[(ev["depth"], ev["tactic"])] = 1.0 if ev["ok"] else 0.0

        with run08.Lean(exe) as L:
            s, why, opening = run08.open_root(L, source, row)
            if s is None:
                raise ValueError(f"TRAIN replay root missing: {row['id']} {why}")
            led = ImmutableReceiptLedger(row["id"] + ":frames")
            m = AuditedMeter(L, led, steps=run08.DEPTH + 1, rpcs=120, hb=run08.HB_CAP)
            tracker = ReceiptTracker(step_cap=run08.STEP_CAP, rpc_cap=run08.RPC_CAP, hb_cap=run08.HB_CAP)
            warm = []
            root_frames = 0
            for d, tac in enumerate(r["path"]):
                acts = run08.enumerate_menu(m, s)
                tacs = [a[-1] for a in acts]
                if tac not in tacs:
                    raise ValueError(f"replay action missing from enumerator: {tac}")
                y_idx = tacs.index(tac)
                pcosts = _all_prior_costs(acts, prior, d)
                # Which base prior assigns the lowest rank / cost to target action y_idx?
                best_prior_idx = min(range(4), key=lambda pi: pcosts[BASE_PRIORS[pi]][y_idx])

                g = m.call("graph", sid=s["sid"])
                names = sorted(
                    {arg for _, k, arg, _ in acts if k.endswith("_lemma") or k == "rw_lemma_rev" or k in ("simp_def", "unfold")}
                )
                l_resp = m.call("lemma_graph", sid=s["sid"], names=names)
                if g.get("ok") is True and l_resp.get("ok") is True and isinstance(l_resp.get("lemmas"), list):
                    g16 = tracker.global_cond16(d * 15, d * 30, d * 5_000_000)
                    p = build_packet(g, l_resp["lemmas"], acts, d, pcosts, g16, parent_ngoals=len(s.get("goals", [1])))
                    if p is not None:
                        ok_mask = torch.tensor(
                            [ok_by_depth_tac.get((d, t), 1.0 if j == y_idx else -1.0) for j, t in enumerate(tacs)],
                            dtype=torch.float32,
                        )
                        frames.append(
                            {
                                "root_id": row["id"],
                                "file": row["file"],
                                "depth": d,
                                "packet": p,
                                "target_idx": y_idx,
                                "ok_mask": ok_mask,
                                "best_prior_idx": best_prior_idx,
                            }
                        )
                        root_frames += 1

                if tac == "exact?":
                    w = L.request(op="step", sid=s["sid"], tactic=tac, hb=run08.REPLAY_WARM_HB)
                    warm.append({"depth": d, "ok": w.get("ok"), "hb_raw": w.get("request_hb_raw"), "server_ms": w.get("request_ms")})
                s = m.call("step", sid=s["sid"], tactic=tac, hb=500 if tac == "exact?" else 200)
                if s.get("ok") is not True:
                    raise ValueError("fresh native path replay failed during frame extraction")
            if s.get("ngoals") != 0 or not run08.close_cert(m.call("cert", sid=s["sid"]).get("cert", {})):
                raise ValueError("fresh native certification failed during frame extraction")
            frame_meters.append(
                {
                    "id": row["id"],
                    "path": r["path"],
                    "frames": root_frames,
                    "open": opening,
                    "meter": m.report(),
                    "library_search_warmup": warm,
                    "receipt_chain_sha256": led.chain_sha256,
                }
            )
    return frames, frame_meters


def run_mms_budget_sweep(
    exe: Path,
    source: Path,
    roots: list[dict[str, Any]],
    prior: dict[str, Any],
    explicit_policy: ExplicitMemoryPolicy,
    out: Path,
    *,
    budgets_mib: tuple[int, ...] = VALID_MMS_BUDGETS_MIB,
    workers: int = 2,
    limit: int = run08.STEP_CAP,
) -> dict[str, Any]:
    """Measure explicit_memory across all five configurable MMS budgets (0, 64, 256, 1024, 4096 MiB)."""
    sweep: dict[str, Any] = {}
    for mib in budgets_mib:
        rows = run08.stage(
            out,
            f"mms_sweep_{mib}mib",
            roots,
            lambda r, mb=mib: search_olr08b(
                exe,
                source,
                r,
                "explicit_memory",
                prior=prior,
                explicit_policy=explicit_policy,
                limit=limit,
                mms_budget_mib=float(mb),
            ),
            workers,
        )
        sweep[f"{mib}MiB"] = rate_summary(rows)
    return sweep


def verify_reissues(reissue_a: Path | None, reissue_b: Path | None) -> dict[str, str]:
    prereq: dict[str, str] = {}
    for k, p in (("07a", reissue_a), ("07b", reissue_b)):
        if p is None:
            raise ValueError(f"full profile requires exact-registry reissue manifest {k}")
        r = run08.read(p)
        if r.get("scope_version") != "pre_command_v1" or r.get("kind") != k or not r.get("old_receipts_superseded") or not r.get("files"):
            raise ValueError("invalid reissue manifest " + k)
        expected = {"07a": 207, "07b": 1441}[k]
        if sum(v["selected"] for v in r.get("census", {}).values()) != expected:
            raise ValueError("reissue census mismatch " + k)
        for n, h in r["files"].items():
            f = p.parent / n
            if not f.is_file() or run08.sha(f) != h:
                raise ValueError("reissue artifact hash mismatch " + n)
        prereq[k] = run08.sha(p)
    return prereq


def main(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    exe = args.exe.resolve()

    if (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != run08.COMMIT
        or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    ):
        raise ValueError("pinned clean Lean 4.19.0 source checkout (6caaee84) required")

    prereq = verify_reissues(args.reissue_a, args.reissue_b) if args.profile == "full" else {}
    roster_pre_audit = verify_rule_matches_source(source, RESEARCH_DIR)
    if not roster_pre_audit["passed"]:
        raise ValueError("pre-outcome roster audit failed")

    contract = {
        "schema": "olr08b.cpu.v1",
        "profile": args.profile,
        "source": run08.COMMIT,
        "exe_sha256": run08.sha(exe),
        "torch": torch.__version__,
        "threads": args.threads,
        "workers": args.workers,
        "code_olr08": {p.name: run08.sha(p) for p in sorted(OLR08_DIR.glob("*.py"))},
        "code_olr08b": {p.name: run08.sha(p) for p in sorted(OLR08B_DIR.glob("*.py"))},
        "data": {n: run08.sha(OLR08_DIR / n) for n in ("train_registry.json", "spent.json")},
        "reissues": prereq,
        "caps": [run08.STEP_CAP, run08.RPC_CAP, run08.HB_CAP, run08.DEPTH],
        "mms_budgets_mib": list(VALID_MMS_BUDGETS_MIB),
        "gates": {
            "min_train_hard": MIN_TRAIN_HARD,
            "min_cal_hard": MIN_CAL_HARD,
            "min_cal_file_clusters": MIN_CAL_FILE_CLUSTERS,
            "max_control_closure_rate": MAX_CONTROL_CLOSURE_RATE,
            "min_offline_headroom_pp": MIN_HEADROOM_PP,
            "min_source_frames": MIN_SOURCE_FRAMES,
        },
        "cal_files_frozen": list(FROZEN_CAL_FILES_08B),
        "protected_eval_roots": roster_pre_audit["protected_eval_roots_total"],
        "pristine_eval_roots": roster_pre_audit["protected_eval_pristine_roots"],
        "smoke_eval_excluded_id": SMOKE_EVAL_ROOT_ID,
        "sealed_SInt_opened": False,
        "stop_before_protected_eval": True,
    }
    cp = out / "contract.json"
    if cp.exists() and run08.read(cp) != contract:
        raise ValueError("resume contract differs; use a new --out directory")
    run08.write(cp, contract)

    # Gate 0A: Native scope sentinel
    run08.sentinel(exe, out / "sentinel")

    # Pre-outcome registry freeze
    reg_p = out / "registry.json"
    reg_seal = out / "registry.sha256"
    if reg_p.exists() and reg_seal.exists():
        if reg_seal.read_text().strip() != run08.sha(reg_p):
            raise ValueError("registry hash mismatch")
        rows = run08.read(reg_p)
        roster_audit = run08.read(out / "roster_audit.json")
    else:
        rows, roster_audit = freeze_olr08b_registry(source, exe, RESEARCH_DIR, out, profile=args.profile, workers=args.workers)
        reg_seal.write_text(run08.sha(reg_p) + "\n")

    for r in rows:
        if run08.sha(source / r["file"]) != r["file_sha256"]:
            raise ValueError("source file hash changed after freeze: " + r["file"])

    if args.stop_after_freeze:
        print("PREFLIGHT_FROZEN_BEFORE_OUTCOMES", run08.sha(reg_p), json.dumps(roster_audit["trace_census_summary"]), flush=True)
        return {"status": "preflight_frozen", "registry_sha256": run08.sha(reg_p), "roster_audit": roster_audit}

    # Gate 0B: Menu + human replay validity on TRAIN and fresh CAL ONLY (protected EVAL untouched!)
    sc = [r for r in rows if r["role"] != "evaluation"]
    menu_receipts = run08.stage(out, "menu_source", sc, lambda r: run08.menu(exe, source, r), args.workers)
    midx = {r["id"]: r for r in menu_receipts}

    def eligible(rs: list[dict]) -> list[dict]:
        return [r for r in rs if midx[r["id"]]["scored"] and not midx[r["id"]]["any"]]

    def validity(rs: list[dict]) -> dict[str, Any]:
        x = [midx[r["id"]] for r in rs if midx[r["id"]]["scored"]]
        hum = [v["human"] for v in x if v["human"] is not None]
        ex = Counter(midx[r["id"]]["reason"] for r in rs if not midx[r["id"]]["scored"])
        return {
            "selected": len(rs),
            "scored": len(x),
            "exclusions": dict(ex),
            "T_human": len(hum),
            "human_closed": sum(1 for v in hum if v["closed"]),
            "human_rate": sum(1 for v in hum if v["closed"]) / max(1, len(hum)),
            "menu_any": sum(1 for v in x if v["any"]),
            "hard": sum(1 for v in x if not v["any"]),
        }

    tr_all = [r for r in sc if r["role"] == "train"]
    cal_all = [r for r in sc if r["role"] == "calibration"]
    tv = validity(tr_all)
    cv = validity(cal_all)
    tr = eligible(tr_all)
    cal = eligible(cal_all)
    cal_hard_files = sorted({r["file"] for r in cal})
    cal_hard_by_file = dict(sorted(Counter(r["file"] for r in cal).items()))

    gate0_pass = (
        args.profile != "full"
        or (tv["human_rate"] >= 0.95 and cv["human_rate"] >= 0.95 and tv["T_human"] >= 20 and cv["T_human"] >= 10)
    )
    if not gate0_pass:
        res = {
            "schema": "olr08b.cpu.result.v1",
            "profile": args.profile,
            "status": "gate0_source_validity_failed",
            "source_validity": tv,
            "calibration_validity": cv,
            "protected_eval_opened": False,
            "sealed_SInt_opened": False,
            "architecture_promotion_eligible": False,
        }
        run08.write(out / "summary.json", res)
        return res

    # Gate 1A: Feasibility check (>= 120 menu-hard CAL roots across >= 12 independent file clusters)
    feasibility_pass = (
        args.profile != "full"
        or (len(tr) >= MIN_TRAIN_HARD and len(cal) >= MIN_CAL_HARD and len(cal_hard_files) >= MIN_CAL_FILE_CLUSTERS)
    )

    # Fit TRAIN source S2 teacher & counted prior (reused source cohort, never CAL/EVAL)
    teacher = run08.stage(
        out,
        "source_s2",
        tr,
        lambda r: search_olr08b(exe, source, r, "s2", limit=run08.STEP_CAP, mms_budget_mib=0.0),
        args.workers,
    )
    prior = run08.fit_prior(teacher)
    run08.write(out / "prior.json", prior)

    explicit_pol = ExplicitMemoryPolicy()
    explicit_fit_meta = explicit_pol.fit_from_source_rows(teacher)

    # Calibration fixed controls: 4 single priors + fixed-budget portfolios + online UCB + explicit memory
    cal_runs: dict[str, list[dict[str, Any]]] = {}
    for arm in (*BASE_PRIORS, "portfolio_rr2", "portfolio_rr4", "adaptive_ucb"):
        cal_runs[arm] = run08.stage(
            out,
            f"cal_{arm}",
            cal,
            lambda r, a=arm: search_olr08b(
                exe,
                source,
                r,
                a,
                prior=prior if a not in ("s1", "s2") else None,
                limit=run08.STEP_CAP,
                mms_budget_mib=0.0,
                lazy_expand=(a in ("portfolio_rr2", "portfolio_rr4", "adaptive_ucb")),
            ),
            args.workers,
        )
    cal_runs["explicit_memory_64m"] = run08.stage(
        out,
        "cal_explicit_memory_64m",
        cal,
        lambda r: search_olr08b(
            exe,
            source,
            r,
            "explicit_memory",
            prior=prior,
            explicit_policy=explicit_pol,
            limit=run08.STEP_CAP,
            mms_budget_mib=64.0,
            lazy_expand=True,
        ),
        args.workers,
    )

    # Offline longer-budget headroom reference (1200 attempted tactics, 4x RPC/HB caps)
    cal_long = (
        run08.stage(
            out,
            "cal_s3_long",
            cal,
            lambda r: search_olr08b(exe, source, r, "s3", prior=prior, limit=1200, mms_budget_mib=0.0),
            args.workers,
        )
        if args.profile == "full"
        else cal_runs["s3"]
    )
    cal_portfolio_long = cal_long  # single authoritative 1200-step S3 reference matching OLR-08A contract

    # Competing fixed-budget controls include ALL 4 base priors AND native fixed-budget portfolios!
    competing_fixed_controls = (*BASE_PRIORS, "portfolio_rr2", "portfolio_rr4")
    best_fixed_arm = max(competing_fixed_controls, key=lambda a: sum(x["solved"] for x in cal_runs[a]))
    best_fixed_solved = sum(x["solved"] for x in cal_runs[best_fixed_arm])
    best_fixed_rate = best_fixed_solved / max(1, len(cal))

    long_s3_solved = sum(x["solved"] for x in cal_long)
    long_port_solved = sum(x["solved"] for x in cal_portfolio_long)
    # Retrospective union oracle across the 2 long references + all 300-step controls
    cal_ids = [r["id"] for r in cal]
    oracle_300_solved = sum(
        any(cal_runs[a][idx]["solved"] for a in competing_fixed_controls) for idx in range(len(cal))
    )
    oracle_1200_solved = sum(
        (cal_long[idx]["solved"] or cal_portfolio_long[idx]["solved"] or any(cal_runs[a][idx]["solved"] for a in competing_fixed_controls))
        for idx in range(len(cal))
    )

    headroom_s3_pp = 100.0 * (long_s3_solved - best_fixed_solved) / max(1, len(cal))
    headroom_port_pp = 100.0 * (long_port_solved - best_fixed_solved) / max(1, len(cal))
    headroom_gate_s3 = (
        args.profile == "full"
        and feasibility_pass
        and best_fixed_rate <= MAX_CONTROL_CLOSURE_RATE
        and headroom_s3_pp >= MIN_HEADROOM_PP
    )
    headroom_gate_port = (
        args.profile == "full"
        and feasibility_pass
        and best_fixed_rate <= MAX_CONTROL_CLOSURE_RATE
        and headroom_port_pp >= MIN_HEADROOM_PP
    )
    # Strict preregistered Gate 1 requires feasibility + <=80% control + >=10pp offline headroom (either S3-long or portfolio-long; primary reported as S3-long!)
    gate1_primary_pass = headroom_gate_s3
    gate1_any_long_pass = headroom_gate_s3 or headroom_gate_port

    paired_controls = {
        f"{a}_vs_{best_fixed_arm}": paired_file_ci(cal_runs[a], cal_runs[best_fixed_arm])
        for a in cal_runs
        if a != best_fixed_arm
    }
    paired_controls["cal_s3_long_vs_best_fixed"] = paired_file_ci(cal_long, cal_runs[best_fixed_arm])
    paired_controls["cal_portfolio_rr2_long_vs_best_fixed"] = paired_file_ci(cal_portfolio_long, cal_runs[best_fixed_arm])

    # Configurable MMS memory-budget sweep (0, 64, 256, 1024, 4096 MiB) on deterministic 24-root calibration slice
    # (or full CAL if <= 24) to measure memory-budget scaling & cache accounting cleanly!
    mms_slice = sorted(cal, key=lambda r: run08.digest([SEED_08B, "mms_sweep", r["id"]]))[: min(24, len(cal))]
    mms_sweep = run_mms_budget_sweep(
        exe, source, mms_slice, prior, explicit_pol, out, budgets_mib=VALID_MMS_BUDGETS_MIB, workers=args.workers
    )

    frozen: dict[str, Any] = {
        "source_validity": tv,
        "calibration_validity": cv,
        "train_hard": len(tr),
        "cal_hard": len(cal),
        "cal_hard_file_clusters": len(cal_hard_files),
        "cal_hard_by_file": cal_hard_by_file,
        "feasibility_gate": feasibility_pass,
        "best_fixed_control": best_fixed_arm,
        "best_fixed_control_solved": best_fixed_solved,
        "best_fixed_control_rate": best_fixed_rate,
        "cal_s3_long_solved": long_s3_solved,
        "cal_portfolio_rr2_long_solved": long_port_solved,
        "retrospective_oracle_300_solved": oracle_300_solved,
        "retrospective_oracle_1200_solved": oracle_1200_solved,
        "headroom_s3_long_pp": round(headroom_s3_pp, 4),
        "headroom_portfolio_long_pp": round(headroom_port_pp, 4),
        "headroom_gate_primary_s3": gate1_primary_pass,
        "headroom_gate_portfolio_long": headroom_gate_port,
        "calibration": {k: rate_summary(v) for k, v in cal_runs.items()},
        "cal_s3_long": rate_summary(cal_long),
        "cal_portfolio_rr2_long": rate_summary(cal_portfolio_long),
        "paired_vs_best_fixed": paired_controls,
        "explicit_memory_fit": explicit_fit_meta,
        "mms_budget_sweep_24roots": mms_sweep,
        "source_prior_sha256": run08.sha(out / "prior.json"),
        "models": {},
        "bpd_predictiveness": None,
    }

    # Gate 2: Conditional source-only neural fitting (only when feasibility + headroom gate pass, OR explicit --diagnostic-fit for smoke)
    should_fit = gate1_any_long_pass or (args.profile == "smoke")
    if should_fit:
        tridx = {r["id"]: r for r in tr}
        fp = out / "source_frames.pt"
        fm = out / "source_frames_manifest.json"
        if fp.exists() and fm.exists():
            meta = run08.read(fm)
            if run08.sha(fp) != meta["sha256"] or meta["prior_sha256"] != run08.sha(out / "prior.json"):
                raise ValueError("source frame checkpoint changed")
            frames = torch.load(fp, map_location="cpu", weights_only=False)
        else:
            frames, f_meters = extract_certified_frames(exe, source, teacher, tridx, prior)
            run08.write(out / "frame_meter.json", f_meters)
            tmp = fp.with_suffix(".part")
            torch.save(frames, tmp)
            os.replace(tmp, fp)
            run08.write(
                fm,
                {
                    "sha256": run08.sha(fp),
                    "prior_sha256": run08.sha(out / "prior.json"),
                    "frames": len(frames),
                },
            )
        frozen["source_frames"] = len(frames)

        if len(frames) >= (4 if args.profile == "smoke" else MIN_SOURCE_FRAMES):
            fd = out / "fits"
            fd.mkdir(exist_ok=True)
            seeds_run = (SEEDS[0],) if args.profile == "smoke" else SEEDS
            train_steps = 20 if args.profile == "smoke" else 400
            bpd_probe_train_recs: list[dict[str, Any]] = []
            bpd_probe_cal_recs: list[dict[str, Any]] = []

            for seed in seeds_run:
                matched_sigs = []
                for arm in NEURAL_ARMS:
                    key = f"{seed}_{arm}"
                    pt_p = fd / f"{key}.pt"
                    meta_p = fd / f"{key}.json"
                    if pt_p.exists() and meta_p.exists():
                        stats = run08.read(meta_p)
                        m = UtilityPredictor(arm)
                        m.load_state_dict(torch.load(pt_p, map_location="cpu", weights_only=True))
                        m.eval()
                        if hash_weights(m) != stats["weights_sha256"]:
                            raise ValueError("fit weight hash mismatch " + key)
                        bpd_dec = (
                            BPDFrontierDecomposer.from_model(m, rank_c=8, learned=False, seed=seed)
                            if arm in ("recurrent", "recurrent_bpd")
                            else None
                        )
                    else:
                        m, bpd_dec, stats = fit_utility_model(
                            frames,
                            arm,
                            seed,
                            steps=train_steps,
                            bpd_rank=8,
                            bpd_learned=(arm == "recurrent_bpd"),
                        )
                        torch.save(m.state_dict(), pt_p)
                        run08.write(meta_p, stats)

                    if arm in MATCHED_PARAM_ARMS:
                        matched_sigs.append((stats["parameters"], stats["initial_sha256"], stats["sampler_sha256"]))

                    collect_probe = (seed == seeds_run[0] and arm == "recurrent_bpd")
                    rr = run08.stage(
                        out,
                        f"cal_model_{key}",
                        cal,
                        lambda row, m=m, bpd_dec=bpd_dec, arm=arm, cp=collect_probe: search_olr08b(
                            exe,
                            source,
                            row,
                            arm,
                            prior=prior,
                            model=m,
                            bpd_decomposer=bpd_dec,
                            alpha=1.0,
                            limit=run08.STEP_CAP,
                            mms_budget_mib=64.0,
                            collect_bpd_probe=cp,
                        ),
                        1,
                    )
                    if collect_probe:
                        for r_item in rr:
                            bpd_probe_cal_recs.extend(r_item.get("bpd_probe_records", []))
                        # Also collect TRAIN BPD probe records on a 24-root TRAIN slice (strictly TRAIN!)
                        tr_probe_rows = sorted(tr, key=lambda x: run08.digest([SEED_08B, "bpd_tr", x["id"]]))[: min(24, len(tr))]
                        tr_rr = run08.stage(
                            out,
                            f"train_bpd_probe_{key}",
                            tr_probe_rows,
                            lambda row, m=m, bpd_dec=bpd_dec: search_olr08b(
                                exe,
                                source,
                                row,
                                "recurrent_bpd",
                                prior=prior,
                                model=m,
                                bpd_decomposer=bpd_dec,
                                alpha=1.0,
                                limit=150,
                                mms_budget_mib=64.0,
                                collect_bpd_probe=True,
                            ),
                            1,
                        )
                        for r_item in tr_rr:
                            bpd_probe_train_recs.extend(r_item.get("bpd_probe_records", []))

                    # Route alpha = 1 only if strictly beats strongest fixed control (ties prefer control)
                    alpha = float(sum(r["solved"] for r in rr) > best_fixed_solved)
                    frozen["models"][key] = {
                        "arm": arm,
                        "seed": seed,
                        "fit": stats,
                        "alpha": alpha,
                        "calibration": rate_summary(rr),
                        "paired_vs_best_fixed": paired_file_ci(rr, cal_runs[best_fixed_arm]),
                        "paired_vs_s3": paired_file_ci(rr, cal_runs["s3"]),
                    }
                if len(set(matched_sigs)) != 1:
                    raise ValueError(f"matched parameter/init/sampler drift on seed {seed}")

            # Evaluate BPD component predictiveness (utility, applicability, recovery after failure)
            if bpd_probe_train_recs and bpd_probe_cal_recs:
                frozen["bpd_predictiveness"] = evaluate_bpd_predictiveness(bpd_probe_train_recs, bpd_probe_cal_recs)

            # Seed-level variation summary per arm
            seed_summary: dict[str, Any] = {}
            for arm in NEURAL_ARMS:
                rates = [frozen["models"][f"{s}_{arm}"]["calibration"]["rate"] for s in seeds_run if f"{s}_{arm}" in frozen["models"]]
                if rates:
                    seed_summary[arm] = {
                        "seeds": list(seeds_run),
                        "rates": [round(x, 4) for x in rates],
                        "mean_rate": round(float(np.mean(rates)), 4),
                        "std_rate": round(float(np.std(rates)), 4),
                        "min_rate": round(float(np.min(rates)), 4),
                        "max_rate": round(float(np.max(rates)), 4),
                    }
            frozen["seed_variation"] = seed_summary

    fp_out = out / "frozen_policy.json"
    run08.write(fp_out, frozen)

    # STRICT STOP BEFORE PROTECTED 275-ROOT EVALUATION REGISTRY!
    # Neither OLR-08B calibration nor exploratory runs open the protected 275-root evaluation cohort or SInt.
    status = (
        "cohort_feasibility_failed"
        if not feasibility_pass
        else ("headroom_gate_failed_no_model_fitted" if not gate1_any_long_pass and args.profile == "full" else "calibration_completed_protected_eval_sealed")
    )
    summary = {
        "schema": "olr08b.cpu.result.v1",
        "profile": args.profile,
        "status": status,
        "frozen": frozen,
        "feasibility_gate": feasibility_pass,
        "headroom_gate_primary_s3": gate1_primary_pass,
        "headroom_gate_portfolio_long": headroom_gate_port,
        "models_fitted": len(frozen["models"]),
        "training_updates": len(frozen["models"]) * (20 if args.profile == "smoke" else 400),
        "protected_eval_opened": False,
        "protected_eval_roots_total": roster_pre_audit["protected_eval_roots_total"],
        "protected_eval_pristine_roots": roster_pre_audit["protected_eval_pristine_roots"],
        "protected_eval_smoke_excluded_id": SMOKE_EVAL_ROOT_ID,
        "sealed_SInt_opened": False,
        "architecture_promotion_eligible": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    run08.write(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": status,
                "train_hard": len(tr),
                "cal_hard": len(cal),
                "cal_hard_file_clusters": len(cal_hard_files),
                "best_fixed_control": best_fixed_arm,
                "best_fixed_rate": round(best_fixed_rate, 4),
                "headroom_s3_long_pp": round(headroom_s3_pp, 2),
                "headroom_portfolio_long_pp": round(headroom_port_pp, 2),
                "models_fitted": len(frozen["models"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--reissue-a", type=Path)
    p.add_argument("--reissue-b", type=Path)
    p.add_argument("--stop-after-freeze", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
