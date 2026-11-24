"""Audit 04 - Statistical re-analysis of the archived LBF-17 calibration outcomes.

Inputs are read-only: results/lbf17/episodes.jsonl.gz, manifest.json and
abstraction_diagnostics.json.  Standard library only.

The experimental unit is the calibration ROOT (48 independent rule-system lineages).  The
three training seeds are repeated measures on each root and are averaged within root before
any inference, exactly as the registered analysis does; seed-to-seed variation is reported
separately because three checkpoints cannot support seed-level inference.

Produces:
  * a record-level recomputation of every headline count / mean in the manifest;
  * exact (Clopper-Pearson) bounds for the 48/48 root-level closure;
  * the registered percentile bootstrap, re-derived, next to sign-flip permutation p-values,
    Holm-adjusted across the registered family of comparators;
  * discordance counts (how many roots carry any information for each contrast);
  * a bootstrap interval for the "80 / 20" envelope-versus-matrix attribution ratio;
  * the AUROC that Boolean support alone would give, from the archived confusion counts;
  * sample sizes needed to confirm each contrast, and what 32 / 64 sealed roots can detect.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

WORK = 128
SEEDS = (107, 109, 113)
CONTROLS = ("astar", "bidir", "geom_mm", "graph_bridge", "exact_lookahead", "uniform")


def registered_bootstrap(vals):
    rng = random.Random(20261123)
    samples = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(2000))
    return [samples[50], samples[1949]]


def signflip_p(diffs, n_perm=200000, seed=7):
    """Two-sided paired sign-flip (randomisation) test on root-level differences."""
    nz = [d for d in diffs if d != 0]
    if not nz:
        return 1.0
    obs = abs(sum(nz))
    n = len(nz)
    if n <= 18:
        cnt = tot = 0
        for mask in range(1 << n):
            s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(nz))
            tot += 1
            cnt += abs(s) >= obs - 1e-12
        return cnt / tot
    rng = random.Random(seed)
    cnt = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in nz)
        cnt += abs(s) >= obs - 1e-12
    return (cnt + 1) / (n_perm + 1)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(x, n, conf=0.95):
    a = (1 - conf) / 2

    def solve(f, lo, hi):
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if x == 0 else solve(lambda p: 1 - binom_cdf(x - 1, n, p) < a, 0.0, 1.0)
    upper = 1.0 if x == n else solve(lambda p: binom_cdf(x, n, p) > a, 0.0, 1.0)
    return [lower, upper]


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0


def n_for_power(delta, s, z_alpha=1.959964, z_beta=0.841621):
    if delta == 0:
        return None
    return math.ceil(((z_alpha + z_beta) * s / abs(delta)) ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fsi-export-dir", default=None,
                    help="optional directory for per-case receipts in the evalmath_fsi_gate.py column format")
    args = ap.parse_args()
    res = common.lbf_dir(Path(args.repo)) / "results" / "lbf17"
    rows = common.read_jsonl_gz(res / "episodes.jsonl.gz")
    manifest = json.loads((res / "manifest.json").read_text())
    diag = json.loads((res / "abstraction_diagnostics.json").read_text())

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    roots = sorted({r["root"] for r in rows})

    # ---- 1. record-level recomputation against the manifest --------------------------------
    recompute = {}
    mismatches = []
    for cond, rr in by_cond.items():
        closures = sum(int(r["native_certified"]) for r in rr)
        pen = mean([r["penalized_work"] for r in rr])
        recompute[cond] = {"episodes": len(rr), "certified_closures": closures, "mean_penalized_work": pen,
                           "mean_wall_clock_ms": mean([r["wall_clock_ms"] for r in rr]),
                           "model_calls": sum(r["model_calls"] for r in rr)}
        mc = manifest["costs"][cond]
        if mc["certified_closures"] != closures or abs(mc["mean_penalized_work"] - pen) > 1e-9:
            mismatches.append(cond)
        if any((r["penalized_work"] != (r["work"] if r["native_certified"] else WORK + 1)) for r in rr):
            mismatches.append(cond + ":penalized_work_rule")
        if any(r["closed"] != r["native_certified"] for r in rr):
            mismatches.append(cond + ":closed_vs_certified")
    out = {"records": len(rows), "roots": len(roots), "manifest_recomputation_mismatches": mismatches,
           "recomputed": recompute}

    def root_mean(cond, key):
        acc = defaultdict(list)
        for r in by_cond[cond]:
            acc[r["root"]].append(float(r[key]))
        return [mean(acc[x]) for x in roots]

    # ---- 2. the 144/144 headline at the level of the independent unit ----------------------
    rp = by_cond["rel_prov_opened"]
    per_root_all = defaultdict(list)
    for r in rp:
        per_root_all[r["root"]].append(r["native_certified"])
    n_root_all = sum(all(v) for v in per_root_all.values())
    out["headline_closure"] = {
        "root_seed_episodes_closed": sum(r["native_certified"] for r in rp), "root_seed_episodes": len(rp),
        "independent_roots_closed_by_all_seeds": n_root_all, "independent_roots": len(roots),
        "clopper_pearson_95_two_sided_on_roots": clopper_pearson(n_root_all, len(roots)),
        "one_sided_95_lower_bound_on_roots": 0.05 ** (1 / len(roots)) if n_root_all == len(roots) else None,
        "rule_of_three_upper_bound_on_failure_rate": 3 / len(roots),
        "note": "144 is 48 roots x 3 checkpoints; the checkpoints share every root, so n = 48.",
        "work_distribution_rel_prov_opened": {
            "min": min(r["work"] for r in rp), "median": sorted(r["work"] for r in rp)[len(rp) // 2],
            "max": max(r["work"] for r in rp),
            "episodes_within_25pct_of_budget_(work>=96)": sum(r["work"] >= 96 for r in rp),
        },
    }
    saturated = [c for c in by_cond if c not in CONTROLS and recompute[c]["certified_closures"] == recompute[c]["episodes"]]
    out["headline_closure"]["learned_or_sham_arms_also_at_100pct"] = sorted(saturated)

    # ---- 3. paired contrasts -----------------------------------------------------------------
    ref_w = root_mean("rel_prov_opened", "penalized_work")
    ref_c = root_mean("rel_prov_opened", "native_certified")
    contrasts, p_work, p_clos = {}, {}, {}
    for cond in sorted(by_cond):
        if cond == "rel_prov_opened":
            continue
        dw = [a - b for a, b in zip(ref_w, root_mean(cond, "penalized_work"))]
        dc = [a - b for a, b in zip(ref_c, root_mean(cond, "native_certified"))]
        reg = manifest["paired"]["rel_prov_minus_" + cond]
        boot = registered_bootstrap(dw)
        p_work[cond] = signflip_p(dw)
        p_clos[cond] = signflip_p(dc)
        per_seed = None
        if cond not in CONTROLS:
            per_seed = {}
            for s in SEEDS:
                a = mean([r["penalized_work"] for r in by_cond["rel_prov_opened"] if r["seed"] == s])
                b = mean([r["penalized_work"] for r in by_cond[cond] if r["seed"] == s])
                per_seed[str(s)] = a - b
        s_dw = sd(dw)
        contrasts[cond] = {
            "work_delta_mean": mean(dw), "work_delta_sd_across_roots": s_dw,
            "work_95_registered_bootstrap_recomputed": boot,
            "work_95_registered_bootstrap_in_manifest": reg["penalized_work_95"],
            "bootstrap_reproduced": all(abs(a - b) < 1e-9 for a, b in zip(boot, reg["penalized_work_95"])),
            "work_signflip_p": p_work[cond],
            "closure_delta_mean_pp": 100 * mean(dc), "closure_signflip_p": p_clos[cond],
            "roots_with_any_work_difference": sum(d != 0 for d in dw),
            "roots_with_any_closure_difference": sum(d != 0 for d in dc),
            "per_seed_work_delta": per_seed,
            "roots_needed_80pct_power_alpha05": n_for_power(mean(dw), s_dw),
            "min_detectable_work_delta_n32": (1.959964 + 0.841621) * s_dw / math.sqrt(32),
            "min_detectable_work_delta_n64": (1.959964 + 0.841621) * s_dw / math.sqrt(64),
        }
    hw, hc = holm(p_work), holm(p_clos)
    for cond in contrasts:
        contrasts[cond]["work_signflip_p_holm"] = hw[cond]
        contrasts[cond]["closure_signflip_p_holm"] = hc[cond]
    # Holm within the registered screen families as well as across all 20 comparators.
    families = {
        "screen_1_symbolic_controls": ["astar", "bidir", "geom_mm", "graph_bridge", "exact_lookahead", "uniform"],
        "screen_2A_recurrence": ["param_nonrecurrent", "compute_nonrecurrent", "rel_prov_standard"],
        "screen_2C_factored_shams": ["rel_prov_scramble_window", "rel_prov_scramble_ribbon", "rel_prov_ablate_axial",
                                     "rel_prov_env_zero", "rel_prov_env_roll", "rel_prov_mat_zero", "rel_prov_mat_roll"],
        "architecture_variants": ["lbf16_pos1_opened", "lowrank_opened", "adj2_noprov_opened", "sym_prov_readout"],
    }
    fam_out = {}
    for fname, members in families.items():
        adj = holm({m: p_work[m] for m in members})
        fam_out[fname] = {m: {"work_signflip_p": p_work[m], "holm_within_family": adj[m]} for m in members}
        for m in members:
            contrasts[m]["work_signflip_p_holm_within_family"] = adj[m]
            contrasts[m]["family"] = fname
    out["holm_by_registered_family"] = fam_out
    out["paired_contrasts_rel_prov_opened_minus"] = contrasts
    out["contrasts_significant_after_holm_work"] = sorted(c for c in contrasts if hw[c] < 0.05)
    out["contrasts_not_significant_after_holm_work"] = sorted(c for c in contrasts if hw[c] >= 0.05)

    # seed-level spread of the primary arm versus the gaps it is claimed to open
    seed_pw = manifest["seed_penalized_work"]
    out["seed_level_penalized_work"] = {
        c: [seed_pw[str(s)][c] for s in SEEDS] for c in sorted(seed_pw[str(SEEDS[0])])}
    out["seed_level_range"] = {c: max(v) - min(v) for c, v in out["seed_level_penalized_work"].items()}

    # ---- 4. the 80/20 attribution ratio ------------------------------------------------------
    d_env = [b - a for a, b in zip(ref_w, root_mean("rel_prov_env_zero", "penalized_work"))]
    d_mat = [b - a for a, b in zip(ref_w, root_mean("rel_prov_mat_zero", "penalized_work"))]
    rng = random.Random(20261124)
    shares = []
    for _ in range(20000):
        idx = [rng.randrange(len(roots)) for _ in roots]
        e = sum(d_env[i] for i in idx)
        m_ = sum(d_mat[i] for i in idx)
        if e + m_ != 0:
            shares.append(e / (e + m_))
    shares.sort()
    out["attribution_80_20"] = {
        "env_zero_work_cost": mean(d_env), "mat_zero_work_cost": mean(d_mat),
        "reported_envelope_share": mean(d_env) / (mean(d_env) + mean(d_mat)),
        "root_bootstrap_95_of_envelope_share": [shares[int(0.025 * len(shares))], shares[int(0.975 * len(shares))]],
        "bootstrap_draws_with_share_outside_[0,1]": sum(s < 0 or s > 1 for s in shares) / len(shares),
        "roots_where_env_zero_changes_work": sum(d != 0 for d in d_env),
        "roots_where_mat_zero_changes_work": sum(d != 0 for d in d_mat),
        "roots_where_mat_zero_HELPS": sum(d < 0 for d in d_mat),
        "roots_where_env_zero_HELPS": sum(d < 0 for d in d_env),
        "correlation_env_mat_root_effects": (
            sum((a - mean(d_env)) * (b - mean(d_mat)) for a, b in zip(d_env, d_mat))
            / max(1e-12, (len(roots) - 1) * sd(d_env) * sd(d_mat))),
        "note": "a ratio of two marginal ablation costs is an attribution only if the effects are additive; "
                "that requires the joint cell, which audit_03 supplies",
    }

    # ---- 5. what Boolean support alone gives for the glyph AUROC -----------------------------
    auroc = {}
    for split in ("train", "calibration"):
        g = diag[split]["glyph_calibration"]
        d = diag[split]["domains"]["neural_rel_prov"]
        pos, neg = g["certified_reachable_subterms"], g["spurious_plausible_subterms"]
        tpr = 1.0 if d["symbol_fn"] == 0 else None
        fpr = d["symbol_fp"] / neg
        auroc[split] = {
            "certified": pos, "spurious_A1_plausible": neg, "spurious_inside_prov4_adj2_support": d["symbol_fp"],
            "auroc_from_boolean_support_alone": None if tpr is None else tpr * (1 - fpr) + 0.5 * tpr * fpr,
            "reported_auroc_led4": g["auroc_rel_prov_led4_quantized"],
            "reported_auroc_continuous": g["auroc_rel_prov_continuous"],
            "reported_auroc_within_supported": g["auroc_within_rel_prov_supported"],
            "negatives_available_to_within_supported_auroc": d["symbol_fp"],
        }
    out["glyph_auroc_decomposition"] = auroc
    out["abstraction_diagnostic_scope"] = {
        "symbol_cells_scored_calibration_neural_rel_prov": sum(
            diag["calibration"]["domains"]["neural_rel_prov"][k] for k in ("symbol_tp", "symbol_fp", "symbol_fn")),
        "neural_equals_symbolic_counts_calibration": all(
            diag["calibration"]["domains"]["neural_rel_prov"][k] == diag["calibration"]["domains"]["sym_prov4_adj2"][k]
            for k in ("symbol_tp", "symbol_fp", "symbol_fn", "bridge_tp", "bridge_fp", "bridge_fn")),
    }

    # ---- 6. registered screens, recomputed -----------------------------------------------------
    auc = manifest["budget_curve_mean_closure_auc"]
    out["screens_recomputed_margins"] = {
        "2A_auc_margin_pp": {o: 100 * (auc["rel_prov_opened"] - auc[o])
                             for o in ("param_nonrecurrent", "compute_nonrecurrent", "rel_prov_standard")},
        "2A_auc_margin_in_episodes_out_of_720": {
            o: round((auc["rel_prov_opened"] - auc[o]) * 720)
            for o in ("param_nonrecurrent", "compute_nonrecurrent", "rel_prov_standard")},
        "2C_auc_margin_pp": {md: 100 * (auc["rel_prov_opened"] - auc["rel_prov_" + md])
                             for md in ("scramble_window", "scramble_ribbon", "ablate_axial", "env_zero",
                                        "env_roll", "mat_zero", "mat_roll")},
        "unregistered_strong_control_auc_margin_pp_vs_sym_prov_readout": 100 * (auc["rel_prov_opened"] - auc["sym_prov_readout"]),
        "manifest_screens": manifest["screens"],
        "hard_coded_custody_flags": {k: manifest[k] for k in (
            "sealed_outcomes_opened", "lbf16_sealed_cohorts_preserved_unopened", "native_online_tactic_calls")},
    }

    # ---- 7. optional export in the FSI-01 bridge column format ----------------------------------
    if args.fsi_export_dir:
        d = Path(args.fsi_export_dir)
        d.mkdir(parents=True, exist_ok=True)
        for ctrl in ("sym_prov_readout", "compute_nonrecurrent", "param_nonrecurrent", "graph_bridge", "bidir"):
            lines = []
            ctrl_by = defaultdict(list)
            for r in by_cond[ctrl]:
                ctrl_by[r["root"]].append(r)
            for r in by_cond["rel_prov_opened"]:
                pool = ctrl_by[r["root"]]
                match = [x for x in pool if x["seed"] == r["seed"]] or pool
                lines.append(json.dumps({
                    "case_id": f'{r["root"]}:{r["seed"]}', "root_id": r["root"],
                    "candidate_correct": bool(r["native_certified"]),
                    "control_correct": bool(match[0]["native_certified"])}, sort_keys=True))
            (d / f"fsi_cases_rel_prov_opened_vs_{ctrl}.jsonl").write_text("\n".join(lines) + "\n")

    common.write_json(Path(args.out), out)
    hc_ = out["headline_closure"]
    print("mismatches vs manifest:", mismatches)
    print("roots closed by all seeds:", hc_["independent_roots_closed_by_all_seeds"], "/", hc_["independent_roots"],
          "CP95:", [round(x, 4) for x in hc_["clopper_pearson_95_two_sided_on_roots"]],
          "one-sided LB:", round(hc_["one_sided_95_lower_bound_on_roots"], 4))
    print("arms at 100%:", hc_["learned_or_sham_arms_also_at_100pct"])
    for c, v in sorted(contrasts.items(), key=lambda kv: kv[1]["work_delta_mean"]):
        print(f'{c:28s} dW={v["work_delta_mean"]:+7.2f} sd={v["work_delta_sd_across_roots"]:6.2f} '
              f'boot={[round(x, 2) for x in v["work_95_registered_bootstrap_recomputed"]]} repro={v["bootstrap_reproduced"]} '
              f'p={v["work_signflip_p"]:.4f} holm20={v["work_signflip_p_holm"]:.4f} holmFam={v["work_signflip_p_holm_within_family"]:.4f} '
              f'nroots!=0:{v["roots_with_any_work_difference"]:2d} n80={v["roots_needed_80pct_power_alpha05"]} '
              f'seed={None if v["per_seed_work_delta"] is None else [round(x, 2) for x in v["per_seed_work_delta"].values()]}')
    print("attribution:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["attribution_80_20"].items() if k != "note"})
    print("auroc:", auroc["calibration"])
    print("screens:", out["screens_recomputed_margins"]["2A_auc_margin_pp"], out["screens_recomputed_margins"]["2A_auc_margin_in_episodes_out_of_720"])


if __name__ == "__main__":
    main()
