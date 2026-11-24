"""Audit 09 - Synthesis of the re-run outcomes (audit_03) with the symbolic controls (audit_06).

Reads only audit outputs and the archived manifest.  Produces:
  1. record-level replication rates of the registered B = 128 outcomes;
  2. reproduction of the registered budget curves and root-paired intervals for the AUC
     contrasts that decide Screen 2A (never computable from the archive, which kept means only);
  3. the 2 x 2 envelope-bank x matrix-bank factorial with its interaction, and Shapley shares;
  4. successor accounting that includes the one-hop expansions the envelope performs for free;
  5. the reuse available to a word-/pair-level cache (the LBF-18 direction);
  6. uninstrumented wall-clock next to the archived figures.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from audit_04_statistics import signflip_p  # noqa: E402

BUDGETS = (16, 32, 64, 96, 128)
SEEDS = (107, 109, 113)


def boot_ci(vals, n=10000, seed=20261124):
    rng = random.Random(seed)
    m = len(vals)
    s = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return [s[int(0.025 * n)], s[int(0.975 * n) - 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rows", required=True, help="merged audit_03_rows.jsonl")
    ap.add_argument("--symbolic", required=True, help="audit_06_symbolic_controls.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if str(args.rows).endswith(".gz"):
        import gzip
        with gzip.open(args.rows, "rt", encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
    else:
        rows = [json.loads(x) for x in Path(args.rows).read_text().splitlines() if x]
    sym = json.loads(Path(args.symbolic).read_text())
    manifest = json.loads((common.lbf_dir(Path(args.repo)) / "results" / "lbf17" / "manifest.json").read_text())
    roots = sorted({r["root"] for r in rows})
    out: dict = {"rows": len(rows), "roots": len(roots)}

    # ---- 1. replication -------------------------------------------------------------------
    rep = defaultdict(lambda: [0, 0])
    for r in rows:
        if "matches_archive" in r:
            rep[r["condition"]][0] += int(r["matches_archive"])
            rep[r["condition"]][1] += 1
    out["replication_B128"] = {
        "records_compared": sum(v[1] for v in rep.values()),
        "records_identical_closed_work_path": sum(v[0] for v in rep.values()),
        "by_condition": {k: {"identical": v[0], "compared": v[1]} for k, v in sorted(rep.items())},
    }

    cell = defaultdict(dict)   # (cond, budget) -> root -> list over seeds of (closed, pen)
    for r in rows:
        if r["seed"] is None:
            continue
        pen = r["work"] if r["closed"] else r["budget"] + 1
        cell[(r["condition"], r["budget"])].setdefault(r["root"], []).append((float(r["closed"]), float(pen)))

    def root_vec(cond, b, j):
        d = cell[(cond, b)]
        return [st.mean(x[j] for x in d[rt]) for rt in roots]

    # ---- 2. budget curves and AUC contrasts ------------------------------------------------
    conds_b = sorted({c for (c, b) in cell if all((c, bb) in cell for bb in BUDGETS)})
    curves, curve_match = {}, {}
    for c in conds_b:
        curves[c] = {str(b): st.mean(root_vec(c, b, 0)) for b in BUDGETS}
        curves[c]["auc"] = st.mean(curves[c][str(b)] for b in BUDGETS)
        if c in manifest["budget_curves"]["128"]["closure_rate"]:
            curve_match[c] = all(abs(curves[c][str(b)] - manifest["budget_curves"][str(b)]["closure_rate"][c]) < 1e-9
                                 for b in BUDGETS)
    out["budget_curves_rerun"] = curves
    out["budget_curves_identical_to_manifest"] = curve_match

    def auc_vec(c):
        per_b = [root_vec(c, b, 0) for b in BUDGETS]
        return [st.mean(per_b[k][i] for k in range(len(BUDGETS))) for i in range(len(roots))]

    ref_auc = auc_vec("rel_prov_opened")
    auc_contrasts = {}
    for c in conds_b:
        if c == "rel_prov_opened":
            continue
        d = [a - b for a, b in zip(ref_auc, auc_vec(c))]
        auc_contrasts[c] = {"auc_delta_pp": 100 * st.mean(d), "root_bootstrap_95_pp": [100 * x for x in boot_ci(d)],
                            "signflip_p": signflip_p(d), "roots_with_any_difference": sum(x != 0 for x in d)}
    out["auc_contrasts_rel_prov_opened_minus"] = auc_contrasts
    per_budget = {}
    for c in ("compute_nonrecurrent", "param_nonrecurrent", "rel_prov_standard", "sym_prov_readout"):
        per_budget[c] = {}
        for b in BUDGETS:
            d = [a - x for a, x in zip(root_vec("rel_prov_opened", b, 0), root_vec(c, b, 0))]
            per_budget[c][str(b)] = {"closure_delta_pp": 100 * st.mean(d), "root_bootstrap_95_pp": [100 * x for x in boot_ci(d)],
                                     "signflip_p": signflip_p(d)}
    out["per_budget_closure_contrasts"] = per_budget

    # ---- 3. factorial ------------------------------------------------------------------------
    def fact(metric_vec):
        i_, e_, m_, b_ = (metric_vec(c) for c in ("rel_prov_opened", "rel_prov_env_zero", "rel_prov_mat_zero",
                                                  "rel_prov_env_mat_zero"))
        env_main_at_mat_intact = [e - i for e, i in zip(e_, i_)]
        mat_main_at_env_intact = [m - i for m, i in zip(m_, i_)]
        env_main_at_mat_zero = [b - m for b, m in zip(b_, m_)]
        mat_main_at_env_zero = [b - e for b, e in zip(b_, e_)]
        inter = [b - e - m + i for b, e, m, i in zip(b_, e_, m_, i_)]
        total = [b - i for b, i in zip(b_, i_)]
        sh_env = [(x + y) / 2 for x, y in zip(env_main_at_mat_intact, env_main_at_mat_zero)]
        sh_mat = [(x + y) / 2 for x, y in zip(mat_main_at_env_intact, mat_main_at_env_zero)]
        rng = random.Random(20261124)
        shares = []
        for _ in range(10000):
            idx = [rng.randrange(len(roots)) for _ in roots]
            t = sum(total[k] for k in idx)
            if t != 0:
                shares.append(sum(sh_env[k] for k in idx) / t)
        shares.sort()
        return {
            "cell_means": {"intact": st.mean(i_), "env_zero": st.mean(e_), "mat_zero": st.mean(m_), "both_zero": st.mean(b_)},
            "env_effect_with_matrix_bank_intact": st.mean(env_main_at_mat_intact),
            "env_effect_with_matrix_bank_zeroed": st.mean(env_main_at_mat_zero),
            "mat_effect_with_envelope_intact": st.mean(mat_main_at_env_intact),
            "mat_effect_with_envelope_zeroed": st.mean(mat_main_at_env_zero),
            "interaction": st.mean(inter), "interaction_root_bootstrap_95": boot_ci(inter),
            "interaction_signflip_p": signflip_p(inter),
            "both_zero_minus_intact": st.mean(total), "both_zero_minus_intact_95": boot_ci(total),
            "additive_prediction_for_both_zero": st.mean(i_) + st.mean(env_main_at_mat_intact) + st.mean(mat_main_at_env_intact),
            "shapley_share_envelope": st.mean(sh_env) / st.mean(total) if st.mean(total) else None,
            "shapley_share_envelope_root_bootstrap_95": ([shares[int(0.025 * len(shares))], shares[int(0.975 * len(shares)) - 1]]
                                                         if shares else None),
            "naive_ratio_share_envelope": st.mean(env_main_at_mat_intact)
            / (st.mean(env_main_at_mat_intact) + st.mean(mat_main_at_env_intact)),
        }
    out["factorial_penalized_work_B128"] = fact(lambda c: root_vec(c, 128, 1))
    out["factorial_auc_closure"] = fact(auc_vec)
    nr = {c: st.mean(root_vec(c, 128, 1)) for c in ("param_nonrecurrent", "compute_nonrecurrent", "sym_prov_readout")}
    out["reference_single_pass_arms_penalized_work_B128"] = nr

    # ---- 4. successor accounting including implicit one-hop expansion -------------------------
    # Every word the simulator expands was a frontier word at that decision, so its successor set
    # had already been enumerated by Pass 1.  The equalised count is therefore the number of
    # successors of all DISTINCT words ever presented to the network ("enumerated once"); adding
    # the charged work on top would double count.  "As implemented" re-enumerates every frontier
    # word at every decision and enumerates again on explicit expansion.
    bid = {r["root"]: r["work"] for r in sym["rows"] if r["condition"] == "bidir" and r["budget"] == 4096 and r["closed"]}
    acc = {}
    for c in ("rel_prov_opened", "sym_prov_readout", "param_nonrecurrent", "compute_nonrecurrent", "lbf16_pos1_opened"):
        rr = [r for r in rows if r["condition"] == c and r["budget"] == 128 and r["seed"] is not None]
        by_root = defaultdict(list)
        for r in rr:
            by_root[r["root"]].append(r)
        charged = [st.mean(x["work"] for x in by_root[rt]) for rt in roots]
        once = [st.mean(x["implicit_onehop_successors_cached"] for x in by_root[rt]) for rt in roots]
        impl = [st.mean(x["work"] + x["implicit_onehop_successors_recomputed"] for x in by_root[rt]) for rt in roots]
        b = [bid[rt] for rt in roots]
        diff = [y - x for x, y in zip(b, once)]
        acc[c] = {
            "all_roots_closed_at_B128": all(x["closed"] for x in rr),
            "mean_charged_successor_checks": st.mean(charged), "median_charged_successor_checks": st.median(charged),
            "mean_successors_enumerated_once_(explicit_or_implicit)": st.mean(once),
            "median_successors_enumerated_once": st.median(once),
            "mean_successor_enumerations_as_implemented": st.mean(impl),
            "ratio_enumerated_once_over_charged": st.mean(once) / st.mean(charged),
            "bidir_bfs_mean_work_to_closure_unbounded_budget": st.mean(b),
            "bidir_bfs_median_work_to_closure": st.median(b), "bidir_bfs_max_work_to_closure": max(b),
            "paired_mean_enumerated_once_minus_bidir": st.mean(diff), "paired_root_bootstrap_95": boot_ci(diff),
            "paired_signflip_p": signflip_p(diff),
            "roots_where_bidir_enumerates_fewer_successors": sum(x < y for x, y in zip(b, once)),
            "roots_where_bidir_needs_fewer_than_charged_only": sum(x < y for x, y in zip(b, charged)),
        }
    out["successor_accounting_B128"] = acc

    # ---- 5. cache reuse (LBF-18 direction) -----------------------------------------------------
    rr = [r for r in rows if r["condition"] == "rel_prov_opened" and r["budget"] == 128]
    out["cache_reuse_rel_prov_opened_B128"] = {
        "episodes": len(rr), "mean_decisions": st.mean(r["model_calls"] for r in rr),
        "word_envelope_evaluations": sum(r["word_evals"] for r in rr),
        "distinct_words": sum(r["distinct_words"] for r in rr),
        "word_level_fraction_avoidable_by_exact_cache": 1 - sum(r["distinct_words"] for r in rr) / sum(r["word_evals"] for r in rr),
        "pair_evaluations": sum(r["pair_evals"] for r in rr), "distinct_pairs": sum(r["distinct_pairs"] for r in rr),
        "pair_level_fraction_recurring": 1 - sum(r["distinct_pairs"] for r in rr) / sum(r["pair_evals"] for r in rr),
        "note": "word envelopes depend only on (word, rules) and are exactly cacheable; pair ribbons depend only on "
                "(u, v, rules) up to the scramble modes; everything from pair_in onward depends on the whole frontier, "
                "the remaining budget and prev_pot, and is not cacheable without changing the function computed",
    }

    # ---- 6. wall clock ---------------------------------------------------------------------------
    wc = {}
    for c in sorted({r["condition"] for r in rows if r["seed"] is not None and r["budget"] == 128}):
        rr = [r for r in rows if r["condition"] == c and r["budget"] == 128]
        clean = st.mean(r["wall_clock_ms_clean"] for r in rr)
        arch = [r["archive_wall_clock_ms"] for r in rr if r.get("archive_wall_clock_ms")]
        wc[c] = {"clean_ms_per_episode": clean,
                 "clean_ms_per_decision": sum(r["wall_clock_ms_clean"] for r in rr) / max(1, sum(r["model_calls"] for r in rr)),
                 "archived_ms_per_episode": st.mean(arch) if arch else None,
                 "archived_over_clean": (st.mean(arch) / clean) if arch else None}
    ref = wc["sym_prov_readout"]["clean_ms_per_episode"]
    for c in wc:
        wc[c]["clean_ratio_vs_sym_prov_readout"] = wc[c]["clean_ms_per_episode"] / ref
        if wc[c]["archived_ms_per_episode"]:
            wc[c]["archived_ratio_vs_sym_prov_readout"] = wc[c]["archived_ms_per_episode"] / wc["sym_prov_readout"]["archived_ms_per_episode"]
    out["wall_clock_B128"] = wc
    out["wall_clock_note"] = ("clean timings were taken while two audit shards shared a 2-core machine; arms are "
                              "interleaved within root, so ratios are meaningful and absolute values are indicative only")
    # break-even successor cost versus plain bidirectional BFS and versus the parameter-free exact control
    s128 = {c: sym["summary"][c]["128"] for c in ("bidir", "geom_mm", "graph_bridge", "absint_mm", "exact_mm")}
    rp = wc["rel_prov_opened"]["clean_ms_per_episode"]
    rp_w = st.mean(root_vec("rel_prov_opened", 128, 1))
    out["break_even_cost_per_successor_check_ms"] = {
        c: {"baseline_mean_penalized_work": s128[c]["mean_penalized_work"],
            "baseline_closure": s128[c]["closure_rate"],
            "successor_checks_saved": s128[c]["mean_penalized_work"] - rp_w,
            "break_even_ms_per_successor_check": (rp / (s128[c]["mean_penalized_work"] - rp_w)
                                                   if s128[c]["mean_penalized_work"] > rp_w else None)}
        for c in s128}
    common.write_json(Path(args.out), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
