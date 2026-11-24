"""Audit 01 - Soundness of root-backtracking suppression in A_{prov4 x adj2}.

Claim under test (LBF_17_RELATIONAL_ABSTRACT_INTERPRETATION.md, Theorem 2.2(3) and section 2.3):
    suppressing Pass-2 rewrites at a fiber's origin window whose RHS equals w0's slice is
    "100% sound for all word lengths L", and both the symbolic domain and the neural
    FastProvenanceBigramCell have 100% recall of concrete <=k-step reachability.

Counterexample R* (three reversible rules, |Sigma| = 6):
        abc <-> def        ab <-> de        bf <-> cc
    w0 = "abc" + padding.  Concretely  abc -> def -> abf -> acc  in three steps, so symbol
    'c' is reachable at position 1 within k = 3.  The registered suppression drops the
    Pass-2 rewrite  de -> ab  at p = 0 (its RHS equals w0[0:2] and 0 = 0 mod 4), although on
    the fiber word "def..." it produces "abf..." != w0, because the fiber was created by a
    LENGTH-3 redex at the same position.  The lost word reappears one pass late, so its
    own successors are missing at the final pass.

This script (i) evaluates the registered symbolic interpreter and the archived neural
checkpoints on R*, (ii) scans the open LBF-17 cohorts (train + calibration only), (iii)
estimates how often the benchmark generator produces affected systems using audit-only
seeds, and (iv) checks a minimal sound repair.  No sealed file is parsed, nothing is trained.
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RSTAR = (("abc", "def"), ("ab", "de"), ("bf", "cc"))
AUDIT_GENERATOR_SEEDS = (990000001, 990000002, 990000003, 990000004)  # never registered anywhere


def nested_pairs(rules) -> list[dict]:
    """Rule pairs (A length 3, B length 2) with B = {A_lhs[:2], A_rhs[:2]} and differing 3rd symbol."""
    out = []
    twos = [(a, b) for a, b in rules if len(a) == 2]
    for a, b in rules:
        if len(a) != 3:
            continue
        for u, v in twos:
            if {u, v} == {a[:2], b[:2]} and a[:2] != b[:2] and a[2] != b[2]:
                out.append({"triple": [a, b], "pair": [u, v]})
    return out


def sym_support_pos1(m17, rules, w0: str, k: int, fn=None) -> torch.Tensor:
    fn = fn or m17.symbolic_prov4_adj2
    base_a2, fibers = fn(rules, w0, k)[k]
    pos = m17.adj2_to_pos1(base_a2)
    for fb in fibers:
        if fb.any():
            pos |= m17.adj2_to_pos1(fb)
    return pos


def symbol_fn_cells(m17, rules, w0: str, k: int, fn=None) -> list[tuple[int, str]]:
    ball = [w for w, d in common.bfs_ball(rules, w0, k).items() if d <= k]
    gt = m17.alpha1_bool(ball, len(w0))
    pr = sym_support_pos1(m17, rules, w0, k, fn)
    miss = (gt & ~pr).nonzero().tolist()
    return [(p, "abcdef"[s]) for p, s in miss]


def repaired_prov4_adj2(m17):
    """Minimal sound repair: only suppress a rewrite at p if no Pass-1 redex LONGER than it
    was routed into this fiber at p (then restoring the slice provably returns w0)."""
    CH = m17.CHAR2ID

    def run(rules, w0, k_max=3, fp=m17.FP_FIBERS):
        L = len(w0)
        Lm1 = L - 1
        ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
        base_a2 = m17.alpha2_bool((w0,), L)
        history = [(base_a2, [torch.zeros((Lm1, 6, 6), dtype=torch.bool) for _ in range(fp)])]
        if k_max == 0:
            return history
        fibers_init = [torch.zeros((Lm1, 6, 6), dtype=torch.bool) for _ in range(fp)]
        active = [False] * fp
        long_origin = set()  # positions p at which a length-3 Pass-1 redex exists
        for lhs, rhs in ors:
            ll = len(lhs)
            for p in range(L - ll + 1):
                if w0[p:p + ll] == lhs:
                    w1 = w0[:p] + rhs + w0[p + ll:]
                    fibers_init[p % fp] |= m17.alpha2_bool((w1,), L)
                    active[p % fp] = True
                    if ll == 3:
                        long_origin.add(p)
        fibers = [m17.trellis_filter_adj2_bool(fibers_init[f]) if active[f] else fibers_init[f] for f in range(fp)]
        history.append((base_a2, [x.clone() for x in fibers]))
        for step_idx in range(1, k_max):
            new_fibers = []
            for f in range(fp):
                if not active[f]:
                    new_fibers.append(fibers[f])
                    continue
                ta = m17.trellis_filter_adj2_bool(fibers[f])
                nxt = ta.clone()
                for lhs, rhs in ors:
                    ll = len(lhs)
                    x = [CH[c] for c in lhs]
                    y = [CH[c] for c in rhs]
                    if ll == 2:
                        for p in range(Lm1):
                            if (step_idx == 1 and p % fp == f and w0[p:p + 2] == rhs
                                    and p not in long_origin):
                                continue
                            if ta[p, x[0], x[1]]:
                                nxt[p, y[0], y[1]] = True
                                if p > 0:
                                    nxt[p - 1, ta[p - 1, :, x[0]], y[0]] = True
                                if p + 1 < Lm1:
                                    nxt[p + 1, y[1], ta[p + 1, x[1], :]] = True
                    else:
                        for p in range(L - 2):
                            if step_idx == 1 and p % fp == f and w0[p:p + 3] == rhs:
                                continue
                            if ta[p, x[0], x[1]] and ta[p + 1, x[1], x[2]]:
                                nxt[p, y[0], y[1]] = True
                                nxt[p + 1, y[1], y[2]] = True
                                if p > 0:
                                    nxt[p - 1, ta[p - 1, :, x[0]], y[0]] = True
                                if p + 2 < Lm1:
                                    nxt[p + 2, y[2], ta[p + 2, x[2], :]] = True
                new_fibers.append(m17.trellis_filter_adj2_bool(nxt))
            fibers = new_fibers
            history.append((base_a2, [x.clone() for x in fibers]))
        return history

    return run


def neural_bigram_support(m17, model, rules, w0: str, passes: int = 3):
    """Run the registered bigram path of RelationalProvenanceFrescoNet exactly as forward() does."""
    _, oh_bi, _, _ = model.init_words([w0])
    r2, r3 = m17.FastProvenanceBigramCell.encode_bigram_rules(tuple(rules))
    fp = m17.FP_FIBERS if model.use_prov else 1
    bi = torch.zeros((1, fp, len(w0) - 1, 6, 6, model.cb))
    hist = []
    with torch.no_grad():
        for k in range(passes):
            if k == 0:
                gbi = torch.zeros_like(bi)
            else:
                gbi, _ = model.bi_gate(oh_bi.unsqueeze(1).expand_as(bi), bi)
            bi, _ = model.bi_cell(oh_bi, k, gbi, bi, r2, r3)
            hist.append(bi.clone())
    return hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mc-systems", type=int, default=6000, help="phase B: systems generated")
    ap.add_argument("--mc-full", type=int, default=400, help="phase A: systems fully checked")
    args = ap.parse_args()
    repo = Path(args.repo)
    scratch = common.scratch_dir(args.scratch)
    common.set_single_thread()
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    report: dict = {"review_commit": common.REVIEW_COMMIT, "rules": [list(r) for r in RSTAR]}

    # ---- (i) hand-built counterexample: symbolic ------------------------------------------
    sym_rows = []
    for L in (3, 4, 10, 11, 13):
        for pad in ("a", "d"):
            w0 = "abc" + pad * (L - 3)
            ball = common.bfs_ball(RSTAR, w0, 3)
            witness = "acc" + pad * (L - 3)
            fn3 = symbol_fn_cells(m17, RSTAR, w0, 3)
            sym_rows.append({
                "w0": w0, "L": L,
                "witness": witness, "witness_distance": ball.get(witness),
                "symbol_false_negatives_k1": symbol_fn_cells(m17, RSTAR, w0, 1),
                "symbol_false_negatives_k2": symbol_fn_cells(m17, RSTAR, w0, 2),
                "symbol_false_negatives_k3": fn3,
                "sym_prov_pos1_false_negatives_k3": [
                    (p, "abcdef"[s]) for p, s in (
                        m17.alpha1_bool([w for w, d in ball.items() if d <= 3], L)
                        & ~torch.stack(m17.symbolic_prov_pos1(RSTAR, w0, 3)[3], 0).any(0)).nonzero().tolist()],
                "sym_adj2_false_negatives_k3": [
                    (p, "abcdef"[s]) for p, s in (
                        m17.alpha1_bool([w for w, d in ball.items() if d <= 3], L)
                        & ~m17.adj2_to_pos1(m17.symbolic_adj2(RSTAR, w0, 3)[3])).nonzero().tolist()],
            })
    report["symbolic_prov4_adj2_on_Rstar"] = sym_rows

    # word-level witness: the 2-step word "abf.." is in no fiber concretisation at k = 2
    w0 = "abcaaaaaaa"
    base_a2, fibers = m17.symbolic_prov4_adj2(RSTAR, w0, 2)[2]
    lost = "abfaaaaaaa"
    a2_lost = m17.alpha2_bool((lost,), len(w0))
    in_any = bool((a2_lost & ~base_a2).sum() == 0) or any(bool((a2_lost & ~fb).sum() == 0) for fb in fibers)
    report["word_level_k2"] = {
        "w0": w0, "word": lost, "concrete_distance": common.bfs_ball(RSTAR, w0, 3).get(lost),
        "contained_in_some_fiber_concretisation_at_pass2": in_any,
    }

    # ---- (i') neural cell in the archived checkpoints -------------------------------------
    neural_rows = []
    ball3 = [w for w, d in common.bfs_ball(RSTAR, w0, 3).items() if d <= 3]
    gt = m17.alpha1_bool(ball3, len(w0))
    oh = m17.alpha1_bool((w0,), len(w0))
    for seed in m17.SEEDS:
        for arch in ("rel_prov_opened", "rel_prov_standard", "adj2_noprov_opened"):
            model, rec = common.load_checkpoint(repo, m17, seed, arch, scratch)
            hist = neural_bigram_support(m17, model, RSTAR, w0, 3)
            pos = model.bigram_to_pos1_channels(hist[-1])[0]
            strict = oh | (pos > 0).any(-1)
            thr = oh | (pos > 1e-6).any(-1)
            neural_rows.append({
                "seed": seed, "arch": arch, "checkpoint_sha256": rec["checkpoint_sha256"],
                "fn_strict_gt0": [(p, "abcdef"[s]) for p, s in (gt & ~strict).nonzero().tolist()],
                "fn_registered_threshold_1e-6": [(p, "abcdef"[s]) for p, s in (gt & ~thr).nonzero().tolist()],
                "activation_at_(1,c)_max_over_channels": float(pos[1, m17.CHAR2ID["c"]].max()),
            })
    report["neural_bigram_cell_on_Rstar"] = neural_rows

    # ---- (ii) open cohorts (train + calibration) -------------------------------------------
    repaired = repaired_prov4_adj2(m17)
    cohort = {}
    for split in common.OPEN_SPLITS:
        cases, _ = common.load_open_split(repo, rg, split)
        with_nested = 0
        fn_cases = 0
        fn_words = 0
        words_checked = 0
        examples = []
        for c in cases:
            npairs = nested_pairs(c.rules)
            with_nested += int(bool(npairs))
            words = set()
            for end in (c.source, c.target):
                words.update(w for w, d in common.bfs_ball(c.rules, end, 2).items())
            words = sorted(words)[:400]
            case_hit = False
            for w in words:
                words_checked += 1
                miss = symbol_fn_cells(m17, c.rules, w, 3)
                if miss:
                    fn_words += 1
                    case_hit = True
                    if len(examples) < 6:
                        examples.append({"root": c.root, "word": w, "missing": miss, "nested_pairs": npairs})
            fn_cases += int(case_hit)
        cohort[split] = {
            "systems": len(cases), "systems_with_nested_3_over_2_pair": with_nested,
            "words_checked_(2-balls_of_endpoints)": words_checked,
            "systems_with_symbol_false_negative_k3": fn_cases,
            "words_with_symbol_false_negative_k3": fn_words, "examples": examples,
        }
    report["open_cohort_scan"] = cohort

    # ---- (ii') targeted scan inside OPEN benchmark systems that contain a nested pair ---------
    targeted = []
    for split in common.OPEN_SPLITS:
        cases, _ = common.load_open_split(repo, rg, split)
        for c in cases:
            npairs = nested_pairs(c.rules)
            if not npairs:
                continue
            sides = {s_ for np_ in npairs for s_ in np_["triple"]}
            # (a) words an actual search can reach: radius-5 balls of both endpoints, capped
            reach_words = set()
            for end in (c.source, c.target):
                ball = common.bfs_ball(c.rules, end, 5)
                reach_words.update(w for w in ball if any(t in w for t in sides))
            reach_words = sorted(reach_words)[:1500]
            reach_fn = [w for w in reach_words if symbol_fn_cells(m17, c.rules, w, 3)]
            # (b) existence: plant each triple side at each position of the source word
            planted = sorted({c.source[:p] + t + c.source[p + 3:] for t in sides for p in range(len(c.source) - 2)})
            planted_fn = [w for w in planted if symbol_fn_cells(m17, c.rules, w, 3)]
            targeted.append({
                "split": split, "root": c.root, "nested_pairs": npairs,
                "reachable_words_containing_triple_(radius<=5)": len(reach_words),
                "reachable_words_with_symbol_fn_k3": len(reach_fn),
                "reachable_fn_examples": [{"word": w, "missing": symbol_fn_cells(m17, c.rules, w, 3),
                                           "distance_from_source": common.bfs_ball(c.rules, c.source, 5).get(w),
                                           "distance_from_target": common.bfs_ball(c.rules, c.target, 5).get(w)}
                                          for w in reach_fn[:3]],
                "planted_words": len(planted), "planted_words_with_symbol_fn_k3": len(planted_fn),
                "planted_fn_examples": [{"word": w, "missing": symbol_fn_cells(m17, c.rules, w, 3)} for w in planted_fn[:3]],
            })
    report["open_cohort_targeted_scan"] = targeted

    # ---- (iii) generator Monte Carlo with audit-only seeds ----------------------------------
    rng_words = random.Random(20261124)

    def frontier_like_words(c):
        words = {c.source, c.target}
        for end in (c.source, c.target):
            w = end
            for _ in range(4):
                nb = rg.neighbors(c, w)
                if not nb:
                    break
                w = rng_words.choice(nb).successor
                words.add(w)
        return sorted(words)

    # Phase A: every system fully checked (tests the characterisation "FN => nested pair").
    a_sys = a_nested = a_fn_sys = a_fn_not_nested = a_words = a_fn_words = 0
    a_fp = 0
    cases_a, _ = rg.generate(AUDIT_GENERATOR_SEEDS[0], {"train": args.mc_full})
    for c in cases_a["train"]:
        a_sys += 1
        npairs = nested_pairs(c.rules)
        a_nested += int(bool(npairs))
        hit = False
        for w in frontier_like_words(c):
            a_words += 1
            ball = [z for z, d in common.bfs_ball(c.rules, w, 3).items() if d <= 3]
            gtw = m17.alpha1_bool(ball, len(w))
            pr = sym_support_pos1(m17, c.rules, w, 3)
            a_fp += int((pr & ~gtw).sum())
            if bool((gtw & ~pr).any()):
                a_fn_words += 1
                hit = True
        a_fn_sys += int(hit)
        a_fn_not_nested += int(hit and not npairs)

    # Phase B: many systems, only those with the precondition are checked (rate given precondition).
    b_sys = b_nested = b_fn_sys = b_words = b_fn_words = 0
    b_planted = b_planted_fn = b_planted_fn_sys = 0
    rep_fn_words = rep_fp = orig_fp = 0
    mc_examples = []
    per_seed = max(1, args.mc_systems // (len(AUDIT_GENERATOR_SEEDS) - 1))
    for gs in AUDIT_GENERATOR_SEEDS[1:]:
        cases_b, _ = rg.generate(gs, {"train": per_seed})
        for c in cases_b["train"]:
            b_sys += 1
            npairs = nested_pairs(c.rules)
            if not npairs:
                continue
            b_nested += 1
            hit = False
            for w in frontier_like_words(c):
                b_words += 1
                ball = [z for z, d in common.bfs_ball(c.rules, w, 3).items() if d <= 3]
                gtw = m17.alpha1_bool(ball, len(w))
                pr = sym_support_pos1(m17, c.rules, w, 3)
                pr_r = sym_support_pos1(m17, c.rules, w, 3, repaired)
                orig_fp += int((pr & ~gtw).sum())
                rep_fp += int((pr_r & ~gtw).sum())
                rep_fn_words += int(bool((gtw & ~pr_r).any()))
                if bool((gtw & ~pr).any()):
                    b_fn_words += 1
                    hit = True
                    if len(mc_examples) < 5:
                        mc_examples.append({
                            "generator_seed": gs, "root": c.root, "word": w, "rules": [list(r) for r in c.rules],
                            "missing": [(p, "abcdef"[s_]) for p, s_ in (gtw & ~pr).nonzero().tolist()],
                            "nested_pairs": npairs,
                        })
            b_fn_sys += int(hit)
            sides = {s_ for np_ in npairs for s_ in np_["triple"]}
            planted = sorted({c.source[:p] + t + c.source[p + 3:] for t in sides for p in range(len(c.source) - 2)})
            p_hit = False
            for w in planted:
                b_planted += 1
                ball = [z for z, d in common.bfs_ball(c.rules, w, 3).items() if d <= 3]
                gtw = m17.alpha1_bool(ball, len(w))
                bad = bool((gtw & ~sym_support_pos1(m17, c.rules, w, 3)).any())
                b_planted_fn += int(bad)
                p_hit = p_hit or bad
                rep_fn_words += int(bool((gtw & ~sym_support_pos1(m17, c.rules, w, 3, repaired)).any()))
            b_planted_fn_sys += int(p_hit)
    report["generator_monte_carlo"] = {
        "audit_only_generator_seeds": list(AUDIT_GENERATOR_SEEDS),
        "analytic_nested_pair_rate_approx": 4 * (8 / 630) * (35 / 36) * (5 / 6),
        "phase_A_all_systems_checked": {
            "systems": a_sys, "systems_with_nested_pair": a_nested,
            "frontier_like_words_checked": a_words, "words_with_symbol_fn_k3": a_fn_words,
            "systems_with_symbol_fn_k3": a_fn_sys, "systems_with_fn_but_no_nested_pair": a_fn_not_nested,
            "symbol_false_positives": a_fp,
        },
        "phase_B_precondition_systems_only": {
            "systems_generated": b_sys, "systems_with_nested_pair": b_nested,
            "nested_pair_rate": b_nested / max(1, b_sys),
            "frontier_like_words_checked": b_words, "frontier_like_words_with_symbol_fn_k3": b_fn_words,
            "systems_with_fn_on_frontier_like_words": b_fn_sys,
            "planted_words_checked": b_planted, "planted_words_with_symbol_fn_k3": b_planted_fn,
            "systems_with_fn_on_some_planted_word": b_planted_fn_sys,
            "repaired_domain_words_with_symbol_fn_k3": rep_fn_words,
            "symbol_false_positives_original": orig_fp, "symbol_false_positives_repaired": rep_fp,
        },
        "examples": mc_examples,
    }

    # ---- (iv) the repair on R* ---------------------------------------------------------------
    report["repair_on_Rstar"] = {
        "symbol_false_negatives_k3": symbol_fn_cells(m17, RSTAR, w0, 3, repaired),
        "cex1_still_exact": symbol_fn_cells(m17, (("ab", "de"), ("bc", "fa"), ("df", "cc")), "abc", 3, repaired) == []
        and not bool(sym_support_pos1(m17, (("ab", "de"), ("bc", "fa"), ("df", "cc")), "abc", 3, repaired)[0, m17.CHAR2ID["c"]]),
    }
    common.write_json(Path(args.out), report)
    print("symbolic R* (L=10, pad a) k=3 FN:", sym_rows[4]["symbol_false_negatives_k3"])
    print("word-level:", report["word_level_k2"])
    for r in neural_rows:
        print("neural", r["seed"], r["arch"], "FN strict:", r["fn_strict_gt0"], "act(1,c)=", r["activation_at_(1,c)_max_over_channels"])
    print("open cohorts:", {k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in cohort.items()})
    print("targeted:", [{k: v for k, v in t.items() if "examples" not in k and k != "nested_pairs"} for t in targeted])
    print("monte carlo:", {k: v for k, v in report["generator_monte_carlo"].items() if k != "examples"})
    print("repair:", report["repair_on_Rstar"])


if __name__ == "__main__":
    main()
