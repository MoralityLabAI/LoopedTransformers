"""Audit 07 - Finite-precision support preservation, horizon semantics and fiber collisions.

Claims under test (LBF_17_RELATIONAL_ABSTRACT_INTERPRETATION.md sections 1.2, 2.1, 2.3 and
LBF_17_RESULTS.md section 4.1):
  (a) the neural bank's strict-positivity support EQUALS the symbolic abstract support "for
      all k >= 1", and the rational-saturation trellis filter preserves it "without
      exponential attenuation";
  (b) Pass k support means "<= k-step" reachability, and residual false positives are
      chimeras;
  (c) routing Pass-1 redexes by p mod 4 separates incompatible branches.

Method (open calibration words only, archived checkpoints, inference only):
  * For every archived rel_prov_opened / rel_prov_standard / adj2_noprov_opened checkpoint,
    run the registered bigram path for K in {1..12} passes and compare, per fiber and per
    bigram cell, with the registered symbolic interpreter iterated the same number of
    steps.  Count symbolic-supported cells whose float32 activation is (i) exactly 0.0,
    (ii) <= 1e-6 (the threshold the registered diagnostics use), and record the smallest
    positive activation.  K = 3 is the registered depth; larger K probes the "for all k" claim.
  * Split the K = 3 symbol-level false positives of each symbolic domain into cells that
    become concretely reachable within 6 steps (horizon overshoot) and cells that do not.
  * Count Pass-1 redexes per word and how many share a congruence fiber, and re-run the
    symbolic domain with fiber counts fp in {1, 2, 3, 4, 5, L} to expose the precision and
    soundness consequences of the choice fp = 4.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PASS_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12)


def calibration_words(rg, m15, cases, per_side: int = 4) -> list[tuple[object, str]]:
    """Same sampling recipe as evaluate_abstraction_and_glyphs, but both frontiers are kept."""
    out = []
    for c in cases:
        s = m15.FringeState.start(c)
        for _ in range(3):
            if not s.live(128):
                break
            (sd, ix), _, _ = m15.geom_mm_select(rg, c, s)
            m15.expand_node(rg, c, s, sd, ix)
        if not s.fronts[0] or not s.fronts[1]:
            s = m15.FringeState.start(c)
        seen = set()
        for w in [c.source, c.target] + s.fronts[0][:per_side] + s.fronts[1][:per_side]:
            if w not in seen:
                seen.add(w)
                out.append((c, w))
    return out


def neural_bigram_history(m17, model, rules, w0: str, passes: int):
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
            hist.append(bi[0].clone())
    return hist  # list of (Fp, Lm1, 6, 6, Cb)


def symbolic_fibers(m17, rules, w0: str, passes: int, use_prov: bool):
    if use_prov:
        hist = m17.symbolic_prov4_adj2(tuple(rules), w0, passes)
        return [torch.stack(hist[k][1]) for k in range(1, passes + 1)]
    # merged adj2 bank: the cell with use_provenance=False has no root fiber, so the root word's
    # own bigrams are part of the bank only through the context term; compare on non-root cells.
    hist = m17.symbolic_adj2(tuple(rules), w0, passes)
    return [hist[k].unsqueeze(0) for k in range(1, passes + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-roots", type=int, default=48)
    args = ap.parse_args()
    repo = Path(args.repo)
    scratch = common.scratch_dir(args.scratch)
    common.set_single_thread()
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    cases, _ = common.load_open_split(repo, rg, "calibration")
    words = calibration_words(rg, m15, cases[:args.max_roots])
    out: dict = {"words": len(words), "roots": len({c.root for c, _ in words}), "pass_grid": list(PASS_GRID)}
    kmax = max(PASS_GRID)

    # ---- (a) float32 support versus symbolic support -----------------------------------------
    sym_cache = {}
    def sym(c, w, use_prov):
        key = (c.root, w, use_prov)
        if key not in sym_cache:
            sym_cache[key] = symbolic_fibers(m17, c.rules, w, kmax, use_prov)
        return sym_cache[key]

    precision = {}
    # Only the provenance models are compared: for them exact cell-for-cell equality with
    # sym_prov4_adj2 is claimed. (The merged no-provenance bank omits the root word, so its
    # relation to symbolic_adj2 is inclusion, not equality.)
    for arch in ("rel_prov_opened", "rel_prov_standard"):
        for seed in m17.SEEDS:
            model, _ = common.load_checkpoint(repo, m17, seed, arch, scratch)
            stats = {k: {"symbolic_cells": 0, "exact_zero": 0, "le_1e-6": 0, "le_1e-12": 0,
                         "neural_positive_outside_symbolic": 0, "min_positive": float("inf"),
                         "words_with_exact_zero_loss": 0, "words_with_threshold_loss": 0}
                     for k in PASS_GRID}
            for c, w in words:
                nh = neural_bigram_history(m17, model, c.rules, w, kmax)
                sh = sym(c, w, model.use_prov)
                root_bi = m17.alpha2_bool((w,), len(w))
                for k in PASS_GRID:
                    act = nh[k - 1].amax(dim=-1)            # (Fp, Lm1, 6, 6) max over channels
                    s_k = sh[k - 1]
                    if not model.use_prov:
                        s_k = s_k & ~root_bi.unsqueeze(0)   # see symbolic_fibers note
                        act_cmp = act.masked_fill(root_bi.unsqueeze(0), 0.0)
                    else:
                        act_cmp = act
                    st = stats[k]
                    n_sym = int(s_k.sum())
                    zero = int((s_k & (act == 0)).sum())
                    thr = int((s_k & (act <= 1e-6)).sum())
                    st["symbolic_cells"] += n_sym
                    st["exact_zero"] += zero
                    st["le_1e-6"] += thr
                    st["le_1e-12"] += int((s_k & (act <= 1e-12)).sum())
                    st["neural_positive_outside_symbolic"] += int(((act_cmp > 0) & ~s_k).sum())
                    st["words_with_exact_zero_loss"] += int(zero > 0)
                    st["words_with_threshold_loss"] += int(thr > 0)
                    pos = act[s_k & (act > 0)]
                    if pos.numel():
                        st["min_positive"] = min(st["min_positive"], float(pos.min()))
            precision[f"{seed}_{arch}"] = {str(k): v for k, v in stats.items()}
            print(f"[audit07] {seed} {arch}: " + " | ".join(
                f"K={k}: zero {stats[k]['exact_zero']}/{stats[k]['symbolic_cells']} thr {stats[k]['le_1e-6']} "
                f"min+ {stats[k]['min_positive']:.1e} extra {stats[k]['neural_positive_outside_symbolic']}"
                for k in (1, 2, 3, 4, 6, 12)), flush=True)
    out["float32_support_vs_symbolic"] = precision

    # ---- (a') existence: a legal parameter value at which float32 positivity fails ------------
    gate = m16.SparseEnvGate(4, init_bias=-110.0)
    with torch.no_grad():
        g, s_val = gate(torch.zeros(1, 4), torch.ones(1, 4))
    out["float32_sigmoid_underflow_witness"] = {
        "gate_bias": -110.0, "sigmoid_value": float(s_val.max()), "gated_message_max": float(g.max()),
        "real_arithmetic_value_is_positive": True,
        "float32_smallest_subnormal": float(torch.finfo(torch.float32).smallest_normal),
        "note": "sigma(z) > 0 and softplus(z) > 0 hold over the reals but not in float32 for z <~ -104; "
                "the positivity argument of section 1.2 is therefore a statement about ideal arithmetic",
    }
    eps = 1e-4
    x = torch.tensor(1e-9)
    out["rational_saturation_attenuation"] = {
        "eps": eps, "entry": 1e-9, "rho": float(x / (x + eps)),
        "mask_after_9_such_edges": float((x / (x + eps)) ** 9),
        "float32_result_is_zero": bool(((x / (x + eps)) ** 9) == 0),
        "note": "rho_eps(x) ~ x/eps for x << eps, so a chain of small entries attenuates geometrically; "
                "'without exponential attenuation' holds only while entries stay >> eps",
    }

    # ---- (b) horizon overshoot versus genuine chimeras at K = 3 -------------------------------
    horizon = {d: {"fp_cells": 0, "reachable_within_6": 0, "not_reachable_within_6": 0, "fn_cells": 0}
               for d in ("sym_pos1", "sym_adj2", "sym_sub3", "sym_prov4_adj2")}
    endpoint_words = [(c, w) for c in cases[:args.max_roots] for w in (c.source, c.target)]
    for c, w in endpoint_words:
        ball = common.bfs_ball(c.rules, w, 6)
        L = len(w)
        gt3 = m17.alpha1_bool([z for z, d in ball.items() if d <= 3], L)
        gt6 = m17.alpha1_bool(list(ball), L)
        p42 = m17.symbolic_prov4_adj2(c.rules, w, 3)[3]
        p42_pos = m17.adj2_to_pos1(p42[0])
        for fb in p42[1]:
            if fb.any():
                p42_pos |= m17.adj2_to_pos1(fb)
        preds = {
            "sym_pos1": m17.symbolic_pos1(c.rules, w, 3)[3],
            "sym_adj2": m17.adj2_to_pos1(m17.symbolic_adj2(c.rules, w, 3)[3]),
            "sym_sub3": m17.sub3_to_pos1(m17.symbolic_sub3(c.rules, w, 3)[3]),
            "sym_prov4_adj2": p42_pos,
        }
        for name, pr in preds.items():
            fp = pr & ~gt3
            horizon[name]["fp_cells"] += int(fp.sum())
            horizon[name]["reachable_within_6"] += int((fp & gt6).sum())
            horizon[name]["not_reachable_within_6"] += int((fp & ~gt6).sum())
            horizon[name]["fn_cells"] += int((gt3 & ~pr).sum())
    out["k3_false_positive_decomposition_endpoints"] = {"words": len(endpoint_words), **horizon}

    # ---- (c) fiber collisions and the choice fp = 4 --------------------------------------------
    redex_hist, collide_same_p, collide_diff_p, words_with_collision = Counter(), 0, 0, 0
    for c, w in words:
        ors = [(u, v) for u, v in c.rules if u != v] + [(v, u) for u, v in c.rules if u != v]
        redexes = [(p, len(lhs)) for lhs, rhs in ors for p in range(len(w) - len(lhs) + 1) if w[p:p + len(lhs)] == lhs]
        redex_hist[len(redexes)] += 1
        by_f = Counter(p % 4 for p, _ in redexes)
        by_p = Counter(p for p, _ in redexes)
        same_p = sum(n - 1 for n in by_p.values() if n > 1)
        total_coll = sum(n - 1 for n in by_f.values() if n > 1)
        collide_same_p += same_p
        collide_diff_p += total_coll - same_p
        words_with_collision += int(total_coll > 0)
    out["pass1_fiber_collisions"] = {
        "words": len(words), "redex_count_histogram": dict(sorted(redex_hist.items())),
        "mean_redexes_per_word": sum(k * v for k, v in redex_hist.items()) / len(words),
        "words_with_two_or_more_redexes_in_one_fiber": words_with_collision,
        "excess_redexes_sharing_fiber_same_position": collide_same_p,
        "excess_redexes_sharing_fiber_different_position": collide_diff_p,
    }
    fp_sweep = {}
    for fpn in ("1", "2", "3", "4", "5", "L"):
        acc = {"tp": 0, "fp": 0, "fn": 0}
        for c, w in endpoint_words:
            L = len(w)
            ball = common.bfs_ball(c.rules, w, 3)
            gt3 = m17.alpha1_bool([z for z, d in ball.items() if d <= 3], L)
            nf = L if fpn == "L" else int(fpn)
            h = m17.symbolic_prov4_adj2(c.rules, w, 3, fp=nf)[3]
            pr = m17.adj2_to_pos1(h[0])
            for fb in h[1]:
                if fb.any():
                    pr |= m17.adj2_to_pos1(fb)
            acc["tp"] += int((pr & gt3).sum())
            acc["fp"] += int((pr & ~gt3).sum())
            acc["fn"] += int((gt3 & ~pr).sum())
        acc["precision"] = acc["tp"] / max(1, acc["tp"] + acc["fp"])
        fp_sweep[fpn] = acc
    out["fiber_count_sweep_k3_endpoints"] = fp_sweep

    common.write_json(Path(args.out), out)
    print(json.dumps({k: v for k, v in out.items() if k != "float32_support_vs_symbolic"}, indent=1))


if __name__ == "__main__":
    main()
