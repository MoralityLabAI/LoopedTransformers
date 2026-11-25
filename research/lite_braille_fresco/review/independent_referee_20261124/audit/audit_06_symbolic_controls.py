"""Audit 06 - Missing symbolic controls on the open calibration cohort (post hoc, unregistered).

The learned controller is charged only for concrete successors it asks the simulator to
generate.  Its envelope, however, is computed from the rule system on every frontier word at
every decision at zero charged cost, and Pass 1 of that envelope is exactly the abstraction of
the word's complete one-step successor set.  The registered symbolic heuristics (geom_mm,
graph_bridge) are given no such information, and the one control that uses concrete lookahead
(exact_lookahead) is charged for it.  Two things therefore remain untested in LBF-17:

  (1) how far the SAME abstract domain gets with ZERO learned parameters, and
  (2) what plain bidirectional BFS achieves when it is allowed the successor generations the
      learned controller performs implicitly.

Controls defined here (all fixed before any of them was run; nothing was tuned):

  absint_mm   geom_mm with its admissible bound ceil(d_H/3) replaced by
              max(ceil(d_H/3), d_hat), where d_hat(u, v) = min{ku + kv : the registered
              prov4 x adj2 envelopes of u at horizon ku and of v at horizon kv intersect},
              ku, kv in {0..3}, and d_hat = 7 if none intersect (no intersection within
              horizons (K, K) proves only d >= 2K + 1).  Tie-breaks are geom_mm's.
              Zero parameters, zero charged lookahead - the same privilege the network has.
  absint1_mm  the same with horizons restricted to ku, kv <= 1 (single-pass information).
  exact_mm    the same with d_hat taken from exact concrete radius-3 balls (the quantity the
              training teacher is built from).  Uncharged; an upper reference only.
  bidir@B     registered smaller-frontier bidirectional BFS at budgets up to 4096.

This uses calibration roots that have already been scored many times; it is exploratory
evidence about alternative explanations, not a confirmatory result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BIDIR_BUDGETS = (128, 192, 256, 384, 512, 768, 1024, 2048, 4096)
BUDGETS = (16, 32, 64, 96, 128)
NO_MEET = 7


def nonempty_paths(a: torch.Tensor) -> torch.Tensor:
    """a: (..., Lm1, 6, 6) bool trellis tables -> (...) bool: some full word is consistent."""
    reach = a[..., 0, :, :].any(dim=-2)
    for p in range(1, a.shape[-3]):
        reach = (reach.unsqueeze(-1) & a[..., p, :, :]).any(dim=-2)
    return reach.any(dim=-1)


class EnvelopeOracle:
    def __init__(self, m17, rules, kmax: int):
        self.m17, self.rules, self.kmax = m17, rules, kmax
        self.cache: dict[str, torch.Tensor] = {}
        self.sums = torch.tensor([[ku + kv for kv in range(kmax + 1)] for ku in range(kmax + 1)])
        self.no_meet = 2 * kmax + 1  # no intersection within (kmax, kmax) only proves d >= 2*kmax + 1

    def env(self, w: str) -> torch.Tensor:
        if w not in self.cache:
            hist = self.m17.symbolic_prov4_adj2(self.rules, w, self.kmax)
            self.cache[w] = torch.stack([torch.stack([hist[k][0]] + list(hist[k][1])) for k in range(self.kmax + 1)])
        return self.cache[w]  # (K+1, 5, Lm1, 6, 6)

    def dhat(self, us: list[str], vs: list[str]) -> torch.Tensor:
        V = torch.stack([self.env(v) for v in vs])  # (m1, K+1, 5, Lm1, 6, 6)
        out = torch.full((len(us), len(vs)), self.no_meet, dtype=torch.long)
        for i, u in enumerate(us):
            U = self.env(u)
            inter = U[None, :, None, :, None] & V[:, None, :, None, :]      # (m1, K+1, K+1, 5, 5, Lm1, 6, 6)
            hit = nonempty_paths(inter).any(dim=-1).any(dim=-1)             # (m1, K+1, K+1)
            cand = torch.where(hit, self.sums[None], torch.full_like(self.sums[None], self.no_meet))
            out[i] = cand.reshape(len(vs), -1).min(dim=1).values
        return out


class ExactOracle:
    def __init__(self, rg, c, radius: int = 3):
        self.rg, self.c, self.radius = rg, c, radius
        self.cache: dict[str, dict[str, int]] = {}
        self.generated = 0

    def ball(self, w: str) -> dict[str, int]:
        if w not in self.cache:
            seen = {w: 0}
            frontier = [w]
            for d in range(1, self.radius + 1):
                nxt = []
                for x in frontier:
                    for mv in self.rg.neighbors(self.c, x):
                        self.generated += 1
                        if mv.successor not in seen:
                            seen[mv.successor] = d
                            nxt.append(mv.successor)
                frontier = nxt
            self.cache[w] = seen
        return self.cache[w]

    def dhat(self, us, vs) -> torch.Tensor:
        out = torch.full((len(us), len(vs)), NO_MEET, dtype=torch.long)
        vb = [self.ball(v) for v in vs]
        for i, u in enumerate(us):
            bu = self.ball(u)
            for j, bv in enumerate(vb):
                small, big = (bu, bv) if len(bu) <= len(bv) else (bv, bu)
                best = NO_MEET
                for w, d in small.items():
                    e = big.get(w)
                    if e is not None and d + e < best:
                        best = d + e
                out[i, j] = best
        return out


def select_with_bound(rg, s, dh_bound: torch.Tensor, depth: int):
    """geom_mm selection with the admissible bound replaced by max(ceil(dH/3), dh_bound)."""
    best_key, best_act = None, (0, 0)
    for side in (0, 1):
        opp = s.fronts[1 - side]
        ms = len(s.fronts[side])
        for idx, w in enumerate(s.fronts[side]):
            g = s.dist[side][w]
            best_f = (999, 999, 999)
            for jdx, v in enumerate(opp):
                gv = s.dist[1 - side][v]
                uw, vw = (w, v) if side == 0 else (v, w)
                dh = rg.hamming(uw, vw)
                lb = max((dh + 2) // 3, int(dh_bound[idx, jdx] if side == 0 else dh_bound[jdx, idx]))
                f_tot = g + gv + max(1, lb)
                if f_tot > depth:
                    continue
                diffs = [k for k, (ca, cb) in enumerate(zip(uw, vw)) if ca != cb]
                span = (diffs[-1] - diffs[0] + 1) if diffs else 0
                cand = (f_tot, dh, span)
                if cand < best_f:
                    best_f = cand
            key = (*best_f, ms, g, side, idx)
            if best_key is None or key < best_key:
                best_key, best_act = key, (side, idx)
    return best_act


def run_bound_policy(rg, m15, c, oracle, budget: int, depth: int = 16) -> dict:
    s = m15.FringeState.start(c)
    deg = {}
    words = set()
    implicit_recomputed = decisions = 0
    t0 = time.perf_counter()
    while s.live(budget):
        decisions += 1
        for w in s.fronts[0] + s.fronts[1]:
            if w not in deg:
                deg[w] = len(rg.neighbors(c, w))
            implicit_recomputed += deg[w]
            words.add(w)
        bound = oracle.dhat(s.fronts[0], s.fronts[1])
        side, idx = select_with_bound(rg, s, bound, depth)
        m15.expand_node(rg, c, s, side, idx, budget, depth)
    ms = (time.perf_counter() - t0) * 1000.0
    if s.path is not None:
        rg.replay(c, s.path)
    return {"closed": s.path is not None, "work": s.work, "decisions": decisions, "wall_ms": ms,
            "implicit_onehop_successors_recomputed": implicit_recomputed,
            "implicit_onehop_successors_cached": sum(deg[w] for w in words),
            "path_len": 0 if s.path is None else len(s.path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-roots", type=int, default=48)
    args = ap.parse_args()
    repo = Path(args.repo)
    common.set_single_thread()
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    cases, _ = common.load_open_split(repo, rg, "calibration")
    cases = cases[:args.max_roots]
    rows = []
    for ci, c in enumerate(cases):
        for b in BIDIR_BUDGETS:
            r = rg.search(c, "bidir", budget=b, depth=16)
            rows.append({"root": c.root, "condition": "bidir", "budget": b, "closed": r["closed"], "work": r["work"]})
        for b in (128, 256, 512):
            for pol in ("geom_mm", "graph_bridge"):
                r = m17.run_lbf17_episode(rg, c, pol, budget=b)
                rows.append({"root": c.root, "condition": pol, "budget": b, "closed": r["closed"], "work": r["work"]})
        env3 = EnvelopeOracle(m17, c.rules, 3)
        env1 = EnvelopeOracle(m17, c.rules, 1)
        for name, mk in (("absint_mm", lambda: env3), ("absint1_mm", lambda: env1),
                         ("exact_mm", lambda: ExactOracle(rg, c, 3))):
            for b in BUDGETS:
                oracle = mk()
                r = run_bound_policy(rg, m15, c, oracle, b)
                row = {"root": c.root, "condition": name, "budget": b, **r}
                if name == "exact_mm":
                    row["exact_ball_successor_generations"] = oracle.generated
                rows.append(row)
        print(f"[audit06] {ci + 1}/{len(cases)} {c.root}", flush=True)

    def agg(cond, b):
        rr = [r for r in rows if r["condition"] == cond and r["budget"] == b]
        if not rr:
            return None
        n = len(rr)
        return {"roots": n, "closures": sum(r["closed"] for r in rr), "closure_rate": sum(r["closed"] for r in rr) / n,
                "mean_penalized_work": sum((r["work"] if r["closed"] else b + 1) for r in rr) / n,
                "mean_implicit_onehop_cached": (sum(r.get("implicit_onehop_successors_cached", 0) for r in rr) / n),
                "mean_wall_ms": (sum(r.get("wall_ms", 0.0) for r in rr) / n)}
    summary = {}
    for cond in ("bidir", "geom_mm", "graph_bridge", "absint_mm", "absint1_mm", "exact_mm"):
        summary[cond] = {str(b): agg(cond, b) for b in sorted({r["budget"] for r in rows if r["condition"] == cond})}
    for cond in ("absint_mm", "absint1_mm", "exact_mm"):
        summary[cond]["auc_5_budget"] = sum(summary[cond][str(b)]["closure_rate"] for b in BUDGETS) / len(BUDGETS)
    # bidir work-to-closure at the largest budget, per root
    wtc = {r["root"]: (r["work"] if r["closed"] else None) for r in rows
           if r["condition"] == "bidir" and r["budget"] == BIDIR_BUDGETS[-1]}
    summary["bidir_work_to_closure_at_4096"] = {
        "closed_roots": sum(v is not None for v in wtc.values()),
        "median": sorted(v for v in wtc.values() if v is not None)[len([v for v in wtc.values() if v is not None]) // 2],
        "mean": sum(v for v in wtc.values() if v is not None) / max(1, sum(v is not None for v in wtc.values())),
        "max": max(v for v in wtc.values() if v is not None),
    }
    common.write_json(Path(args.out), {"definition": __doc__, "summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
