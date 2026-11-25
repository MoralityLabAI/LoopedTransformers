"""LBF-19 engineering: vectorised dense pair store for `IncrementalSymController` (exactly the same function).

The LBF-18 controller keeps a Python dict of per-pair ribbon entries and rebuilds `(m0, m1)` grids with
per-element tensor assignments and `.item()` calls.  At frontiers of several hundred words this dominates the
decision time (11 of 18 s in a profiled episode).  `FastSymController` replaces those loops with a dense
`(Nu, Nv, K, 2*dm)` store and index gathers.  Numerics are untouched: the same blocks are computed by the same
code, and `tests/test_lbf19_fast_controller.py` checks path / work equality against the LBF-18 controller.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

import lbf15_meeting_fresco as m15
import lbf18_incremental_fresco as m18


class FastSymController(m18.IncrementalSymController):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._reset_store()

    def _reset_store(self):
        self._uid, self._vid = {}, {}
        self._cap = (0, 0)
        self._RIB = self._DH = self._LB = self._SP = self._HAVE = None

    def clear_caches(self):
        super().clear_caches()
        self._reset_store()

    def _ids(self, table, words):
        out = []
        for w in words:
            i = table.get(w)
            if i is None:
                i = len(table)
                table[w] = i
            out.append(i)
        return out

    def _ensure(self, nu, nv, K, D2):
        cu, cv = self._cap
        if self._RIB is not None and nu <= cu and nv <= cv:
            return
        ncu, ncv = max(nu, 2 * cu, 32), max(nv, 2 * cv, 32)
        RIB = torch.zeros((ncu, ncv, K, D2))
        DH, LB, SP = (torch.zeros((ncu, ncv)) for _ in range(3))
        HAVE = torch.zeros((ncu, ncv), dtype=torch.bool)
        if self._RIB is not None:
            RIB[:cu, :cv] = self._RIB
            DH[:cu, :cv], LB[:cu, :cv], SP[:cu, :cv] = self._DH, self._LB, self._SP
            HAVE[:cu, :cv] = self._HAVE
        self._RIB, self._DH, self._LB, self._SP, self._HAVE, self._cap = RIB, DH, LB, SP, HAVE, (ncu, ncv)

    def _blocks(self, case, sub_u, sub_v, L):
        """Identical computation to `IncrementalSymController.forward` (pair ribbon pools)."""
        if self.is_hybrid:
            pools_blk, dh_blk, lb_blk, sp_blk = self.model.compute_rib_pools(sub_u, sub_v)
            est = len(sub_u) * len(sub_v) * self.model.passes * L * (2 * 14 * 38 + 2 * 2 * 3 * 38 * 38)
            return pools_blk, dh_blk, lb_blk, sp_blk, est
        m = self.model
        du_s, dv_s = len(sub_u), len(sub_v)
        u_ids = torch.stack([e.w_ids for e in sub_u], dim=0)
        v_ids = torch.stack([e.w_ids for e in sub_v], dim=0)
        mismatch = (u_ids[:, None, :] != v_ids[None, :, :])
        dh_blk = mismatch.sum(dim=-1).float()
        lb_blk = torch.div(dh_blk.long() + 2, 3, rounding_mode="floor").float()
        pos_idx = torch.arange(L, dtype=torch.float32).view(1, 1, L)
        fmm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, float(L))).min(dim=-1).values
        lmm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, -1.0)).max(dim=-1).values
        sp_blk = torch.clamp(lmm - fmm + 1.0, min=0.0)
        ub0 = torch.stack([e.ub0 for e in sub_u], dim=0)
        vb0 = torch.stack([e.ub0 for e in sub_v], dim=0)
        up1 = torch.stack([e.up_horizons[0] for e in sub_u], dim=0)
        vp1 = torch.stack([e.up_horizons[0] for e in sub_v], dim=0)
        up3 = torch.stack([e.up_horizons[2] for e in sub_u], dim=0)
        vp3 = torch.stack([e.up_horizons[2] for e in sub_v], dim=0)
        uf3 = torch.stack([e.uf_horizons[2] for e in sub_u], dim=0)
        vf3 = torch.stack([e.uf_horizons[2] for e in sub_v], dim=0)
        unov = F.relu(up3 - up1)
        vnov = F.relu(vp3 - vp1)
        ov0 = torch.einsum("ipk,jpk->ijp", ub0, vb0).unsqueeze(-1)
        ov1 = torch.einsum("ipk,jpk->ijp", up1, vp1).unsqueeze(-1)
        ovk = (torch.einsum("ipk,jpk->ijp", unov, vnov)
               + 0.5 * torch.einsum("ipk,jpk->ijp", unov, vp1)
               + 0.5 * torch.einsum("ipk,jpk->ijp", up1, vnov)).unsqueeze(-1)
        u2v1 = torch.einsum("ipk,jpk->ijp", up1, vb0).unsqueeze(-1)
        v2u1 = torch.einsum("ipk,jpk->ijp", ub0, vp1).unsqueeze(-1)
        u2v3 = torch.einsum("ipk,jpk->ijp", up3, vb0).unsqueeze(-1)
        v2u3 = torch.einsum("ipk,jpk->ijp", ub0, vp3).unsqueeze(-1)
        fib_p = torch.einsum("ifpk,jgpk->ijfgp", uf3, vf3).amax(dim=(2, 3)).unsqueeze(-1)
        supp = 0.5 * (up3.mean(dim=-1)[:, None, :, None] + vp3.mean(dim=-1)[None, :, :, None])
        edge9 = torch.cat([ov0, ov1, ovk, u2v1, v2u1, u2v3, v2u3, fib_p, supp], dim=-1)
        node9 = m.pad_edges_to_nodes(edge9)
        sym_feat = torch.cat([node9, mismatch.float().unsqueeze(-1)], dim=-1)
        rib = m.ribbon_in(sym_feat).reshape(du_s * dv_s, L, m.dm).transpose(1, 2)
        rc_t = F.gelu(m.sym_rib_conv2(F.gelu(m.ribbon_conv(rib)))).transpose(1, 2).reshape(du_s, dv_s, L, m.dm)
        mm_m = mismatch.float().unsqueeze(-1)
        pools_blk = torch.cat([(rc_t * mm_m).sum(dim=2) / mm_m.sum(dim=2).clamp_min(1.0), rc_t.amax(dim=2)],
                              dim=-1).unsqueeze(2)
        return pools_blk, dh_blk, lb_blk, sp_blk, du_s * dv_s * L * (2 * 10 * 38 + 2 * 2 * 3 * 38 * 38)

    def forward(self, case, s, rem_budget, *, depth_cap=m18.DEPTH):
        if m18.live_param_signature(self.model) != self.bound_signature:
            raise m18.CacheInvalidationError("symbolic-readout model mutated; cache reuse rejected")
        r_key = m18.canonical_rules_key(case.rules)
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0, m1 = len(u_words), len(v_words)
        L = len(case.source)
        missing_w, seen_w = [], set()
        for w in u_words + v_words:
            wk = (self.checkpoint_sha, self.guard_mode, r_key, w)
            if wk in self.word_cache:
                self.word_hits += 1
            else:
                self.word_misses += 1
                if w not in seen_w:
                    seen_w.add(w)
                    missing_w.append(w)
        if missing_w:
            ents, rc = m18.compute_sym_bitset_word_entries(case, missing_w, use_rust=self.use_rust,
                                                           guard_mode=self.guard_mode)
            self.rule_window_checks += rc
            for w, ent in zip(missing_w, ents):
                self.word_cache[(self.checkpoint_sha, self.guard_mode, r_key, w)] = ent
        u_ent = [self.word_cache[(self.checkpoint_sha, self.guard_mode, r_key, w)] for w in u_words]
        v_ent = [self.word_cache[(self.checkpoint_sha, self.guard_mode, r_key, w)] for w in v_words]

        K = self.model.passes if self.is_hybrid else 1
        D2 = 2 * self.model.dm
        uid = self._ids(self._uid, u_words)
        vid = self._ids(self._vid, v_words)
        self._ensure(len(self._uid), len(self._vid), K, D2)
        ut = torch.tensor(uid, dtype=torch.long)
        vt = torch.tensor(vid, dtype=torch.long)
        have = self._HAVE[ut[:, None], vt[None, :]]
        n_miss = int((~have).sum())
        self.pair_hits += m0 * m1 - n_miss
        self.pair_misses += n_miss
        if n_miss:
            rows = torch.nonzero(~have.all(dim=1)).flatten().tolist()
            cols = torch.nonzero(~have.all(dim=0)).flatten().tolist()
            sub_u = [u_ent[i] for i in rows]
            sub_v = [v_ent[j] for j in cols]
            pools_blk, dh_blk, lb_blk, sp_blk, est = self._blocks(case, sub_u, sub_v, L)
            self.neural_flops += int(est)
            rt = ut[rows][:, None].expand(len(rows), len(cols))
            ct = vt[cols][None, :].expand(len(rows), len(cols))
            self._RIB[rt, ct] = pools_blk.reshape(len(rows), len(cols), K, D2)
            self._DH[rt, ct] = dh_blk
            self._LB[rt, ct] = lb_blk
            self._SP[rt, ct] = sp_blk
            self._HAVE[rt, ct] = True
        gi, gj = ut[:, None], vt[None, :]
        rib_grid = self._RIB[gi, gj]
        dh, lb, span = self._DH[gi, gj], self._LB[gi, gj], self._SP[gi, gj]

        if self.is_hybrid:
            outs = self.model.forward_from_pools(case, s, rem_budget, u_ent, v_ent, rib_grid, dh, lb, span,
                                                 depth_cap=depth_cap)
            self.neural_flops += m18._calibrated_stage_c_flops("rel_prov_opened", m0, m1, len(outs))
            return outs

        m = self.model
        g0 = torch.tensor([s.dist[0][w] for w in u_words], dtype=torch.float32)[:, None]
        g1 = torch.tensor([s.dist[1][w] for w in v_words], dtype=torch.float32)[None, :]
        tot_lb = g0 + g1 + torch.clamp(lb, min=1.0)
        infeas = tot_lb > depth_cap
        if s.prev_pot:
            pp = s.prev_pot
            pers = torch.tensor([[1.0 if (u, v) in pp else 0.0 for v in v_words] for u in u_words], dtype=torch.float32)
            prev_v = torch.tensor([[pp.get((u, v), 0.0) for v in v_words] for u in u_words], dtype=torch.float32)
        else:
            pers = torch.zeros((m0, m1), dtype=torch.float32)
            prev_v = torch.zeros((m0, m1), dtype=torch.float32)
        b_frac = rem_budget / m18.WORK
        lm_diff = (math.log1p(m0) - math.log1p(m1)) / math.log1p(m18.WORK)
        pub = torch.stack([
            (g0 / depth_cap).expand(m0, m1), (g1 / depth_cap).expand(m0, m1),
            ((g0 + g1) / depth_cap).expand(m0, m1), dh / L, lb / depth_cap,
            torch.clamp(depth_cap - tot_lb, min=0.0) / depth_cap,
            1.0 - infeas.float(), span / L,
            torch.full((m0, m1), b_frac), torch.full((m0, m1), lm_diff), pers, prev_v,
        ], dim=-1)

        def sym_np(entries, words, side):
            ms, mo = len(words), len(s.fronts[1 - side])
            goal = case.target if side == 0 else case.source
            g_ids = torch.tensor([m18.CHAR2ID[c] for c in goal], dtype=torch.long)
            w_ids = torch.stack([e.w_ids for e in entries], dim=0)
            dh_g = (w_ids != g_ids[None, :]).sum(dim=-1).float() / L
            degs = torch.tensor([e.pub_deg_norm for e in entries], dtype=torch.float32)
            gs = torch.tensor([s.dist[side][w] / depth_cap for w in words], dtype=torch.float32)
            return torch.stack([gs, degs, torch.full((ms,), math.log1p(ms) / 5.0),
                                torch.full((ms,), math.log1p(mo) / 5.0), torch.full((ms,), b_frac), dh_g], dim=-1)

        pbase = m.pair_in(torch.cat([rib_grid[:, :, 0, :], pub], dim=-1))
        pbase = pbase + m.pair_prelude(pbase)
        h_mat, _ = m.fresco_block(pbase, torch.zeros((m0, m1, m.dm)), infeas)
        h_mat = h_mat + m.sym_pair_ff(h_mat)
        pot = m.pot_head(h_mat).squeeze(-1)
        pot_masked = pot.masked_fill(infeas, -12.0)
        tau = F.softplus(m.log_tau) + 0.1
        s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
              + m.node_head(sym_np(u_ent, u_words, 0)).squeeze(-1)).masked_fill(infeas.all(dim=1), -12.0)
        s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
              + m.node_head(sym_np(v_ent, v_words, 1)).squeeze(-1)).masked_fill(infeas.all(dim=0), -12.0)
        self.neural_flops += m18._calibrated_stage_c_flops("param_nonrecurrent", m0, m1, 1)
        return [{"pot": pot, "pot_masked": pot_masked, "s0": s0, "s1": s1, "infeas": infeas}]


def run_fast_episode(rg, case, controller, *, budget: int, depth: int) -> dict:
    """`run_clean_episode` for `IncrementalSymController`-family controllers with a batched `prev_pot` update."""
    controller.clear_caches()
    controller.reset_telemetry()
    s = m15.FringeState.start(case)
    decisions = pair_evals = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        while s.live(budget):
            decisions += 1
            m0, m1 = len(s.fronts[0]), len(s.fronts[1])
            rem = budget - s.work
            outs = controller.forward(case, s, rem, depth_cap=depth)
            pair_evals += m0 * m1
            last = outs[-1]
            psig = last["pot"].sigmoid()
            pm = float(psig.mean().item())
            vals = (psig - 0.0).tolist()
            s.prev_pot = {(u, v): vals[i][j] - pm for i, u in enumerate(s.fronts[0]) for j, v in enumerate(s.fronts[1])}
            s0_l = last["s0"].tolist()
            s1_l = last["s1"].tolist()
            cands = [(s0_l[i], -m0, 0, -i) for i in range(m0)] + [(s1_l[j], -m1, -1, -j) for j in range(m1)]
            best = max(cands)
            side = 0 if best[2] == 0 else 1
            m15.expand_node(rg, case, s, side, -best[3], budget, depth)
    elapsed = (time.perf_counter() - t0) * 1000.0
    res = rg._result(case, "learned", s.path, s.work, s.expanded, decisions, budget, depth)
    return {**res, "pair_evaluations": pair_evals, "fringe_decisions": decisions, "neural_flops": controller.neural_flops,
            "wall_clock_ms": elapsed, "word_hits": controller.word_hits, "word_misses": controller.word_misses,
            "pair_hits": controller.pair_hits, "pair_misses": controller.pair_misses}
