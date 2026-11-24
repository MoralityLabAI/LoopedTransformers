"""LBF-16: S6-equivariant horizon-separated soft-rewrite envelope and set-equivariant meeting fresco.

Resolves LBF-15's two falsified mechanisms:
1. Exact S6 alphabet-permutation equivariance + zero-preserving SparseEnvGate (preserving
   multi-hop support sparsity supp(B_w^(k)) without LayerNorm noise amplification).
2. Horizon separation (anchoring Pass-1 1-hop bridges separately from k>=2 multi-hop novelty)
   and S_{m0}xS_{m1} set-equivariant feasibility-masked axial competition (eliminating planar
   sibling blurring while preserving 1D subterm spatial locality along p=0..L-1).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

from lbf_gym import glyph
import lbf15_meeting_fresco as m15

base = m15.base
WORK = 128
DEPTH = 16
BUDGET_CURVE = (32, 64, 96, 128)
SEED = 20261221
HISTORICAL_SEEDS = m15.HISTORICAL_SEEDS + (
    (20261215, {"train": 64, "calibration": 48, "ordinary": 16, "ood": 16}),
)
COUNTS = {"train": 64, "calibration": 48, "ordinary": 16, "ood": 16}
SEEDS = (107, 109, 113)
STREAMS = (17, 31, 47, 61, 83)
STEPS = 240
BATCH_SIZE = 6
PASSES = 3
CHAR2ID = m15.CHAR2ID
ARCHITECTURES = m15.ARCHITECTURES
OPENED_MODES = m15.OPENED_MODES


# ============================================================================
# 1. S6-Equivariant Support-Preserving Envelope & Set-Equivariant Fresco Net
# ============================================================================
class SparseEnvGate(nn.Module):
    """Zero-preserving nonnegative envelope gate: maps B=0 -> 0 identically, preserving rewrite support."""
    def __init__(self, d: int, init_bias: float = -1.5):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, x: torch.Tensor, msg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.sigmoid(self.gate(torch.cat([x, msg], dim=-1)))
        w_pos = F.softplus(self.proj.weight) / math.sqrt(x.shape[-1])
        return s * F.linear(F.relu(msg), w_pos), s


class MessageGate(nn.Module):
    def __init__(self, d: int, init_bias: float = -1.5):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, x: torch.Tensor, msg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.sigmoid(self.gate(torch.cat([x, msg], dim=-1)))
        return s * self.proj(msg), s


class SparseMultiplicativeRewriteCell(nn.Module):
    """Exact S6-equivariant multiplicative subword match + support-preserving soft-rewrite operator."""
    def __init__(self, ce: int):
        super().__init__()
        self.ce = ce
        self.rule2_mlp = nn.Sequential(nn.Linear(3 * ce, 2 * ce), nn.GELU(), nn.Linear(2 * ce, ce), nn.Softplus())
        self.rule3_mlp = nn.Sequential(nn.Linear(4 * ce, 2 * ce), nn.GELU(), nn.Linear(2 * ce, ce), nn.Softplus())
        self.res_ff = nn.Sequential(nn.Linear(ce, 2 * ce), nn.GELU(), nn.Linear(2 * ce, ce))

    def forward(self, oh_base: torch.Tensor, h_res: torch.Tensor, gated_env: torch.Tensor,
                prev_bank: torch.Tensor,
                r2: tuple[torch.Tensor, torch.Tensor] | None,
                r3: tuple[torch.Tensor, torch.Tensor] | None,
                scramble_pos: torch.Tensor | None = None):
        q_env = torch.clamp(oh_base.expand_as(h_res) + gated_env, 0.0, 1.5)
        if scramble_pos is not None:
            inv = torch.argsort(scramble_pos)
            qw = q_env[:, scramble_pos, :, :]
        else:
            qw = q_env

        N, L, _, Ce = qw.shape
        rew = torch.zeros_like(qw)

        if r2 is not None and L >= 2:
            x2, y2 = r2
            w2 = qw.unfold(1, 2, 1)
            m2 = torch.einsum("nlscq,eqs->nelqc", w2, x2)
            prod2 = m2[:, :, :, 0, :] * m2[:, :, :, 1, :]
            feat2 = torch.cat([prod2, m2.reshape(N, x2.shape[0], L - 1, 2 * Ce)], dim=-1)
            act2 = self.rule2_mlp(feat2) * prod2
            out2 = torch.einsum("nelc,eqs->nlqsc", act2, y2)
            rew[:, :-1] = rew[:, :-1] + out2[:, :, 0]
            rew[:, 1:] = rew[:, 1:] + out2[:, :, 1]

        if r3 is not None and L >= 3:
            x3, y3 = r3
            w3 = qw.unfold(1, 3, 1)
            m3 = torch.einsum("nlscq,eqs->nelqc", w3, x3)
            prod3 = m3[:, :, :, 0, :] * m3[:, :, :, 1, :] * m3[:, :, :, 2, :]
            feat3 = torch.cat([prod3, m3.reshape(N, x3.shape[0], L - 2, 3 * Ce)], dim=-1)
            act3 = self.rule3_mlp(feat3) * prod3
            out3 = torch.einsum("nelc,eqs->nlqsc", act3, y3)
            for q in range(3):
                sl = slice(q, L - 2 + q)
                rew[:, sl] = rew[:, sl] + out3[:, :, q]

        if scramble_pos is not None:
            rew = rew[:, inv, :, :]

        bank_next = torch.tanh(rew + 0.65 * F.relu(prev_bank))
        h_next = torch.tanh(h_res + 0.25 * self.res_ff(h_res))
        return h_next, bank_next, rew, q_env


class AxialSetFrescoBlock(nn.Module):
    """Feasibility-masked S_{m0}xS_{m1} set-equivariant cross-frontier axial attention & competition."""
    def __init__(self, dm: int, structured: bool = True, point_hidden: int = 226):
        super().__init__()
        self.structured = structured
        if structured:
            self.row_q = nn.Linear(dm, dm, bias=False)
            self.row_kv = nn.Linear(dm, 2 * dm, bias=False)
            self.col_q = nn.Linear(dm, dm, bias=False)
            self.col_kv = nn.Linear(dm, 2 * dm, bias=False)
            self.comp_proj = nn.Linear(2 * dm, dm)
            self.mix = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        else:
            self.pointwise = nn.Sequential(
                nn.LayerNorm(dm), nn.Linear(dm, point_hidden), nn.GELU(), nn.Linear(point_hidden, dm),
            )
        self.msg_norm = nn.LayerNorm(dm)

    def forward(self, h: torch.Tensor, prev_bank: torch.Tensor, infeas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.structured:
            upd = self.pointwise(h)
            return h + upd, self.msg_norm(upd + 0.5 * prev_bank)

        dm = h.shape[-1]
        valid = (~infeas).float().unsqueeze(-1)  # (m0, m1, 1)

        # Row-wise attention across feasible backward partners j=1..m1
        rq = self.row_q(h)
        rk, rv = self.row_kv(h).chunk(2, dim=-1)
        r_logits = (rq @ rk.transpose(-1, -2)) / math.sqrt(dm)
        row_mask = infeas.unsqueeze(1) & (~infeas.all(dim=1, keepdim=True).unsqueeze(2))
        r_ctx = torch.softmax(r_logits.masked_fill(row_mask, -1e9), dim=-1) @ rv

        # Column-wise attention across feasible forward partners i=1..m0
        ht = h.transpose(0, 1)
        infeas_t = infeas.transpose(0, 1)
        cq = self.col_q(ht)
        ck, cv = self.col_kv(ht).chunk(2, dim=-1)
        c_logits = (cq @ ck.transpose(-1, -2)) / math.sqrt(dm)
        col_mask = infeas_t.unsqueeze(1) & (~infeas_t.all(dim=1, keepdim=True).unsqueeze(2))
        c_ctx = (torch.softmax(c_logits.masked_fill(col_mask, -1e9), dim=-1) @ cv).transpose(0, 1)

        # Feasibility-masked row/column competitive centering
        col_mean = (h * valid).sum(dim=0, keepdim=True) / valid.sum(dim=0, keepdim=True).clamp_min(1.0)
        row_mean = (h * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        comp = torch.tanh(self.comp_proj(torch.cat([h - col_mean, h - row_mean], dim=-1)))

        rel_msg = 0.5 * (r_ctx + c_ctx) + comp
        h_next = h + self.mix(h + rel_msg)
        return h_next, self.msg_norm(rel_msg + 0.5 * prev_bank)


class EquivariantHorizonFrescoNet(nn.Module):
    def __init__(self, arch: str = "recurrent_opened", ce: int = 21, dm: int = 38, passes: int = PASSES):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch}")
        self.arch = arch
        self.ce = ce
        self.dm = dm
        self.nonrecurrent = (arch == "nonrecurrent")
        self.structured = (arch != "unstructured_recurrent")
        self.passes = 1 if self.nonrecurrent else passes

        self.env_init = nn.Linear(3, ce)
        self.rewrite_cell = SparseMultiplicativeRewriteCell(ce)
        self.ribbon_in = nn.Linear(5 * ce + 1, dm)
        self.ribbon_conv = nn.Conv1d(dm, dm, kernel_size=3, padding=1)
        self.pair_in = nn.Linear(2 * dm + 12, dm)
        self.pair_prelude = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        self.fresco_block = AxialSetFrescoBlock(dm, structured=self.structured, point_hidden=226)

        if self.nonrecurrent:
            self.nr_extra = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 73), nn.GELU(), nn.Linear(73, dm))
        else:
            self.env_gate = SparseEnvGate(ce, init_bias=-6.0)
            self.fresco_gate = MessageGate(dm, init_bias=-6.0)
            self.delta_scale = nn.Parameter(torch.tensor(0.35))

        self.pot_head = nn.Linear(dm, 1)
        self.node_head = nn.Sequential(nn.Linear(2 * ce + 6, dm), nn.GELU(), nn.Linear(dm, 1))
        self.log_tau = nn.Parameter(torch.tensor(-0.7))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def encode_rule_tensors(rules: tuple[tuple[str, str], ...]):
        r2_x, r2_y, r3_x, r3_y = [], [], [], []
        for a, b in rules:
            for src, dst in ((a, b), (b, a)):
                xs = F.one_hot(torch.tensor([CHAR2ID[ch] for ch in src]), num_classes=6).float()
                ys = F.one_hot(torch.tensor([CHAR2ID[ch] for ch in dst]), num_classes=6).float()
                if len(src) == 2:
                    r2_x.append(xs); r2_y.append(ys)
                else:
                    r3_x.append(xs); r3_y.append(ys)
        return ((torch.stack(r2_x), torch.stack(r2_y)) if r2_x else None,
                (torch.stack(r3_x), torch.stack(r3_y)) if r3_x else None)

    def init_words(self, words: list[str]):
        L = len(words[0])
        ids = torch.tensor([[CHAR2ID[ch] for ch in w] for w in words], dtype=torch.long)
        oh = F.one_hot(ids, num_classes=6).float()
        bl = torch.zeros_like(oh); bl[:, 0, :] = oh[:, 0, :]
        br = torch.zeros_like(oh); br[:, -1, :] = oh[:, -1, :]
        h0 = torch.tanh(self.env_init(torch.stack([oh, bl, br], dim=-1)))
        return oh.unsqueeze(-1), h0, ids

    def forward(self, case, s: m15.FringeState, rem_budget: int, *, mode: str = "normal",
                depth_cap: int = DEPTH, return_tensors: bool = False) -> tuple[list[dict], int]:
        if mode not in OPENED_MODES:
            raise ValueError(f"unsupported intervention mode {mode}")
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0, m1 = len(u_words), len(v_words)
        L = len(case.source)
        r2, r3 = self.encode_rule_tensors(case.rules)
        oh_u, hu, u_ids = self.init_words(u_words)
        oh_v, hv, v_ids = self.init_words(v_words)

        g0 = torch.tensor([s.dist[0][w] for w in u_words], dtype=torch.float32)[:, None]
        g1 = torch.tensor([s.dist[1][w] for w in v_words], dtype=torch.float32)[None, :]
        mismatch = (u_ids[:, None, :] != v_ids[None, :, :])
        dh = mismatch.sum(dim=-1).float()
        lb = torch.div(dh.long() + 2, 3, rounding_mode="floor").float()
        tot_lb = g0 + g1 + torch.clamp(lb, min=1.0)
        infeas = tot_lb > depth_cap

        pos_idx = torch.arange(L, dtype=torch.float32).view(1, 1, L)
        first_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, float(L))).min(dim=-1).values
        last_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, -1.0)).max(dim=-1).values
        span = torch.clamp(last_mm - first_mm + 1.0, min=0.0)

        pers = torch.zeros((m0, m1), dtype=torch.float32)
        prev_v = torch.zeros((m0, m1), dtype=torch.float32)
        if mode != "no_persist" and s.prev_pot:
            for i, uw in enumerate(u_words):
                for j, vw in enumerate(v_words):
                    if (uw, vw) in s.prev_pot:
                        pers[i, j] = 1.0
                        prev_v[i, j] = s.prev_pot[(uw, vw)]

        b_frac = rem_budget / WORK
        lm_diff = (math.log1p(m0) - math.log1p(m1)) / math.log1p(WORK)
        pub = torch.stack([
            (g0 / depth_cap).expand(m0, m1),
            (g1 / depth_cap).expand(m0, m1),
            ((g0 + g1) / depth_cap).expand(m0, m1),
            dh / L, lb / depth_cap,
            torch.clamp(depth_cap - tot_lb, min=0.0) / depth_cap,
            1.0 - infeas.float(), span / L,
            torch.full((m0, m1), b_frac),
            torch.full((m0, m1), lm_diff),
            pers, prev_v,
        ], dim=-1)

        sub_checks = 0
        def node_pub(words, w_ids, side):
            nonlocal sub_checks
            ms, mo = len(words), len(s.fronts[1 - side])
            goal = case.target if side == 0 else case.source
            g_ids = torch.tensor([CHAR2ID[ch] for ch in goal], dtype=torch.long)
            dh_goal = (w_ids != g_ids[None, :]).sum(dim=-1).float() / L
            degs = []
            for w in words:
                dg, ch = m15.public_rule_matches(case, w)
                sub_checks += ch
                degs.append(dg / 24.0)
            gs = torch.tensor([s.dist[side][w] / depth_cap for w in words], dtype=torch.float32)
            return torch.stack([
                gs, torch.tensor(degs, dtype=torch.float32),
                torch.full((ms,), math.log1p(ms) / 5.0),
                torch.full((ms,), math.log1p(mo) / 5.0),
                torch.full((ms,), b_frac), dh_goal,
            ], dim=-1)

        np0, np1 = node_pub(u_words, u_ids, 0), node_pub(v_words, v_ids, 1)
        scramble_pos = None
        if mode == "scramble":
            gen = torch.Generator().manual_seed(1337 + m0 * 31 + m1)
            scramble_pos = torch.randperm(L, generator=gen)

        bu = torch.zeros_like(hu)
        bv = torch.zeros_like(hv)
        bu_anchor = bv_anchor = None
        bm = torch.zeros((m0, m1, self.dm), dtype=torch.float32)
        h_mat = pot_accum = None
        outs = []

        for k in range(self.passes):
            if self.nonrecurrent:
                gu = torch.zeros_like(hu)
                gv = torch.zeros_like(hv)
                hu, bu, ru, qu = self.rewrite_cell(oh_u, hu, gu, bu, r2, r3, scramble_pos=scramble_pos)
                hv, bv, rv, qv = self.rewrite_cell(oh_v, hv, gv, bv, r2, r3, scramble_pos=scramble_pos)
                bu_anchor, bv_anchor = bu, bv
                su_gate = sv_gate = sm_gate = gm = mu_in = mm_in = None
            else:
                mu_in, mv_in, mm_in = bu, bv, bm
                if k > 0:
                    if mode == "zero":
                        mu_in = torch.zeros_like(mu_in)
                        mv_in = torch.zeros_like(mv_in)
                        mm_in = torch.zeros_like(mm_in)
                    elif mode == "roll":
                        mu_in = mu_in.roll(1, dims=1)
                        mv_in = mv_in.roll(1, dims=1)
                        mm_in = mm_in.roll(shifts=(1, 1), dims=(0, 1))
                gu, su_gate = self.env_gate(hu, mu_in)
                gv, sv_gate = self.env_gate(hv, mv_in)
                hu, bu, ru, qu = self.rewrite_cell(oh_u, hu, gu, mu_in, r2, r3, scramble_pos=scramble_pos)
                hv, bv, rv, qv = self.rewrite_cell(oh_v, hv, gv, mv_in, r2, r3, scramble_pos=scramble_pos)
                if k == 0:
                    bu_anchor, bv_anchor = bu, bv

            bu_nov = F.relu(bu - bu_anchor)
            bv_nov = F.relu(bv - bv_anchor)
            ov_q = torch.einsum("ilsc,jlsc->ijlc", qu, qv)
            ov_1hop = torch.einsum("ilsc,jlsc->ijlc", bu_anchor, bv_anchor)
            ov_khop = torch.einsum("ilsc,jlsc->ijlc", bu_nov, bv_nov) + 0.5 * (
                torch.einsum("ilsc,jlsc->ijlc", bu_nov, bv_anchor) + torch.einsum("ilsc,jlsc->ijlc", bu_anchor, bv_nov)
            )
            ov_u2v = torch.einsum("ilsc,jlsc->ijlc", bu, oh_v.expand_as(bv))
            ov_v2u = torch.einsum("ilsc,jlsc->ijlc", oh_u.expand_as(bu), bv)

            rib = self.ribbon_in(torch.cat([ov_q, ov_1hop, ov_khop, ov_u2v, ov_v2u, mismatch.float().unsqueeze(-1)], dim=-1))
            if scramble_pos is not None:
                inv_p = torch.argsort(scramble_pos)
                rib_sp = rib[:, :, scramble_pos, :].reshape(m0 * m1, L, self.dm).transpose(1, 2)
                rib_c = F.gelu(self.ribbon_conv(rib_sp)).transpose(1, 2).reshape(m0, m1, L, self.dm)[:, :, inv_p, :]
            else:
                rib_c = F.gelu(self.ribbon_conv(rib.reshape(m0 * m1, L, self.dm).transpose(1, 2))).transpose(1, 2).reshape(m0, m1, L, self.dm)

            mm_mask = mismatch.float().unsqueeze(-1)
            mm_cnt = mm_mask.sum(dim=2).clamp_min(1.0)
            rib_pool = torch.cat([(rib_c * mm_mask).sum(dim=2) / mm_cnt, rib_c.amax(dim=2)], dim=-1)
            pbase = self.pair_in(torch.cat([rib_pool, pub], dim=-1))
            pbase = pbase + self.pair_prelude(pbase)

            if self.nonrecurrent:
                h_mat, bm = self.fresco_block(pbase, bm, infeas)
                h_mat = h_mat + self.nr_extra(h_mat)
                pot = self.pot_head(h_mat).squeeze(-1)
            else:
                if k == 0:
                    gm, sm_gate = self.fresco_gate(pbase, mm_in)
                    h_mat, bm = self.fresco_block(pbase + gm, mm_in, infeas)
                    pot_accum = self.pot_head(h_mat).squeeze(-1)
                else:
                    gm, sm_gate = self.fresco_gate(h_mat, mm_in)
                    h_mat, bm = self.fresco_block(h_mat + 0.5 * pbase + gm, mm_in, infeas)
                    pot_accum = pot_accum + self.delta_scale * self.pot_head(h_mat - pbase).squeeze(-1)
                pot = pot_accum

            pot_masked = pot.masked_fill(infeas, -12.0)
            tau = F.softplus(self.log_tau) + 0.1
            nu_inv = torch.cat([bu_anchor.mean(dim=(1, 2)), bu_nov.amax(dim=(1, 2)), np0], dim=-1)
            nv_inv = torch.cat([bv_anchor.mean(dim=(1, 2)), bv_nov.amax(dim=(1, 2)), np1], dim=-1)
            s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
                  + self.node_head(nu_inv).squeeze(-1)).masked_fill(infeas.all(dim=1), -12.0)
            s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
                  + self.node_head(nv_inv).squeeze(-1)).masked_fill(infeas.all(dim=0), -12.0)

            row = {"pot": pot, "pot_masked": pot_masked, "s0": s0, "s1": s1, "infeas": infeas}
            if return_tensors:
                row["h_mat"] = h_mat.detach().cpu().clone()
                row["bm_out"] = bm.detach().cpu().clone()
                row["bw_u_out"] = bu.detach().cpu().clone()
                row["rib_spatial_autocorr"] = subterm_morans_i(rib_c.detach())
                row["active_envelope_support_frac"] = float((bu.detach() > 1e-6).float().mean())
                if not self.nonrecurrent:
                    row["bm_in"] = mm_in.detach().cpu().clone()
                    row["bw_u_in"] = mu_in.detach().cpu().clone()
                    row["word_gate_strength"] = 0.5 * (float(su_gate.detach().mean()) + float(sv_gate.detach().mean()))
                    row["fresco_gate_strength"] = float(sm_gate.detach().mean())
                    row["fresco_gate_contrib_rms"] = base.rms(gm)
                    row["word_gate_contrib_rms"] = base.rms(gu)
            outs.append(row)
        return outs, sub_checks


def subterm_morans_i(rib_c: torch.Tensor) -> float:
    """1D spatial autocorrelation along the word subterm coordinate lattice p=0..L-1."""
    # rib_c: (m0, m1, L, dm) -> norm along dm -> (m0*m1, L)
    x = rib_c.float().norm(dim=-1).reshape(-1, rib_c.shape[2])
    L = x.shape[1]
    if L < 3:
        return 0.0
    xc = x - x.mean(dim=1, keepdim=True)
    denom = (xc * xc).sum(dim=1)
    valid = denom > 1e-12
    if not bool(valid.any()):
        return 0.0
    num = (xc[:, :-1] * xc[:, 1:]).sum(dim=1)
    vals = (L / (L - 1.0)) * (num[valid] / denom[valid])
    return float(vals.mean())


def new_model(seed: int, arch: str) -> tuple[EquivariantHorizonFrescoNet, str]:
    torch.manual_seed(seed)
    m = EquivariantHorizonFrescoNet(arch=arch).cpu().float()
    if m.parameter_count() > 57475:
        raise ValueError(f"parameter count {m.parameter_count()} exceeds 57,475")
    shared_fp = base._model_fingerprint(m)
    if not m.nonrecurrent:
        for gmod in (m.env_gate, m.fresco_gate):
            if not torch.all(gmod.gate.bias == -6.0):
                raise ValueError("expected base gate bias -6.0")
            if arch != "recurrent_standard":
                with torch.no_grad():
                    gmod.gate.bias.fill_(-1.5)
    return m, shared_fp


# ============================================================================
# 2. Train-Only Relative-Geodesic & Continuation-Advantage Teacher
# ============================================================================
def teacher(rg, suite: Path, out: Path) -> tuple[list[dict], dict]:
    cases, m, refs = rg.load_split(suite, "train", witnesses=True)
    if len(cases) != COUNTS["train"] or m["seed"] != SEED:
        raise ValueError("unexpected LBF-16 training roster")
    out.mkdir(parents=True)
    ref_cert = base.certify(rg, [(c, refs[c.root], "train_" + c.root) for c in cases], out / "references")

    frames = []
    audits = []
    oracle_work = 0
    non_tied = 0
    mixed_closures = 0
    pos_marg = neg_marg = 0
    multi_hop_bridge_frames = 0

    for c in cases:
        nb_cache: dict[str, tuple] = {}
        def nb_fn(word: str):
            if word not in nb_cache:
                nb_cache[word] = rg.neighbors(c, word)
            return nb_cache[word]

        pdist = m15.build_kmer_graph(c)
        for rollout in ("geom", "graph"):
            s = m15.FringeState.start(c)
            prev_snap = None
            prev_rem = WORK
            step = 0
            while s.live(WORK) and step < 8:
                step += 1
                m0, m1 = len(s.fronts[0]), len(s.fronts[1])
                if 2 < m0 + m1 <= 28:
                    rem = WORK - s.work
                    rem_cap = min(rem, 48)
                    sub_b = s.work + rem_cap

                    bu = m15.bounded_balls(c, s.fronts[0], nb_fn, radius=3)
                    bv = m15.bounded_balls(c, s.fronts[1], nb_fn, radius=3)
                    inv_v = defaultdict(list)
                    for j, dmap in enumerate(bv):
                        for w, dv in dmap.items():
                            inv_v[w].append((j, dv))

                    dr_mat = [[99] * m1 for _ in range(m0)]
                    ell_mat = [[99] * m1 for _ in range(m0)]
                    best_node_ell = [[99] * m0, [99] * m1]
                    for i, dmap in enumerate(bu):
                        g0 = s.dist[0][s.fronts[0][i]]
                        for w, du in dmap.items():
                            for j, dv in inv_v.get(w, ()):
                                g1 = s.dist[1][s.fronts[1][j]]
                                dr = max(1, du + dv)
                                ell = g0 + dr + g1
                                if ell <= DEPTH:
                                    if dr < dr_mat[i][j]: dr_mat[i][j] = dr
                                    if ell < ell_mat[i][j]: ell_mat[i][j] = ell
                                    if ell < best_node_ell[0][i]: best_node_ell[0][i] = ell
                                    if ell < best_node_ell[1][j]: best_node_ell[1][j] = ell

                    min_ell = min([x for row in ell_mat for x in row], default=99)
                    min_dr = min([x for row in dr_mat for x in row], default=99)
                    if min_ell < 99:
                        if min_dr >= 3:
                            multi_hop_bridge_frames += 1
                        y_mat = [[math.exp(-0.7 * (ell_mat[i][j] - min_ell) - 0.12 * max(0, dr_mat[i][j] - 1))
                                  if ell_mat[i][j] <= DEPTH else 0.0 for j in range(m1)] for i in range(m0)]

                        costs = [[], []]
                        marg = [[], []]
                        closed_flags = []
                        for side in (0, 1):
                            for idx in range(len(s.fronts[side])):
                                zf = s.clone_light()
                                m15.expand_node(rg, c, zf, side, idx, budget=sub_b, nb_fn=nb_fn)
                                su = zf.work - s.work
                                if zf.path is not None:
                                    cf, cu, cpath = su, 0, zf.path
                                else:
                                    cc, cu, cpath = m15.continuation_cost_geom(rg, c, zf, budget=sub_b, nb_fn=nb_fn)
                                    cf = su + cc if cc <= rem_cap - su else rem_cap + 1
                                oracle_work += su + cu
                                closed_flags.append(cf <= rem_cap)
                                if cpath is not None and sum(x[0].family == c.family for x in audits) < 12:
                                    audits.append((c, cpath, f"cf_{len(audits)}"))

                                if len(s.fronts[side]) > 1:
                                    zr = s.clone_light()
                                    zr.fronts[side].pop(idx)
                                    cr, ru_w, _ = m15.continuation_cost_geom(rg, c, zr, budget=sub_b, nb_fn=nb_fn)
                                    oracle_work += ru_w
                                else:
                                    cr = rem_cap + 1
                                dm_val = (cr - cf) / (rem_cap + 1.0)
                                costs[side].append(cf)
                                marg[side].append(dm_val)
                                pos_marg += int(dm_val > 1e-9)
                                neg_marg += int(dm_val < -1e-9)

                        flat_c = costs[0] + costs[1]
                        if len(set(flat_c)) > 1:
                            non_tied += 1
                        if any(closed_flags) and not all(closed_flags):
                            mixed_closures += 1
                        min_c = min(flat_c)
                        node_y = []
                        for side in (0, 1):
                            for idx in range(len(s.fronts[side])):
                                adv = -(costs[side][idx] - min_c) / (rem_cap + 1.0)
                                geo = math.exp(-0.8 * (best_node_ell[side][idx] - min_ell)) if best_node_ell[side][idx] < 99 else 0.0
                                deg_pen = 0.04 * len(nb_fn(s.fronts[side][idx]))
                                node_y.append(0.70 * adv + 0.25 * marg[side][idx] + 0.90 * geo - deg_pen)

                        frames.append({
                            "root": c.root, "lineage": c.lineage, "step": step, "rollout": rollout,
                            "case": c, "state": s.clone_light(),
                            "prev_state": None if prev_snap is None else prev_snap.clone_light(),
                            "rem": rem, "prev_rem": prev_rem,
                            "min_dr": min_dr, "min_ell": min_ell,
                            "y_mat": torch.tensor(y_mat, dtype=torch.float32),
                            "node_y": torch.tensor(node_y, dtype=torch.float32),
                        })

                prev_snap = s.clone_light()
                prev_rem = WORK - s.work
                if rollout == "geom":
                    (side, idx), _, _ = m15.geom_mm_select(rg, c, s)
                else:
                    (side, idx), _, _ = m15.graph_bridge_select(rg, c, s, pdist)
                m15.expand_node(rg, c, s, side, idx, nb_fn=nb_fn)

    if len(audits) != 24:
        raise ValueError(f"expected 24 counterfactual audit proofs, got {len(audits)}")
    cf_cert = base.certify(rg, audits, out / "counterfactuals")
    serial = [{
        "root": f["root"], "lineage": f["lineage"], "step": f["step"], "rollout": f["rollout"],
        "m0": len(f["state"].fronts[0]), "m1": len(f["state"].fronts[1]),
        "min_dr": f["min_dr"], "min_ell": f["min_ell"],
    } for f in frames]
    base.write_jsonl(out / "frames_summary.jsonl", serial)
    receipt = {
        "train_roots": len(cases),
        "frames": len(frames),
        "non_tied_frames": non_tied,
        "mixed_closure_frames": mixed_closures,
        "multi_hop_bridge_frames_dr_ge_3": multi_hop_bridge_frames,
        "positive_marginal_actions": pos_marg,
        "negative_marginal_actions": neg_marg,
        "oracle_work": oracle_work,
        "reference_certificates": ref_cert["compiled_axiom_free"],
        "counterfactual_certificates": cf_cert["compiled_axiom_free"],
        "frames_summary_sha256": base.file_sha(out / "frames_summary.jsonl"),
    }
    base.write(out / "manifest.json", receipt)
    return frames, receipt


# ============================================================================
# 3. Matched Mini-Batch Training & Complete Campaign Evaluation
# ============================================================================
def fit(frames: list[dict], seed: int, arch: str, out: Path) -> tuple[EquivariantHorizonFrescoNet, dict]:
    model, shared_init = new_model(seed, arch)
    init_fp = base._model_fingerprint(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=2e-4)
    byroot = defaultdict(list)
    for idx, f in enumerate(frames):
        byroot[f["root"]].append((idx, f))
    roots = sorted(byroot)
    rng = random.Random(seed)
    draws = []
    trace = []
    t0 = time.monotonic()

    for step in range(STEPS):
        if time.monotonic() - t0 > 600:
            raise TimeoutError("CPU fit cap exceeded")
        bloss = torch.zeros(())
        step_indices = []
        for _ in range(BATCH_SIZE):
            rname = rng.choice(roots)
            f_idx, fr = rng.choice(byroot[rname])
            step_indices.append(f_idx)
            st_cur = fr["state"].clone_light()
            if fr["prev_state"] is not None:
                with torch.no_grad():
                    po, _ = model(fr["case"], fr["prev_state"], fr["prev_rem"], mode="normal")
                    psig = po[-1]["pot"].sigmoid()
                    pm = float(psig.mean())
                    st_cur.prev_pot = {
                        (u, v): float(psig[i, j]) - pm
                        for i, u in enumerate(fr["prev_state"].fronts[0])
                        for j, v in enumerate(fr["prev_state"].fronts[1])
                    }
            outs, _ = model(fr["case"], st_cur, fr["rem"], mode="normal")
            ym = fr["y_mat"]
            ny = fr["node_y"]
            pos_w = ym.new_full(ym.shape, 5.0)
            tprob = torch.softmax(ny * 4.0, dim=0)
            fl = torch.zeros(())
            ws = 0.0
            for k, o in enumerate(outs):
                wk = float((k + 1) ** 2)
                lm = F.binary_cross_entropy_with_logits(o["pot"], ym, pos_weight=pos_w)
                sc = torch.cat([o["s0"], o["s1"]], dim=0)
                lr = -(tprob * F.log_softmax(sc, dim=0)).sum()
                fl = fl + wk * (lm + lr)
                ws += wk
            bloss = bloss + (fl / ws) / BATCH_SIZE

        if not torch.isfinite(bloss):
            raise ValueError("nonfinite LBF-16 training loss")
        opt.zero_grad(set_to_none=True)
        bloss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            raise ValueError("nonfinite LBF-16 gradient")
        opt.step()
        sched.step()
        drec = {"step": step + 1, "frames": step_indices}
        draws.append(drec)
        trace.append({**drec, "loss": float(bloss.detach())})

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out.mkdir(parents=True)
    torch.save({"arch": arch, "seed": seed, "parameters": model.parameter_count(), "weights": model.state_dict()},
               out / "checkpoint.pt")
    base.write_jsonl(out / "draws.jsonl", draws)
    base.write_jsonl(out / "trace.jsonl", trace)
    receipt = {
        "seed": seed,
        "arch": arch,
        "parameters": model.parameter_count(),
        "passes": model.passes,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "shared_initial": shared_init,
        "initial": init_fp,
        "final": base._model_fingerprint(model),
        "draws_sha256": base.file_sha(out / "draws.jsonl"),
        "trace_sha256": base.file_sha(out / "trace.jsonl"),
        "checkpoint_sha256": base.file_sha(out / "checkpoint.pt"),
    }
    base.write(out / "manifest.json", receipt)
    return model, receipt


def summarize_lbf16_probes(raw_records: list[dict]) -> dict:
    out = {}
    for tag in ("recurrent_standard", "recurrent_opened"):
        rr = [r for r in raw_records if r["tag"].endswith(tag)]
        if not rr:
            continue
        K = len(rr[0]["normal"])
        wg = [r["normal"][K - 1]["word_gate_strength"] for r in rr]
        fg = [r["normal"][K - 1]["fresco_gate_strength"] for r in rr]
        fc_rms = [r["normal"][K - 1]["fresco_gate_contrib_rms"] for r in rr]
        er_mat = [m15.erank_2d(r["normal"][K - 1]["bm_out"]) for r in rr]
        er_word = [m15.erank_2d(r["normal"][K - 1]["bw_u_out"]) for r in rr]
        sub_mi_p1 = [r["normal"][0]["rib_spatial_autocorr"] for r in rr]
        sub_mi_pk = [r["normal"][K - 1]["rib_spatial_autocorr"] for r in rr]
        supp_p1 = [r["normal"][0]["active_envelope_support_frac"] for r in rr]
        supp_pk = [r["normal"][K - 1]["active_envelope_support_frac"] for r in rr]
        bright = [float((r["normal"][K - 1]["pot"].sigmoid() - r["normal"][0]["pot"].sigmoid()).abs().mean()) for r in rr]
        pos_pol = [float(((r["normal"][K - 1]["pot"] - r["normal"][0]["pot"]) > 0.05).float().mean()) for r in rr]

        ms = {
            "probed_decisions": len(rr),
            "passK_env_gate_mean": sum(wg) / len(wg),
            "passK_fresco_gate_mean": sum(fg) / len(fg),
            "passK_fresco_contrib_rms": sum(fc_rms) / len(fc_rms),
            "fresco_bank_erank_mean": sum(er_mat) / len(er_mat),
            "envelope_bank_erank_mean": sum(er_word) / len(er_word),
            "envelope_support_frac_pass1": sum(supp_p1) / len(supp_p1),
            "envelope_support_frac_passK": sum(supp_pk) / len(supp_pk),
            "subterm_morans_i_pass1": sum(sub_mi_p1) / len(sub_mi_p1),
            "subterm_morans_i_passK": sum(sub_mi_pk) / len(sub_mi_pk),
            "mean_abs_pass_brightening": sum(bright) / len(bright),
            "positive_polarity_tile_fraction": sum(pos_pol) / len(pos_pol),
        }
        for md in ("zero", "roll", "scramble", "no_persist"):
            s_norm = [torch.cat([r["normal"][K - 1]["s0"], r["normal"][K - 1]["s1"]]) for r in rr]
            s_alt = [torch.cat([r["alternatives"][md][K - 1]["s0"], r["alternatives"][md][K - 1]["s1"]]) for r in rr]
            ms[f"{md}_max_score_delta"] = max(float((a - b).abs().max()) for a, b in zip(s_norm, s_alt))
            ms[f"{md}_top_action_changes"] = sum(int(a.argmax()) != int(b.argmax()) for a, b in zip(s_norm, s_alt))
        out[tag] = ms
    return out


def verify_historical_disjointness(rg, suite_dir: Path) -> int:
    current_lineages = set()
    for sp in COUNTS:
        rows = rg.read_rows(suite_dir / f"{sp}.jsonl")
        current_lineages.update(r["lineage"] for r in rows)
    if len(current_lineages) != sum(COUNTS.values()):
        raise ValueError("internal lineage collision inside LBF-16 suite")
    historical = set()
    for h_seed, h_counts in HISTORICAL_SEEDS:
        h_cases, _ = rg.generate(h_seed, h_counts)
        for sp_cases in h_cases.values():
            historical.update(x.lineage for x in sp_cases)
    if current_lineages & historical:
        raise ValueError("lineage overlap detected with historical LBF/LSA suite")
    return len(historical)


def campaign(rg, out: Path, threads: int = 2) -> dict:
    if out.exists():
        raise ValueError("fresh LBF-16 output directory required")
    out.mkdir(parents=True)
    torch.set_num_threads(max(1, min(2, threads)))
    torch.backends.mkldnn.enabled = False

    suite = out / "suite"
    sm = rg.freeze(suite, SEED, COUNTS)
    if sm["counts"] != COUNTS or sm["unique_rule_system_lineages"] != sum(COUNTS.values()):
        raise ValueError("LBF-16 suite census mismatch")
    checked_historical = verify_historical_disjointness(rg, suite)

    frames, tm = teacher(rg, suite, out / "teacher")
    models = {}
    receipts = {}
    for seed in SEEDS:
        for arch in ARCHITECTURES:
            k = (seed, arch)
            models[k], receipts[k] = fit(frames, seed, arch, out / f"{seed}_{arch}")
        ref_draws = receipts[(seed, "recurrent_opened")]["draws_sha256"]
        for arch in ARCHITECTURES:
            if receipts[(seed, arch)]["draws_sha256"] != ref_draws:
                raise ValueError("draw schedule diverged across architectures")
        if receipts[(seed, "recurrent_standard")]["shared_initial"] != receipts[(seed, "recurrent_opened")]["shared_initial"]:
            raise ValueError("standard vs opened shared initialization mismatch")

    cal_cases, _ = rg.load_split(suite, "calibration")
    if len(cal_cases) != COUNTS["calibration"] or ({c.lineage for c in cal_cases} & {f["lineage"] for f in frames}):
        raise ValueError("calibration roster or lineage disjointness failed")

    rows = []
    raw_probes = []
    observations = []
    controls = ("astar", "bidir", "geom_mm", "graph_bridge", "uniform")

    for c in cal_cases:
        r_bidir = m15.run_fringe_episode(rg, c, "bidir")
        r_lbf14 = m15.lbf14.run(rg, c, "bidir", budget=WORK, depth=DEPTH)
        if any(r_bidir[k] != r_lbf14[k] for k in ("closed", "work", "expanded", "path")):
            raise ValueError("pinned bidir equivalence check failed")
        rows.append({**r_bidir, "condition": "bidir", "seed": None, "stream": None})

        for cname in ("astar", "geom_mm", "graph_bridge"):
            rc = m15.run_fringe_episode(rg, c, cname)
            rows.append({**rc, "condition": cname, "seed": None, "stream": None})

        for stream in STREAMS:
            ru = m15.run_fringe_episode(rg, c, "uniform", seed=int(base.digest([stream, c.root])[:16], 16))
            rows.append({**ru, "condition": "uniform", "seed": None, "stream": stream})

        for seed in SEEDS:
            for arch in ("nonrecurrent", "unstructured_recurrent", "recurrent_standard"):
                m = models[(seed, arch)]
                pctx = ({"tag": f"{seed}:{arch}", "checkpoint": receipts[(seed, arch)]["checkpoint_sha256"],
                         "raw": raw_probes, "obs": observations}
                        if seed == SEEDS[0] and arch == "recurrent_standard" else None)
                rl = m15.run_fringe_episode(rg, c, "learned", model=m, mode="normal", probe_ctx=pctx)
                rows.append({**rl, "condition": arch, "seed": seed, "stream": None})

            m_open = models[(seed, "recurrent_opened")]
            ck_open = receipts[(seed, "recurrent_opened")]["checkpoint_sha256"]
            for md in OPENED_MODES:
                pctx = ({"tag": f"{seed}:recurrent_opened", "checkpoint": ck_open,
                         "raw": raw_probes, "obs": observations}
                        if seed == SEEDS[0] and md == "normal" else None)
                ro = m15.run_fringe_episode(rg, c, "learned", model=m_open, mode=md, probe_ctx=pctx)
                rows.append({**ro, "condition": f"opened_{md}", "seed": seed, "stream": None})

    for k, m in models.items():
        if base._model_fingerprint(m) != receipts[k]["final"]:
            raise ValueError("frozen checkpoint mutated during evaluation")

    lookup = {c.root: c for c in cal_cases}
    proofs = []
    for i, row in enumerate(rows):
        if row["closed"]:
            rg.replay(lookup[row["root"]], row["path"])
            proofs.append((lookup[row["root"]], row["path"], f"policy_{i}"))
    n_policy_proofs = len(proofs)
    _, _, cal_refs = rg.load_split(suite, "calibration", witnesses=True)
    proofs.extend((c, cal_refs[c.root], "reference_" + c.root) for c in cal_cases)
    cert = base.certify(rg, proofs, out / "calibration_certificates")
    if cert["compiled_axiom_free"] != n_policy_proofs + len(cal_cases):
        raise ValueError("standalone Lean certificate count mismatch")
    cert_set = set(cert["declarations"])
    for i, row in enumerate(rows):
        row["native_certified"] = bool(row["closed"] and f"policy_{i}" in cert_set)
        row["penalized_work"] = row["work"] if row["native_certified"] else WORK + 1

    # Full matched-budget curves across B in {32, 64, 96, 128} including all shams
    budget_curves = {}
    learned_names = ("nonrecurrent", "unstructured_recurrent", "recurrent_standard") + tuple(f"opened_{m}" for m in OPENED_MODES)
    for b_cap in BUDGET_CURVE:
        b_rates = {}
        b_pen = {}
        for cname in ("astar", "bidir", "geom_mm", "graph_bridge"):
            res_c = [m15.run_fringe_episode(rg, c, cname, budget=b_cap) for c in cal_cases]
            b_rates[cname] = sum(int(r["closed"]) for r in res_c) / len(cal_cases)
            b_pen[cname] = sum((r["work"] if r["closed"] else b_cap + 1) for r in res_c) / len(cal_cases)
        for arch in ("nonrecurrent", "unstructured_recurrent", "recurrent_standard"):
            cl_tot = pw_tot = 0
            for seed in SEEDS:
                res_a = [m15.run_fringe_episode(rg, c, "learned", model=models[(seed, arch)], mode="normal", budget=b_cap)
                         for c in cal_cases]
                cl_tot += sum(int(r["closed"]) for r in res_a)
                pw_tot += sum((r["work"] if r["closed"] else b_cap + 1) for r in res_a)
            b_rates[arch] = cl_tot / (len(SEEDS) * len(cal_cases))
            b_pen[arch] = pw_tot / (len(SEEDS) * len(cal_cases))
        for md in OPENED_MODES:
            cl_tot = pw_tot = 0
            for seed in SEEDS:
                res_o = [m15.run_fringe_episode(rg, c, "learned", model=models[(seed, "recurrent_opened")], mode=md, budget=b_cap)
                         for c in cal_cases]
                cl_tot += sum(int(r["closed"]) for r in res_o)
                pw_tot += sum((r["work"] if r["closed"] else b_cap + 1) for r in res_o)
            b_rates[f"opened_{md}"] = cl_tot / (len(SEEDS) * len(cal_cases))
            b_pen[f"opened_{md}"] = pw_tot / (len(SEEDS) * len(cal_cases))
        budget_curves[str(b_cap)] = {"closure_rate": b_rates, "mean_penalized_work": b_pen}

    budget_auc = {}
    for cond_k in budget_curves["128"]["closure_rate"]:
        budget_auc[cond_k] = sum(budget_curves[str(b)]["closure_rate"][cond_k] for b in BUDGET_CURVE) / len(BUDGET_CURVE)

    base.write_jsonl(out / "episodes.jsonl", rows)
    base.write_jsonl(out / "observations.jsonl", observations)
    encoded = [base.encode(o) for o in observations]
    base.write_jsonl(out / "encoded.jsonl", encoded)
    torch.save(raw_probes, out / "raw_fresco_tensors.pt")
    (out / "fresco.svg").write_text(base.fresco_svg(encoded[0]), encoding="utf-8")

    open_probes = [r for r in raw_probes if r["tag"].endswith("recurrent_opened")]
    showcase = max(open_probes, key=lambda r: len(r["u_words"]) * len(r["v_words"]))
    ms_svg = m15.render_manuscript_svg(
        lookup[showcase["root"]], showcase["u_words"], showcase["v_words"],
        [o["pot"] for o in showcase["normal"]],
        [o["s0"] for o in showcase["normal"]],
        [o["s1"] for o in showcase["normal"]],
        showcase["persisted_mask"],
    )
    (out / "fresco_manuscript.svg").write_text(ms_svg, encoding="utf-8")

    all_names = controls + learned_names
    rates = {}
    costs = {}
    seed_rates = {str(s): {} for s in SEEDS}
    seed_pen_work = {str(s): {} for s in SEEDS}
    for name in all_names:
        rr = [r for r in rows if r["condition"] == name]
        rates[name] = sum(r["native_certified"] for r in rr) / len(rr)
        costs[name] = {
            "episodes": len(rr),
            "certified_closures": sum(r["native_certified"] for r in rr),
            "work": sum(r["work"] for r in rr),
            "mean_work": sum(r["work"] for r in rr) / len(rr),
            "mean_penalized_work": sum(r["penalized_work"] for r in rr) / len(rr),
            "expanded": sum(r["expanded"] for r in rr),
            "model_calls": sum(r["model_calls"] for r in rr),
            "recurrent_passes": sum(r["recurrent_passes"] for r in rr),
            "pair_evaluations": sum(r["pair_evaluations"] for r in rr),
            "substring_checks": sum(r["substring_checks"] for r in rr),
        }
    for seed in SEEDS:
        for name in learned_names:
            rr = [r for r in rows if r["condition"] == name and r["seed"] == seed]
            seed_rates[str(seed)][name] = sum(r["native_certified"] for r in rr) / len(rr)
            seed_pen_work[str(seed)][name] = sum(r["penalized_work"] for r in rr) / len(rr)

    def root_mean(c, name: str, key: str) -> float:
        rr = [r for r in rows if r["root"] == c.root and r["condition"] == name]
        expected = len(STREAMS) if name == "uniform" else (1 if name in controls else len(SEEDS))
        if len(rr) != expected:
            raise ValueError(f"incomplete root pairing for {name}")
        return sum(r[key] for r in rr) / len(rr)

    paired = {}
    for name in all_names:
        if name == "opened_normal":
            continue
        q = [root_mean(c, "opened_normal", "native_certified") - root_mean(c, name, "native_certified") for c in cal_cases]
        v = [root_mean(c, "opened_normal", "penalized_work") - root_mean(c, name, "penalized_work") for c in cal_cases]
        paired["opened_minus_" + name] = {
            "closure_mean": sum(q) / len(q),
            "closure_95": base.bootstrap(q),
            "penalized_work_mean": sum(v) / len(v),
            "penalized_work_95": base.bootstrap(v),
            "wins": sum(x > 0 for x in q),
            "losses": sum(x < 0 for x in q),
            "ties": sum(x == 0 for x in q),
        }

    # Evaluate pre-declared LBF-16 screens
    bidir_gain = rates["opened_normal"] - rates["bidir"]
    bidir_save = 1.0 - costs["opened_normal"]["mean_penalized_work"] / costs["bidir"]["mean_penalized_work"]
    screen_1a = (
        bidir_gain >= 0.10
        and bidir_save >= 0.25
        and all(seed_rates[str(s)]["opened_normal"] > rates["bidir"] for s in SEEDS)
    )

    best_ctrl = max(controls, key=lambda n: (rates[n], -costs[n]["mean_penalized_work"]))
    ctrl_gain = rates["opened_normal"] - rates[best_ctrl]
    ctrl_save = 1.0 - costs["opened_normal"]["mean_penalized_work"] / costs[best_ctrl]["mean_penalized_work"]
    screen_1b = (
        ctrl_gain >= 0.05
        and ctrl_save >= 0.20
        and all(seed_rates[str(s)]["opened_normal"] > rates[best_ctrl] for s in SEEDS)
    )

    screen_2a = all(
        (budget_auc["opened_normal"] > budget_auc[other] or rates["opened_normal"] > rates[other])
        and rates["opened_normal"] >= rates[other] - 1e-9
        for other in ("nonrecurrent", "unstructured_recurrent", "recurrent_standard")
    )
    screen_2b = all(
        budget_auc["opened_normal"] > budget_auc[f"opened_{m}"]
        and rates["opened_normal"] >= rates[f"opened_{m}"] - 1e-9
        for m in ("zero", "roll", "scramble")
    )

    result = {
        "schema": "lbf16.equivariant_horizon_fresco.v1",
        "source_commit": base.UPSTREAM_COMMIT,
        "source_blob": base.UPSTREAM_BLOB,
        "generator_seed": SEED,
        "historical_lineages_verified_disjoint": checked_historical,
        "suite_manifest_sha256": base.file_sha(suite / "manifest.json"),
        "training": tm,
        "fits": [receipts[k] for k in sorted(receipts)],
        "rates": rates,
        "seed_rates": seed_rates,
        "seed_penalized_work": seed_pen_work,
        "costs": costs,
        "paired": paired,
        "budget_curves": budget_curves,
        "budget_curve_mean_closure_auc": budget_auc,
        "strongest_non_neural_control": best_ctrl,
        "screens": {
            "primary_1a_vs_symbolic_bidir": screen_1a,
            "primary_1b_vs_strongest_heuristic": screen_1b,
            "secondary_2a_recurrence_and_structure": screen_2a,
            "secondary_2b_causal_memory_and_spatial": screen_2b,
        },
        "fresco_probe_summary": summarize_lbf16_probes(raw_probes),
        "policy_certificates": n_policy_proofs,
        "reference_certificates": len(cal_cases),
        "sealed_outcomes_opened": False,
        "native_online_tactic_calls": 0,
        "artifacts": {
            x: base.file_sha(out / x)
            for x in ("episodes.jsonl", "observations.jsonl", "encoded.jsonl",
                      "raw_fresco_tensors.pt", "fresco.svg", "fresco_manuscript.svg")
        },
    }
    base.write(out / "manifest.json", result)
    print("LBF16_RESULT", base.canonical(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    campaign(base.load_upstream(args.upstream), args.out, threads=args.threads)


if __name__ == "__main__":
    main()
