"""LBF-17: Relational Abstract Interpretation and Provenance-Preserving Equivariant Frescoes.

Implements:
1. Symbolic relational abstract interpreters: Pos1 (A_1), Prov-Pos1 (A_prov1), Adj2 (A_2),
   Sub3 (A_3), and Spatial-Provenance x Adj2 (A_{prov4 x adj2}), plus bounded concrete k-balls.
2. Differentiable S6-equivariant, zero-preserving FastProvenanceBigramCell with rational-saturation
   trellis filtering and 1-step inverse-redex suppression, plus calibrated subterm glyph brightness.
3. Parameter-matched (~53,047) and compute-matched controller hierarchy:
   rel_prov_opened, rel_prov_standard, adj2_noprov_opened, lowrank_opened, lbf16_pos1_opened,
   param_nonrecurrent, compute_nonrecurrent (>=1.04x 3-pass FLOPs at K=1), sym_prov_readout,
   and successor-charged exact_lookahead.
4. Five orthogonal factored causal interventions: scramble_window, scramble_ribbon, ablate_axial,
   env_zero / env_roll, and mat_zero / mat_roll.
5. Prospective evaluation across budgets {16, 32, 64, 96, 128}, abstraction-error & glyph-AUROC
   diagnostics, and standalone axiom-free Lean 4.19.0 certification (Proofs + LBF17AbstractInterp.lean).
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import functools
import gc
import html
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from lbf_gym import glyph
import lbf15_meeting_fresco as m15
import lbf16_equivariant_fresco as m16

base = m16.base
WORK = 128
DEPTH = 16
BUDGET_CURVE = (16, 32, 64, 96, 128)
SEED = 20261227
HISTORICAL_SEEDS = m16.HISTORICAL_SEEDS + (
    (20261221, {"train": 64, "calibration": 48, "ordinary": 16, "ood": 16}),
)
COUNTS = {"train": 64, "calibration": 48, "ordinary": 16, "ood": 16}
SEEDS = (107, 109, 113)
STREAMS = (17, 31, 47, 61, 83)
STEPS = 240
BATCH_SIZE = 6
PASSES = 3
FP_FIBERS = 4
CHAR2ID = m16.CHAR2ID

LBF17_ARCHS = (
    "rel_prov_opened",
    "rel_prov_standard",
    "adj2_noprov_opened",
    "lowrank_opened",
    "param_nonrecurrent",
    "compute_nonrecurrent",
    "sym_prov_readout",
)
ALL_LEARNED_ARCHS = LBF17_ARCHS + ("lbf16_pos1_opened",)

FACTORED_MODES = (
    "normal",
    "scramble_window",
    "scramble_ribbon",
    "ablate_axial",
    "env_zero",
    "env_roll",
    "mat_zero",
    "mat_roll",
)
SHAM_MODES = FACTORED_MODES[1:]


# ============================================================================
# 1. Concrete Bounded Reachability & Relational Abstract Interpreter Hierarchy
# ============================================================================
def concrete_k_balls(rg, c, w0: str, k_max: int = 3) -> list[set[str]]:
    """Exact concrete <=k-step reachable word sets F_R^k({w0}) for k=0..k_max."""
    seen = {w0: 0}
    q = deque([w0])
    while q:
        u = q.popleft()
        d = seen[u]
        if d >= k_max:
            continue
        for mv in rg.neighbors(c, u):
            v = mv.successor
            if v not in seen:
                seen[v] = d + 1
                q.append(v)
    layers = [set() for _ in range(k_max + 1)]
    for w, d in seen.items():
        for k in range(d, k_max + 1):
            layers[k].add(w)
    return layers


def alpha1_bool(words: set[str] | list[str], L: int) -> torch.Tensor:
    a = torch.zeros((L, 6), dtype=torch.bool)
    for w in words:
        for p, ch in enumerate(w):
            a[p, CHAR2ID[ch]] = True
    return a


def alpha2_bool(words: set[str] | list[str], L: int) -> torch.Tensor:
    a = torch.zeros((L - 1, 6, 6), dtype=torch.bool)
    for w in words:
        for p in range(L - 1):
            a[p, CHAR2ID[w[p]], CHAR2ID[w[p + 1]]] = True
    return a


def alpha3_bool(words: set[str] | list[str], L: int) -> torch.Tensor:
    a = torch.zeros((L - 2, 6, 6, 6), dtype=torch.bool)
    for w in words:
        for p in range(L - 2):
            a[p, CHAR2ID[w[p]], CHAR2ID[w[p + 1]], CHAR2ID[w[p + 2]]] = True
    return a


def adj2_to_pos1(a2: torch.Tensor) -> torch.Tensor:
    Lm1 = a2.shape[0]
    p1 = torch.zeros((Lm1 + 1, 6), dtype=torch.bool)
    p1[:-1] = a2.any(dim=2)
    p1[-1] = a2[-1].any(dim=0)
    return p1


def sub3_to_pos1(a3: torch.Tensor) -> torch.Tensor:
    Lm2 = a3.shape[0]
    p1 = torch.zeros((Lm2 + 2, 6), dtype=torch.bool)
    p1[:-2] = a3.any(dim=(2, 3))
    p1[-2] = a3[-1].any(dim=(0, 2))
    p1[-1] = a3[-1].any(dim=(0, 1))
    return p1


def trellis_filter_adj2_bool(a2: torch.Tensor) -> torch.Tensor:
    """Exact O(L * 36) forward-backward trellis projection alpha_2(gamma_2(a2))."""
    Lm1 = a2.shape[0]
    fwd = torch.zeros((Lm1 + 1, 6), dtype=torch.bool)
    fwd[0] = a2[0].any(dim=1)
    for p in range(Lm1):
        fwd[p + 1] = (fwd[p, :, None] & a2[p]).any(dim=0)
    bwd = torch.zeros((Lm1 + 1, 6), dtype=torch.bool)
    bwd[Lm1] = a2[Lm1 - 1].any(dim=0)
    for p in range(Lm1 - 1, -1, -1):
        bwd[p] = (a2[p] & bwd[p + 1, None, :]).any(dim=1)
    return fwd[:-1, :, None] & a2 & bwd[1:, None, :]


def trellis_filter_sub3_bool(a3: torch.Tensor) -> torch.Tensor:
    """Exact O(L * 216) forward-backward bigram-chain projection alpha_3(gamma_3(a3))."""
    Lm2 = a3.shape[0]
    fwd = torch.zeros((Lm2 + 1, 6, 6), dtype=torch.bool)
    fwd[0] = a3[0].any(dim=2)
    for p in range(Lm2):
        fwd[p + 1] = (fwd[p, :, :, None] & a3[p]).any(dim=0)
    bwd = torch.zeros((Lm2 + 1, 6, 6), dtype=torch.bool)
    bwd[Lm2] = a3[Lm2 - 1].any(dim=0)
    for p in range(Lm2 - 1, -1, -1):
        bwd[p] = (a3[p] & bwd[p + 1, None, :, :]).any(dim=2)
    return fwd[:-1, :, :, None] & a3 & bwd[1:, None, :, :]


def symbolic_pos1(rules: tuple[tuple[str, str], ...], w0: str, k_max: int = 3) -> list[torch.Tensor]:
    L = len(w0)
    a = alpha1_bool((w0,), L)
    outs = [a.clone()]
    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    for _ in range(k_max):
        nxt = a.clone()
        for lhs, rhs in ors:
            ll = len(lhs)
            lids = [CHAR2ID[ch] for ch in lhs]
            rids = [CHAR2ID[ch] for ch in rhs]
            for p in range(L - ll + 1):
                if all(bool(a[p + q, lids[q]]) for q in range(ll)):
                    for q in range(ll):
                        nxt[p + q, rids[q]] = True
        a = nxt
        outs.append(a.clone())
    return outs


def symbolic_prov_pos1(rules: tuple[tuple[str, str], ...], w0: str, k_max: int = 3) -> list[list[torch.Tensor]]:
    """Non-backtracking 1-step redex-origin provenance fibers over Pos1."""
    L = len(w0)
    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    base_a = alpha1_bool((w0,), L)
    history = [[base_a.clone()]]
    if k_max == 0:
        return history
    fibers = []
    for lhs, rhs in ors:
        ll = len(lhs)
        for p in range(L - ll + 1):
            if w0[p:p + ll] == lhs:
                w1 = w0[:p] + rhs + w0[p + ll:]
                fibers.append([alpha1_bool((w1,), L), p, ll, lhs])
    history.append([base_a.clone()] + [f[0].clone() for f in fibers])
    for step_idx in range(1, k_max):
        for item in fibers:
            fa, orig_p, orig_ll, orig_lhs = item
            nxt = fa.clone()
            for lhs, rhs in ors:
                ll = len(lhs)
                lids = [CHAR2ID[ch] for ch in lhs]
                rids = [CHAR2ID[ch] for ch in rhs]
                for p in range(L - ll + 1):
                    if step_idx == 1 and p == orig_p and ll == orig_ll and rhs == orig_lhs:
                        continue
                    if all(bool(fa[p + q, lids[q]]) for q in range(ll)):
                        for q in range(ll):
                            nxt[p + q, rids[q]] = True
            item[0] = nxt
        history.append([base_a.clone()] + [f[0].clone() for f in fibers])
    return history


def symbolic_adj2(rules: tuple[tuple[str, str], ...], w0: str, k_max: int = 3) -> list[torch.Tensor]:
    L = len(w0)
    Lm1 = L - 1
    a2 = alpha2_bool((w0,), L)
    outs = [a2.clone()]
    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    for _ in range(k_max):
        ta = trellis_filter_adj2_bool(a2)
        nxt = ta.clone()
        for lhs, rhs in ors:
            ll = len(lhs)
            x = [CHAR2ID[ch] for ch in lhs]
            y = [CHAR2ID[ch] for ch in rhs]
            if ll == 2:
                for p in range(Lm1):
                    if ta[p, x[0], x[1]]:
                        nxt[p, y[0], y[1]] = True
                        if p > 0:
                            nxt[p - 1, ta[p - 1, :, x[0]], y[0]] = True
                        if p + 1 < Lm1:
                            nxt[p + 1, y[1], ta[p + 1, x[1], :]] = True
            else:
                for p in range(L - 2):
                    if ta[p, x[0], x[1]] and ta[p + 1, x[1], x[2]]:
                        nxt[p, y[0], y[1]] = True
                        nxt[p + 1, y[1], y[2]] = True
                        if p > 0:
                            nxt[p - 1, ta[p - 1, :, x[0]], y[0]] = True
                        if p + 2 < Lm1:
                            nxt[p + 2, y[2], ta[p + 2, x[2], :]] = True
        a2 = trellis_filter_adj2_bool(nxt)
        outs.append(a2.clone())
    return outs


def symbolic_sub3(rules: tuple[tuple[str, str], ...], w0: str, k_max: int = 3) -> list[torch.Tensor]:
    L = len(w0)
    Lm2 = L - 2
    a3 = alpha3_bool((w0,), L)
    outs = [a3.clone()]
    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    for _ in range(k_max):
        ta = trellis_filter_sub3_bool(a3)
        nxt = ta.clone()
        for lhs, rhs in ors:
            ll = len(lhs)
            x = [CHAR2ID[ch] for ch in lhs]
            y = [CHAR2ID[ch] for ch in rhs]
            if ll == 2:
                for p in range(L - 1):
                    has_m = bool(ta[p, x[0], x[1], :].any()) if p < Lm2 else bool(ta[Lm2 - 1, :, x[0], x[1]].any())
                    if not has_m:
                        continue
                    if p - 2 >= 0:
                        nxt[p - 2, :, :, y[0]] |= ta[p - 2, :, :, x[0]]
                    if p - 1 >= 0 and p < L - 1:
                        nxt[p - 1, ta[p - 1, :, x[0], x[1]], y[0], y[1]] = True
                    if p < Lm2:
                        nxt[p, y[0], y[1], ta[p, x[0], x[1], :]] = True
                    if p + 1 < Lm2:
                        c_from = ta[p, x[0], x[1], :]
                        nxt[p + 1, y[1], :, :] |= (ta[p + 1, x[1], :, :] & c_from[:, None])
            else:
                for p in range(Lm2):
                    if not ta[p, x[0], x[1], x[2]]:
                        continue
                    nxt[p, y[0], y[1], y[2]] = True
                    if p - 2 >= 0:
                        nxt[p - 2, :, :, y[0]] |= (ta[p - 2, :, :, x[0]] & ta[p - 1, :, x[0], x[1]][None, :])
                    if p - 1 >= 0:
                        nxt[p - 1, ta[p - 1, :, x[0], x[1]], y[0], y[1]] = True
                    if p + 1 < Lm2:
                        nxt[p + 1, y[1], y[2], ta[p + 1, x[1], x[2], :]] = True
                    if p + 2 < Lm2:
                        d_from = ta[p + 1, x[1], x[2], :]
                        nxt[p + 2, y[2], :, :] |= (ta[p + 2, x[2], :, :] & d_from[:, None])
        a3 = trellis_filter_sub3_bool(nxt)
        outs.append(a3.clone())
    return outs


def symbolic_prov4_adj2(rules: tuple[tuple[str, str], ...], w0: str,
                        k_max: int = 3, fp: int = FP_FIBERS) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
    """Spatial-congruence (f = p mod fp) provenance fibers x trellis-filtered Adj2."""
    L = len(w0)
    Lm1 = L - 1
    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    base_a2 = alpha2_bool((w0,), L)
    history = [(base_a2, [torch.zeros((Lm1, 6, 6), dtype=torch.bool) for _ in range(fp)])]
    if k_max == 0:
        return history

    fibers_init = [torch.zeros((Lm1, 6, 6), dtype=torch.bool) for _ in range(fp)]
    active = [False] * fp
    for lhs, rhs in ors:
        ll = len(lhs)
        for p in range(L - ll + 1):
            if w0[p:p + ll] == lhs:
                w1 = w0[:p] + rhs + w0[p + ll:]
                f = p % fp
                fibers_init[f] |= alpha2_bool((w1,), L)
                active[f] = True
    fibers = [trellis_filter_adj2_bool(fibers_init[f]) if active[f] else fibers_init[f] for f in range(fp)]
    history.append((base_a2, [x.clone() for x in fibers]))

    for step_idx in range(1, k_max):
        new_fibers = []
        for f in range(fp):
            if not active[f]:
                new_fibers.append(fibers[f])
                continue
            ta = trellis_filter_adj2_bool(fibers[f])
            nxt = ta.clone()
            for lhs, rhs in ors:
                ll = len(lhs)
                x = [CHAR2ID[ch] for ch in lhs]
                y = [CHAR2ID[ch] for ch in rhs]
                if ll == 2:
                    for p in range(Lm1):
                        if step_idx == 1 and (p % fp == f) and w0[p:p + 2] == rhs:
                            continue
                        if ta[p, x[0], x[1]]:
                            nxt[p, y[0], y[1]] = True
                            if p > 0:
                                nxt[p - 1, ta[p - 1, :, x[0]], y[0]] = True
                            if p + 1 < Lm1:
                                nxt[p + 1, y[1], ta[p + 1, x[1], :]] = True
                else:
                    for p in range(L - 2):
                        if step_idx == 1 and (p % fp == f) and w0[p:p + 3] == rhs:
                            continue
                        if ta[p, x[0], x[1]] and ta[p + 1, x[1], x[2]]:
                            nxt[p, y[0], y[1]] = True
                            nxt[p + 1, y[1], y[2]] = True
                            if p > 0:
                                nxt[p - 1, ta[p - 1, :, x[0]], y[0]] = True
                            if p + 2 < Lm1:
                                nxt[p + 2, y[2], ta[p + 2, x[2], :]] = True
            new_fibers.append(trellis_filter_adj2_bool(nxt))
        fibers = new_fibers
        history.append((base_a2, [x.clone() for x in fibers]))
    return history


# ============================================================================
# 2. Differentiable S6-Equivariant Relational-Provenance Fresco Architecture
# ============================================================================
@functools.lru_cache(maxsize=512)
def _cached_pos_rules(rules: tuple[tuple[str, str], ...]):
    return m16.EquivariantHorizonFrescoNet.encode_rule_tensors(rules)


@functools.lru_cache(maxsize=32768)
def _cached_pub_matches(rules: tuple[tuple[str, str], ...], word: str) -> tuple[int, int]:
    matches = 0
    checks = 0
    L = len(word)
    for a, b in rules:
        la = len(a)
        for p in range(L - la + 1):
            checks += 1
            sub = word[p:p + la]
            if sub == a or sub == b:
                matches += 1
    return matches, checks


def trellis_arc_filter_soft(e: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Zero-preserving S6-equivariant forward-backward trellis filter (exact support mask)."""
    with torch.no_grad():
        Lm1 = e.shape[-4]
        rho = e / (e + eps)
        fwd = [None] * Lm1
        cur_f = rho[..., 0, :, :, :].amax(dim=-2)
        for p in range(Lm1):
            fwd[p] = cur_f
            if p + 1 < Lm1:
                cur_f = (cur_f.unsqueeze(-2) * rho[..., p, :, :, :]).amax(dim=-3)
        bwd = [None] * Lm1
        cur_b = rho[..., Lm1 - 1, :, :, :].amax(dim=-3)
        for p in range(Lm1 - 1, -1, -1):
            bwd[p] = cur_b
            if p > 0:
                cur_b = (rho[..., p, :, :, :] * cur_b.unsqueeze(-3)).amax(dim=-2)
        mask = torch.stack(fwd, dim=-3).unsqueeze(-2) * torch.stack(bwd, dim=-3).unsqueeze(-3)
    return e * mask


class FastProvenanceBigramCell(nn.Module):
    """Exact S6-equivariant provenance-fibered (f = p mod 4) adjacent-2-gram soft-rewrite operator."""
    def __init__(self, cb: int, use_provenance: bool = True, learned: bool = True):
        super().__init__()
        self.cb = cb
        self.use_provenance = use_provenance
        self.learned = learned
        self.fp = FP_FIBERS if use_provenance else 1
        if learned:
            self.rule2_mlp = nn.Sequential(nn.Linear(cb, 2 * cb), nn.GELU(), nn.Linear(2 * cb, cb), nn.Softplus())
            self.rule3_mlp = nn.Sequential(nn.Linear(3 * cb, 2 * cb), nn.GELU(), nn.Linear(2 * cb, cb), nn.Softplus())

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def encode_bigram_rules(rules: tuple[tuple[str, str], ...]):
        r2_x, r2_y, r3_x, r3_y = [], [], [], []
        for a, b in rules:
            for src, dst in ((a, b), (b, a)):
                if src == dst:
                    continue
                xs = F.one_hot(torch.tensor([CHAR2ID[ch] for ch in src]), num_classes=6).float()
                ys = F.one_hot(torch.tensor([CHAR2ID[ch] for ch in dst]), num_classes=6).float()
                if len(src) == 2:
                    r2_x.append(xs); r2_y.append(ys)
                else:
                    r3_x.append(xs); r3_y.append(ys)
        out2 = None
        if r2_x:
            x2 = torch.stack(r2_x)
            y2 = torch.stack(r2_y)
            x2_bi = (x2[:, 0, :, None] * x2[:, 1, None, :]).reshape(-1, 36)
            y2_bi = (y2[:, 0, :, None] * y2[:, 1, None, :]).reshape(-1, 36)
            ml2 = (x2[:, 0, :, None] * y2[:, 0, None, :]).reshape(-1, 36)
            mr2 = (x2[:, 1, :, None] * y2[:, 1, None, :]).reshape(-1, 36)
            out2 = (x2, y2, x2_bi, y2_bi, ml2, mr2)
        out3 = None
        if r3_x:
            x3 = torch.stack(r3_x)
            y3 = torch.stack(r3_y)
            x3_b0 = (x3[:, 0, :, None] * x3[:, 1, None, :]).reshape(-1, 36)
            x3_b1 = (x3[:, 1, :, None] * x3[:, 2, None, :]).reshape(-1, 36)
            y3_b0 = (y3[:, 0, :, None] * y3[:, 1, None, :]).reshape(-1, 36)
            y3_b1 = (y3[:, 1, :, None] * y3[:, 2, None, :]).reshape(-1, 36)
            ml3 = (x3[:, 0, :, None] * y3[:, 0, None, :]).reshape(-1, 36)
            mr3 = (x3[:, 2, :, None] * y3[:, 2, None, :]).reshape(-1, 36)
            out3 = (x3, y3, x3_b0, x3_b1, y3_b0, y3_b1, ml3, mr3)
        return out2, out3

    def forward(self, oh_bi: torch.Tensor, pass_idx: int, gated_bi: torch.Tensor,
                prev_bank: torch.Tensor, r2_pack, r3_pack,
                scramble_pos: torch.Tensor | None = None):
        N, Lm1, _, _, _ = oh_bi.shape
        Fp, Cb = self.fp, self.cb

        if pass_idx == 0:
            q = oh_bi.unsqueeze(1).expand(N, 1, Lm1, 6, 6, Cb)
        else:
            q = torch.clamp(gated_bi, 0.0, 1.5)

        if scramble_pos is not None:
            inv = torch.argsort(scramble_pos)
            qw = q[:, :, scramble_pos, :, :, :].contiguous()
        else:
            qw = q.contiguous()

        QFp = qw.shape[1]
        qw_c36 = qw.reshape(N, QFp, Lm1, 36, Cb).movedim(-1, -2).contiguous()
        with torch.no_grad():
            q_ind = qw / (qw + 1e-4)
            q36_ind = q_ind.reshape(N, QFp, Lm1, 36, Cb)
            oh36 = oh_bi.reshape(N, Lm1, 36)
            p_idx = torch.arange(Lm1)
            f_idx = torch.arange(Fp)
            q_edge = torch.arange(Lm1)

        rew36 = torch.zeros((N, Fp, Lm1, 36, Cb), dtype=oh_bi.dtype)
        rew_bd = torch.zeros((N, Fp, Lm1, 6, 6, Cb), dtype=oh_bi.dtype)

        if r2_pack is not None and Lm1 >= 1:
            x2, y2, x2_bi, y2_bi, ml2, mr2 = r2_pack[:6] if len(r2_pack) == 6 else (
                *r2_pack,
                (r2_pack[0][:, 0, :, None] * r2_pack[1][:, 0, None, :]).reshape(-1, 36),
                (r2_pack[0][:, 1, :, None] * r2_pack[1][:, 1, None, :]).reshape(-1, 36),
            )
            pi2 = (qw_c36 @ x2_bi.t()).permute(0, 1, 4, 2, 3).contiguous()
            if pass_idx == 1 and self.use_provenance:
                with torch.no_grad():
                    rhs_is_root = (oh36 @ y2_bi.t()).transpose(1, 2) > 0.99
                    same_fib = (p_idx[None, :] % Fp == f_idx[:, None])
                    sup2 = (rhs_is_root[:, None, :, :] & same_fib[None, :, None, :]).unsqueeze(-1)
                pi2 = pi2.masked_fill(sup2, 0.0)
            act2 = (self.rule2_mlp(pi2) * pi2) if self.learned else pi2
            if pass_idx == 0 and self.use_provenance:
                route2 = (p_idx[None, :] % Fp == f_idx[:, None]).float()
                act2 = act2.expand(N, Fp, -1, Lm1, Cb) * route2[None, :, None, :, None]
                q_ind_ctx = q_ind.expand(N, Fp, Lm1, 6, 6, Cb)
                q36_ind_ctx = q36_ind.expand(N, Fp, Lm1, 36, Cb)
            else:
                q_ind_ctx = q_ind
                q36_ind_ctx = q36_ind

            act2_p = act2.permute(0, 1, 3, 4, 2).contiguous()
            rew36 = rew36 + (act2_p @ y2_bi).movedim(-2, -1)
            if Lm1 >= 2:
                tl2 = (act2_p[:, :, 1:] @ ml2).reshape(N, Fp, Lm1 - 1, Cb, 6, 6)
                qi_l = q_ind_ctx[:, :, :-1].movedim(-1, -3)
                rew_bd[:, :, :-1] = rew_bd[:, :, :-1] + (qi_l @ tl2).movedim(-3, -1)

                tr2 = (act2_p[:, :, :-1] @ mr2).reshape(N, Fp, Lm1 - 1, Cb, 6, 6)
                qi_r = q_ind_ctx[:, :, 1:].movedim(-1, -3)
                rew_bd[:, :, 1:] = rew_bd[:, :, 1:] + (tr2.transpose(-1, -2) @ qi_r).movedim(-3, -1)

            ctx_m2 = ((q_edge[None, :] < p_idx[:, None] - 1) | (q_edge[None, :] > p_idx[:, None] + 1)).float()
            ctx_w2 = (act2_p.sum(dim=-1).movedim(-1, -2) @ ctx_m2).movedim(-2, -1)
            rew36 = rew36 + ctx_w2.unsqueeze(-2) * q36_ind_ctx

        if r3_pack is not None and Lm1 >= 2:
            x3, y3, x3_b0, x3_b1, y3_b0, y3_b1, ml3, mr3 = r3_pack[:8] if len(r3_pack) == 8 else (
                *r3_pack,
                (r3_pack[0][:, 0, :, None] * r3_pack[1][:, 0, None, :]).reshape(-1, 36),
                (r3_pack[0][:, 2, :, None] * r3_pack[1][:, 2, None, :]).reshape(-1, 36),
            )
            Lm2 = Lm1 - 1
            p3_idx = torch.arange(Lm2)
            m3_0 = (qw_c36[:, :, :-1] @ x3_b0.t()).permute(0, 1, 4, 2, 3).contiguous()
            m3_1 = (qw_c36[:, :, 1:] @ x3_b1.t()).permute(0, 1, 4, 2, 3).contiguous()
            pi3 = m3_0 * m3_1
            if pass_idx == 1 and self.use_provenance:
                with torch.no_grad():
                    r0 = (oh36[:, :-1] @ y3_b0.t()).transpose(1, 2) > 0.99
                    r1 = (oh36[:, 1:] @ y3_b1.t()).transpose(1, 2) > 0.99
                    same_fib3 = (p3_idx[None, :] % Fp == f_idx[:, None])
                    sup3 = ((r0 & r1)[:, None, :, :] & same_fib3[None, :, None, :]).unsqueeze(-1)
                pi3 = pi3.masked_fill(sup3, 0.0)
            act3 = (self.rule3_mlp(torch.cat([pi3, m3_0, m3_1], dim=-1)) * pi3) if self.learned else pi3
            if pass_idx == 0 and self.use_provenance:
                route3 = (p3_idx[None, :] % Fp == f_idx[:, None]).float()
                act3 = act3.expand(N, Fp, -1, Lm2, Cb) * route3[None, :, None, :, None]
                q_ind_ctx = q_ind.expand(N, Fp, Lm1, 6, 6, Cb)
                q36_ind_ctx = q36_ind.expand(N, Fp, Lm1, 36, Cb)
            else:
                q_ind_ctx = q_ind
                q36_ind_ctx = q36_ind

            act3_p = act3.permute(0, 1, 3, 4, 2).contiguous()
            rew36[:, :, :-1] = rew36[:, :, :-1] + (act3_p @ y3_b0).movedim(-2, -1)
            rew36[:, :, 1:] = rew36[:, :, 1:] + (act3_p @ y3_b1).movedim(-2, -1)
            if Lm2 >= 2:
                tl3 = (act3_p[:, :, 1:] @ ml3).reshape(N, Fp, Lm2 - 1, Cb, 6, 6)
                qi_l3 = q_ind_ctx[:, :, :-2].movedim(-1, -3)
                rew_bd[:, :, :-2] = rew_bd[:, :, :-2] + (qi_l3 @ tl3).movedim(-3, -1)

                tr3 = (act3_p[:, :, :-1] @ mr3).reshape(N, Fp, Lm2 - 1, Cb, 6, 6)
                qi_r3 = q_ind_ctx[:, :, 2:].movedim(-1, -3)
                rew_bd[:, :, 2:] = rew_bd[:, :, 2:] + (tr3.transpose(-1, -2) @ qi_r3).movedim(-3, -1)

            ctx_m3 = ((q_edge[None, :] < p3_idx[:, None] - 1) | (q_edge[None, :] > p3_idx[:, None] + 2)).float()
            ctx_w3 = (act3_p.sum(dim=-1).movedim(-1, -2) @ ctx_m3).movedim(-2, -1)
            rew36 = rew36 + ctx_w3.unsqueeze(-2) * q36_ind_ctx

        rew = rew36.reshape(N, Fp, Lm1, 6, 6, Cb) + rew_bd
        if scramble_pos is not None:
            rew = rew[:, :, inv, :, :, :]

        bank_raw = torch.tanh(rew) if pass_idx == 0 else torch.tanh(rew + 0.65 * F.relu(prev_bank))
        bank_next = trellis_arc_filter_soft(bank_raw)
        return bank_next, q


class LowRankJointRewriteCell(nn.Module):
    """Learned S6-equivariant low-rank bilinear representation of adjacent-symbol reachability."""
    def __init__(self, ce: int):
        super().__init__()
        self.ce = ce
        self.left_proj = nn.Linear(ce, ce, bias=False)
        self.right_proj = nn.Linear(ce, ce, bias=False)
        self.rule2_mlp = nn.Sequential(nn.Linear(3 * ce, 2 * ce), nn.GELU(), nn.Linear(2 * ce, ce), nn.Softplus())
        self.rule3_mlp = nn.Sequential(nn.Linear(4 * ce, 2 * ce), nn.GELU(), nn.Linear(2 * ce, ce), nn.Softplus())
        self.res_ff = nn.Sequential(nn.Linear(ce, 2 * ce), nn.GELU(), nn.Linear(2 * ce, ce))

    def forward(self, oh_base: torch.Tensor, h_res: torch.Tensor, gated_env: torch.Tensor,
                prev_bank: torch.Tensor, r2, r3, scramble_pos: torch.Tensor | None = None):
        q_env = torch.clamp(oh_base.expand_as(h_res) + gated_env, 0.0, 1.5)
        if scramble_pos is not None:
            inv = torch.argsort(scramble_pos)
            qw = q_env[:, scramble_pos, :, :]
        else:
            qw = q_env
        N, L, _, Ce = qw.shape
        lp = F.relu(self.left_proj(qw[:, :-1]))
        rp = F.relu(self.right_proj(qw[:, 1:]))
        bi_lr = torch.tanh(lp.amax(dim=2, keepdim=True) * rp.amax(dim=2, keepdim=True))
        m_left = F.pad(0.5 + 0.5 * bi_lr, (0, 0, 0, 0, 0, 1), value=1.0)
        m_right = F.pad(0.5 + 0.5 * bi_lr, (0, 0, 0, 0, 1, 0), value=1.0)
        qw_mod = qw * m_left * m_right

        rew = torch.zeros_like(qw)
        if r2 is not None and L >= 2:
            x2, y2 = r2
            w2 = qw_mod.unfold(1, 2, 1)
            m2 = torch.einsum("nlscq,eqs->nelqc", w2, x2)
            prod2 = m2[:, :, :, 0, :] * m2[:, :, :, 1, :]
            feat2 = torch.cat([prod2, m2.reshape(N, x2.shape[0], L - 1, 2 * Ce)], dim=-1)
            act2 = self.rule2_mlp(feat2) * prod2
            out2 = torch.einsum("nelc,eqs->nlqsc", act2, y2)
            rew[:, :-1] = rew[:, :-1] + out2[:, :, 0]
            rew[:, 1:] = rew[:, 1:] + out2[:, :, 1]
        if r3 is not None and L >= 3:
            x3, y3 = r3
            w3 = qw_mod.unfold(1, 3, 1)
            m3 = torch.einsum("nlscq,eqs->nelqc", w3, x3)
            prod3 = m3[:, :, :, 0, :] * m3[:, :, :, 1, :] * m3[:, :, :, 2, :]
            feat3 = torch.cat([prod3, m3.reshape(N, x3.shape[0], L - 2, 3 * Ce)], dim=-1)
            act3 = self.rule3_mlp(feat3) * prod3
            out3 = torch.einsum("nelc,eqs->nlqsc", act3, y3)
            for q in range(3):
                rew[:, q:L - 2 + q] = rew[:, q:L - 2 + q] + out3[:, :, q]
        if scramble_pos is not None:
            rew = rew[:, inv, :, :]
        bank_next = torch.tanh(rew + 0.65 * F.relu(prev_bank))
        h_next = torch.tanh(h_res + 0.25 * self.res_ff(h_res))
        return h_next, bank_next, q_env


class FactoredAxialFrescoBlock(nn.Module):
    """Feasibility-masked S_{m0}xS_{m1} set-equivariant cross-frontier axial block with ablate_axial mode."""
    def __init__(self, dm: int):
        super().__init__()
        self.row_q = nn.Linear(dm, dm, bias=False)
        self.row_kv = nn.Linear(dm, 2 * dm, bias=False)
        self.col_q = nn.Linear(dm, dm, bias=False)
        self.col_kv = nn.Linear(dm, 2 * dm, bias=False)
        self.comp_proj = nn.Linear(2 * dm, dm)
        self.mix = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        self.msg_norm = nn.LayerNorm(dm)

    def forward(self, h: torch.Tensor, prev_bank: torch.Tensor, infeas: torch.Tensor,
                ablate_axial: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dm = h.shape[-1]
        rq = self.row_q(h)
        rk, rv = self.row_kv(h).chunk(2, dim=-1)
        ht = h.transpose(0, 1)
        cq = self.col_q(ht)
        ck, cv = self.col_kv(ht).chunk(2, dim=-1)

        if ablate_axial:
            r_ctx = rv
            c_ctx = cv.transpose(0, 1)
            comp = torch.tanh(self.comp_proj(torch.zeros((h.shape[0], h.shape[1], 2 * dm), dtype=h.dtype)))
        else:
            valid = (~infeas).float().unsqueeze(-1)
            r_logits = (rq @ rk.transpose(-1, -2)) / math.sqrt(dm)
            row_mask = infeas.unsqueeze(1) & (~infeas.all(dim=1, keepdim=True).unsqueeze(2))
            r_ctx = torch.softmax(r_logits.masked_fill(row_mask, -1e9), dim=-1) @ rv

            infeas_t = infeas.transpose(0, 1)
            c_logits = (cq @ ck.transpose(-1, -2)) / math.sqrt(dm)
            col_mask = infeas_t.unsqueeze(1) & (~infeas_t.all(dim=1, keepdim=True).unsqueeze(2))
            c_ctx = (torch.softmax(c_logits.masked_fill(col_mask, -1e9), dim=-1) @ cv).transpose(0, 1)

            col_mean = (h * valid).sum(dim=0, keepdim=True) / valid.sum(dim=0, keepdim=True).clamp_min(1.0)
            row_mean = (h * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            comp = torch.tanh(self.comp_proj(torch.cat([h - col_mean, h - row_mean], dim=-1)))

        rel_msg = 0.5 * (r_ctx + c_ctx) + comp
        h_next = h + self.mix(h + rel_msg)
        return h_next, self.msg_norm(rel_msg + 0.5 * prev_bank)


class RelationalProvenanceFrescoNet(nn.Module):
    """Parameter-matched (~53,047) LBF-17 Relational & Provenance-Preserving Fresco Family."""
    def __init__(self, arch: str = "rel_prov_opened", ce: int = 20, cb: int = 4, dm: int = 38, passes: int = PASSES):
        super().__init__()
        if arch not in LBF17_ARCHS:
            raise ValueError(f"unknown LBF-17 architecture {arch}")
        self.arch = arch
        self.ce = ce
        self.cb = cb
        self.dm = dm
        self.is_lowrank = (arch == "lowrank_opened")
        self.is_sym_readout = (arch == "sym_prov_readout")
        self.nonrecurrent = arch in ("param_nonrecurrent", "compute_nonrecurrent", "sym_prov_readout")
        self.passes = 1 if self.nonrecurrent else passes
        self.use_prov = (arch != "adj2_noprov_opened")

        self.env_init = nn.Linear(3, ce)
        if self.is_lowrank:
            self.lowrank_cell = LowRankJointRewriteCell(ce)
            rib_in_dim = 5 * ce + 1
            self.cal_w = nn.Parameter(torch.zeros(ce))
        elif self.is_sym_readout:
            self.sym_cell = FastProvenanceBigramCell(cb=1, use_provenance=True, learned=False)
            rib_in_dim = 10
        else:
            self.pos_cell = m16.SparseMultiplicativeRewriteCell(ce)
            self.bi_cell = FastProvenanceBigramCell(cb=cb, use_provenance=self.use_prov, learned=True)
            rib_in_dim = 5 * ce + 5 * cb + 1
            self.cal_w = nn.Parameter(torch.zeros(cb))

        self.ribbon_in = nn.Linear(rib_in_dim, dm)
        self.ribbon_conv = nn.Conv1d(dm, dm, kernel_size=3, padding=1)

        if arch == "compute_nonrecurrent":
            # Evaluates all 16 cross-fiber + 8 self-fiber 1-hop views along L in a single K=1 pass:
            # 53,070 parameters AND >= 1.04x the 3-pass FLOPs of rel_prov_opened!
            self.fib_rib_in = nn.Linear(2 * cb + 1, dm)
            self.fib_rib_conv = nn.Conv1d(dm, dm, kernel_size=3, padding=1)
            self.nr_rib_ff = nn.Sequential(nn.Linear(dm, 11), nn.GELU(), nn.Linear(11, dm))
        elif arch == "param_nonrecurrent":
            self.nr_pair_ff = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 72), nn.GELU(), nn.Linear(72, dm))
        elif arch == "sym_prov_readout":
            self.sym_rib_conv2 = nn.Conv1d(dm, dm, kernel_size=3, padding=1)
            self.sym_pair_ff = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 213), nn.GELU(), nn.Linear(213, dm))
        elif self.is_lowrank:
            self.lr_pair_ff = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 5), nn.GELU(), nn.Linear(5, dm))

        self.pair_in = nn.Linear(2 * dm + 12, dm)
        self.pair_prelude = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        self.fresco_block = FactoredAxialFrescoBlock(dm)

        if not self.nonrecurrent:
            self.env_gate = m16.SparseEnvGate(ce, init_bias=-6.0)
            if not self.is_lowrank:
                self.bi_gate = m16.SparseEnvGate(cb, init_bias=-6.0)
            self.fresco_gate = m16.MessageGate(dm, init_bias=-6.0)
            self.delta_scale = nn.Parameter(torch.tensor(0.35))

        node_in_dim = (6 if self.is_sym_readout else (2 * ce + 6 if self.is_lowrank else 2 * ce + 2 * cb + 6))
        self.pot_head = nn.Linear(dm, 1)
        self.node_head = nn.Sequential(nn.Linear(node_in_dim, dm), nn.GELU(), nn.Linear(dm, 1))
        self.log_tau = nn.Parameter(torch.tensor(-0.7))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_words(self, words: list[str]):
        ids = torch.tensor([[CHAR2ID[ch] for ch in w] for w in words], dtype=torch.long)
        oh = F.one_hot(ids, num_classes=6).float()
        bl = torch.zeros_like(oh); bl[:, 0, :] = oh[:, 0, :]
        br = torch.zeros_like(oh); br[:, -1, :] = oh[:, -1, :]
        h0 = torch.tanh(self.env_init(torch.stack([oh, bl, br], dim=-1)))
        oh_bi = (oh[:, :-1, :, None] * oh[:, 1:, None, :]).unsqueeze(-1)
        return oh.unsqueeze(-1), oh_bi, h0, ids

    @staticmethod
    def pad_edges_to_nodes(edge_feat: torch.Tensor) -> torch.Tensor:
        left_pad = F.pad(edge_feat, (0, 0, 0, 1))
        right_pad = F.pad(edge_feat, (0, 0, 1, 0))
        deg = torch.ones_like(left_pad[..., :1])
        deg[..., 1:-1, :] = 2.0
        return (left_pad + right_pad) / deg

    @staticmethod
    def bigram_to_pos1_channels(bi: torch.Tensor) -> torch.Tensor:
        """Map (N, Fp, L-1, 6, 6, Cb) -> zero-preserving node symbol channels (N, L, 6, Cb)."""
        b_max_f = bi.amax(dim=1)  # (N, L-1, 6, 6, Cb)
        left_s = F.pad(b_max_f.amax(dim=3), (0, 0, 0, 0, 0, 1))
        right_s = F.pad(b_max_f.amax(dim=2), (0, 0, 0, 0, 1, 0))
        return torch.maximum(left_s, right_s)

    def subterm_brightness(self, oh_u: torch.Tensor, bu: torch.Tensor,
                           bi_u: torch.Tensor | None) -> torch.Tensor:
        """Zero-preserving S6-equivariant subterm brightness in [0, 1) on (N, L, 6)."""
        base_pos = oh_u.squeeze(-1)
        if self.is_sym_readout:
            return base_pos
        w_pos = F.softplus(self.cal_w)
        if self.is_lowrank or bi_u is None:
            rate = F.linear(F.relu(bu), w_pos)
        else:
            pos_from_bi = self.bigram_to_pos1_channels(F.relu(bi_u))
            rate = F.linear(pos_from_bi, w_pos)
        learned_prob = 1.0 - torch.exp(-rate)
        return torch.maximum(base_pos, learned_prob.clamp(0.0, 0.9999))

    def _symbolic_ribbon_features(self, case, oh_bi_u, oh_bi_v, mismatch, win_perm_edge):
        m0, Lm1 = oh_bi_u.shape[0], oh_bi_u.shape[1]
        m1 = oh_bi_v.shape[0]
        r2_b, r3_b = FastProvenanceBigramCell.encode_bigram_rules(case.rules)
        bu_b = torch.zeros((m0, FP_FIBERS, Lm1, 6, 6, 1))
        bv_b = torch.zeros((m1, FP_FIBERS, Lm1, 6, 6, 1))
        bu_h, bv_h = [], []
        for kk in range(3):
            bu_b, _ = self.sym_cell(oh_bi_u, kk, bu_b, bu_b, r2_b, r3_b, scramble_pos=win_perm_edge)
            bv_b, _ = self.sym_cell(oh_bi_v, kk, bv_b, bv_b, r2_b, r3_b, scramble_pos=win_perm_edge)
            bu_h.append((bu_b > 1e-5).float())
            bv_h.append((bv_b > 1e-5).float())
        ub0 = oh_bi_u.reshape(m0, Lm1, 36)
        vb0 = oh_bi_v.reshape(m1, Lm1, 36)
        up1 = bu_h[0].amax(dim=1).reshape(m0, Lm1, 36)
        vp1 = bv_h[0].amax(dim=1).reshape(m1, Lm1, 36)
        uf3 = bu_h[2].reshape(m0, FP_FIBERS, Lm1, 36)
        vf3 = bv_h[2].reshape(m1, FP_FIBERS, Lm1, 36)
        up3 = uf3.amax(dim=1)
        vp3 = vf3.amax(dim=1)
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
        fib_pair = torch.einsum("ifpk,jgpk->ijfgp", uf3, vf3).amax(dim=(2, 3)).unsqueeze(-1)
        usupp = up3.mean(dim=-1)[:, None, :, None].expand(m0, m1, Lm1, 1)
        vsupp = vp3.mean(dim=-1)[None, :, :, None].expand(m0, m1, Lm1, 1)
        edge9 = torch.cat([ov0, ov1, ovk, u2v1, v2u1, u2v3, v2u3, fib_pair, 0.5 * (usupp + vsupp)], dim=-1)
        node9 = self.pad_edges_to_nodes(edge9)
        n_r2 = 0 if r2_b is None else r2_b[0].shape[0]
        n_r3 = 0 if r3_b is None else r3_b[0].shape[0]
        sym_win = (m0 + m1) * 3 * (n_r2 * Lm1 + n_r3 * max(0, Lm1 - 1))
        return torch.cat([node9, mismatch.float().unsqueeze(-1)], dim=-1), bu_h[2], sym_win

    def forward(self, case, s: m15.FringeState, rem_budget: int, *, mode: str = "normal",
                depth_cap: int = DEPTH, return_tensors: bool = False) -> tuple[list[dict], int]:
        if mode not in FACTORED_MODES:
            raise ValueError(f"unsupported LBF-17 intervention mode {mode}")
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0, m1 = len(u_words), len(v_words)
        L = len(case.source)
        Lm1 = L - 1

        oh_all, oh_bi_all, h_all, all_ids = self.init_words(u_words + v_words)
        oh_u, oh_v = oh_all[:m0], oh_all[m0:]
        oh_bi_u, oh_bi_v = oh_bi_all[:m0], oh_bi_all[m0:]
        u_ids, v_ids = all_ids[:m0], all_ids[m0:]

        g0 = torch.tensor([s.dist[0][w] for w in u_words], dtype=torch.float32)[:, None]
        g1 = torch.tensor([s.dist[1][w] for w in v_words], dtype=torch.float32)[None, :]
        mismatch = (u_ids[:, None, :] != v_ids[None, :, :])
        dh_mat = mismatch.sum(dim=-1).float()
        lb = torch.div(dh_mat.long() + 2, 3, rounding_mode="floor").float()
        tot_lb = g0 + g1 + torch.clamp(lb, min=1.0)
        infeas = tot_lb > depth_cap

        pos_idx = torch.arange(L, dtype=torch.float32).view(1, 1, L)
        first_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, float(L))).min(dim=-1).values
        last_mm = torch.where(mismatch, pos_idx, torch.full_like(pos_idx, -1.0)).max(dim=-1).values
        span = torch.clamp(last_mm - first_mm + 1.0, min=0.0)

        pers = torch.zeros((m0, m1), dtype=torch.float32)
        prev_v = torch.zeros((m0, m1), dtype=torch.float32)
        if s.prev_pot:
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
            dh_mat / L, lb / depth_cap,
            torch.clamp(depth_cap - tot_lb, min=0.0) / depth_cap,
            1.0 - infeas.float(), span / L,
            torch.full((m0, m1), b_frac),
            torch.full((m0, m1), lm_diff),
            pers, prev_v,
        ], dim=-1)

        rule_win_checks = 0
        def node_pub(words, w_ids, side):
            nonlocal rule_win_checks
            ms, mo = len(words), len(s.fronts[1 - side])
            goal = case.target if side == 0 else case.source
            g_ids = torch.tensor([CHAR2ID[ch] for ch in goal], dtype=torch.long)
            dh_goal = (w_ids != g_ids[None, :]).sum(dim=-1).float() / L
            degs = []
            for w in words:
                dg, ch = _cached_pub_matches(case.rules, w)
                rule_win_checks += ch
                degs.append(dg / 24.0)
            gs = torch.tensor([s.dist[side][w] / depth_cap for w in words], dtype=torch.float32)
            return torch.stack([
                gs, torch.tensor(degs, dtype=torch.float32),
                torch.full((ms,), math.log1p(ms) / 5.0),
                torch.full((ms,), math.log1p(mo) / 5.0),
                torch.full((ms,), b_frac), dh_goal,
            ], dim=-1)

        np0, np1 = node_pub(u_words, u_ids, 0), node_pub(v_words, v_ids, 1)

        win_perm_node = win_perm_edge = rib_perm = None
        if mode == "scramble_window":
            gen = torch.Generator().manual_seed(1337 + m0 * 31 + m1)
            win_perm_node = torch.randperm(L, generator=gen)
            win_perm_edge = torch.randperm(Lm1, generator=gen)
        elif mode == "scramble_ribbon":
            gen = torch.Generator().manual_seed(2468 + m0 * 31 + m1)
            rib_perm = torch.randperm(L, generator=gen)
        ablate_ax = (mode == "ablate_axial")

        if self.is_sym_readout:
            sym_feat, sym_bu3, sym_w = self._symbolic_ribbon_features(case, oh_bi_u, oh_bi_v, mismatch, win_perm_edge)
            rule_win_checks += sym_w
            rib = self.ribbon_in(sym_feat)
            if rib_perm is not None:
                inv_r = torch.argsort(rib_perm)
                rc = F.gelu(self.ribbon_conv(rib[:, :, rib_perm, :].reshape(m0 * m1, L, self.dm).transpose(1, 2)))
                rc = F.gelu(self.sym_rib_conv2(rc)).transpose(1, 2).reshape(m0, m1, L, self.dm)[:, :, inv_r, :]
            else:
                rc = F.gelu(self.ribbon_conv(rib.reshape(m0 * m1, L, self.dm).transpose(1, 2)))
                rc = F.gelu(self.sym_rib_conv2(rc)).transpose(1, 2).reshape(m0, m1, L, self.dm)
            mm_mask = mismatch.float().unsqueeze(-1)
            rib_pool = torch.cat([(rc * mm_mask).sum(dim=2) / mm_mask.sum(dim=2).clamp_min(1.0), rc.amax(dim=2)], dim=-1)
            pbase = self.pair_in(torch.cat([rib_pool, pub], dim=-1))
            pbase = pbase + self.pair_prelude(pbase)
            bm0 = torch.zeros((m0, m1, self.dm), dtype=torch.float32)
            h_mat, bm = self.fresco_block(pbase, bm0, infeas, ablate_axial=ablate_ax)
            h_mat = h_mat + self.sym_pair_ff(h_mat)
            pot = self.pot_head(h_mat).squeeze(-1)
            pot_masked = pot.masked_fill(infeas, -12.0)
            tau = F.softplus(self.log_tau) + 0.1
            s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
                  + self.node_head(np0).squeeze(-1)).masked_fill(infeas.all(dim=1), -12.0)
            s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
                  + self.node_head(np1).squeeze(-1)).masked_fill(infeas.all(dim=0), -12.0)
            row = {"pot": pot, "pot_masked": pot_masked, "s0": s0, "s1": s1, "infeas": infeas}
            if return_tensors:
                row["h_mat"] = h_mat.detach().cpu().clone()
                row["bm_out"] = bm.detach().cpu().clone()
                row["bw_u_out"] = self.bigram_to_pos1_channels(sym_bu3).detach().cpu().clone()
                row["bi_u_out"] = sym_bu3.detach().cpu().clone()
            return [row], rule_win_checks

        r2_pos, r3_pos = _cached_pos_rules(case.rules)
        r2_bi, r3_bi = (None, None) if self.is_lowrank else FastProvenanceBigramCell.encode_bigram_rules(case.rules)

        b_all = torch.zeros_like(h_all)
        Fp = 1 if self.is_lowrank else (FP_FIBERS if self.use_prov else 1)
        bi_all = None if self.is_lowrank else torch.zeros((m0 + m1, Fp, Lm1, 6, 6, self.cb))

        bu_anchor = bv_anchor = bi_u_anchor = bi_v_anchor = None
        bm = torch.zeros((m0, m1, self.dm), dtype=torch.float32)
        h_mat = pot_accum = None
        outs = []

        n_r2 = 0 if r2_pos is None else r2_pos[0].shape[0]
        n_r3 = 0 if r3_pos is None else r3_pos[0].shape[0]
        win_per_word = n_r2 * max(0, L - 1) + n_r3 * max(0, L - 2)
        bi_mult = 0 if self.is_lowrank else Fp

        for k in range(self.passes):
            rule_win_checks += (m0 + m1) * win_per_word * (1 + (1 if k == 0 and bi_mult else bi_mult))
            if self.nonrecurrent:
                g_all = torch.zeros_like(h_all)
                h_all, b_all, _, q_all = self.pos_cell(oh_all, h_all, g_all, b_all, r2_pos, r3_pos, scramble_pos=win_perm_node)
                gbi_all = torch.zeros_like(bi_all)
                bi_all, _ = self.bi_cell(oh_bi_all, 0, gbi_all, bi_all, r2_bi, r3_bi, scramble_pos=win_perm_edge)
                bu, bv = b_all[:m0], b_all[m0:]
                qu, qv = q_all[:m0], q_all[m0:]
                bi_u, bi_v = bi_all[:m0], bi_all[m0:]
                bu_anchor, bv_anchor = bu, bv
                bi_u_anchor, bi_v_anchor = bi_u, bi_v
                mu_in = mbi_u = mm_in = None
                s_pos_gate = sm_gate = sbi_u = gm = None
            else:
                m_in = b_all
                mbi_in = bi_all
                mm_in = bm
                if k > 0:
                    if mode == "env_zero":
                        m_in = torch.zeros_like(m_in)
                        if mbi_in is not None:
                            mbi_in = torch.zeros_like(mbi_in)
                    elif mode == "env_roll":
                        m_in = m_in.roll(1, dims=1)
                        if mbi_in is not None:
                            mbi_in = mbi_in.roll(1, dims=2)
                    elif mode == "mat_zero":
                        mm_in = torch.zeros_like(mm_in)
                    elif mode == "mat_roll":
                        mm_in = mm_in.roll(shifts=(1, 1), dims=(0, 1))

                g_all, s_pos_gate = self.env_gate(h_all, m_in)
                if self.is_lowrank:
                    h_all, b_all, q_all = self.lowrank_cell(oh_all, h_all, g_all, m_in, r2_pos, r3_pos, scramble_pos=win_perm_node)
                    sbi_u = None
                else:
                    h_all, b_all, _, q_all = self.pos_cell(oh_all, h_all, g_all, m_in, r2_pos, r3_pos, scramble_pos=win_perm_node)
                    if k == 0:
                        gbi_all = torch.zeros_like(mbi_in)
                        sbi_u = torch.zeros((1,))
                    else:
                        gbi_all, sbi_all = self.bi_gate(oh_bi_all.unsqueeze(1).expand_as(mbi_in), mbi_in)
                        sbi_u = sbi_all[:m0]
                    bi_all, _ = self.bi_cell(oh_bi_all, k, gbi_all, mbi_in, r2_bi, r3_bi, scramble_pos=win_perm_edge)
                bu, bv = b_all[:m0], b_all[m0:]
                qu, qv = q_all[:m0], q_all[m0:]
                bi_u = None if bi_all is None else bi_all[:m0]
                bi_v = None if bi_all is None else bi_all[m0:]
                mu_in = m_in[:m0]
                mbi_u = None if mbi_in is None else mbi_in[:m0]
                if k == 0:
                    bu_anchor, bv_anchor = bu, bv
                    bi_u_anchor, bi_v_anchor = bi_u, bi_v

            bu_nov = F.relu(bu - bu_anchor)
            bv_nov = F.relu(bv - bv_anchor)
            # Single batched matmul across all 6 positionwise overlap terms (L, 6*ce, m0, 6) @ (L, 6*ce, 6, m1)
            u_pos_stack = torch.cat([qu, bu_anchor, bu_nov, 0.5 * bu_anchor, bu, oh_u.expand_as(bu)], dim=-1)
            v_pos_stack = torch.cat([qv, bv_anchor, bv_nov + 0.5 * bv_anchor, bv_nov, oh_v.expand_as(bv), bv], dim=-1)
            # u_pos_stack: (m0, L, 6, 6*ce) -> (L, 6*ce, m0, 6)
            up_m = u_pos_stack.permute(1, 3, 0, 2).contiguous()
            vp_m = v_pos_stack.permute(1, 3, 2, 0).contiguous()
            ov_all6 = (up_m @ vp_m).permute(2, 3, 0, 1)  # (m0, m1, L, 6*ce)
            ov_q, ov_1hop, ov_k1, ov_k2, ov_u2v, ov_v2u = ov_all6.split(self.ce, dim=-1)
            ov_khop = ov_k1 + ov_k2

            if self.is_lowrank:
                rib_raw = torch.cat([ov_q, ov_1hop, ov_khop, ov_u2v, ov_v2u, mismatch.float().unsqueeze(-1)], dim=-1)
            else:
                au_bi = bi_u_anchor.amax(dim=1).reshape(m0, Lm1, 36, self.cb)
                av_bi = bi_v_anchor.amax(dim=1).reshape(m1, Lm1, 36, self.cb)
                cu_bi = bi_u.amax(dim=1).reshape(m0, Lm1, 36, self.cb)
                cv_bi = bi_v.amax(dim=1).reshape(m1, Lm1, 36, self.cb)
                nu_bi = F.relu(cu_bi - au_bi)
                nv_bi = F.relu(cv_bi - av_bi)
                obu0 = oh_bi_u.reshape(m0, Lm1, 36, 1).expand(m0, Lm1, 36, self.cb)
                obv0 = oh_bi_v.reshape(m1, Lm1, 36, 1).expand(m1, Lm1, 36, self.cb)

                u_bi_stack = torch.cat([obu0, au_bi, nu_bi, 0.5 * au_bi, cu_bi, obu0], dim=-1)
                v_bi_stack = torch.cat([obv0, av_bi, nv_bi + 0.5 * av_bi, nv_bi, obv0, cv_bi], dim=-1)
                ub_m = u_bi_stack.permute(1, 3, 0, 2).contiguous()
                vb_m = v_bi_stack.permute(1, 3, 2, 0).contiguous()
                bi_all6 = (ub_m @ vb_m).permute(2, 3, 0, 1)  # (m0, m1, Lm1, 6*cb)
                b_0, b_1hop, b_k1, b_k2, b_u2v, b_v2u = bi_all6.split(self.cb, dim=-1)
                b_khop = b_k1 + b_k2
                bi_node = self.pad_edges_to_nodes(torch.cat([b_0, b_1hop, b_khop, b_u2v, b_v2u], dim=-1))
                rib_raw = torch.cat([ov_q, ov_1hop, ov_khop, ov_u2v, ov_v2u, bi_node, mismatch.float().unsqueeze(-1)], dim=-1)

            rib = self.ribbon_in(rib_raw)
            if rib_perm is not None:
                inv_r = torch.argsort(rib_perm)
                rib_sp = rib[:, :, rib_perm, :].reshape(m0 * m1, L, self.dm).transpose(1, 2)
                rib_c = F.gelu(self.ribbon_conv(rib_sp)).transpose(1, 2).reshape(m0, m1, L, self.dm)[:, :, inv_r, :]
            else:
                rib_c = F.gelu(self.ribbon_conv(rib.reshape(m0 * m1, L, self.dm).transpose(1, 2))).transpose(1, 2).reshape(m0, m1, L, self.dm)

            if self.arch == "compute_nonrecurrent":
                # Match BOTH O(m0 + m1) word-level and O(m0 * m1) pair-level 3-pass FLOPs in K=1 pass
                # using zero extra parameters (53,070 params total):
                fu_f = bi_u.reshape(m0, FP_FIBERS, Lm1, 36, self.cb)
                fv_f = bi_v.reshape(m1, FP_FIBERS, Lm1, 36, self.cb)
                # 1. Word-level 1-hop inter-fiber consistency views: (m0+m1) x 32 views along L
                all_f = torch.cat([fu_f, fv_f], dim=0)  # (m0+m1, 4, Lm1, 36, cb)
                all_ob = torch.cat([obu0, obv0], dim=0)
                wf_pair = torch.einsum("nfpkc,ngpkc->nfgpc", all_f, all_f).reshape(m0 + m1, 16, Lm1, self.cb)
                wf_root = torch.einsum("nfpkc,npkc->nfpc", all_f, all_ob).repeat(1, 4, 1, 1)
                wf_a = torch.cat([wf_pair, wf_root], dim=1)                                   # (m0+m1, 32, Lm1, cb)
                wf_b = torch.cat([wf_root, wf_pair], dim=1)                                   # (m0+m1, 32, Lm1, cb)
                wf_node = self.pad_edges_to_nodes(torch.cat([wf_a, wf_b], dim=-1))            # (m0+m1, 32, L, 2*cb)
                wf_in = torch.cat([wf_node, wf_node.new_ones((m0 + m1, 32, L, 1))], dim=-1)
                wh = self.fib_rib_in(wf_in).reshape((m0 + m1) * 32, L, self.dm)
                if rib_perm is not None:
                    inv_r = torch.argsort(rib_perm)
                    wc = F.gelu(self.fib_rib_conv(wh[:, rib_perm, :].transpose(1, 2))).transpose(1, 2)[:, inv_r, :]
                else:
                    wc = F.gelu(self.fib_rib_conv(wh.transpose(1, 2))).transpose(1, 2)
                wc = (wc + self.nr_rib_ff(wc)).reshape(m0 + m1, 32, L, self.dm).mean(dim=1)
                wu_ctx, wv_ctx = wc[:m0], wc[m0:]

                # 2. Pair-level 1-hop cross-fiber views: m0 x m1 x 24 views along L
                cross_ov = torch.einsum("ifpkc,jgpkc->ijfgpc", fu_f, fv_f).reshape(m0, m1, 16, Lm1, self.cb)
                cross_dir = (torch.einsum("ifpkc,jpkc->ijfpc", fu_f, obv0)[:, :, :, None, :, :]
                             + torch.einsum("ipkc,jgpkc->ijgpc", obu0, fv_f)[:, :, None, :, :, :]).reshape(m0, m1, 16, Lm1, self.cb)
                self_u = torch.einsum("ifpkc,jpkc->ijfpc", fu_f, obv0)
                self_v = torch.einsum("ipkc,jgpkc->ijgpc", obu0, fv_f)
                views_a = torch.cat([cross_ov, self_u, self_v], dim=2)
                views_b = torch.cat([cross_dir, self_v, self_u], dim=2)
                v_node = self.pad_edges_to_nodes(torch.cat([views_a, views_b], dim=-1))
                mm_v = mismatch.float().unsqueeze(2).unsqueeze(-1).expand(m0, m1, 24, L, 1)
                vh = self.fib_rib_in(torch.cat([v_node, mm_v], dim=-1)).reshape(m0 * m1 * 24, L, self.dm)
                if rib_perm is not None:
                    inv_r = torch.argsort(rib_perm)
                    vc = F.gelu(self.fib_rib_conv(vh[:, rib_perm, :].transpose(1, 2))).transpose(1, 2)[:, inv_r, :]
                else:
                    vc = F.gelu(self.fib_rib_conv(vh.transpose(1, 2))).transpose(1, 2)
                vc = vc + self.nr_rib_ff(vc)
                rib_c = rib_c + vc.reshape(m0, m1, 24, L, self.dm).amax(dim=2) + 0.25 * (wu_ctx[:, None] + wv_ctx[None, :])

            mm_mask = mismatch.float().unsqueeze(-1)
            mm_cnt = mm_mask.sum(dim=2).clamp_min(1.0)
            rib_pool = torch.cat([(rib_c * mm_mask).sum(dim=2) / mm_cnt, rib_c.amax(dim=2)], dim=-1)
            pbase = self.pair_in(torch.cat([rib_pool, pub], dim=-1))
            pbase = pbase + self.pair_prelude(pbase)

            if self.nonrecurrent:
                h_mat, bm = self.fresco_block(pbase, bm, infeas, ablate_axial=ablate_ax)
                if self.arch == "param_nonrecurrent":
                    h_mat = h_mat + self.nr_pair_ff(h_mat)
                pot = self.pot_head(h_mat).squeeze(-1)
            else:
                if k == 0:
                    gm, sm_gate = self.fresco_gate(pbase, mm_in)
                    h_mat, bm = self.fresco_block(pbase + gm, mm_in, infeas, ablate_axial=ablate_ax)
                    if self.is_lowrank:
                        h_mat = h_mat + self.lr_pair_ff(h_mat)
                    pot_accum = self.pot_head(h_mat).squeeze(-1)
                else:
                    gm, sm_gate = self.fresco_gate(h_mat, mm_in)
                    h_mat, bm = self.fresco_block(h_mat + 0.5 * pbase + gm, mm_in, infeas, ablate_axial=ablate_ax)
                    if self.is_lowrank:
                        h_mat = h_mat + self.lr_pair_ff(h_mat)
                    pot_accum = pot_accum + self.delta_scale * self.pot_head(h_mat - pbase).squeeze(-1)
                pot = pot_accum

            pot_masked = pot.masked_fill(infeas, -12.0)
            tau = F.softplus(self.log_tau) + 0.1
            if self.is_lowrank:
                nu_inv = torch.cat([bu_anchor.mean(dim=(1, 2)), bu_nov.amax(dim=(1, 2)), np0], dim=-1)
                nv_inv = torch.cat([bv_anchor.mean(dim=(1, 2)), bv_nov.amax(dim=(1, 2)), np1], dim=-1)
            else:
                nu_inv = torch.cat([
                    bu_anchor.mean(dim=(1, 2)), bu_nov.amax(dim=(1, 2)),
                    bi_u_anchor.mean(dim=(1, 2, 3, 4)), F.relu(bi_u - bi_u_anchor).amax(dim=(1, 2, 3, 4)),
                    np0,
                ], dim=-1)
                nv_inv = torch.cat([
                    bv_anchor.mean(dim=(1, 2)), bv_nov.amax(dim=(1, 2)),
                    bi_v_anchor.mean(dim=(1, 2, 3, 4)), F.relu(bi_v - bi_v_anchor).amax(dim=(1, 2, 3, 4)),
                    np1,
                ], dim=-1)
            s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
                  + self.node_head(nu_inv).squeeze(-1)).masked_fill(infeas.all(dim=1), -12.0)
            s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
                  + self.node_head(nv_inv).squeeze(-1)).masked_fill(infeas.all(dim=0), -12.0)

            sub_u = self.subterm_brightness(oh_u, bu, bi_u)
            sub_v = self.subterm_brightness(oh_v, bv, bi_v)
            row = {
                "pot": pot, "pot_masked": pot_masked, "s0": s0, "s1": s1, "infeas": infeas,
                "sub_u": sub_u, "sub_v": sub_v,
            }
            if return_tensors:
                row["h_mat"] = h_mat.detach().cpu().clone()
                row["bm_out"] = bm.detach().cpu().clone()
                row["bw_u_out"] = bu.detach().cpu().clone()
                row["rib_spatial_autocorr"] = m16.subterm_morans_i(rib_c.detach())
                if bi_u is not None:
                    row["bi_u_out"] = bi_u.detach().cpu().clone()
                    row["active_bi_support_frac"] = float((bi_u.detach() > 1e-6).any(dim=(1, -1)).float().mean())
                else:
                    row["active_bi_support_frac"] = float((bu.detach() > 1e-6).any(dim=-1).float().mean())
                if not self.nonrecurrent:
                    row["bm_in"] = mm_in.detach().cpu().clone()
                    row["bw_u_in"] = mu_in.detach().cpu().clone()
                    if mbi_u is not None:
                        row["bi_u_in"] = mbi_u.detach().cpu().clone()
                    row["word_gate_strength"] = 0.5 * (
                        float(s_pos_gate[:m0].detach().mean()) + float(s_pos_gate[m0:].detach().mean())
                    )
                    row["bi_gate_strength"] = 0.0 if sbi_u is None else float(sbi_u.detach().mean())
                    row["fresco_gate_strength"] = float(sm_gate.detach().mean())
                    row["fresco_gate_contrib_rms"] = base.rms(gm)
            outs.append(row)
        return outs, rule_win_checks


def new_lbf17_model(seed: int, arch: str):
    if arch == "lbf16_pos1_opened":
        return m16.new_model(seed, "recurrent_opened")
    torch.manual_seed(seed)
    m = RelationalProvenanceFrescoNet(arch=arch).cpu().float()
    if not (53000 <= m.parameter_count() <= 53200):
        raise ValueError(f"parameter count {m.parameter_count()} for {arch} outside [53000, 53200]")
    shared_fp = base._model_fingerprint(m)
    if not m.nonrecurrent:
        gates = [m.env_gate, m.fresco_gate] + ([] if m.is_lowrank else [m.bi_gate])
        for gmod in gates:
            if not torch.all(gmod.gate.bias == -6.0):
                raise ValueError("expected base gate bias -6.0")
            if arch != "rel_prov_standard":
                with torch.no_grad():
                    gmod.gate.bias.fill_(-1.5)
    return m, shared_fp


# ============================================================================
# 3. Charged Bounded Exact-Lookahead Controller & Episode Runner
# ============================================================================
def estimate_peak_tensor_bytes(arch: str, m0: int, m1: int, L: int) -> int:
    """Deterministic active tensor memory footprint (bytes, float32) for one decision."""
    n_tot = m0 + m1
    Lm1 = max(1, L - 1)
    pos_bytes = n_tot * L * 6 * 20 * 4 * 3
    pair_bytes = m0 * m1 * L * 38 * 4 * 2 + m0 * m1 * 38 * 4 * 4
    if arch in ("lbf16_pos1_opened", "lowrank_opened"):
        return pos_bytes + pair_bytes
    fp = 1 if arch == "adj2_noprov_opened" else FP_FIBERS
    bi_bytes = n_tot * fp * Lm1 * 36 * 4 * 4 * 3
    extra = (m0 * m1 * 24 * L * 38 * 4) if arch == "compute_nonrecurrent" else 0
    return pos_bytes + bi_bytes + pair_bytes + extra


_FLOP_CACHE: dict[tuple[str, int, int, int], int] = {}


def cached_decision_flops(model, arch: str, case, s: m15.FringeState, rem: int, mode: str = "normal") -> int:
    key = (arch, len(s.fronts[0]), len(s.fronts[1]), len(case.source))
    if key not in _FLOP_CACHE:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            if arch == "lbf16_pos1_opened":
                model(case, s, rem, mode="normal")
            else:
                model(case, s, rem, mode=mode)
        _FLOP_CACHE[key] = int(fc.get_total_flops())
    return _FLOP_CACHE[key]


def run_exact_lookahead_episode(rg, c, budget: int = WORK, depth: int = DEPTH, max_probe_per_side: int = 3) -> dict:
    """Bounded exact 1-step reachable-set lookahead controller charging EVERY examined legal successor."""
    t0 = time.perf_counter()
    s = m15.FringeState.start(c)
    probed_moves: dict[tuple[int, str], list[str]] = {}
    pair_evals = decisions = 0
    L = len(c.source)
    peak_mem = 0

    while s.live(budget):
        decisions += 1
        m0, m1 = len(s.fronts[0]), len(s.fronts[1])
        # Rank candidates initially by public geometric heuristic to pick top-M to probe with exact 1-balls
        for side in (0, 1):
            opp = s.fronts[1 - side]
            ranked = []
            for idx, w in enumerate(s.fronts[side]):
                g = s.dist[side][w]
                if g >= depth:
                    continue
                best_lb = min(
                    (g + s.dist[1 - side][v] + max(1, (rg.hamming(w, v) + 2) // 3) for v in opp),
                    default=999,
                )
                ranked.append((best_lb, idx, w))
            ranked.sort()
            for _, _, w in ranked[:max_probe_per_side]:
                if (side, w) in probed_moves or not s.live(budget):
                    continue
                g = s.dist[side][w]
                novel_succs = []
                for mv in rg.neighbors(c, w):
                    if s.work >= budget:
                        break
                    # Charge EVERY examined legal successor (including duplicates)!
                    s.work += 1
                    nxt = mv.successor
                    if nxt in s.dist[side]:
                        continue
                    s.dist[side][nxt] = g + 1
                    s.parent[side][nxt] = (w, mv)
                    if nxt in s.dist[1 - side] and g + 1 + s.dist[1 - side][nxt] <= depth:
                        s.path = m15.lbf14.meeting_path(
                            rg, c, m15.lbf14.State(s.fronts, s.parent, s.dist, s.work, s.expanded), nxt
                        )
                        rg.replay(c, s.path)
                        break
                    if g + 1 < depth:
                        novel_succs.append(nxt)
                probed_moves[(side, w)] = novel_succs
                if s.path is not None or s.work >= budget:
                    break
            if s.path is not None or s.work >= budget:
                break

        if not s.live(budget):
            break

        # Score candidates using exact probed 1-step sets S_1(u) x S_1(v)
        best_key = None
        best_act = (0, 0)
        sets_0 = [([u] + probed_moves.get((0, u), [])) for u in s.fronts[0]]
        sets_1 = [([v] + probed_moves.get((1, v), [])) for v in s.fronts[1]]
        peak_mem = max(peak_mem, sum(len(x) for x in sets_0 + sets_1) * L * 4)

        for side in (0, 1):
            ms = len(s.fronts[side])
            for idx, w in enumerate(s.fronts[side]):
                g = s.dist[side][w]
                my_set = sets_0[idx] if side == 0 else sets_1[idx]
                opp_sets = sets_1 if side == 0 else sets_0
                opp_words = s.fronts[1 - side]
                best_f = (999, 999)
                for j, oset in enumerate(opp_sets):
                    gv = s.dist[1 - side][opp_words[j]]
                    pair_evals += len(my_set) * len(oset)
                    for wu in my_set:
                        du = 0 if wu == w else 1
                        for wv in oset:
                            dv = 0 if wv == opp_words[j] else 1
                            dh = rg.hamming(wu, wv)
                            lb = 0 if dh == 0 else max(1, (dh + 2) // 3)
                            tot = g + du + gv + dv + lb
                            if tot <= depth and (tot, dh) < best_f:
                                best_f = (tot, dh)
                key = (*best_f, ms, g, side, idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_act = (side, idx)

        b_side, b_idx = best_act
        b_word = s.fronts[b_side][b_idx]
        if (b_side, b_word) in probed_moves:
            s.expanded += 1
            s.fronts[b_side][b_idx:b_idx + 1] = probed_moves.pop((b_side, b_word))
        else:
            m15.expand_node(rg, c, s, b_side, b_idx, budget, depth)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    res = rg._result(c, "exact_lookahead", s.path, s.work, s.expanded, 0, budget, depth)
    return {
        **res,
        "pair_evaluations": pair_evals,
        "substring_checks": 0,
        "recurrent_passes": 0,
        "fringe_decisions": decisions,
        "neural_flops": 0,
        "wall_clock_ms": elapsed_ms,
        "peak_tensor_bytes": peak_mem,
    }


def probe_lbf17_decision(model: RelationalProvenanceFrescoNet, case, s: m15.FringeState, rem: int,
                         tag: str, ordinal: int, checkpoint_sha: str,
                         raw_store: list[dict], obs_store: list[dict]) -> tuple[list[dict], int]:
    outs_norm, sub_ch = model(case, s, rem, mode="normal", return_tensors=True)
    alts = {}
    for md in SHAM_MODES:
        ao, _ = model(case, s, rem, mode=md, return_tensors=True)
        alts[md] = ao

    # Fail-closed Pass-1 equivalence for recurrence-only interventions & roll norm isometries!
    for md in ("env_zero", "env_roll", "mat_zero", "mat_roll"):
        torch.testing.assert_close(outs_norm[0]["pot"], alts[md][0]["pot"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outs_norm[0]["s0"], alts[md][0]["s0"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outs_norm[0]["s1"], alts[md][0]["s1"], rtol=1e-6, atol=1e-6)

    if model.passes > 1:
        # Orthogonal decoupling checks at Pass 2 (k=1):
        # 1. env_zero/env_roll leave incoming matrix bank bm_in IDENTICAL to normal at k=1!
        torch.testing.assert_close(outs_norm[1]["bm_in"], alts["env_zero"][1]["bm_in"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outs_norm[1]["bm_in"], alts["env_roll"][1]["bm_in"], rtol=1e-6, atol=1e-6)
        # 2. mat_zero/mat_roll leave outgoing Pass-2 envelope banks IDENTICAL to normal!
        torch.testing.assert_close(outs_norm[1]["bw_u_out"], alts["mat_zero"][1]["bw_u_out"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outs_norm[1]["bw_u_out"], alts["mat_roll"][1]["bw_u_out"], rtol=1e-6, atol=1e-6)
        if "bi_u_out" in outs_norm[1]:
            torch.testing.assert_close(outs_norm[1]["bi_u_out"], alts["mat_zero"][1]["bi_u_out"], rtol=1e-6, atol=1e-6)
        # 3. Roll Frobenius-norm isometries across passes k>=1
        for k in range(1, model.passes):
            torch.testing.assert_close(
                alts["mat_roll"][k - 1]["bm_out"].norm(), alts["mat_roll"][k]["bm_in"].norm(), rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                alts["env_roll"][k - 1]["bw_u_out"].norm(), alts["env_roll"][k]["bw_u_in"].norm(), rtol=1e-6, atol=1e-6
            )

    obs = m15.build_fresco_observation(case, s, outs_norm, checkpoint_sha, ordinal)
    obs_store.append(obs)
    pers_mask = [[(u, v) in s.prev_pot for v in s.fronts[1]] for u in s.fronts[0]]
    raw_store.append({
        "decision_id": obs["root_id"],
        "root": case.root,
        "tag": tag,
        "ordinal": ordinal,
        "u_words": list(s.fronts[0]),
        "v_words": list(s.fronts[1]),
        "persisted_mask": pers_mask,
        "normal": [{k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in o.items()}
                   for o in outs_norm],
        "alternatives": {
            md: [{k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                  for k, v in o.items() if k in ("pot", "s0", "s1")}
                 for o in ao]
            for md, ao in alts.items()
        },
    })
    return outs_norm, sub_ch


def run_lbf17_episode(rg, c, policy: str, *, model=None, arch: str = "", mode: str = "normal",
                      seed: int = 0, budget: int = WORK, depth: int = DEPTH,
                      probe_ctx: dict | None = None) -> dict:
    if policy == "exact_lookahead":
        return run_exact_lookahead_episode(rg, c, budget=budget, depth=depth)
    if policy in ("astar", "bidir"):
        t0 = time.perf_counter()
        r = rg.search(c, policy, budget=budget, depth=depth)
        return {
            **r, "pair_evaluations": 0, "substring_checks": 0, "recurrent_passes": 0,
            "fringe_decisions": 0, "neural_flops": 0,
            "wall_clock_ms": (time.perf_counter() - t0) * 1000.0, "peak_tensor_bytes": 0,
        }

    t0 = time.perf_counter()
    s = m15.FringeState.start(c)
    pdist = m15.build_kmer_graph(c) if policy == "graph_bridge" else None
    rng = random.Random(seed)
    calls = pair_evals = sub_checks = rec_passes = decisions = probed = flops_tot = peak_mem = 0
    L = len(c.source)

    while s.live(budget):
        decisions += 1
        m0, m1 = len(s.fronts[0]), len(s.fronts[1])
        if policy == "uniform":
            tot = m0 + m1
            pick = rng.randrange(tot)
            side, idx = (0, pick) if pick < m0 else (1, pick - m0)
        elif policy == "geom_mm":
            (side, idx), pe, sc = m15.geom_mm_select(rg, c, s, budget, depth)
            pair_evals += pe
            sub_checks += sc
        elif policy == "graph_bridge":
            (side, idx), pe, sc = m15.graph_bridge_select(rg, c, s, pdist, budget, depth)
            pair_evals += pe
            sub_checks += sc
        elif policy == "learned":
            if model is None:
                raise ValueError("learned policy requires frozen model")
            rem = budget - s.work
            with torch.no_grad():
                if (probe_ctx is not None and mode == "normal" and arch != "lbf16_pos1_opened"
                        and m0 * m1 >= 4 and probed < 2):
                    probed += 1
                    outs, sc = probe_lbf17_decision(
                        model, c, s, rem, probe_ctx["tag"], decisions,
                        probe_ctx["checkpoint"], probe_ctx["raw"], probe_ctx["obs"],
                    )
                elif arch == "lbf16_pos1_opened":
                    outs, sc = model(c, s, rem, mode="normal", depth_cap=depth)
                else:
                    outs, sc = model(c, s, rem, mode=mode, depth_cap=depth)
            flops_tot += cached_decision_flops(model, arch, c, s, rem, mode="normal" if arch == "lbf16_pos1_opened" else mode)
            peak_mem = max(peak_mem, estimate_peak_tensor_bytes(arch, m0, m1, L))
            calls += 1
            rec_passes += len(outs)
            pair_evals += m0 * m1
            sub_checks += sc
            last = outs[-1]
            psig = last["pot"].sigmoid()
            pm = float(psig.mean())
            s.prev_pot = {(u, v): float(psig[i, j]) - pm
                          for i, u in enumerate(s.fronts[0])
                          for j, v in enumerate(s.fronts[1])}
            s0_l = last["s0"].tolist()
            s1_l = last["s1"].tolist()
            if any(not math.isfinite(x) for x in s0_l + s1_l):
                raise ValueError("nonfinite learned node scores")
            cands = [(s0_l[i], -m0, 0, -i) for i in range(m0)] + [(s1_l[j], -m1, -1, -j) for j in range(m1)]
            best = max(cands)
            side = 0 if best[2] == 0 else 1
            idx = -best[3]
        else:
            raise ValueError(f"unknown policy {policy}")
        m15.expand_node(rg, c, s, side, idx, budget, depth)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    res = rg._result(c, policy, s.path, s.work, s.expanded, calls, budget, depth)
    return {
        **res,
        "pair_evaluations": pair_evals,
        "substring_checks": sub_checks,
        "recurrent_passes": rec_passes,
        "fringe_decisions": decisions,
        "neural_flops": flops_tot,
        "wall_clock_ms": elapsed_ms,
        "peak_tensor_bytes": peak_mem,
    }


# ============================================================================
# 4. Train-Only Teacher (with Concrete k-Ball Subterm Targets) & Matched Fit
# ============================================================================
def _ball_k_alpha1(ball_map: dict[str, int], L: int, k: int) -> torch.Tensor:
    a = torch.zeros((L, 6), dtype=torch.float32)
    for w, dist in ball_map.items():
        if dist <= k:
            for p, ch in enumerate(w):
                a[p, CHAR2ID[ch]] = 1.0
    return a


def teacher(rg, suite: Path, out: Path, resume: bool = False) -> tuple[list[dict], dict]:
    cases, m, refs = rg.load_split(suite, "train", witnesses=True)
    if len(cases) != COUNTS["train"] or m["seed"] != SEED:
        raise ValueError("unexpected LBF-17 training roster")
    saved_manifest = None
    if resume and (out / "manifest.json").exists():
        saved_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    else:
        out.mkdir(parents=True)
        ref_cert = base.certify(rg, [(c, refs[c.root], "train_" + c.root) for c in cases], out / "references")

    frames = []
    audits = []
    oracle_work = non_tied = mixed_closures = pos_marg = neg_marg = multi_hop = 0

    for c in cases:
        nb_cache: dict[str, tuple] = {}
        def nb_fn(word: str):
            if word not in nb_cache:
                nb_cache[word] = rg.neighbors(c, word)
            return nb_cache[word]

        pdist = m15.build_kmer_graph(c)
        L = len(c.source)
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
                            multi_hop += 1
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

                        # Train-only concrete k-step symbol reachability targets (zero extra oracle work!)
                        sub_u_y = [torch.stack([_ball_k_alpha1(dmap, L, k=kk) for dmap in bu]) for kk in (1, 2, 3)]
                        sub_v_y = [torch.stack([_ball_k_alpha1(dmap, L, k=kk) for dmap in bv]) for kk in (1, 2, 3)]

                        frames.append({
                            "root": c.root, "lineage": c.lineage, "step": step, "rollout": rollout,
                            "case": c, "state": s.clone_light(),
                            "prev_state": None if prev_snap is None else prev_snap.clone_light(),
                            "rem": rem, "prev_rem": prev_rem, "rem_cap": rem_cap,
                            "min_dr": min_dr, "min_ell": min_ell,
                            "costs": flat_c,
                            "y_mat": torch.tensor(y_mat, dtype=torch.float32),
                            "node_y": torch.tensor(node_y, dtype=torch.float32),
                            "sub_u_y": sub_u_y,
                            "sub_v_y": sub_v_y,
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
    serial = [{
        "root": f["root"], "lineage": f["lineage"], "step": f["step"], "rollout": f["rollout"],
        "m0": len(f["state"].fronts[0]), "m1": len(f["state"].fronts[1]),
        "min_dr": f["min_dr"], "min_ell": f["min_ell"],
    } for f in frames]
    if saved_manifest is not None:
        tmp_check = out / "_verify_frames_summary.jsonl"
        base.write_jsonl(tmp_check, serial)
        sha_now = base.file_sha(tmp_check)
        tmp_check.unlink(missing_ok=True)
        if sha_now != saved_manifest["frames_summary_sha256"]:
            raise ValueError("resumed teacher frames SHA-256 mismatch against saved manifest")
        return frames, saved_manifest

    cf_cert = base.certify(rg, audits, out / "counterfactuals")
    base.write_jsonl(out / "frames_summary.jsonl", serial)
    receipt = {
        "train_roots": len(cases),
        "frames": len(frames),
        "non_tied_frames": non_tied,
        "mixed_closure_frames": mixed_closures,
        "multi_hop_bridge_frames_dr_ge_3": multi_hop,
        "positive_marginal_actions": pos_marg,
        "negative_marginal_actions": neg_marg,
        "oracle_work": oracle_work,
        "reference_certificates": ref_cert["compiled_axiom_free"],
        "counterfactual_certificates": cf_cert["compiled_axiom_free"],
        "frames_summary_sha256": base.file_sha(out / "frames_summary.jsonl"),
    }
    base.write(out / "manifest.json", receipt)
    return frames, receipt


def fit(frames: list[dict], seed: int, arch: str, out: Path,
        pre_init: tuple | None = None):
    model, shared_init = pre_init if pre_init is not None else new_lbf17_model(seed, arch)
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
        if time.monotonic() - t0 > 1800:
            raise TimeoutError(f"CPU fit cap exceeded for {arch}")
        opt.zero_grad(set_to_none=True)
        bloss_val = 0.0
        step_indices = []
        for _ in range(BATCH_SIZE):
            rname = rng.choice(roots)
            f_idx, fr = rng.choice(byroot[rname])
            step_indices.append(f_idx)
            st_cur = fr["state"].clone_light()
            if fr["prev_state"] is not None and step >= 160:
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
                lsub = torch.zeros(())
                if "sub_u" in o and arch not in ("sym_prov_readout", "lbf16_pos1_opened"):
                    hk = min(2, k)
                    pu = o["sub_u"].clamp(1e-6, 1.0 - 1e-6)
                    pv = o["sub_v"].clamp(1e-6, 1.0 - 1e-6)
                    mu_s = (o["sub_u"].detach() > 1e-6).float()
                    mv_s = (o["sub_v"].detach() > 1e-6).float()
                    if mu_s.sum() > 0:
                        lsub = lsub + (F.binary_cross_entropy(pu, fr["sub_u_y"][hk], reduction="none") * mu_s).sum() / mu_s.sum()
                    if mv_s.sum() > 0:
                        lsub = lsub + (F.binary_cross_entropy(pv, fr["sub_v_y"][hk], reduction="none") * mv_s).sum() / mv_s.sum()
                    lsub = 0.15 * lsub
                fl = fl + wk * (lm + lr + lsub)
                ws += wk
            item_loss = (fl / ws) / BATCH_SIZE
            if not torch.isfinite(item_loss):
                raise ValueError(f"nonfinite LBF-17 training loss for {arch}")
            # Backpropagate per frame immediately so at most 1 autograd graph resides in RAM!
            item_loss.backward()
            bloss_val += float(item_loss.detach())

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            raise ValueError(f"nonfinite LBF-17 gradient for {arch}")
        opt.step()
        sched.step()
        drec = {"step": step + 1, "frames": step_indices}
        draws.append(drec)
        trace.append({**drec, "loss": bloss_val})

    fit_sec = time.monotonic() - t0
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
        "fit_wall_clock_sec": round(fit_sec, 3),
        "shared_initial": shared_init,
        "initial": init_fp,
        "final": base._model_fingerprint(model),
        "draws_sha256": base.file_sha(out / "draws.jsonl"),
        "trace_sha256": base.file_sha(out / "trace.jsonl"),
        "checkpoint_sha256": base.file_sha(out / "checkpoint.pt"),
    }
    base.write(out / "manifest.json", receipt)
    return model, receipt


# ============================================================================
# 5. Abstraction-Error, Glyph-AUROC, Regret & Lean Formalization Diagnostics
# ============================================================================
def compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Exact tie-aware Mann-Whitney U AUROC without external dependencies."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        pos_in_tie = sum(pairs[t][1] for t in range(i, j))
        rank_sum_pos += pos_in_tie * avg_rank
        i = j
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def evaluate_abstraction_and_glyphs(rg, cases: list, models: dict, seed: int, max_roots: int = 48) -> tuple[dict, dict[str, float]]:
    """Compare concrete K=3 reachability against symbolic & neural envelopes; measure glyph AUROC."""
    domains = (
        "sym_pos1", "sym_prov_pos1", "sym_adj2", "sym_sub3", "sym_prov4_adj2",
        "neural_lbf16_pos1", "neural_adj2_noprov", "neural_rel_prov",
    )
    acc = {d: {"tp": 0, "fp": 0, "fn": 0, "b_tp": 0, "b_fp": 0, "b_fn": 0, "b_tn": 0} for d in domains}
    glyph_scores = {"lbf16_pos1": [], "adj2_noprov": [], "rel_prov": [], "rel_prov_led4": []}
    glyph_labels = []
    within_prov_scores, within_prov_labels = [], []
    root_pos1_fpr: dict[str, float] = {}

    m_16 = models[(seed, "lbf16_pos1_opened")]
    m_np = models[(seed, "adj2_noprov_opened")]
    m_rp = models[(seed, "rel_prov_opened")]

    for c in cases[:max_roots]:
        L = len(c.source)
        s = m15.FringeState.start(c)
        for _ in range(3):
            if not s.live(WORK):
                break
            (sd, ix), _, _ = m15.geom_mm_select(rg, c, s)
            m15.expand_node(rg, c, s, sd, ix)
        if not s.fronts[0] or not s.fronts[1]:
            s = m15.FringeState.start(c)

        with torch.no_grad():
            o16, _ = m_16(c, s, WORK - s.work, mode="normal", return_tensors=True)
            onp, _ = m_np(c, s, WORK - s.work, mode="normal", return_tensors=True)
            orp, _ = m_rp(c, s, WORK - s.work, mode="normal", return_tensors=True)

        u_sub = s.fronts[0][:4]
        v_sub = s.fronts[1][:4]
        u_balls = [concrete_k_balls(rg, c, w, k_max=3)[3] for w in u_sub]
        v_balls = [concrete_k_balls(rg, c, w, k_max=3)[3] for w in v_sub]

        # Extract neural Pass-3 supports & brightness on u_sub
        oh_u = F.one_hot(torch.tensor([[CHAR2ID[ch] for ch in w] for w in u_sub]), num_classes=6).bool()
        oh_v = F.one_hot(torch.tensor([[CHAR2ID[ch] for ch in w] for w in v_sub]), num_classes=6).bool()

        b16_u = o16[-1]["bw_u_out"][:len(u_sub)]
        n16_supp_u = oh_u | (b16_u > 1e-6).any(dim=-1)
        b16_bright_u = torch.maximum(oh_u.float(), b16_u.mean(dim=-1).clamp(0.0, 1.0))

        bi_np_u = onp[-1]["bi_u_out"][:len(u_sub)]
        nnp_pos_u = m_np.bigram_to_pos1_channels(bi_np_u)
        nnp_supp_u = oh_u | (nnp_pos_u > 1e-6).any(dim=-1)
        nnp_bright_u = onp[-1]["sub_u"][:len(u_sub)].detach().cpu()

        bi_rp_u = orp[-1]["bi_u_out"][:len(u_sub)]
        nrp_pos_u = m_rp.bigram_to_pos1_channels(bi_rp_u)
        nrp_supp_u = oh_u | (nrp_pos_u > 1e-6).any(dim=-1)
        nrp_bright_u = orp[-1]["sub_u"][:len(u_sub)].detach().cpu()

        # Also run v_sub through init+cell via reversed fronts for exact neural bridge test:
        s_swap = m15.FringeState([s.fronts[1], s.fronts[0]], [s.parent[1], s.parent[0]],
                                 [s.dist[1], s.dist[0]], s.work, s.expanded)
        with torch.no_grad():
            o16_v, _ = m_16(c, s_swap, WORK - s.work, mode="normal", return_tensors=True)
            onp_v, _ = m_np(c, s_swap, WORK - s.work, mode="normal", return_tensors=True)
            orp_v, _ = m_rp(c, s_swap, WORK - s.work, mode="normal", return_tensors=True)
        n16_supp_v = oh_v | (o16_v[-1]["bw_u_out"][:len(v_sub)] > 1e-6).any(dim=-1)
        bi_np_v = onp_v[-1]["bi_u_out"][:len(v_sub)]
        bi_rp_v = orp_v[-1]["bi_u_out"][:len(v_sub)]
        oh_bi_u_b = (oh_u[:, :-1, :, None] & oh_u[:, 1:, None, :])
        oh_bi_v_b = (oh_v[:, :-1, :, None] & oh_v[:, 1:, None, :])

        u_sym, v_sym = [], []
        r_fp = r_neg = 0
        for iu, uw in enumerate(u_sub):
            gt1 = alpha1_bool(u_balls[iu], L)
            sp1 = symbolic_pos1(c.rules, uw, 3)[3]
            spr1_f = symbolic_prov_pos1(c.rules, uw, 3)[3]
            spr1 = torch.stack(spr1_f, dim=0).any(dim=0)
            sa2 = symbolic_adj2(c.rules, uw, 3)[3]
            ss3 = symbolic_sub3(c.rules, uw, 3)[3]
            sp42 = symbolic_prov4_adj2(c.rules, uw, 3)[3]
            sp42_pos = adj2_to_pos1(sp42[0])
            for fb in sp42[1]:
                if fb.any():
                    sp42_pos |= adj2_to_pos1(fb)
            u_sym.append((sp1, spr1_f, sa2, ss3, sp42))

            r_fp += int((sp1 & ~gt1).sum())
            r_neg += int((~gt1).sum())

            preds = {
                "sym_pos1": sp1,
                "sym_prov_pos1": spr1,
                "sym_adj2": adj2_to_pos1(sa2),
                "sym_sub3": sub3_to_pos1(ss3),
                "sym_prov4_adj2": sp42_pos,
                "neural_lbf16_pos1": n16_supp_u[iu],
                "neural_adj2_noprov": nnp_supp_u[iu],
                "neural_rel_prov": nrp_supp_u[iu],
            }
            for dname, pr in preds.items():
                acc[dname]["tp"] += int((pr & gt1).sum())
                acc[dname]["fp"] += int((pr & ~gt1).sum())
                acc[dname]["fn"] += int((~pr & gt1).sum())

            # Collect glyph brightness on NON-ROOT A_1-plausible subterms (sp1 & ~oh_u[iu])!
            plaus_mask = sp1 & (~oh_u[iu])
            for p in range(L):
                for sym_i in range(6):
                    if plaus_mask[p, sym_i]:
                        cert = int(bool(gt1[p, sym_i]))
                        glyph_labels.append(cert)
                        glyph_scores["lbf16_pos1"].append(float(b16_bright_u[iu, p, sym_i]))
                        glyph_scores["adj2_noprov"].append(float(nnp_bright_u[iu, p, sym_i]))
                        br_rp = float(nrp_bright_u[iu, p, sym_i])
                        glyph_scores["rel_prov"].append(br_rp)
                        glyph_scores["rel_prov_led4"].append(float(min(3, int(4.0 * br_rp))))
                    if nrp_supp_u[iu, p, sym_i] and (not oh_u[iu, p, sym_i]):
                        within_prov_labels.append(int(bool(gt1[p, sym_i])))
                        within_prov_scores.append(float(nrp_bright_u[iu, p, sym_i]))

        root_pos1_fpr[c.root] = r_fp / max(1, r_neg)

        for iv, vw in enumerate(v_sub):
            v_sym.append((
                symbolic_pos1(c.rules, vw, 3)[3],
                symbolic_prov_pos1(c.rules, vw, 3)[3],
                symbolic_adj2(c.rules, vw, 3)[3],
                symbolic_sub3(c.rules, vw, 3)[3],
                symbolic_prov4_adj2(c.rules, vw, 3)[3],
            ))

        for iu in range(len(u_sub)):
            for iv in range(len(v_sub)):
                gt_b = bool(u_balls[iu] & v_balls[iv])
                sp1_u, spr1_u, sa2_u, ss3_u, sp42_u = u_sym[iu]
                sp1_v, spr1_v, sa2_v, ss3_v, sp42_v = v_sym[iv]

                b_map = {
                    "sym_pos1": bool((sp1_u & sp1_v).any(dim=1).all()),
                    "sym_prov_pos1": any(bool((fu & fv).any(dim=1).all()) for fu in spr1_u for fv in spr1_v),
                    "sym_adj2": bool(trellis_filter_adj2_bool(sa2_u & sa2_v).any()),
                    "sym_sub3": bool(trellis_filter_sub3_bool(ss3_u & ss3_v).any()),
                    "sym_prov4_adj2": any(
                        bool(trellis_filter_adj2_bool(fu & fv).any())
                        for fu in ([sp42_u[0]] + [x for x in sp42_u[1] if x.any()])
                        for fv in ([sp42_v[0]] + [x for x in sp42_v[1] if x.any()])
                    ),
                    "neural_lbf16_pos1": bool((n16_supp_u[iu] & n16_supp_v[iv]).any(dim=1).all()),
                    "neural_adj2_noprov": bool(trellis_filter_adj2_bool(
                        (oh_bi_u_b[iu] | (bi_np_u[iu, 0] > 1e-6).any(dim=-1))
                        & (oh_bi_v_b[iv] | (bi_np_v[iv, 0] > 1e-6).any(dim=-1))
                    ).any()),
                    "neural_rel_prov": any(
                        bool(trellis_filter_adj2_bool(fu & fv).any())
                        for fu in ([oh_bi_u_b[iu]] + [(bi_rp_u[iu, f] > 1e-6).any(dim=-1)
                                                      for f in range(FP_FIBERS) if (bi_rp_u[iu, f] > 1e-6).any()])
                        for fv in ([oh_bi_v_b[iv]] + [(bi_rp_v[iv, f] > 1e-6).any(dim=-1)
                                                      for f in range(FP_FIBERS) if (bi_rp_v[iv, f] > 1e-6).any()])
                    ),
                }
                for dname, bp in b_map.items():
                    if gt_b and bp: acc[dname]["b_tp"] += 1
                    elif gt_b and not bp: acc[dname]["b_fn"] += 1
                    elif (not gt_b) and bp: acc[dname]["b_fp"] += 1
                    else: acc[dname]["b_tn"] += 1

    summary_domains = {}
    for dname, d in acc.items():
        summary_domains[dname] = {
            "symbol_precision": d["tp"] / max(1, d["tp"] + d["fp"]),
            "symbol_recall": d["tp"] / max(1, d["tp"] + d["fn"]),
            "symbol_tp": d["tp"],
            "symbol_fp": d["fp"],
            "symbol_fn": d["fn"],
            "bridge_tp": d["b_tp"],
            "bridge_fn": d["b_fn"],
            "bridge_fp": d["b_fp"],
            "false_bridge_rate": d["b_fp"] / max(1, d["b_fp"] + d["b_tn"]),
        }

    cert_rp = [s for s, y in zip(glyph_scores["rel_prov"], glyph_labels) if y == 1]
    spur_rp = [s for s, y in zip(glyph_scores["rel_prov"], glyph_labels) if y == 0]
    cert_16 = [s for s, y in zip(glyph_scores["lbf16_pos1"], glyph_labels) if y == 1]
    spur_16 = [s for s, y in zip(glyph_scores["lbf16_pos1"], glyph_labels) if y == 0]

    glyph_cal = {
        "evaluated_plausible_subterms": len(glyph_labels),
        "certified_reachable_subterms": sum(glyph_labels),
        "spurious_plausible_subterms": len(glyph_labels) - sum(glyph_labels),
        "auroc_lbf16_pos1": compute_auroc(glyph_scores["lbf16_pos1"], glyph_labels),
        "auroc_adj2_noprov": compute_auroc(glyph_scores["adj2_noprov"], glyph_labels),
        "auroc_rel_prov_continuous": compute_auroc(glyph_scores["rel_prov"], glyph_labels),
        "auroc_rel_prov_led4_quantized": compute_auroc(glyph_scores["rel_prov_led4"], glyph_labels),
        "auroc_within_rel_prov_supported": compute_auroc(within_prov_scores, within_prov_labels),
        "rel_prov_mean_brightness_certified": sum(cert_rp) / max(1, len(cert_rp)),
        "rel_prov_mean_brightness_spurious": sum(spur_rp) / max(1, len(spur_rp)),
        "lbf16_mean_brightness_certified": sum(cert_16) / max(1, len(cert_16)),
        "lbf16_mean_brightness_spurious": sum(spur_16) / max(1, len(spur_16)),
    }
    return {"domains": summary_domains, "glyph_calibration": glyph_cal}, root_pos1_fpr


def evaluate_action_ranking_regret(rg, cal_cases: list, models: dict, seed: int) -> dict:
    """Compute oracle continuation-cost action-ranking regret on calibration frontier states."""
    regrets = {a: [] for a in ALL_LEARNED_ARCHS}
    for c in cal_cases[:24]:
        nb_cache = {}
        def nb_fn(w):
            if w not in nb_cache:
                nb_cache[w] = rg.neighbors(c, w)
            return nb_cache[w]
        s = m15.FringeState.start(c)
        for _ in range(2):
            if not s.live(WORK):
                break
            (sd, ix), _, _ = m15.geom_mm_select(rg, c, s)
            m15.expand_node(rg, c, s, sd, ix, nb_fn=nb_fn)
        if not s.live(WORK) or len(s.fronts[0]) + len(s.fronts[1]) <= 2:
            continue
        rem_cap = min(WORK - s.work, 48)
        sub_b = s.work + rem_cap
        costs = []
        for side in (0, 1):
            for idx in range(len(s.fronts[side])):
                zf = s.clone_light()
                m15.expand_node(rg, c, zf, side, idx, budget=sub_b, nb_fn=nb_fn)
                su = zf.work - s.work
                if zf.path is not None:
                    cf = su
                else:
                    cc, _, _ = m15.continuation_cost_geom(rg, c, zf, budget=sub_b, nb_fn=nb_fn)
                    cf = su + cc if cc <= rem_cap - su else rem_cap + 1
                costs.append(cf)
        min_c = min(costs)
        for arch in ALL_LEARNED_ARCHS:
            mod = models[(seed, arch)]
            with torch.no_grad():
                outs, _ = mod(c, s, WORK - s.work, mode="normal")
            sc = torch.cat([outs[-1]["s0"], outs[-1]["s1"]], dim=0)
            pick = int(sc.argmax().item())
            regrets[arch].append((costs[pick] - min_c) / (rem_cap + 1.0))
    return {a: (sum(v) / max(1, len(v))) for a, v in regrets.items()}


def verify_lean_theory_module(out_dir: Path) -> dict:
    """Compile standalone lean/LBF17AbstractInterp.lean in Lean 4.19.0 and verify zero axioms."""
    lean_src = Path(__file__).resolve().parent / "lean" / "LBF17AbstractInterp.lean"
    cfg = base.config
    version = subprocess.run([str(cfg.LEAN_EXE), "--version"], capture_output=True, text=True,
                             check=True, env=cfg.lean_env()).stdout.strip()
    if "version 4.19.0" not in version:
        raise ValueError("pinned Lean 4.19.0 required")
    res = subprocess.run([str(cfg.LEAN_EXE), str(lean_src)], capture_output=True, text=True,
                         timeout=120, env=cfg.lean_env())
    log = res.stdout + res.stderr
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lbf17_theory_compiler.log").write_text(log, encoding="utf-8")
    expected_theorems = {
        "abstract_iter_soundness",
        "best_transformer_soundness",
        "galois_pos1",
        "galois_adj2",
        "galois_sub3",
        "adj2_refines_pos1",
        "sub3_refines_adj2",
        "provenance_fibers_refine_merged",
        "repaired_lbf16_horizon_support_soundness",
        "nonbacktracking_provenance_soundness",
        "cex1_concrete_component_exact",
        "cex1_pos1_pass1_exact_pass2_spurious",
        "cex1_relational_and_provenance_exact",
        "cex2_adj2_spurious_sub3_and_prov_exact",
        "cex3_sub3_loses_global_consistency_prov_exact",
    }
    passed = set(re.findall(r"'LBF17AbstractInterp\.([A-Za-z0-9_]+)' does not depend on any axioms", log))
    if res.returncode != 0 or passed != expected_theorems:
        raise RuntimeError(f"LBF17AbstractInterp.lean verification failed: {log[-2000:]}")
    receipt = {
        "lean_version": version,
        "source_sha256": base.file_sha(lean_src),
        "log_sha256": base.file_sha(out_dir / "lbf17_theory_compiler.log"),
        "compiled_axiom_free": len(passed),
        "theorems": sorted(passed),
    }
    base.write(out_dir / "manifest.json", receipt)
    return receipt


def summarize_lbf17_probes(raw_records: list[dict]) -> dict:
    out = {}
    for tag in ("rel_prov_standard", "rel_prov_opened", "adj2_noprov_opened"):
        rr = [r for r in raw_records if r["tag"].endswith(tag)]
        if not rr:
            continue
        K = len(rr[0]["normal"])
        wg = [r["normal"][K - 1]["word_gate_strength"] for r in rr]
        bg = [r["normal"][K - 1]["bi_gate_strength"] for r in rr]
        fg = [r["normal"][K - 1]["fresco_gate_strength"] for r in rr]
        er_mat = [m15.erank_2d(r["normal"][K - 1]["bm_out"]) for r in rr]
        er_pos = [m15.erank_2d(r["normal"][K - 1]["bw_u_out"]) for r in rr]
        er_bi = [m15.erank_2d(r["normal"][K - 1]["bi_u_out"]) for r in rr if "bi_u_out" in r["normal"][K - 1]]
        sub_mi_p1 = [r["normal"][0]["rib_spatial_autocorr"] for r in rr]
        sub_mi_pk = [r["normal"][K - 1]["rib_spatial_autocorr"] for r in rr]
        supp_p1 = [r["normal"][0]["active_bi_support_frac"] for r in rr]
        supp_pk = [r["normal"][K - 1]["active_bi_support_frac"] for r in rr]
        bright = [float((r["normal"][K - 1]["pot"].sigmoid() - r["normal"][0]["pot"].sigmoid()).abs().mean()) for r in rr]

        ms = {
            "probed_decisions": len(rr),
            "passK_pos_gate_mean": sum(wg) / len(wg),
            "passK_bigram_gate_mean": sum(bg) / len(bg),
            "passK_fresco_gate_mean": sum(fg) / len(fg),
            "fresco_bank_erank_mean": sum(er_mat) / len(er_mat),
            "pos_envelope_erank_mean": sum(er_pos) / len(er_pos),
            "provenance_bigram_erank_mean": sum(er_bi) / max(1, len(er_bi)),
            "bigram_support_frac_pass1": sum(supp_p1) / len(supp_p1),
            "bigram_support_frac_passK": sum(supp_pk) / len(supp_pk),
            "subterm_morans_i_pass1": sum(sub_mi_p1) / len(sub_mi_p1),
            "subterm_morans_i_passK": sum(sub_mi_pk) / len(sub_mi_pk),
            "mean_abs_pass_brightening": sum(bright) / len(bright),
        }
        for md in SHAM_MODES:
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
        raise ValueError("internal lineage collision inside LBF-17 suite")
    historical = set()
    for h_seed, h_counts in HISTORICAL_SEEDS:
        h_cases, _ = rg.generate(h_seed, h_counts)
        for sp_cases in h_cases.values():
            historical.update(x.lineage for x in sp_cases)
    if current_lineages & historical:
        raise ValueError("lineage overlap detected with historical LBF/LSA suite")
    return len(historical)


def _serialize_frames_for_workers(frames: list[dict], dest: Path) -> None:
    def pack_st(st: m15.FringeState | None):
        if st is None:
            return None
        return {"fronts": st.fronts, "dist": st.dist, "work": st.work, "expanded": st.expanded}

    payload = [{
        "root": f["root"], "lineage": f["lineage"], "step": f["step"], "rollout": f["rollout"],
        "state": pack_st(f["state"]), "prev_state": pack_st(f["prev_state"]),
        "rem": f["rem"], "prev_rem": f["prev_rem"], "rem_cap": f["rem_cap"],
        "min_dr": f["min_dr"], "min_ell": f["min_ell"], "costs": f["costs"],
        "y_mat": f["y_mat"], "node_y": f["node_y"],
        "sub_u_y": f["sub_u_y"], "sub_v_y": f["sub_v_y"],
    } for f in frames]
    torch.save(payload, dest)


def _rehydrate_frames_for_worker(rg, suite: Path, src_pt: Path) -> list[dict]:
    train_cases, _ = rg.load_split(suite, "train")
    by_root = {c.root: c for c in train_cases}
    raw = torch.load(src_pt, weights_only=False)
    def unpack_st(d):
        if d is None:
            return None
        return m15.FringeState(d["fronts"], [{}, {}], d["dist"], d["work"], d["expanded"])

    return [{
        **r,
        "case": by_root[r["root"]],
        "state": unpack_st(r["state"]),
        "prev_state": unpack_st(r["prev_state"]),
    } for r in raw]


def run_fit_worker(upstream_path: Path, suite: Path, frames_pt: Path, jobs_spec: str, out: Path) -> None:
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    rg = base.load_upstream(upstream_path)
    frames = _rehydrate_frames_for_worker(rg, suite, frames_pt)
    for item in jobs_spec.split(","):
        if not item:
            continue
        sd_s, arch = item.split(":")
        sd = int(sd_s)
        mod, rec = fit(frames, sd, arch, out / f"{sd}_{arch}")
        del mod
        gc.collect()
        print(f"[LBF17] fit {sd}:{arch} ({rec['parameters']}p) in {rec['fit_wall_clock_sec']}s", flush=True)


# ============================================================================
# 6. Registered End-to-End LBF-17 Prospective Campaign
# ============================================================================
def campaign(rg, out: Path, threads: int = 2, upstream_path: Path | None = None,
             resume: bool = False) -> dict:
    if out.exists() and not resume:
        raise ValueError("fresh LBF-17 output directory required")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False

    lt_dir = out / "lean_theory_certificates"
    if resume and (lt_dir / "manifest.json").exists():
        theory_cert = json.loads((lt_dir / "manifest.json").read_text(encoding="utf-8"))
        if theory_cert.get("compiled_axiom_free") != 15:
            raise ValueError("invalid cached Lean theory certificate")
    else:
        theory_cert = verify_lean_theory_module(lt_dir)

    suite = out / "suite"
    if resume and (suite / "manifest.json").exists():
        sm = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    else:
        sm = rg.freeze(suite, SEED, COUNTS)
    if sm["counts"] != COUNTS or sm["unique_rule_system_lineages"] != sum(COUNTS.values()):
        raise ValueError("LBF-17 suite census mismatch")
    checked_historical = verify_historical_disjointness(rg, suite)

    all_jobs = [(s, a) for s in SEEDS for a in ALL_LEARNED_ARCHS]
    pending_jobs = []
    for sd, ar in all_jobs:
        mdir = out / f"{sd}_{ar}"
        if resume and (mdir / "manifest.json").exists() and (mdir / "checkpoint.pt").exists():
            rec = json.loads((mdir / "manifest.json").read_text(encoding="utf-8"))
            if (base.file_sha(mdir / "checkpoint.pt") != rec["checkpoint_sha256"]
                    or base.file_sha(mdir / "draws.jsonl") != rec["draws_sha256"]):
                raise ValueError(f"corrupted existing checkpoint for {sd}:{ar}")
            print(f"[LBF17] verified existing {sd}:{ar} ({rec['parameters']}p)", flush=True)
        else:
            if mdir.exists():
                for f in mdir.glob("*"):
                    f.unlink()
                mdir.rmdir()
            pending_jobs.append((sd, ar))

    if not pending_jobs and resume and (out / "teacher" / "manifest.json").exists():
        tm = json.loads((out / "teacher" / "manifest.json").read_text(encoding="utf-8"))
        if base.file_sha(out / "teacher" / "frames_summary.jsonl") != tm["frames_summary_sha256"]:
            raise ValueError("teacher frames_summary SHA-256 mismatch")
        frames = None
        print(f"[LBF17] teacher manifest verified ({tm['frames']} frames); all 24 checkpoints present", flush=True)
    else:
        frames, tm = teacher(rg, suite, out / "teacher", resume=resume)
        print(f"[LBF17] teacher ready: {len(frames)} frames, {tm['reference_certificates']} ref certs", flush=True)

    n_workers = max(1, min(2, threads))
    models = {}
    receipts = {}

    if pending_jobs:
        if n_workers == 2 and upstream_path is not None and len(pending_jobs) >= 2:
            # Spawn ONLY 1 child process for half the jobs while executing the other half in this
            # parent process -> strictly 2 Python/PyTorch processes total (<2.5 GB commit charge)!
            frames_pt = out / "teacher" / "_worker_frames.pt"
            _serialize_frames_for_workers(frames, frames_pt)
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            parent_jobs = pending_jobs[0::2]
            child_jobs = pending_jobs[1::2]
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--fit-worker",
                "--upstream", str(upstream_path),
                "--suite", str(suite),
                "--frames-pt", str(frames_pt),
                "--jobs", ",".join(f"{sd}:{ar}" for sd, ar in child_jobs),
                "--out", str(out),
            ]
            child_proc = subprocess.Popen(cmd, env=env)
            try:
                for sd, ar in parent_jobs:
                    mod, rec = fit(frames, sd, ar, out / f"{sd}_{ar}")
                    del mod
                    gc.collect()
                    print(f"[LBF17] fit {sd}:{ar} ({rec['parameters']}p) in {rec['fit_wall_clock_sec']}s", flush=True)
            finally:
                rc_child = child_proc.wait()
                frames_pt.unlink(missing_ok=True)
            if rc_child != 0:
                raise RuntimeError(f"child fit worker failed with exit code {rc_child}")
        else:
            for sd, ar in pending_jobs:
                mod, rec = fit(frames, sd, ar, out / f"{sd}_{ar}")
                del mod
                gc.collect()
                print(f"[LBF17] fit {sd}:{ar} ({rec['parameters']}p) in {rec['fit_wall_clock_sec']}s", flush=True)

    del frames
    gc.collect()

    for sd, ar in all_jobs:
        mdir = out / f"{sd}_{ar}"
        rec = json.loads((mdir / "manifest.json").read_text(encoding="utf-8"))
        mod, _ = new_lbf17_model(sd, ar)
        ckpt = torch.load(mdir / "checkpoint.pt", map_location="cpu", weights_only=True)
        mod.load_state_dict(ckpt["weights"])
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
        models[(sd, ar)] = mod
        receipts[(sd, ar)] = rec

    for seed in SEEDS:
        ref_draws = receipts[(seed, "rel_prov_opened")]["draws_sha256"]
        for arch in ALL_LEARNED_ARCHS:
            if receipts[(seed, arch)]["draws_sha256"] != ref_draws:
                raise ValueError(f"draw schedule diverged for {arch}")
        if receipts[(seed, "rel_prov_standard")]["shared_initial"] != receipts[(seed, "rel_prov_opened")]["shared_initial"]:
            raise ValueError("rel_prov_standard vs rel_prov_opened shared initialization mismatch")

    train_cases, _ = rg.load_split(suite, "train")
    cal_cases, _ = rg.load_split(suite, "calibration")
    if len(cal_cases) != COUNTS["calibration"] or ({c.lineage for c in cal_cases} & {c.lineage for c in train_cases}):
        raise ValueError("calibration roster or lineage disjointness failed")

    # Separate train-only and calibration abstraction-error & glyph-brightness diagnostics
    train_abs_diag, _ = evaluate_abstraction_and_glyphs(rg, train_cases, models, SEEDS[0], max_roots=32)
    cal_abs_diag, cal_root_pos1_fpr = evaluate_abstraction_and_glyphs(rg, cal_cases, models, SEEDS[0], max_roots=48)
    cal_regret = evaluate_action_ranking_regret(rg, cal_cases, models, SEEDS[0])

    rows = []
    raw_probes = []
    observations = []
    controls = ("astar", "bidir", "geom_mm", "graph_bridge", "exact_lookahead", "uniform")
    sham_conds = tuple(f"rel_prov_{md}" for md in SHAM_MODES)
    learned_conds = ALL_LEARNED_ARCHS + sham_conds

    for c in cal_cases:
        r_bidir = run_lbf17_episode(rg, c, "bidir")
        r_lbf14 = m15.lbf14.run(rg, c, "bidir", budget=WORK, depth=DEPTH)
        if any(r_bidir[k] != r_lbf14[k] for k in ("closed", "work", "expanded", "path")):
            raise ValueError("pinned bidir equivalence check failed")
        rows.append({**r_bidir, "condition": "bidir", "seed": None, "stream": None})

        for cname in ("astar", "geom_mm", "graph_bridge", "exact_lookahead"):
            rc = run_lbf17_episode(rg, c, cname)
            rows.append({**rc, "condition": cname, "seed": None, "stream": None})

        for stream in STREAMS:
            ru = run_lbf17_episode(rg, c, "uniform", seed=int(base.digest([stream, c.root])[:16], 16))
            rows.append({**ru, "condition": "uniform", "seed": None, "stream": stream})

        for seed in SEEDS:
            for arch in ALL_LEARNED_ARCHS:
                m = models[(seed, arch)]
                pctx = (
                    {"tag": f"{seed}:{arch}", "checkpoint": receipts[(seed, arch)]["checkpoint_sha256"],
                     "raw": raw_probes, "obs": observations}
                    if seed == SEEDS[0] and arch in ("rel_prov_standard", "rel_prov_opened", "adj2_noprov_opened")
                    else None
                )
                rl = run_lbf17_episode(rg, c, "learned", model=m, arch=arch, mode="normal", probe_ctx=pctx)
                rows.append({**rl, "condition": arch, "seed": seed, "stream": None})

            m_open = models[(seed, "rel_prov_opened")]
            for md in SHAM_MODES:
                rs = run_lbf17_episode(rg, c, "learned", model=m_open, arch="rel_prov_opened", mode=md)
                rows.append({**rs, "condition": f"rel_prov_{md}", "seed": seed, "stream": None})

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
    print(f"[LBF17] Lean 4.19 certified {cert['compiled_axiom_free']} calibration proofs axiom-free", flush=True)
    if cert["compiled_axiom_free"] != n_policy_proofs + len(cal_cases):
        raise ValueError("standalone Lean certificate count mismatch")
    cert_set = set(cert["declarations"])
    for i, row in enumerate(rows):
        row["native_certified"] = bool(row["closed"] and f"policy_{i}" in cert_set)
        row["penalized_work"] = row["work"] if row["native_certified"] else WORK + 1

    # Matched compute-budget curves across B in {16, 32, 64, 96, 128}
    budget_curves = {}
    curve_controls = ("astar", "bidir", "geom_mm", "graph_bridge", "exact_lookahead")
    for b_cap in BUDGET_CURVE:
        b_rates = {}
        b_pen = {}
        b_flops = {}
        b_ms = {}
        if b_cap == WORK:
            for cname in curve_controls + learned_conds:
                rr = [r for r in rows if r["condition"] == cname]
                b_rates[cname] = sum(int(r["native_certified"]) for r in rr) / len(rr)
                b_pen[cname] = sum(r["penalized_work"] for r in rr) / len(rr)
                b_flops[cname] = sum(r["neural_flops"] for r in rr) / len(rr)
                b_ms[cname] = sum(r["wall_clock_ms"] for r in rr) / len(rr)
        else:
            for cname in curve_controls:
                res_c = [run_lbf17_episode(rg, c, cname, budget=b_cap) for c in cal_cases]
                b_rates[cname] = sum(int(r["closed"]) for r in res_c) / len(cal_cases)
                b_pen[cname] = sum((r["work"] if r["closed"] else b_cap + 1) for r in res_c) / len(cal_cases)
                b_flops[cname] = 0.0
                b_ms[cname] = sum(r["wall_clock_ms"] for r in res_c) / len(cal_cases)
            for arch in ALL_LEARNED_ARCHS:
                cl_tot = pw_tot = fl_tot = ms_tot = 0.0
                for seed in SEEDS:
                    res_a = [run_lbf17_episode(rg, c, "learned", model=models[(seed, arch)],
                                               arch=arch, mode="normal", budget=b_cap) for c in cal_cases]
                    cl_tot += sum(int(r["closed"]) for r in res_a)
                    pw_tot += sum((r["work"] if r["closed"] else b_cap + 1) for r in res_a)
                    fl_tot += sum(r["neural_flops"] for r in res_a)
                    ms_tot += sum(r["wall_clock_ms"] for r in res_a)
                denom = len(SEEDS) * len(cal_cases)
                b_rates[arch] = cl_tot / denom
                b_pen[arch] = pw_tot / denom
                b_flops[arch] = fl_tot / denom
                b_ms[arch] = ms_tot / denom
            for md in SHAM_MODES:
                cname = f"rel_prov_{md}"
                cl_tot = pw_tot = fl_tot = ms_tot = 0.0
                for seed in SEEDS:
                    res_s = [run_lbf17_episode(rg, c, "learned", model=models[(seed, "rel_prov_opened")],
                                               arch="rel_prov_opened", mode=md, budget=b_cap) for c in cal_cases]
                    cl_tot += sum(int(r["closed"]) for r in res_s)
                    pw_tot += sum((r["work"] if r["closed"] else b_cap + 1) for r in res_s)
                    fl_tot += sum(r["neural_flops"] for r in res_s)
                    ms_tot += sum(r["wall_clock_ms"] for r in res_s)
                denom = len(SEEDS) * len(cal_cases)
                b_rates[cname] = cl_tot / denom
                b_pen[cname] = pw_tot / denom
                b_flops[cname] = fl_tot / denom
                b_ms[cname] = ms_tot / denom

        budget_curves[str(b_cap)] = {
            "closure_rate": b_rates,
            "mean_penalized_work": b_pen,
            "mean_neural_flops": b_flops,
            "mean_wall_clock_ms": b_ms,
        }
        print(f"[LBF17] budget B={b_cap} done: rel_prov={b_rates['rel_prov_opened']:.4f} lbf16={b_rates['lbf16_pos1_opened']:.4f} bidir={b_rates['bidir']:.4f}", flush=True)

    budget_auc = {
        cond_k: sum(budget_curves[str(b)]["closure_rate"][cond_k] for b in BUDGET_CURVE) / len(BUDGET_CURVE)
        for cond_k in budget_curves["128"]["closure_rate"]
    }

    # Conditional proof closure stratified by median A_1 false-positive rate across calibration roots
    sorted_fprs = sorted(cal_root_pos1_fpr.values())
    med_fpr = sorted_fprs[len(sorted_fprs) // 2]
    high_err_roots = {r for r, v in cal_root_pos1_fpr.items() if v > med_fpr}
    low_err_roots = {r for r in cal_root_pos1_fpr if r not in high_err_roots}
    cond_by_abs_err = {}
    for cond_k in ("bidir", "geom_mm", "exact_lookahead", "lbf16_pos1_opened",
                   "adj2_noprov_opened", "param_nonrecurrent", "compute_nonrecurrent", "rel_prov_opened"):
        rr_hi = [r for r in rows if r["condition"] == cond_k and r["root"] in high_err_roots]
        rr_lo = [r for r in rows if r["condition"] == cond_k and r["root"] in low_err_roots]
        cond_by_abs_err[cond_k] = {
            "low_pos1_error_closure": sum(int(r["native_certified"]) for r in rr_lo) / max(1, len(rr_lo)),
            "high_pos1_error_closure": sum(int(r["native_certified"]) for r in rr_hi) / max(1, len(rr_hi)),
            "low_pos1_error_pen_work": sum(r["penalized_work"] for r in rr_lo) / max(1, len(rr_lo)),
            "high_pos1_error_pen_work": sum(r["penalized_work"] for r in rr_hi) / max(1, len(rr_hi)),
        }

    base.write_jsonl(out / "episodes.jsonl", rows)
    base.write_jsonl(out / "observations.jsonl", observations)
    encoded = [base.encode(o) for o in observations]
    base.write_jsonl(out / "encoded.jsonl", encoded)
    torch.save(raw_probes, out / "raw_fresco_tensors.pt")
    base.write(out / "abstraction_diagnostics.json", {
        "train": train_abs_diag,
        "calibration": cal_abs_diag,
        "calibration_action_ranking_regret": cal_regret,
        "closure_conditional_on_pos1_abstraction_error": {
            "median_pos1_fpr": med_fpr,
            "low_error_roots": len(low_err_roots),
            "high_error_roots": len(high_err_roots),
            "conditions": cond_by_abs_err,
        },
    })
    (out / "fresco.svg").write_text(base.fresco_svg(encoded[0]), encoding="utf-8")

    open_probes = [r for r in raw_probes if r["tag"].endswith("rel_prov_opened")]
    showcase = max(open_probes, key=lambda r: len(r["u_words"]) * len(r["v_words"]))
    ms_svg = m15.render_manuscript_svg(
        lookup[showcase["root"]], showcase["u_words"], showcase["v_words"],
        [o["pot"] for o in showcase["normal"]],
        [o["s0"] for o in showcase["normal"]],
        [o["s1"] for o in showcase["normal"]],
        showcase["persisted_mask"],
    ).replace("LBF-15 · GEOMETRIC MEETING-POTENTIAL FRESCO",
              "LBF-17 · RELATIONAL PROVENANCE MEETING FRESCO")
    (out / "fresco_manuscript.svg").write_text(ms_svg, encoding="utf-8")

    all_names = controls + learned_conds
    rates = {}
    costs = {}
    seed_rates = {str(s): {} for s in SEEDS}
    seed_pen_work = {str(s): {} for s in SEEDS}
    for name in all_names:
        rr = [r for r in rows if r["condition"] == name]
        decs_tot = max(1, sum(r["fringe_decisions"] for r in rr))
        rates[name] = sum(int(r["native_certified"]) for r in rr) / len(rr)
        costs[name] = {
            "episodes": len(rr),
            "certified_closures": sum(int(r["native_certified"]) for r in rr),
            "work": sum(r["work"] for r in rr),
            "mean_work": sum(r["work"] for r in rr) / len(rr),
            "mean_penalized_work": sum(r["penalized_work"] for r in rr) / len(rr),
            "expanded": sum(r["expanded"] for r in rr),
            "model_calls": sum(r["model_calls"] for r in rr),
            "recurrent_passes": sum(r["recurrent_passes"] for r in rr),
            "pair_evaluations": sum(r["pair_evaluations"] for r in rr),
            "rule_window_checks": sum(r["substring_checks"] for r in rr),
            "total_neural_flops": sum(r["neural_flops"] for r in rr),
            "mean_neural_flops_per_decision": sum(r["neural_flops"] for r in rr) / decs_tot,
            "total_wall_clock_ms": round(sum(r["wall_clock_ms"] for r in rr), 2),
            "mean_wall_clock_ms_per_episode": round(sum(r["wall_clock_ms"] for r in rr) / len(rr), 3),
            "max_peak_tensor_bytes": max(r["peak_tensor_bytes"] for r in rr),
        }
    for seed in SEEDS:
        for name in learned_conds:
            rr = [r for r in rows if r["condition"] == name and r["seed"] == seed]
            seed_rates[str(seed)][name] = sum(int(r["native_certified"]) for r in rr) / len(rr)
            seed_pen_work[str(seed)][name] = sum(r["penalized_work"] for r in rr) / len(rr)

    def root_mean(c, name: str, key: str) -> float:
        rr = [r for r in rows if r["root"] == c.root and r["condition"] == name]
        expected = len(STREAMS) if name == "uniform" else (1 if name in controls else len(SEEDS))
        if len(rr) != expected:
            raise ValueError(f"incomplete root pairing for {name}")
        return sum(r[key] for r in rr) / len(rr)

    paired = {}
    for name in all_names:
        if name == "rel_prov_opened":
            continue
        q = [root_mean(c, "rel_prov_opened", "native_certified") - root_mean(c, name, "native_certified") for c in cal_cases]
        v = [root_mean(c, "rel_prov_opened", "penalized_work") - root_mean(c, name, "penalized_work") for c in cal_cases]
        paired["rel_prov_minus_" + name] = {
            "closure_mean": sum(q) / len(q),
            "closure_95": base.bootstrap(q),
            "penalized_work_mean": sum(v) / len(v),
            "penalized_work_95": base.bootstrap(v),
            "wins": sum(x > 0 for x in q),
            "losses": sum(x < 0 for x in q),
            "ties": sum(x == 0 for x in q),
        }

    # Evaluate pre-registered LBF-17 decision screens
    bidir_gain = rates["rel_prov_opened"] - rates["bidir"]
    bidir_save = 1.0 - costs["rel_prov_opened"]["mean_penalized_work"] / costs["bidir"]["mean_penalized_work"]
    screen_1a = (
        bidir_gain >= 0.15
        and bidir_save >= 0.30
        and all(seed_rates[str(s)]["rel_prov_opened"] > rates["bidir"] for s in SEEDS)
        and rates["rel_prov_opened"] > rates["exact_lookahead"]
        and budget_auc["rel_prov_opened"] > budget_auc["exact_lookahead"]
    )

    heur_names = ("geom_mm", "graph_bridge")
    best_heur = max(heur_names, key=lambda n: (rates[n], -costs[n]["mean_penalized_work"]))
    heur_gain = rates["rel_prov_opened"] - rates[best_heur]
    heur_save = 1.0 - costs["rel_prov_opened"]["mean_penalized_work"] / costs[best_heur]["mean_penalized_work"]
    screen_1b = (
        heur_gain >= 0.05
        and heur_save >= 0.20
        and all(seed_rates[str(s)]["rel_prov_opened"] > rates[best_heur] for s in SEEDS)
        and all(budget_curves[str(b)]["closure_rate"]["rel_prov_opened"]
                > budget_curves[str(b)]["closure_rate"][best_heur] for b in BUDGET_CURVE)
    )

    screen_2a = all(
        budget_auc["rel_prov_opened"] > budget_auc[other]
        and costs["rel_prov_opened"]["mean_penalized_work"] <= costs[other]["mean_penalized_work"] + 1e-9
        for other in ("param_nonrecurrent", "compute_nonrecurrent", "rel_prov_standard")
    )

    nrp_dom = cal_abs_diag["domains"]["neural_rel_prov"]
    n16_dom = cal_abs_diag["domains"]["neural_lbf16_pos1"]
    fp_reduction = 1.0 - (nrp_dom["symbol_fp"] / max(1, n16_dom["symbol_fp"]))
    glyph_auroc = cal_abs_diag["glyph_calibration"]["auroc_rel_prov_continuous"]
    screen_2b = (
        nrp_dom["symbol_fn"] == 0
        and fp_reduction >= 0.90
        and nrp_dom["symbol_precision"] >= 0.95
        and glyph_auroc >= 0.90
    )

    screen_2c = (
        all(budget_auc["rel_prov_opened"] >= budget_auc[f"rel_prov_{md}"] - 0.005 for md in SHAM_MODES)
        and all(budget_auc["rel_prov_opened"] > budget_auc[f"rel_prov_{md}"]
                for md in ("scramble_window", "ablate_axial", "env_roll"))
    )

    result = {
        "schema": "lbf17.relational_provenance_fresco.v1",
        "source_commit": base.UPSTREAM_COMMIT,
        "source_blob": base.UPSTREAM_BLOB,
        "generator_seed": SEED,
        "historical_lineages_verified_disjoint": checked_historical,
        "suite_manifest_sha256": base.file_sha(suite / "manifest.json"),
        "lean_theory_formalization": theory_cert,
        "training": tm,
        "fits": [receipts[k] for k in sorted(receipts)],
        "rates": rates,
        "seed_rates": seed_rates,
        "seed_penalized_work": seed_pen_work,
        "costs": costs,
        "paired": paired,
        "budget_curves": budget_curves,
        "budget_curve_mean_closure_auc": budget_auc,
        "strongest_explicit_heuristic": best_heur,
        "abstraction_summary_calibration": cal_abs_diag,
        "action_ranking_regret_calibration": cal_regret,
        "conditional_closure_by_pos1_error": cond_by_abs_err,
        "screens": {
            "primary_1a_vs_bidir_and_exact_lookahead": screen_1a,
            "primary_1b_vs_strongest_heuristic": screen_1b,
            "secondary_2a_recurrence_vs_param_and_compute_matched_nr": screen_2a,
            "secondary_2b_abstraction_precision_and_glyph_auroc": screen_2b,
            "secondary_2c_factored_causal_interventions": screen_2c,
        },
        "fresco_probe_summary": summarize_lbf17_probes(raw_probes),
        "policy_certificates": n_policy_proofs,
        "reference_certificates": len(cal_cases),
        "sealed_outcomes_opened": False,
        "lbf16_sealed_cohorts_preserved_unopened": True,
        "native_online_tactic_calls": 0,
        "artifacts": {
            x: base.file_sha(out / x)
            for x in ("episodes.jsonl", "observations.jsonl", "encoded.jsonl",
                      "raw_fresco_tensors.pt", "abstraction_diagnostics.json",
                      "fresco.svg", "fresco_manuscript.svg")
        },
    }
    base.write(out / "manifest.json", result)
    print("LBF17_RESULT", base.canonical(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fit-worker", action="store_true")
    p.add_argument("--suite", type=Path, default=None)
    p.add_argument("--frames-pt", type=Path, default=None)
    p.add_argument("--jobs", type=str, default="")
    args = p.parse_args()
    if args.fit_worker:
        run_fit_worker(args.upstream, args.suite, args.frames_pt, args.jobs, args.out)
        return
    campaign(
        base.load_upstream(args.upstream), args.out, threads=args.threads,
        upstream_path=args.upstream, resume=args.resume,
    )


if __name__ == "__main__":
    main()
