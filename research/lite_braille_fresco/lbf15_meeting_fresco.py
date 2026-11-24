"""LBF-15: Geometric meeting-potential fresco for bidirectional equational proof search.

Implements planar tree-fringe bidirectional search, cross-frontier potential matrix
M_ij^(t,k) = phi_theta(u_i, v_j, R, b), associative rule-transition recurrence,
2D axial + cousin-neighborhood fresco propagation, matched nonrecurrent/unstructured/gate
controls, causal memory & spatial scramble interventions, illuminated manuscript SVG
rendering, and standalone axiom-free Lean 4.19.0 certification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
import html
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

from lbf_gym import glyph
import lbf14_bidirectional_fresco as lbf14

base = lbf14.base
WORK = 128
DEPTH = 16
BUDGET_CURVE = (32, 64, 96, 128)
SEED = 20261215
HISTORICAL_SEEDS = (
    (20261204, {"train": 64, "calibration": 32, "ordinary": 32, "ood": 32}),
    (20261205, {"train": 24, "calibration": 16, "ordinary": 16, "ood": 16}),
    (20261207, {"train": 32, "calibration": 24, "ordinary": 16, "ood": 16}),
    (20261209, {"train": 48, "calibration": 32, "ordinary": 16, "ood": 16}),
)
COUNTS = {"train": 64, "calibration": 48, "ordinary": 16, "ood": 16}
SEEDS = (107, 109, 113)
STREAMS = (17, 31, 47, 61, 83)
STEPS = 480
PASSES = 3
ALPHABET = "abcdef"
CHAR2ID = {c: i for i, c in enumerate(ALPHABET)}
PAD_ID = 6

ARCHITECTURES = ("nonrecurrent", "unstructured_recurrent", "recurrent_standard", "recurrent_opened")
OPENED_MODES = ("normal", "zero", "roll", "scramble", "no_persist")


# ============================================================================
# 1. Planar Tree-Fringe Bidirectional Search State & Update Law
# ============================================================================
class FringeState:
    def __init__(self, fronts: list[list[str]], parent: list[dict], dist: list[dict[str, int]],
                 work: int = 0, expanded: int = 0, path: list | None = None,
                 prev_pot: dict[tuple[str, str], float] | None = None):
        self.fronts = fronts
        self.parent = parent
        self.dist = dist
        self.work = work
        self.expanded = expanded
        self.path = path
        self.prev_pot = {} if prev_pot is None else dict(prev_pot)

    @classmethod
    def start(cls, c) -> "FringeState":
        return cls([[c.source], [c.target]], [{c.source: None}, {c.target: None}],
                   [{c.source: 0}, {c.target: 0}])

    def live(self, budget: int = WORK) -> bool:
        return bool(self.path is None and self.work < budget and self.fronts[0] and self.fronts[1])

    def clone_light(self) -> "FringeState":
        return FringeState(
            [list(self.fronts[0]), list(self.fronts[1])],
            [dict(self.parent[0]), dict(self.parent[1])],
            [dict(self.dist[0]), dict(self.dist[1])],
            self.work, self.expanded,
            None if self.path is None else list(self.path),
            dict(self.prev_pot),
        )


def expand_node(rg, c, s: FringeState, side: int, idx: int,
                budget: int = WORK, depth: int = DEPTH, nb_fn=None) -> None:
    """Expand one open frontier state in canonical Move order; splice novel children in place."""
    if side not in (0, 1) or not (0 <= idx < len(s.fronts[side])) or not s.live(budget):
        raise ValueError("invalid frontier node intervention")
    word = s.fronts[side][idx]
    g = s.dist[side][word]
    if g >= depth:
        s.fronts[side].pop(idx)
        return
    s.expanded += 1
    novel = []
    moves = nb_fn(word) if nb_fn is not None else rg.neighbors(c, word)
    for mv in moves:
        if s.work >= budget:
            break
        s.work += 1
        nxt = mv.successor
        if nxt in s.dist[side]:
            continue
        s.dist[side][nxt] = g + 1
        s.parent[side][nxt] = (word, mv)
        if nxt in s.dist[1 - side] and g + 1 + s.dist[1 - side][nxt] <= depth:
            s.path = lbf14.meeting_path(rg, c, lbf14.State(s.fronts, s.parent, s.dist, s.work, s.expanded), nxt)
            rg.replay(c, s.path)
            return
        if g + 1 < depth:
            novel.append(nxt)
    s.fronts[side][idx:idx + 1] = novel


# ============================================================================
# 2. Public Pre-Expansion Observations & Explicit Geometric / Graph Heuristics
# ============================================================================
def public_rule_matches(c, word: str) -> tuple[int, int]:
    """Count matching rule substrings on an already-discovered word without generating successors."""
    matches = 0
    checks = 0
    L = len(word)
    for a, b in c.rules:
        la = len(a)
        for p in range(L - la + 1):
            checks += 1
            sub = word[p:p + la]
            if sub == a or sub == b:
                matches += 1
    return matches, checks


def geom_mm_select(rg, c, s: FringeState, budget: int = WORK, depth: int = DEPTH) -> tuple[tuple[int, int], int, int]:
    """Parameter-free Front-to-Front Geometric MM heuristic on discovered words only."""
    best_key = None
    best_act = (0, 0)
    pair_evals = len(s.fronts[0]) * len(s.fronts[1])
    for side in (0, 1):
        opp = s.fronts[1 - side]
        ms = len(s.fronts[side])
        for idx, w in enumerate(s.fronts[side]):
            g = s.dist[side][w]
            best_f = (999, 999, 999)
            for v in opp:
                gv = s.dist[1 - side][v]
                uw, vw = (w, v) if side == 0 else (v, w)
                dh = rg.hamming(uw, vw)
                lb = (dh + 2) // 3
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
                best_key = key
                best_act = (side, idx)
    return best_act, pair_evals, 0


def build_kmer_graph(c) -> dict[str, dict[str, int]]:
    adj = defaultdict(set)
    for a, b in c.rules:
        adj[a].add(b)
        adj[b].add(a)
    pdist = {}
    for src in adj:
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        pdist[src] = d
    return pdist


def graph_bridge_select(rg, c, s: FringeState, pdist: dict[str, dict[str, int]],
                        budget: int = WORK, depth: int = DEPTH) -> tuple[tuple[int, int], int, int]:
    """Parameter-free Graph-Theoretic Meeting Heuristic: k-mer rule-graph shortest-path conductance."""
    best_tuple = None
    best_act = (0, 0)
    pair_evals = len(s.fronts[0]) * len(s.fronts[1])
    sub_checks = 0
    deg_cache = [{}, {}]
    for side in (0, 1):
        for w in s.fronts[side]:
            if w not in deg_cache[side]:
                deg_w, ch = public_rule_matches(c, w)
                deg_cache[side][w] = deg_w
                sub_checks += ch

    for side in (0, 1):
        opp = s.fronts[1 - side]
        ms = len(s.fronts[side])
        for idx, w in enumerate(s.fronts[side]):
            g = s.dist[side][w]
            deg_w = deg_cache[side][w]
            best_pair = -1e18
            for v in opp:
                gv = s.dist[1 - side][v]
                uw, vw = (w, v) if side == 0 else (v, w)
                dh = rg.hamming(uw, vw)
                lb = (dh + 2) // 3
                if g + gv + max(1, lb) > depth:
                    continue
                cond = 0.0
                for L in (2, 3):
                    for p in range(len(uw) - L + 1):
                        su, sv = uw[p:p + L], vw[p:p + L]
                        if su != sv and su in pdist and sv in pdist[su]:
                            cond += 1.0 / pdist[su][sv]
                deg_v = deg_cache[1 - side][v]
                sc = -(g + gv + lb) - dh / len(uw) + cond / math.sqrt(1.0 + deg_w + deg_v)
                if sc > best_pair:
                    best_pair = sc
            node_sc = best_pair - math.log1p(ms) / math.log1p(budget) if best_pair > -1e17 else -1e18
            cand = (node_sc, -ms, -side, -idx)
            if best_tuple is None or cand > best_tuple:
                best_tuple = cand
                best_act = (side, idx)
    return best_act, pair_evals, sub_checks


# ============================================================================
# 3. Geometric Meeting-Potential Fresco Architecture (~55.9K params)
# ============================================================================
class MessageGate(nn.Module):
    def __init__(self, d: int, init_bias: float = -2.0):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, x: torch.Tensor, msg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        strength = torch.sigmoid(self.gate(torch.cat([x, msg], dim=-1)))
        return strength * self.proj(msg), strength


class RuleTransitionCell(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, rule_k: torch.Tensor, rule_v: torch.Tensor,
                prev_bank: torch.Tensor, scramble_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if scramble_pos is not None:
            inv = torch.argsort(scramble_pos)
            loc = F.gelu(self.conv(x[:, scramble_pos, :].transpose(1, 2)).transpose(1, 2))[:, inv, :]
        else:
            loc = F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        q = self.q_proj(loc)
        attn = torch.softmax((q @ rule_k.T) / math.sqrt(q.shape[-1]), dim=-1)
        rule_msg = attn @ rule_v
        h = x + self.ff(x + loc + 0.5 * rule_msg)
        bank_next = self.out_norm(rule_msg + 0.5 * prev_bank)
        return h, bank_next


class CrossFrontierFrescoBlock(nn.Module):
    def __init__(self, dm: int, structured: bool = True):
        super().__init__()
        self.structured = structured
        if structured:
            self.spatial = nn.Conv2d(dm, dm, kernel_size=3, padding=1, groups=2)
            self.row_q = nn.Linear(dm, dm, bias=False)
            self.row_kv = nn.Linear(dm, 2 * dm, bias=False)
            self.col_q = nn.Linear(dm, dm, bias=False)
            self.col_kv = nn.Linear(dm, 2 * dm, bias=False)
            self.mix = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        else:
            self.pointwise = nn.Sequential(
                nn.LayerNorm(dm), nn.Linear(dm, 230), nn.GELU(), nn.Linear(230, dm),
            )
        self.msg_norm = nn.LayerNorm(dm)

    def forward(self, h: torch.Tensor, prev_bank: torch.Tensor,
                scramble_rc: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.structured:
            upd = self.pointwise(h)
            return h + upd, self.msg_norm(upd + 0.5 * prev_bank)
        dm = h.shape[-1]
        rq = self.row_q(h)
        rk, rv = self.row_kv(h).chunk(2, dim=-1)
        r_ctx = torch.softmax((rq @ rk.transpose(-1, -2)) / math.sqrt(dm), dim=-1) @ rv

        ht = h.transpose(0, 1)
        cq = self.col_q(ht)
        ck, cv = self.col_kv(ht).chunk(2, dim=-1)
        c_ctx = (torch.softmax((cq @ ck.transpose(-1, -2)) / math.sqrt(dm), dim=-1) @ cv).transpose(0, 1)

        if scramble_rc is not None:
            pr, pc = scramble_rc
            ir, ic = torch.argsort(pr), torch.argsort(pc)
            sp = F.gelu(self.spatial(h[pr][:, pc].permute(2, 0, 1).unsqueeze(0))).squeeze(0).permute(1, 2, 0)[ir][:, ic]
        else:
            sp = F.gelu(self.spatial(h.permute(2, 0, 1).unsqueeze(0))).squeeze(0).permute(1, 2, 0)

        rel_msg = 0.5 * (r_ctx + c_ctx) + sp
        h_next = h + self.mix(h + rel_msg)
        return h_next, self.msg_norm(rel_msg + 0.5 * prev_bank)


class MeetingPotentialFrescoNet(nn.Module):
    def __init__(self, arch: str = "recurrent_opened", d: int = 40, dm: int = 32, passes: int = PASSES):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch}")
        self.arch = arch
        self.d = d
        self.dm = dm
        self.nonrecurrent = (arch == "nonrecurrent")
        self.structured = (arch != "unstructured_recurrent")
        self.gate_bias = -6.0 if arch == "recurrent_standard" else -2.0
        self.passes = 1 if self.nonrecurrent else passes

        self.sym_emb = nn.Embedding(7, d)
        self.pos_emb = nn.Embedding(16, d)
        self.side_emb = nn.Embedding(2, d)
        self.rule_lhs = nn.Linear(3 * d + 1, d)
        self.rule_rhs = nn.Linear(3 * d + 1, d)
        self.word_cell = RuleTransitionCell(d)
        self.word_to_dm = nn.Linear(d, dm, bias=False)
        self.pair_in = nn.Linear(dm + 12, dm)
        self.pair_prelude = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.GELU(), nn.Linear(2 * dm, dm))
        self.fresco_block = CrossFrontierFrescoBlock(dm, structured=self.structured)

        if self.nonrecurrent:
            self.nr_extra = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 122), nn.GELU(), nn.Linear(122, dm))
        else:
            self.word_gate = MessageGate(d, init_bias=-6.0)
            self.fresco_gate = MessageGate(dm, init_bias=-6.0)

        self.pot_head = nn.Linear(dm, 1)
        self.node_head = nn.Sequential(nn.Linear(d + 6, d), nn.GELU(), nn.Linear(d, 1))
        self.log_tau = nn.Parameter(torch.tensor(-0.7))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_rules(self, rules: tuple[tuple[str, str], ...]) -> tuple[torch.Tensor, torch.Tensor]:
        lhs_list, rhs_list = [], []
        for a, b in rules:
            for src, dst in ((a, b), (b, a)):
                ids_s = [CHAR2ID[ch] for ch in src] + ([PAD_ID] if len(src) == 2 else [])
                ids_d = [CHAR2ID[ch] for ch in dst] + ([PAD_ID] if len(dst) == 2 else [])
                es = self.sym_emb(torch.tensor(ids_s)).reshape(-1)
                ed = self.sym_emb(torch.tensor(ids_d)).reshape(-1)
                lf = es.new_tensor([len(src) / 3.0])
                lhs_list.append(torch.cat([es, lf]))
                rhs_list.append(torch.cat([ed - es, lf]))
        return torch.tanh(self.rule_lhs(torch.stack(lhs_list))), self.rule_rhs(torch.stack(rhs_list))

    def forward(self, case, s: FringeState, rem_budget: int, *, mode: str = "normal",
                depth_cap: int = DEPTH, return_tensors: bool = False) -> tuple[list[dict], int]:
        if mode not in OPENED_MODES:
            raise ValueError(f"unsupported intervention mode {mode}")
        u_words, v_words = s.fronts[0], s.fronts[1]
        m0, m1 = len(u_words), len(v_words)
        L = len(case.source)
        rule_k, rule_v = self.encode_rules(case.rules)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        u_h = self.sym_emb(torch.tensor([[CHAR2ID[ch] for ch in w] for w in u_words])) + self.pos_emb(pos) + self.side_emb(torch.tensor(0))
        v_h = self.sym_emb(torch.tensor([[CHAR2ID[ch] for ch in w] for w in v_words])) + self.pos_emb(pos) + self.side_emb(torch.tensor(1))

        pub = torch.zeros((m0, m1, 12), dtype=torch.float32)
        infeas = torch.zeros((m0, m1), dtype=torch.bool)
        b_frac = rem_budget / WORK
        lm0, lm1 = math.log1p(m0) / math.log1p(WORK), math.log1p(m1) / math.log1p(WORK)
        for i, uw in enumerate(u_words):
            g0 = s.dist[0][uw]
            for j, vw in enumerate(v_words):
                g1 = s.dist[1][vw]
                diffs = [p for p, (ca, cb) in enumerate(zip(uw, vw)) if ca != cb]
                dh = len(diffs)
                lb = (dh + 2) // 3
                span = (diffs[-1] - diffs[0] + 1) if diffs else 0
                tot_lb = g0 + g1 + max(1, lb)
                bad = tot_lb > depth_cap
                infeas[i, j] = bad
                pers, prev_v = 0.0, 0.0
                if mode != "no_persist" and (uw, vw) in s.prev_pot:
                    pers, prev_v = 1.0, s.prev_pot[(uw, vw)]
                pub[i, j] = torch.tensor([
                    g0 / depth_cap, g1 / depth_cap, (g0 + g1) / depth_cap,
                    dh / L, lb / depth_cap, max(0, depth_cap - tot_lb) / depth_cap,
                    1.0 - float(bad), span / L, b_frac, lm0 - lm1, pers, prev_v,
                ], dtype=torch.float32)

        sub_checks = 0
        def node_pub(words, side):
            nonlocal sub_checks
            ms, mo = len(words), len(s.fronts[1 - side])
            goal = case.target if side == 0 else case.source
            rows = []
            for w in words:
                deg, ch = public_rule_matches(case, w)
                sub_checks += ch
                rows.append([
                    s.dist[side][w] / depth_cap, deg / 24.0,
                    math.log1p(ms) / 5.0, math.log1p(mo) / 5.0, b_frac,
                    sum(ca != cb for ca, cb in zip(w, goal)) / L,
                ])
            return torch.tensor(rows, dtype=torch.float32)

        np0, np1 = node_pub(u_words, 0), node_pub(v_words, 1)
        scramble_pos = scramble_rc = None
        if mode == "scramble":
            gen = torch.Generator().manual_seed(1337 + m0 * 31 + m1)
            scramble_pos = torch.randperm(L, generator=gen)
            scramble_rc = (torch.randperm(m0, generator=gen), torch.randperm(m1, generator=gen))

        bw_u = torch.zeros_like(u_h)
        bw_v = torch.zeros_like(v_h)
        bm = torch.zeros((m0, m1, self.dm), dtype=torch.float32)
        h_mat = None
        outs = []

        for k in range(self.passes):
            if self.nonrecurrent:
                u_h, bw_u = self.word_cell(u_h, rule_k, rule_v, bw_u, scramble_pos=scramble_pos)
                v_h, bw_v = self.word_cell(v_h, rule_k, rule_v, bw_v, scramble_pos=scramble_pos)
                align = torch.einsum("ild,jld->ijd", self.word_to_dm(u_h), self.word_to_dm(v_h)) / L
                h_mat = self.pair_in(torch.cat([align, pub], dim=-1))
                h_mat = h_mat + self.pair_prelude(h_mat)
                h_mat, bm = self.fresco_block(h_mat, bm, scramble_rc=scramble_rc)
                h_mat = h_mat + self.nr_extra(h_mat)
                su_gate = sv_gate = sm_gate = gu = gm = None
                mu_in = mm_in = None
            else:
                mu_in, mv_in, mm_in = bw_u, bw_v, bm
                if k > 0:
                    if mode == "zero":
                        mu_in = torch.zeros_like(mu_in)
                        mv_in = torch.zeros_like(mv_in)
                        mm_in = torch.zeros_like(mm_in)
                    elif mode == "roll":
                        mu_in = mu_in.roll(1, dims=1)
                        mv_in = mv_in.roll(1, dims=1)
                        mm_in = mm_in.roll(shifts=(1, 1), dims=(0, 1))
                gu, su_gate = self.word_gate(u_h, mu_in)
                gv, sv_gate = self.word_gate(v_h, mv_in)
                u_h, bw_u = self.word_cell(u_h + gu, rule_k, rule_v, mu_in, scramble_pos=scramble_pos)
                v_h, bw_v = self.word_cell(v_h + gv, rule_k, rule_v, mv_in, scramble_pos=scramble_pos)
                align = torch.einsum("ild,jld->ijd", self.word_to_dm(u_h), self.word_to_dm(v_h)) / L
                pbase = self.pair_in(torch.cat([align, pub], dim=-1))
                pbase = pbase + self.pair_prelude(pbase)
                h_mat = pbase if h_mat is None else (h_mat + 0.5 * pbase)
                gm, sm_gate = self.fresco_gate(h_mat, mm_in)
                h_mat, bm = self.fresco_block(h_mat + gm, mm_in, scramble_rc=scramble_rc)

            pot = self.pot_head(h_mat).squeeze(-1)
            pot_masked = pot.masked_fill(infeas, -12.0)
            tau = F.softplus(self.log_tau) + 0.1
            s0 = (tau * torch.logsumexp(pot_masked / tau, dim=1) - tau * math.log(m1)
                  + self.node_head(torch.cat([u_h.mean(1), np0], dim=-1)).squeeze(-1)).masked_fill(infeas.all(dim=1), -12.0)
            s1 = (tau * torch.logsumexp(pot_masked / tau, dim=0) - tau * math.log(m0)
                  + self.node_head(torch.cat([v_h.mean(1), np1], dim=-1)).squeeze(-1)).masked_fill(infeas.all(dim=0), -12.0)

            row = {"pot": pot, "pot_masked": pot_masked, "s0": s0, "s1": s1, "infeas": infeas}
            if return_tensors:
                row["h_mat"] = h_mat.detach().cpu().clone()
                row["bm_out"] = bm.detach().cpu().clone()
                row["bw_u_out"] = bw_u.detach().cpu().clone()
                if not self.nonrecurrent:
                    row["bm_in"] = mm_in.detach().cpu().clone()
                    row["bw_u_in"] = mu_in.detach().cpu().clone()
                    row["word_gate_strength"] = 0.5 * (float(su_gate.detach().mean()) + float(sv_gate.detach().mean()))
                    row["fresco_gate_strength"] = float(sm_gate.detach().mean())
                    row["fresco_gate_contrib_rms"] = base.rms(gm)
                    row["word_gate_contrib_rms"] = base.rms(gu)
            outs.append(row)
        return outs, sub_checks


def new_model(seed: int, arch: str) -> tuple[MeetingPotentialFrescoNet, str]:
    torch.manual_seed(seed)
    m = MeetingPotentialFrescoNet(arch=arch).cpu().float()
    if m.parameter_count() > 57475:
        raise ValueError(f"architecture {arch} exceeds 57,475 parameter cap: {m.parameter_count()}")
    shared_fp = base._model_fingerprint(m)
    if not m.nonrecurrent:
        for gmod in (m.word_gate, m.fresco_gate):
            if not torch.all(gmod.gate.bias == -6.0):
                raise ValueError("expected base gate bias -6.0")
            if arch != "recurrent_standard":
                with torch.no_grad():
                    gmod.gate.bias.fill_(-2.0)
    return m, shared_fp


# ============================================================================
# 4. Spatial Coherence (Moran's I), Fresco Encoding & Illuminated Manuscript SVG
# ============================================================================
def morans_i(mat: torch.Tensor) -> float:
    """4-neighbor rook spatial autocorrelation Moran's I on 2D matrix."""
    x = mat.detach().float()
    m0, m1 = x.shape
    if m0 * m1 < 4:
        return 0.0
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 1e-12:
        return 0.0
    num = 0.0
    w_sum = 0
    if m0 > 1:
        num += 2.0 * float((xc[:-1, :] * xc[1:, :]).sum())
        w_sum += 2 * (m0 - 1) * m1
    if m1 > 1:
        num += 2.0 * float((xc[:, :-1] * xc[:, 1:]).sum())
        w_sum += 2 * m0 * (m1 - 1)
    if w_sum == 0:
        return 0.0
    return float((m0 * m1 / w_sum) * (num / denom))


def erank_2d(mat: torch.Tensor) -> float:
    x = mat.detach().float()
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if min(x.shape) == 0:
        return 0.0
    s = torch.linalg.svdvals(x)
    p = s.square()
    tot = float(p.sum())
    if tot <= 1e-12:
        return 0.0
    p = p / tot
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def build_fresco_observation(case, s: FringeState, outs: list[dict], checkpoint_sha: str, ordinal: int) -> dict:
    """Label-free canonical lbf.observation.v1 record encoding candidate utilities, polarity & fresco memory."""
    cands = [f"u{i}:{w}" for i, w in enumerate(s.fronts[0])] + [f"v{j}:{w}" for j, w in enumerate(s.fronts[1])]
    if len(cands) > 64:
        cands = cands[:64]
    ident = base.digest([case.root, checkpoint_sha, ordinal, cands])
    src_sha = base.digest([case.source, case.target, [list(r) for r in case.rules], cands, s.work])
    record = {
        "schema": base.SCHEMA_OBS,
        "root_id": "meeting:" + ident,
        "lineage": case.lineage,
        "split": "calibration",
        "source": {"kind": "ppt_replay", "source_sha256": src_sha, "checkpoint_sha256": checkpoint_sha},
        "candidates": cands,
        "visits": [],
    }
    p1_pot = outs[0]["pot"].detach()
    p1_scores = torch.cat([outs[0]["s0"].detach(), outs[0]["s1"].detach()], dim=0)
    for k, o in enumerate(outs):
        pot_sig = o["pot"].detach().sigmoid()
        scores = torch.cat([o["s0"].detach(), o["s1"].detach()], dim=0)
        probs = torch.softmax(scores, dim=0)
        bm_rms = base.rms(o["bm_out"])
        bw_rms = base.rms(o["bw_u_out"])
        mi = morans_i(pot_sig)
        mi_act = min(1.0, max(0.0, 0.5 * (mi + 1.0)))
        fg = o.get("fresco_gate_strength", 0.0)
        tiles = [
            {"candidate_id": "shared", "role": "goal", "frame": "state",
             "activation": float(pot_sig.max()), "polarity": 0},
            {"candidate_id": "shared", "role": "lemma", "frame": "closure",
             "activation": mi_act, "polarity": 1 if mi > 0.05 else (-1 if mi < -0.05 else 0)},
            {"candidate_id": "shared", "role": "plan", "frame": "gate",
             "activation": min(1.0, max(0.0, fg)), "polarity": 1 if k > 0 and fg > 0.05 else 0},
            {"candidate_id": "shared", "role": "message", "frame": "memory",
             "activation": bm_rms / (1.0 + bm_rms), "polarity": 0},
            {"candidate_id": "shared", "role": "context", "frame": "memory",
             "activation": bw_rms / (1.0 + bw_rms), "polarity": 0},
        ]
        for idx, cid in enumerate(cands):
            delta = float(scores[idx] - p1_scores[idx]) if k > 0 else 0.0
            pol = 1 if delta > 0.05 else (-1 if delta < -0.05 else 0)
            tiles.append({
                "candidate_id": cid,
                "role": "value",
                "frame": "action",
                "activation": min(1.0, max(0.0, float(probs[idx]))),
                "polarity": pol,
            })
        record["visits"].append({"attempt": 1, "pass": k + 1, "tiles": tiles})
    base.validate_observation(record)
    return record


def render_manuscript_svg(case, u_words: list[str], v_words: list[str],
                          pass_pots: list[torch.Tensor], pass_s0: list[torch.Tensor],
                          pass_s1: list[torch.Tensor], persisted_mask: list[list[bool]]) -> str:
    """Render an illuminated 2D meeting-potential manuscript plate across recurrent passes k=1..K."""
    m0, m1 = len(u_words), len(v_words)
    K = len(pass_pots)
    cell = 34
    pad_left, pad_top = 138, 128
    panel_w = pad_left + m1 * cell + 36
    total_w = 36 + K * panel_w + 20
    total_h = pad_top + m0 * cell + 96
    braille_pts = ((-5, -8), (-5, -2), (-5, 4), (5, -8), (5, -2), (5, 4), (-5, 10), (5, 10))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" width="{total_w}" height="{total_h}" '
        f'role="img" aria-label="LBF-15 Geometric Meeting-Potential Fresco for {html.escape(case.root, quote=True)}">',
        f'<rect width="{total_w}" height="{total_h}" fill="#1f2426"/>',
        f'<rect x="10" y="10" width="{total_w-20}" height="{total_h-20}" rx="10" fill="#cbbba0" stroke="#8c7853" stroke-width="2.5"/>',
        f'<rect x="18" y="18" width="{total_w-36}" height="{total_h-36}" rx="6" fill="none" stroke="#6e5d3f" stroke-width="1" stroke-dasharray="4,3"/>',
        '<text x="34" y="48" font-family="Georgia, serif" font-size="20" font-weight="bold" fill="#231f1a">'
        'LBF-15 · GEOMETRIC MEETING-POTENTIAL FRESCO</text>',
        f'<text x="34" y="70" font-family="monospace" font-size="11.5" fill="#3f382e">'
        f'root={html.escape(case.root)} | src={html.escape(case.source)} → dst={html.escape(case.target)} | '
        f'fringe={m0}×{m1} | predecision observable</text>',
    ]
    p1 = pass_pots[0].detach()
    for k in range(K):
        ox = 24 + k * panel_w
        pot = pass_pots[k].detach()
        sig = pot.sigmoid()
        mi = morans_i(sig)
        p0_prob = torch.softmax(pass_s0[k].detach(), dim=0)
        p1_prob = torch.softmax(pass_s1[k].detach(), dim=0)
        parts.append(
            f'<text x="{ox + pad_left - 30}" y="96" font-family="Georgia, serif" font-size="13" font-weight="bold" fill="#2b2620">'
            f'Pass {k+1} (Moran I={mi:+.2f})</text>'
        )
        for j, vw in enumerate(v_words):
            cx = ox + pad_left + j * cell
            act_v = float(p1_prob[j])
            parts.append(
                f'<rect x="{cx+2}" y="{pad_top-22}" width="{cell-4}" height="16" rx="3" '
                f'fill="#355c7d" fill-opacity="{0.2 + 0.8*act_v:.2f}" stroke="#5b5344"/>'
            )
            parts.append(
                f'<text x="{cx+5}" y="{pad_top-10}" font-family="monospace" font-size="8.5" fill="#1e1b16">{html.escape(vw[:3])}</text>'
            )
        for i, uw in enumerate(u_words):
            ry = pad_top + i * cell
            act_u = float(p0_prob[i])
            parts.append(
                f'<rect x="{ox+12}" y="{ry+3}" width="{pad_left-20}" height="{cell-6}" rx="3" '
                f'fill="#355c7d" fill-opacity="{0.15 + 0.75*act_u:.2f}" stroke="#6c6252"/>'
            )
            parts.append(
                f'<text x="{ox+16}" y="{ry+20}" font-family="monospace" font-size="9.5" fill="#1e1b16">'
                f'u{i}:{html.escape(uw[:8])}</text>'
            )
            for j, vw in enumerate(v_words):
                cx = ox + pad_left + j * cell
                a_val = float(sig[i, j])
                led = min(3, int(4 * a_val))
                delta = float(pot[i, j] - p1[i, j]) if k > 0 else 0.0
                pol = 1 if delta > 0.05 else (-1 if delta < -0.05 else 0)
                pers = persisted_mask[i][j]
                g_char = glyph("lemma" if led >= 2 else "value", "closure" if pers else "state", pol)
                mask = ord(g_char) - 0x2800
                bg = "#efe4c8" if not pers else "#f5ebd0"
                stroke = "#b8860b" if pers else "#8d8068"
                sw = "1.8" if pers else "1.0"
                parts.append(
                    f'<rect x="{cx+1}" y="{ry+1}" width="{cell-2}" height="{cell-2}" rx="4" '
                    f'fill="{bg}" stroke="{stroke}" stroke-width="{sw}"/>'
                )
                dot_col = "#2e8b57" if pol == 1 else ("#b25035" if pol == -1 else "#2f7f88")
                op = [0.16, 0.42, 0.72, 1.0][led]
                for b_idx, (dx, dy) in enumerate(braille_pts):
                    on = bool(mask & (1 << b_idx))
                    parts.append(
                        f'<circle cx="{cx + cell/2 + dx:.1f}" cy="{ry + cell/2 + dy:.1f}" '
                        f'r="{2.3 if on else 1.1}" fill="{dot_col if on else "#9c927e"}" opacity="{op if on else 0.28}"/>'
                    )
                parts.append(
                    f'<title>({html.escape(uw)}, {html.escape(vw)}) pass={k+1} pot={a_val:.3f} '
                    f'led={led} pol={pol} persisted={int(pers)}</title>'
                )
    parts.append(
        f'<text x="34" y="{total_h-28}" font-family="Georgia, serif" font-size="11.5" fill="#2f2a22">'
        'Gold border = cross-step persistence; Emerald/Terracotta/Teal = recurrent polarity (+1/-1/0); '
        'Dot code = injective Braille glyph; Lean verification external.</text>'
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


# ============================================================================
# 5. Train-Only Supervision (Pairwise Bridge Geodesic + Marginal Frontier Utility)
# ============================================================================
def bounded_balls(c, words: list[str], nb_fn, radius: int = 3) -> list[dict[str, int]]:
    balls = []
    for w in words:
        d = {w: 0}
        q = deque([w])
        while q:
            cur = q.popleft()
            if d[cur] >= radius:
                continue
            for mv in nb_fn(cur):
                nx = mv.successor
                if nx not in d:
                    d[nx] = d[cur] + 1
                    q.append(nx)
        balls.append(d)
    return balls


def continuation_cost_geom(rg, c, s: FringeState, budget: int, depth: int = DEPTH, nb_fn=None) -> tuple[int, int, list | None]:
    z = s.clone_light()
    start_w = z.work
    rem = budget - start_w
    while z.live(budget):
        (side, idx), _, _ = geom_mm_select(rg, c, z, budget, depth)
        expand_node(rg, c, z, side, idx, budget, depth, nb_fn=nb_fn)
    used = z.work - start_w
    if not 0 <= used <= rem:
        raise ValueError("unaccounted continuation work")
    if z.path is not None:
        rg.replay(c, z.path)
        if len(z.path) > depth:
            raise ValueError("continuation path exceeded cap")
    return (used if z.path is not None else rem + 1), used, z.path


def teacher(rg, suite: Path, out: Path) -> tuple[list[dict], dict]:
    cases, m, refs = rg.load_split(suite, "train", witnesses=True)
    if len(cases) != COUNTS["train"] or m["seed"] != SEED:
        raise ValueError("unexpected LBF-15 training roster")
    out.mkdir(parents=True)
    ref_cert = base.certify(rg, [(c, refs[c.root], "train_" + c.root) for c in cases], out / "references")

    frames = []
    audits = []
    oracle_work = 0
    cost_ties = 0
    mixed_closures = 0
    pos_marg_count = 0
    neg_marg_count = 0
    bridge_pairs_total = 0
    pairs_total = 0
    rng = random.Random(SEED + 15)

    for c in cases:
        nb_cache: dict[str, tuple] = {}
        def nb_fn(word: str):
            if word not in nb_cache:
                nb_cache[word] = rg.neighbors(c, word)
            return nb_cache[word]

        for rollout in ("geom", "explore"):
            s = FringeState.start(c)
            prev_snap = None
            prev_rem = WORK
            step = 0
            while s.live(WORK) and step < 10:
                step += 1
                m0, m1 = len(s.fronts[0]), len(s.fronts[1])
                if m0 + m1 > 2 and m0 <= 28 and m1 <= 28 and (step <= 6 or step % 2 == 0):
                    rem = WORK - s.work
                    rem_cap = min(rem, 64)
                    sub_budget = s.work + rem_cap

                    # 1. Exact bounded pairwise rewrite bridge distance d_R*(u_i, v_j) <= 6
                    bu = bounded_balls(c, s.fronts[0], nb_fn, radius=3)
                    bv = bounded_balls(c, s.fronts[1], nb_fn, radius=3)
                    inv_v = defaultdict(list)
                    for j, dmap in enumerate(bv):
                        for w, dv in dmap.items():
                            inv_v[w].append((j, dv))
                    y_mat = [[0.0] * m1 for _ in range(m0)]
                    best_node_d = [[99] * m0, [99] * m1]
                    for i, dmap in enumerate(bu):
                        g0 = s.dist[0][s.fronts[0][i]]
                        for w, du in dmap.items():
                            for j, dv in inv_v.get(w, ()):
                                g1 = s.dist[1][s.fronts[1][j]]
                                dr = max(1, du + dv)
                                if g0 + dr + g1 <= DEPTH:
                                    val = math.exp(-0.45 * (dr - 1))
                                    if val > y_mat[i][j]:
                                        y_mat[i][j] = val
                                    tot_d = g0 + dr + g1
                                    if tot_d < best_node_d[0][i]:
                                        best_node_d[0][i] = tot_d
                                    if tot_d < best_node_d[1][j]:
                                        best_node_d[1][j] = tot_d
                    pairs_total += m0 * m1
                    bridge_pairs_total += sum(v > 0 for row in y_mat for v in row)

                    # 2. Fixed-policy continuation cost C_pi0(S, a) & marginal deletion utility Delta U^marg(S, a)
                    costs = [[], []]
                    marg = [[], []]
                    closed_flags = []
                    for side in (0, 1):
                        for idx in range(len(s.fronts[side])):
                            zf = s.clone_light()
                            expand_node(rg, c, zf, side, idx, budget=sub_budget, nb_fn=nb_fn)
                            step_used = zf.work - s.work
                            if zf.path is not None:
                                cf, cont_used, cpath = step_used, 0, zf.path
                            else:
                                cont_c, cont_used, cpath = continuation_cost_geom(rg, c, zf, budget=sub_budget, nb_fn=nb_fn)
                                cf = step_used + cont_c if cont_c <= rem_cap - step_used else rem_cap + 1
                            oracle_work += step_used + cont_used
                            closed_flags.append(cf <= rem_cap)
                            if cpath is not None and sum(x[0].family == c.family for x in audits) < 12:
                                audits.append((c, cpath, f"cf_{len(audits)}"))

                            if len(s.fronts[side]) > 1:
                                zr = s.clone_light()
                                zr.fronts[side].pop(idx)
                                cr, r_used, _ = continuation_cost_geom(rg, c, zr, budget=sub_budget, nb_fn=nb_fn)
                                oracle_work += r_used
                            else:
                                cr = rem_cap + 1
                            du_marg = (cr - cf) / (rem_cap + 1.0)
                            costs[side].append(cf)
                            marg[side].append(du_marg)
                            pos_marg_count += int(du_marg > 1e-9)
                            neg_marg_count += int(du_marg < -1e-9)

                    flat_c = costs[0] + costs[1]
                    cost_ties += int(len(set(flat_c)) == 1)
                    mixed_closures += int(any(closed_flags) and not all(closed_flags))
                    min_c = min(flat_c)
                    min_geo = min(best_node_d[0] + best_node_d[1])
                    node_y = []
                    for side in (0, 1):
                        for idx in range(len(s.fronts[side])):
                            u_cont = 1.0 - costs[side][idx] / (rem_cap + 1.0)
                            adv = -(costs[side][idx] - min_c) / (rem_cap + 1.0)
                            geo = 1.0 if best_node_d[side][idx] <= min_geo else (
                                0.5 if best_node_d[side][idx] <= min_geo + 2 else 0.0
                            )
                            node_y.append(0.45 * u_cont + 0.45 * adv + 0.40 * marg[side][idx] + 0.50 * geo)

                    frames.append({
                        "root": c.root,
                        "lineage": c.lineage,
                        "step": step,
                        "rollout": rollout,
                        "case": c,
                        "state": s.clone_light(),
                        "prev_state": None if prev_snap is None else prev_snap.clone_light(),
                        "rem": rem,
                        "prev_rem": prev_rem,
                        "y_mat": torch.tensor(y_mat, dtype=torch.float32),
                        "node_y": torch.tensor(node_y, dtype=torch.float32),
                        "marg_y": torch.tensor(marg[0] + marg[1], dtype=torch.float32),
                    })

                prev_snap = s.clone_light()
                prev_rem = WORK - s.work
                if rollout == "geom" or rng.random() < 0.65:
                    (side, idx), _, _ = geom_mm_select(rg, c, s)
                else:
                    side = 0 if len(s.fronts[0]) <= len(s.fronts[1]) else 1
                    idx = rng.randrange(len(s.fronts[side]))
                expand_node(rg, c, s, side, idx, nb_fn=nb_fn)

    if len(audits) != 24:
        raise ValueError(f"expected 24 counterfactual audit proofs, got {len(audits)}")
    cf_cert = base.certify(rg, audits, out / "counterfactuals")
    serial_rows = [{
        "root": f["root"], "lineage": f["lineage"], "step": f["step"], "rollout": f["rollout"],
        "m0": len(f["state"].fronts[0]), "m1": len(f["state"].fronts[1]), "rem": f["rem"],
        "bridge_positives": int((f["y_mat"] > 0).sum().item()),
        "marg_min": float(f["marg_y"].min().item()), "marg_max": float(f["marg_y"].max().item()),
    } for f in frames]
    base.write_jsonl(out / "frames_summary.jsonl", serial_rows)
    receipt = {
        "train_roots": len(cases),
        "frames": len(frames),
        "all_action_cost_ties": cost_ties,
        "non_tied_frames": len(frames) - cost_ties,
        "mixed_closure_frames": mixed_closures,
        "positive_marginal_actions": pos_marg_count,
        "negative_marginal_actions": neg_marg_count,
        "bridge_positive_pairs": bridge_pairs_total,
        "total_matrix_pairs": pairs_total,
        "oracle_work": oracle_work,
        "reference_certificates": ref_cert["compiled_axiom_free"],
        "counterfactual_certificates": cf_cert["compiled_axiom_free"],
        "frames_summary_sha256": base.file_sha(out / "frames_summary.jsonl"),
    }
    base.write(out / "manifest.json", receipt)
    return frames, receipt


# ============================================================================
# 6. Matched CPU Training & Frozen Closed-Loop Evaluation
# ============================================================================
def fit(frames: list[dict], seed: int, arch: str, out: Path) -> tuple[MeetingPotentialFrescoNet, dict]:
    model, shared_init = new_model(seed, arch)
    init_fp = base._model_fingerprint(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    byroot = defaultdict(list)
    for idx, f in enumerate(frames):
        byroot[f["root"]].append((idx, f))
    roots = sorted(byroot)
    rng = random.Random(seed)
    draws = []
    trace = []
    t0 = time.monotonic()

    for step in range(STEPS):
        if time.monotonic() - t0 > 480:
            raise TimeoutError("CPU fit cap exceeded")
        rname = rng.choice(roots)
        f_idx, fr = rng.choice(byroot[rname])
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
        my = fr["marg_y"]
        pos_w = ym.new_full(ym.shape, 4.0)
        target_probs = torch.softmax(ny * 4.0, dim=0)

        loss = torch.zeros(())
        w_sum = 0.0
        for k, o in enumerate(outs):
            wk = float(k + 1)
            l_mat = F.binary_cross_entropy_with_logits(o["pot"], ym, pos_weight=pos_w)
            scores = torch.cat([o["s0"], o["s1"]], dim=0)
            l_rank = -(target_probs * F.log_softmax(scores, dim=0)).sum()
            l_marg = F.mse_loss(torch.tanh(scores * 0.25), my.clamp(-1.0, 1.0))
            loss = loss + wk * (l_mat + l_rank + 0.5 * l_marg)
            w_sum += wk
        loss = loss / w_sum
        if not torch.isfinite(loss):
            raise ValueError("nonfinite LBF-15 training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            raise ValueError("nonfinite LBF-15 gradient")
        opt.step()
        drec = {"step": step + 1, "root": rname, "frame_index": f_idx}
        draws.append(drec)
        trace.append({**drec, "loss": float(loss.detach())})

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
        "shared_initial": shared_init,
        "initial": init_fp,
        "final": base._model_fingerprint(model),
        "draws_sha256": base.file_sha(out / "draws.jsonl"),
        "trace_sha256": base.file_sha(out / "trace.jsonl"),
        "checkpoint_sha256": base.file_sha(out / "checkpoint.pt"),
    }
    base.write(out / "manifest.json", receipt)
    return model, receipt


def probe_fresco_decision(model: MeetingPotentialFrescoNet, case, s: FringeState, rem: int,
                          tag: str, ordinal: int, checkpoint_sha: str,
                          raw_store: list[dict], obs_store: list[dict]) -> tuple[list[dict], int]:
    """Capture full continuous fresco tensors and verify Pass-1 & roll norm invariants."""
    outs_norm, sub_ch = model(case, s, rem, mode="normal", return_tensors=True)
    alts = {}
    for md in ("zero", "roll", "scramble", "no_persist"):
        ao, _ = model(case, s, rem, mode=md, return_tensors=True)
        alts[md] = ao

    # Fail-closed causal invariants on Pass 1 and roll norm
    for md in ("zero", "roll"):
        torch.testing.assert_close(outs_norm[0]["pot"], alts[md][0]["pot"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outs_norm[0]["s0"], alts[md][0]["s0"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outs_norm[0]["s1"], alts[md][0]["s1"], rtol=1e-6, atol=1e-6)
    if model.passes > 1:
        # At Pass 2 (k=1), incoming banks come from identical Pass-1 states so norms match normal.
        torch.testing.assert_close(
            outs_norm[1]["bm_in"].norm(), alts["roll"][1]["bm_in"].norm(), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            outs_norm[1]["bw_u_in"].norm(), alts["roll"][1]["bw_u_in"].norm(), rtol=1e-5, atol=1e-5
        )
    for k in range(1, model.passes):
        # Across all passes k>=1, cyclic roll is an exact Frobenius-norm isometry of pass (k-1)'s outgoing bank.
        torch.testing.assert_close(
            alts["roll"][k - 1]["bm_out"].norm(), alts["roll"][k]["bm_in"].norm(), rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            alts["roll"][k - 1]["bw_u_out"].norm(), alts["roll"][k]["bw_u_in"].norm(), rtol=1e-6, atol=1e-6
        )

    obs = build_fresco_observation(case, s, outs_norm, checkpoint_sha, ordinal)
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
                  for k, v in o.items() if k in ("pot", "s0", "s1", "bm_out")}
                 for o in ao]
            for md, ao in alts.items()
        },
    })
    return outs_norm, sub_ch


def run_fringe_episode(rg, c, policy: str, *, model: MeetingPotentialFrescoNet | None = None,
                       mode: str = "normal", seed: int = 0, budget: int = WORK, depth: int = DEPTH,
                       probe_ctx: dict | None = None) -> dict:
    if policy in ("astar", "bidir"):
        r = rg.search(c, policy, budget=budget, depth=depth)
        return {**r, "pair_evaluations": 0, "substring_checks": 0, "recurrent_passes": 0, "fringe_decisions": 0}

    s = FringeState.start(c)
    pdist = build_kmer_graph(c) if policy == "graph_bridge" else None
    rng = random.Random(seed)
    calls = pair_evals = sub_checks = rec_passes = decisions = probed = 0

    while s.live(budget):
        decisions += 1
        m0, m1 = len(s.fronts[0]), len(s.fronts[1])
        if policy == "uniform":
            tot = m0 + m1
            pick = rng.randrange(tot)
            side, idx = (0, pick) if pick < m0 else (1, pick - m0)
        elif policy == "geom_mm":
            (side, idx), pe, sc = geom_mm_select(rg, c, s, budget, depth)
            pair_evals += pe
            sub_checks += sc
        elif policy == "graph_bridge":
            (side, idx), pe, sc = graph_bridge_select(rg, c, s, pdist, budget, depth)
            pair_evals += pe
            sub_checks += sc
        elif policy == "learned":
            if model is None:
                raise ValueError("learned policy requires frozen model")
            with torch.no_grad():
                if probe_ctx is not None and mode == "normal" and m0 * m1 >= 4 and probed < 2:
                    probed += 1
                    outs, sc = probe_fresco_decision(
                        model, c, s, budget - s.work, probe_ctx["tag"], decisions,
                        probe_ctx["checkpoint"], probe_ctx["raw"], probe_ctx["obs"],
                    )
                else:
                    outs, sc = model(c, s, budget - s.work, mode=mode, depth_cap=depth)
            calls += 1
            rec_passes += len(outs)
            pair_evals += m0 * m1
            sub_checks += sc
            last = outs[-1]
            psig = last["pot"].sigmoid()
            pm = float(psig.mean())
            if mode != "no_persist":
                s.prev_pot = {(u, v): float(psig[i, j]) - pm
                              for i, u in enumerate(s.fronts[0])
                              for j, v in enumerate(s.fronts[1])}
            else:
                s.prev_pot = {}
            s0 = last["s0"].tolist()
            s1 = last["s1"].tolist()
            if any(not math.isfinite(x) for x in s0 + s1):
                raise ValueError("nonfinite learned node scores")
            cands = [(s0[i], -m0, 0, -i) for i in range(m0)] + [(s1[j], -m1, -1, -j) for j in range(m1)]
            best = max(cands)
            side = 0 if best[2] == 0 else 1
            idx = -best[3]
        else:
            raise ValueError(f"unknown policy {policy}")
        expand_node(rg, c, s, side, idx, budget, depth)

    res = rg._result(c, policy, s.path, s.work, s.expanded, calls, budget, depth)
    return {
        **res,
        "pair_evaluations": pair_evals,
        "substring_checks": sub_checks,
        "recurrent_passes": rec_passes,
        "fringe_decisions": decisions,
    }


def summarize_probes(raw_records: list[dict]) -> dict:
    out = {}
    for tag in ("recurrent_standard", "recurrent_opened"):
        rr = [r for r in raw_records if r["tag"].endswith(tag)]
        if not rr:
            continue
        K = len(rr[0]["normal"])
        wg = [r["normal"][K - 1]["word_gate_strength"] for r in rr]
        fg = [r["normal"][K - 1]["fresco_gate_strength"] for r in rr]
        fc_rms = [r["normal"][K - 1]["fresco_gate_contrib_rms"] for r in rr]
        er_mat = [erank_2d(r["normal"][K - 1]["bm_out"]) for r in rr]
        er_word = [erank_2d(r["normal"][K - 1]["bw_u_out"]) for r in rr]
        mi_p1 = [morans_i(r["normal"][0]["pot"].sigmoid()) for r in rr]
        mi_pk = [morans_i(r["normal"][K - 1]["pot"].sigmoid()) for r in rr]
        # Scramble rook Moran's I on shuffled grid before unpermuting
        mi_scr = []
        for r in rr:
            p = r["normal"][K - 1]["pot"].sigmoid()
            m0, m1 = p.shape
            gen = torch.Generator().manual_seed(1337 + m0 * 31 + m1)
            ps = p[torch.randperm(m0, generator=gen)][:, torch.randperm(m1, generator=gen)]
            mi_scr.append(morans_i(ps))

        brightening = [float((r["normal"][K - 1]["pot"].sigmoid() - r["normal"][0]["pot"].sigmoid()).abs().mean()) for r in rr]
        pos_pol_frac = [
            float(((r["normal"][K - 1]["pot"] - r["normal"][0]["pot"]) > 0.05).float().mean()) for r in rr
        ]
        m_summary = {
            "probed_decisions": len(rr),
            "passK_word_gate_mean": sum(wg) / len(wg),
            "passK_fresco_gate_mean": sum(fg) / len(fg),
            "passK_fresco_contrib_rms": sum(fc_rms) / len(fc_rms),
            "fresco_bank_erank_mean": sum(er_mat) / len(er_mat),
            "word_bank_erank_mean": sum(er_word) / len(er_word),
            "morans_i_pass1_mean": sum(mi_p1) / len(mi_p1),
            "morans_i_passK_mean": sum(mi_pk) / len(mi_pk),
            "morans_i_scrambled_grid_mean": sum(mi_scr) / len(mi_scr),
            "mean_abs_pass_brightening": sum(brightening) / len(brightening),
            "positive_polarity_tile_fraction": sum(pos_pol_frac) / len(pos_pol_frac),
        }
        for md in ("zero", "roll", "scramble", "no_persist"):
            s_norm = [torch.cat([r["normal"][K - 1]["s0"], r["normal"][K - 1]["s1"]]) for r in rr]
            s_alt = [torch.cat([r["alternatives"][md][K - 1]["s0"], r["alternatives"][md][K - 1]["s1"]]) for r in rr]
            m_summary[f"{md}_max_score_delta"] = max(float((a - b).abs().max()) for a, b in zip(s_norm, s_alt))
            m_summary[f"{md}_top_action_changes"] = sum(int(a.argmax()) != int(b.argmax()) for a, b in zip(s_norm, s_alt))
        out[tag] = m_summary
    return out


# ============================================================================
# 7. Registered End-to-End LBF-15 Campaign
# ============================================================================
def verify_historical_disjointness(rg, suite_dir: Path) -> int:
    current_lineages = set()
    for sp in COUNTS:
        rows = rg.read_rows(suite_dir / f"{sp}.jsonl")
        current_lineages.update(r["lineage"] for r in rows)
    if len(current_lineages) != sum(COUNTS.values()):
        raise ValueError("internal lineage collision inside LBF-15 suite")
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
        raise ValueError("fresh LBF-15 output directory required")
    out.mkdir(parents=True)
    torch.set_num_threads(max(1, min(2, threads)))
    torch.backends.mkldnn.enabled = False

    suite = out / "suite"
    sm = rg.freeze(suite, SEED, COUNTS)
    if sm["counts"] != COUNTS or sm["unique_rule_system_lineages"] != sum(COUNTS.values()):
        raise ValueError("LBF-15 suite manifest census mismatch")
    checked_historical = verify_historical_disjointness(rg, suite)

    frames, tm = teacher(rg, suite, out / "teacher")
    models = {}
    receipts = {}
    for seed in SEEDS:
        for arch in ARCHITECTURES:
            k = (seed, arch)
            models[k], receipts[k] = fit(frames, seed, arch, out / f"{seed}_{arch}")
        # Verify identical frame draw schedules across all 4 architectures and shared init for standard vs opened
        ref_draws = receipts[(seed, "recurrent_opened")]["draws_sha256"]
        for arch in ARCHITECTURES:
            if receipts[(seed, arch)]["draws_sha256"] != ref_draws:
                raise ValueError("training frame draw schedule diverged across architectures")
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
        # Verify pinned bidir equivalence against LBF-14 layer scheduler
        r_bidir = run_fringe_episode(rg, c, "bidir")
        r_lbf14 = lbf14.run(rg, c, "bidir", budget=WORK, depth=DEPTH)
        if any(r_bidir[k] != r_lbf14[k] for k in ("closed", "work", "expanded", "path")):
            raise ValueError("pinned bidir equivalence check failed")
        rows.append({**r_bidir, "condition": "bidir", "seed": None, "stream": None})

        for cname in ("astar", "geom_mm", "graph_bridge"):
            rc = run_fringe_episode(rg, c, cname)
            rows.append({**rc, "condition": cname, "seed": None, "stream": None})

        for stream in STREAMS:
            ru = run_fringe_episode(rg, c, "uniform", seed=int(base.digest([stream, c.root])[:16], 16))
            rows.append({**ru, "condition": "uniform", "seed": None, "stream": stream})

        for seed in SEEDS:
            for arch in ("nonrecurrent", "unstructured_recurrent", "recurrent_standard"):
                m = models[(seed, arch)]
                pctx = ({"tag": f"{seed}:{arch}", "checkpoint": receipts[(seed, arch)]["checkpoint_sha256"],
                         "raw": raw_probes, "obs": observations}
                        if seed == SEEDS[0] and arch == "recurrent_standard" else None)
                rl = run_fringe_episode(rg, c, "learned", model=m, mode="normal", probe_ctx=pctx)
                rows.append({**rl, "condition": arch, "seed": seed, "stream": None})

            m_open = models[(seed, "recurrent_opened")]
            ck_open = receipts[(seed, "recurrent_opened")]["checkpoint_sha256"]
            for md in OPENED_MODES:
                pctx = ({"tag": f"{seed}:recurrent_opened", "checkpoint": ck_open,
                         "raw": raw_probes, "obs": observations}
                        if seed == SEEDS[0] and md == "normal" else None)
                ro = run_fringe_episode(rg, c, "learned", model=m_open, mode=md, probe_ctx=pctx)
                rows.append({**ro, "condition": f"opened_{md}", "seed": seed, "stream": None})

    # Verify frozen weights untouched
    for k, m in models.items():
        if base._model_fingerprint(m) != receipts[k]["final"]:
            raise ValueError("frozen checkpoint mutated during evaluation")

    # Standalone axiom-free Lean 4.19.0 certification of all successful calibration episodes + calibration references
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

    # Secondary budget-scaling curve B in {32, 64, 96, 128} on frozen models (zero refitting)
    budget_curves = {}
    for b_cap in BUDGET_CURVE:
        b_rates = {}
        for cname in ("astar", "bidir", "geom_mm", "graph_bridge"):
            b_rates[cname] = sum(int(run_fringe_episode(rg, c, cname, budget=b_cap)["closed"]) for c in cal_cases) / len(cal_cases)
        for arch, cond_label in (("nonrecurrent", "nonrecurrent"),
                                 ("unstructured_recurrent", "unstructured_recurrent"),
                                 ("recurrent_standard", "recurrent_standard"),
                                 ("recurrent_opened", "opened_normal")):
            tot_c = 0
            for seed in SEEDS:
                m = models[(seed, arch)]
                tot_c += sum(int(run_fringe_episode(rg, c, "learned", model=m, mode="normal", budget=b_cap)["closed"])
                             for c in cal_cases)
            b_rates[cond_label] = tot_c / (len(SEEDS) * len(cal_cases))
        budget_curves[str(b_cap)] = b_rates

    # Export observations, encoded Braille glyphs, standard fresco SVG, and illuminated 2D manuscript SVG
    base.write_jsonl(out / "episodes.jsonl", rows)
    base.write_jsonl(out / "observations.jsonl", observations)
    encoded = [base.encode(o) for o in observations]
    base.write_jsonl(out / "encoded.jsonl", encoded)
    torch.save(raw_probes, out / "raw_fresco_tensors.pt")
    (out / "fresco.svg").write_text(base.fresco_svg(encoded[0]), encoding="utf-8")

    # Select richest probed recurrent_opened state for the illuminated 2D manuscript SVG
    open_probes = [r for r in raw_probes if r["tag"].endswith("recurrent_opened")]
    showcase = max(open_probes, key=lambda r: len(r["u_words"]) * len(r["v_words"]))
    showcase_case = lookup[showcase["root"]]
    ms_svg = render_manuscript_svg(
        showcase_case,
        showcase["u_words"],
        showcase["v_words"],
        [o["pot"] for o in showcase["normal"]],
        [o["s0"] for o in showcase["normal"]],
        [o["s1"] for o in showcase["normal"]],
        showcase["persisted_mask"],
    )
    (out / "fresco_manuscript.svg").write_text(ms_svg, encoding="utf-8")

    # Aggregate statistics & pre-declared screens
    learned_names = ("nonrecurrent", "unstructured_recurrent", "recurrent_standard") + tuple(f"opened_{m}" for m in OPENED_MODES)
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

    # Evaluate pre-declared screens
    bidir_gain = rates["opened_normal"] - rates["bidir"]
    bidir_work_saving = 1.0 - costs["opened_normal"]["mean_penalized_work"] / costs["bidir"]["mean_penalized_work"]
    screen_1a_vs_bidir = (
        bidir_gain >= 0.10
        and bidir_work_saving >= 0.20
        and all(seed_rates[str(s)]["opened_normal"] > rates["bidir"] for s in SEEDS)
    )

    best_ctrl = max(controls, key=lambda n: (rates[n], -costs[n]["mean_penalized_work"]))
    ctrl_gain = rates["opened_normal"] - rates[best_ctrl]
    ctrl_work_saving = 1.0 - costs["opened_normal"]["mean_penalized_work"] / costs[best_ctrl]["mean_penalized_work"]
    screen_1b_vs_strongest_control = (
        rates["opened_normal"] >= 0.50
        and ((ctrl_gain >= 0.04) or (ctrl_gain >= 0.0 and ctrl_work_saving >= 0.10))
        and ctrl_gain >= 0.0
    )

    def beats_or_saves(other: str, min_save: float = 0.0) -> bool:
        g = rates["opened_normal"] - rates[other]
        sv = 1.0 - costs["opened_normal"]["mean_penalized_work"] / costs[other]["mean_penalized_work"]
        return (g > 0.0) or (g >= 0.0 and sv >= min_save)

    screen_2a_architecture = (
        beats_or_saves("nonrecurrent", 0.05)
        and beats_or_saves("unstructured_recurrent", 0.05)
        and beats_or_saves("recurrent_standard", 0.0)
    )
    screen_2b_causal_memory = all(
        (rates["opened_normal"] > rates[f"opened_{m}"])
        or (rates["opened_normal"] >= rates[f"opened_{m}"]
            and costs["opened_normal"]["mean_penalized_work"] < costs[f"opened_{m}"]["mean_penalized_work"])
        for m in ("zero", "roll", "scramble")
    )

    result = {
        "schema": "lbf15.meeting_potential_fresco.v1",
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
        "strongest_non_neural_control": best_ctrl,
        "screens": {
            "primary_1a_vs_symbolic_bidir": screen_1a_vs_bidir,
            "primary_1b_vs_strongest_heuristic": screen_1b_vs_strongest_control,
            "secondary_2a_recurrence_and_structure": screen_2a_architecture,
            "secondary_2b_causal_memory_and_spatial": screen_2b_causal_memory,
        },
        "fresco_probe_summary": summarize_probes(raw_probes),
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
    print("LBF15_RESULT", base.canonical(result), flush=True)
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
