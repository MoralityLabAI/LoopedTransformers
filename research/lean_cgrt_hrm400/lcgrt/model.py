"""Hierarchical recurrent reasoner (HRM) over Lean proof-state graphs, and a flat baseline.

HRM, two clocks:
  fast  one shared-weight block, applied T times per decision over atom/hypothesis
        tokens (typed graph attention + read from slow slots + MLP), with input
        re-injection and a post-norm on the recurrent state;
  slow  M slots, updated once every `slow_every` fast steps (slot self-attention +
        read from tokens + MLP); slots carry global search features (budget, open
        goals, failures) and drive the halting head.
Readouts (candidate logits, atom provability, halt) happen at every slow tick; they
are the deep-supervision targets in training and the exit points for adaptive halting.

Flat baseline: L distinct (untied) graph blocks, no slots, one readout.
`graph=False` removes the incidence structure (full attention, one edge type).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .encode import N_EDGE_TYPES, N_FEATS

N_TOK_TYPES = 5
N_GLOBAL = 5


@dataclass
class ModelConfig:
    arch: str = "hrm"          # "hrm" | "flat"
    width: int = 384
    heads: int = 8
    fast_steps: int = 8        # HRM: fast applications per decision
    slow_every: int = 2        # HRM: one slow tick per this many fast steps
    slots: int = 8
    layers: int = 8            # flat: number of untied blocks
    mlp_ratio: int = 4
    graph: bool = True
    slow: bool = True          # HRM: False = single-clock tied loop (no slots, no halting head)

    def to_json(self):
        return asdict(self)


class TypedAttention(nn.Module):
    """Multi-head attention restricted to graph edges, with a learned bias per edge type and head."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.type_bias = nn.Parameter(torch.zeros(N_EDGE_TYPES, heads))

    def make_bias(self, onehot, nonedge):
        """onehot [B,N,N,E] -> additive mask [B,H,N,N]. A one-hot matmul, not an index
        gather: the gather's backward is an atomic scatter that cost ~80% of a step."""
        bias = (onehot @ self.type_bias.to(onehot.dtype)).permute(0, 3, 1, 2)
        return bias.masked_fill(nonedge[:, None], float("-inf"))

    def forward(self, x, bias):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.o(y.transpose(1, 2).reshape(B, N, D))


class CrossAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mem, mem_mask=None):
        B, N, D = x.shape
        M = mem.shape[1]
        q = self.q(x).view(B, N, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(B, M, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        mask = None
        if mem_mask is not None:
            mask = torch.zeros(B, 1, 1, M, dtype=q.dtype, device=x.device)
            mask = mask.masked_fill(~mem_mask[:, None, None], float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, N, D))


def mlp(d, r):
    return nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d))


class FastBlock(nn.Module):
    def __init__(self, d, heads, r, cross: bool):
        super().__init__()
        self.n1, self.n2, self.n3, self.out = (nn.LayerNorm(d) for _ in range(4))
        self.att = TypedAttention(d, heads)
        self.cross = CrossAttention(d, heads) if cross else None
        self.ff = mlp(d, r)

    def forward(self, x, x0, bias, slots=None):
        u = x + x0
        u = u + self.att(self.n1(u), bias)
        if self.cross is not None:
            u = u + self.cross(self.n2(u), slots)
        u = u + self.ff(self.n3(u))
        return self.out(u)


class SlowBlock(nn.Module):
    def __init__(self, d, heads, r):
        super().__init__()
        self.n1, self.n2, self.n3, self.out = (nn.LayerNorm(d) for _ in range(4))
        self.self_att = CrossAttention(d, heads)
        self.read = CrossAttention(d, heads)
        self.ff = mlp(d, r)

    def forward(self, s, x, tok_mask):
        s = s + self.self_att(self.n1(s), self.n1(s))
        s = s + self.read(self.n2(s), x, tok_mask)
        s = s + self.ff(self.n3(s))
        return self.out(s)


class Readout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.cand = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.atom = nn.Linear(d, 1)

    def forward(self, x, main, legal):
        h = self.n(x)
        m = h[torch.arange(h.shape[0], device=h.device), main][:, None].expand_as(h)
        logits = self.cand(torch.cat([h, m, h * m], -1)).squeeze(-1).float()
        logits = logits.masked_fill(~legal, -1e4)
        return logits, self.atom(h).squeeze(-1).float()


class ProofNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_ckpt = False          # recompute each fast step / block in backward
        d = cfg.width
        assert d % cfg.heads == 0
        self.feat = nn.Linear(N_FEATS, d)
        self.ttype = nn.Embedding(N_TOK_TYPES, d)
        self.readout = Readout(d)
        if cfg.arch == "hrm":
            self.fast = FastBlock(d, cfg.heads, cfg.mlp_ratio, cross=cfg.slow)
            if cfg.slow:
                self.slow = SlowBlock(d, cfg.heads, cfg.mlp_ratio)
                self.slot_init = nn.Parameter(torch.randn(cfg.slots, d) * 0.02)
                self.glob = nn.Linear(N_GLOBAL, d)
                self.halt = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        elif cfg.arch == "flat":
            self.blocks = nn.ModuleList(FastBlock(d, cfg.heads, cfg.mlp_ratio, cross=False)
                                        for _ in range(cfg.layers))
        else:
            raise ValueError(cfg.arch)

    def _ckpt(self):
        return self.grad_ckpt and self.training and torch.is_grad_enabled()

    def _etype(self, b):
        if self.cfg.graph:
            return b["etype"]
        m = b["tok_mask"]
        full = (m[:, :, None] & m[:, None, :]).long()
        return torch.maximum(full, torch.eye(m.shape[1], dtype=torch.long, device=m.device)[None])

    def forward(self, b, fast_steps: int | None = None, glob=None, halt_threshold: float | None = None):
        """List of readouts (one per slow tick for HRM, one for flat): dict(logits, atom,
        halt, steps). With `halt_threshold`, HRM stops at the first tick whose halting
        probability exceeds it for every batch row."""
        etype = self._etype(b)
        x0 = self.feat(b["x"]) + self.ttype(b["ttype"])
        onehot = F.one_hot(etype, N_EDGE_TYPES).to(x0.dtype)
        nonedge = etype == 0
        outs = []
        if self.cfg.arch == "flat":
            x = torch.zeros_like(x0)
            for blk in self.blocks:
                bias_b = blk.att.make_bias(onehot, nonedge)
                x = (checkpoint(blk, x, x0, bias_b, use_reentrant=False) if self._ckpt()
                     else blk(x, x0, bias_b))
            lg, at = self.readout(x, b["main"], b["legal"])
            return [dict(logits=lg, atom=at, halt=None, steps=len(self.blocks))]
        T = fast_steps or self.cfg.fast_steps
        k = min(self.cfg.slow_every, T)
        B = x0.shape[0]
        if glob is None:
            glob = torch.zeros(B, N_GLOBAL, device=x0.device)
        s = self.slot_init[None].expand(B, -1, -1) + self.glob(glob)[:, None] if self.cfg.slow else None
        x = torch.zeros_like(x0)
        bias = self.fast.att.make_bias(onehot, nonedge)      # tied weights: one bias for all T
        for t in range(T):
            x = (checkpoint(self.fast, x, x0, bias, s, use_reentrant=False) if self._ckpt()
                 else self.fast(x, x0, bias, s))
            if (t + 1) % k == 0 or t == T - 1:
                lg, hl = None, None
                if self.cfg.slow:
                    s = self.slow(s, x, b["tok_mask"])
                    hl = self.halt(s.mean(1)).squeeze(-1).float()
                lg, at = self.readout(x, b["main"], b["legal"])
                outs.append(dict(logits=lg, atom=at, halt=hl, steps=t + 1))
                if halt_threshold is not None and hl is not None and bool((torch.sigmoid(hl) > halt_threshold).all()):
                    break
        return outs


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    by = {}
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        by[top] = by.get(top, 0) + p.numel()
    return {"total": total, "by_module": by}


def flops_per_decision(cfg: ModelConfig, n_tok: int, fast_steps: int | None = None) -> float:
    """Analytic forward FLOPs (2 x multiply-adds) for one decision on an n_tok graph."""
    d, r, N = cfg.width, cfg.mlp_ratio, n_tok
    attn = 2 * N * (4 * d * d) + 2 * 2 * N * N * d
    ff = 2 * N * (2 * r * d * d)
    head = 2 * N * 3 * d * d
    if cfg.arch == "flat":
        return cfg.layers * (attn + ff) + head
    T = fast_steps or cfg.fast_steps
    M = cfg.slots
    cross = 2 * N * (2 * d * d) + 2 * M * 2 * d * d + 2 * 2 * N * M * d
    ticks = math.ceil(T / min(cfg.slow_every, T))
    slow = 2 * M * (4 * d * d) * 2 + 2 * 2 * M * N * d + 2 * M * 2 * r * d * d
    return T * (attn + cross + ff) + ticks * (slow + head)


PRESETS = {
    # name: (arch, width, heads, layers); widths put totals near 5M / 15M / 30M
    "S": ("hrm", 384, 8, None), "S-loop": ("hrm", 592, 8, None), "M": ("hrm", 672, 8, None), "L": ("hrm", 928, 8, None),
    "flat-S": ("flat", 232, 8, 8), "flat-M": ("flat", 400, 8, 8), "flat-L": ("flat", 552, 8, 8),
    "tiny": ("hrm", 64, 4, None), "flat-tiny": ("flat", 40, 4, 8),
    # 400M handoff presets (SILICO_400M_HANDOFF.md); exact counts 399,131,139 / 398,364,866 / 398,332,194
    "HRM-400M": ("hrm", 3376, 16, None), "flat-400M": ("flat", 2304, 16, 6), "loop-400M": ("hrm", 5152, 16, None),
}


def preset(name: str, **overrides) -> ModelConfig:
    arch, width, heads, layers = PRESETS[name]
    cfg = ModelConfig(arch=arch, width=width, heads=heads, **({"layers": layers} if layers else {}))
    if name.endswith("-loop") or name.startswith("loop-"):
        cfg.slow = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
