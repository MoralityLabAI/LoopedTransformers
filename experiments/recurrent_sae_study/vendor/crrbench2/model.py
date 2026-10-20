"""Weight-tied looped transformer (one block applied K times)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(max_len, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(max_len, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4):
        super().__init__()
        self.h, self.d = heads, d
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, h, cos, sin):
        B, T, d = h.shape
        q, k, v = self.qkv(self.n1(h)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        a = F.scaled_dot_product_attention(q, k, v)  # bidirectional
        h = h + self.o(a.transpose(1, 2).reshape(B, T, d))
        return h + self.fc2(F.gelu(self.fc1(self.n2(h))))


class LoopedTransformer(nn.Module):
    """h_0 = Embed(x);  h_{k+1} = RMS(Block(h_k));  logits_k = Head(LN(h_k[:, -1])).

    The prediction is read at the last token, so the answer must live in that
    position's vector; the rest of the state is working memory.
    """

    def __init__(self, vocab, n_classes, d=128, heads=4, max_len=160):
        super().__init__()
        self.d, self.n_classes, self.max_len = d, n_classes, max_len
        self.emb = nn.Embedding(vocab, d)
        self.block = Block(d, heads)
        self.ln_f = nn.RMSNorm(d)
        self.head = nn.Linear(d, n_classes)
        nn.init.normal_(self.emb.weight, std=0.5)
        self._rope = None

    def rope(self, device):
        if self._rope is None or self._rope[0].device != device:
            self._rope = rope_cache(self.max_len, self.d // self.block.h // 1, device)
        return self._rope

    def embed(self, x):
        return self.emb(x)

    def step(self, h):
        cos, sin = self.rope(h.device)
        return F.rms_norm(self.block(h, cos, sin), (self.d,))

    def read(self, h):
        z = self.ln_f(h[:, -1])
        return self.head(z), z

    def run_from(self, h, K, snap_at=None, keep_z=False):
        """Run K loops from state h.  Returns logits [K,B,C] (and z [K,B,d]).

        snap_at: optional LongTensor [B] of 1-indexed loops; returns the full state
        at that loop for every row (used to freeze/save hidden states).
        """
        logits, zs, snap = [], [], None
        if snap_at is not None:
            snap = torch.zeros_like(h)
        for k in range(1, K + 1):
            h = self.step(h)
            lg, z = self.read(h)
            logits.append(lg)
            if keep_z:
                zs.append(z)
            if snap is not None:
                m = (snap_at == k)
                if m.any():
                    snap[m] = h[m]
        out = [torch.stack(logits)]
        if keep_z:
            out.append(torch.stack(zs))
        if snap is not None:
            out.append(snap)
        return out[0] if len(out) == 1 else tuple(out)

    def run(self, x, K, **kw):
        return self.run_from(self.embed(x), K, **kw)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
