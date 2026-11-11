"""LoRA / loop-specific LoRA, optional bitsandbytes quantisation of frozen base linears."""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_core import Block, Qwen3LM, RecurrentSite, StepCtx


class LoRALinear(nn.Module):
    """y = base(x) + scale * B[i] A[i] x ,  i = min(ctx.t, n-1).

    n == 1  -> ordinary LoRA (also used as the *shared* adapter of a recurrent block)
    n  > 1  -> loop-specific adapters, one (A_t, B_t) pair per recurrent step t.
    """

    def __init__(self, base: nn.Module, rank: int, alpha: float, n: int, ctx: StepCtx):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.n = rank, n
        self.scale = alpha / rank
        self.ctx = ctx
        A = torch.empty(n, rank, self.in_features)
        for i in range(n):
            nn.init.kaiming_uniform_(A[i], a=math.sqrt(5))
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(torch.zeros(n, self.out_features, rank))

    def forward(self, x):
        y = self.base(x)
        i = min(self.ctx.t, self.n - 1)
        a = self.A[i].to(x.dtype)
        b = self.B[i].to(x.dtype)
        return y + F.linear(F.linear(x, a), b) * self.scale

    @property
    def weight(self) -> torch.Tensor:
        """merged weight (only meaningful for n==1) - lets export_hf_state_dict treat a LoRA'd block like a plain one"""
        assert self.n == 1
        return self.merged_weight(0).to(self.base.weight.dtype)

    @torch.no_grad()
    def merged_weight(self, i: int = 0) -> torch.Tensor:
        w = self.base.weight.float() + self.scale * (self.B[i].float() @ self.A[i].float())
        return w


def lora_param_count(block: Block, rank: int, n: int = 1) -> int:
    return n * rank * sum(getattr(block, k).in_features + getattr(block, k).out_features for k in Block.LINEARS)


def install_lora(block: Block, rank: int, alpha: float, n: int, ctx: StepCtx):
    for k in Block.LINEARS:
        setattr(block, k, LoRALinear(getattr(block, k), rank, alpha, n, ctx))


# ------------------------------------------------------------- quantisation ----

def _import_bnb():
    vendor = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
    if os.path.isdir(vendor) and vendor not in sys.path:
        sys.path.insert(0, vendor)
    import bitsandbytes as bnb
    return bnb


def quantize_blocks(model: Qwen3LM, mode: str):
    """Replace frozen base nn.Linear in every Block with bnb 4-bit (nf4) / 8-bit.  Call BEFORE .cuda()
    and BEFORE install_lora.  Embeddings / norms / head stay bf16."""
    if mode == "none":
        return
    bnb = _import_bnb()

    def conv(lin: nn.Linear):
        if mode == "4bit":
            q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.bfloat16,
                                  quant_type="nf4", device="meta")
            q.weight = bnb.nn.Params4bit(lin.weight.data.to(torch.bfloat16), requires_grad=False, quant_type="nf4")
        elif mode == "8bit":
            q = bnb.nn.Linear8bitLt(lin.in_features, lin.out_features, bias=False, has_fp16_weights=False,
                                    threshold=6.0, device="meta")
            q.weight = bnb.nn.Int8Params(lin.weight.data.to(torch.float16), requires_grad=False, has_fp16_weights=False)
        else:
            raise ValueError(mode)
        return q

    blocks = []
    for e in model.entries:
        blocks.append(e.block if isinstance(e, RecurrentSite) else e)
    for b in blocks:
        for k in Block.LINEARS:
            setattr(b, k, conv(getattr(b, k)))
