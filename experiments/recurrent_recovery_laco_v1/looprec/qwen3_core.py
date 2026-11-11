"""Self-contained Qwen3 decoder implementation.

Written from scratch (instead of patching HF modules) so that:
  * every block exposes the residual update F(h) explicitly:   block(h) = h + block.delta(h)
  * LoRA / loop-specific adapters and loop-specific norms can be injected cleanly
  * layers can be merged / removed / looped freely
It is checked against transformers' Qwen3ForCausalLM in scripts/verify_core.py.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class QCfg:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    inter: int
    rope_theta: float
    eps: float
    tie: bool

    @classmethod
    def from_hf_json(cls, path: str) -> "QCfg":
        c = json.load(open(path))
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"], num_kv_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            inter=c["intermediate_size"], rope_theta=float(c["rope_theta"]), eps=c["rms_norm_eps"],
            tie=c.get("tie_word_embeddings", False),
        )


class StepCtx:
    """Mutable pointer to the current recurrent step, shared by all adapters/norms of one block."""
    __slots__ = ("t",)

    def __init__(self):
        self.t = 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.loop_weight: nn.Parameter | None = None
        self.ctx: StepCtx | None = None

    def enable_trainable(self, n: int, ctx: StepCtx):
        """n=1 -> one trainable fp32 copy shared by all loops; n>1 -> one copy per loop step."""
        w = self.weight.detach().float().unsqueeze(0).repeat(n, 1).contiguous()
        self.loop_weight = nn.Parameter(w)
        self.weight.requires_grad_(False)
        self.ctx = ctx

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.loop_weight is None:
            w = self.weight
        else:
            w = self.loop_weight[min(self.ctx.t, self.loop_weight.shape[0] - 1)]
        return w.to(dt) * xf.to(dt)


def rope_cos_sin(T: int, hd: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    t = torch.arange(T, device=device, dtype=torch.float32)
    fr = torch.outer(t, inv)
    emb = torch.cat([fr, fr], dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Block(nn.Module):
    """One Qwen3 decoder layer.  forward(h) = h + delta(h)  (pre-norm, two residual sub-layers)."""

    def __init__(self, c: QCfg):
        super().__init__()
        D, nh, nkv, hd = c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.ln1 = RMSNorm(D, c.eps)
        self.q_proj = nn.Linear(D, nh * hd, bias=False)
        self.k_proj = nn.Linear(D, nkv * hd, bias=False)
        self.v_proj = nn.Linear(D, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, D, bias=False)
        self.q_norm = RMSNorm(hd, c.eps)
        self.k_norm = RMSNorm(hd, c.eps)
        self.ln2 = RMSNorm(D, c.eps)
        self.gate_proj = nn.Linear(D, c.inter, bias=False)
        self.up_proj = nn.Linear(D, c.inter, bias=False)
        self.down_proj = nn.Linear(c.inter, D, bias=False)
        self.t_end = -1       # index of the last original teacher layer this block stands for
        self.merged = False   # True if produced by LaCo collapse

    LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    NORMS = ("ln1", "ln2", "q_norm", "k_norm")

    def delta(self, h, cos, sin):
        """F(h): sum of the attention and MLP contributions (no skip connection)."""
        B, T, _ = h.shape
        x = self.ln1(h)
        q = self.q_norm(self.q_proj(x).view(B, T, self.nh, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.nkv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = q * cos + _rot_half(q) * sin
        k = k * cos + _rot_half(k) * sin
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        a = self.o_proj(o.transpose(1, 2).reshape(B, T, self.nh * self.hd))
        z = self.ln2(h + a)
        m = self.down_proj(F.silu(self.gate_proj(z)) * self.up_proj(z))
        return a + m

    def forward(self, h, cos, sin):
        return h + self.delta(h, cos, sin)

    def flops_per_token(self, ctx: int) -> float:
        """Forward FLOPs / token (2 per MAC) for linears + causal attention (avg ctx/2 keys)."""
        lin = 0
        for n in self.LINEARS:
            m = getattr(self, n)
            lin += 2 * m.in_features * m.out_features
        attn = 2 * 2 * (ctx / 2) * self.nh * self.hd
        return lin + attn


class RecurrentSite(nn.Module):
    """Explicit recurrence  h[t+1] = h[t] + g[t] * F_shared(h[t]; adapter[t], norm[t])  (t = 0..T-1)."""

    def __init__(self, block: Block, anchor: int, span: int, max_loops: int, gate: bool,
                 extrap: str = "clamp"):
        super().__init__()
        self.block = block
        self.anchor, self.span = anchor, span            # teacher layers [anchor, anchor+span) replaced
        self.max_loops = max_loops
        self.ctx = StepCtx()
        self.gate = nn.Parameter(torch.ones(max_loops)) if gate else None
        self.extrap = extrap
        self.t_end = anchor + span - 1

    def step_index(self, t: int) -> int:
        if t < self.max_loops:
            return t
        return self.max_loops - 1 if self.extrap == "clamp" else t % self.max_loops

    def forward(self, h, cos, sin, T: int, ckpt: bool = False, collect: list | None = None):
        for t in range(T):
            self.ctx.t = self.step_index(t)
            if ckpt and torch.is_grad_enabled():
                f = checkpoint(self.block.delta, h, cos, sin, use_reentrant=False)
            else:
                f = self.block.delta(h, cos, sin)
            assert f.shape == h.shape, f"F(h) shape {tuple(f.shape)} != h shape {tuple(h.shape)}"
            if self.gate is not None:
                f = f * self.gate[self.ctx.t].to(f.dtype)
            h = h + f                                    # residual pathway preserved
            if collect is not None:
                tidx = self.anchor + round((t + 1) * self.span / T)
                collect.append((tidx, h, "site_final" if t == T - 1 else "step"))
        return h


class Qwen3LM(nn.Module):
    def __init__(self, c: QCfg, entries: list[nn.Module] | None = None):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.entries = nn.ModuleList(entries if entries is not None else [Block(c) for _ in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.eps)
        if not c.tie:
            self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.grad_ckpt = False
        self.default_loops = 1

    def head_weight(self):
        return self.embed.weight if self.cfg.tie else self.lm_head.weight

    def site(self) -> RecurrentSite | None:
        for e in self.entries:
            if isinstance(e, RecurrentSite):
                return e
        return None

    def forward_hidden(self, ids, n_loops: int | None = None, collect: bool = False):
        """Returns (post-final-norm hidden [B,T,D], states).  states: list of (teacher_state_idx, h, tag)
        where teacher_state_idx refers to H[i] = output after i original layers (H[0] = embeddings)."""
        h = self.embed(ids)
        cos, sin = rope_cos_sin(ids.shape[1], self.cfg.head_dim, self.cfg.rope_theta, ids.device, h.dtype)
        states: list | None = [] if collect else None
        if collect:
            states.append((0, h, "embed"))
        T = n_loops if n_loops is not None else self.default_loops
        use_ck = self.grad_ckpt and torch.is_grad_enabled()
        for e in self.entries:
            if isinstance(e, RecurrentSite):
                h = e(h, cos, sin, T, ckpt=use_ck, collect=states)
            else:
                h = checkpoint(e, h, cos, sin, use_reentrant=False) if use_ck else e(h, cos, sin)
                if collect:
                    states.append((e.t_end + 1, h, "merged" if e.merged else "plain"))
        return self.norm(h), states

    def logits(self, hidden):
        return hidden @ self.head_weight().t()

    # ----- accounting -----
    def flops_per_token(self, ctx: int, n_loops: int | None = None, include_head: bool = True) -> float:
        T = n_loops if n_loops is not None else self.default_loops
        f = 0.0
        for e in self.entries:
            if isinstance(e, RecurrentSite):
                f += T * e.block.flops_per_token(ctx)
            else:
                f += e.flops_per_token(ctx)
        if include_head:
            f += 2 * self.cfg.hidden_size * self.cfg.vocab_size
        return f

    def n_layer_evals(self, n_loops: int | None = None) -> int:
        T = n_loops if n_loops is not None else self.default_loops
        return sum(T if isinstance(e, RecurrentSite) else 1 for e in self.entries)


# ---------------------------------------------------------------- loading ----

def resolve_model_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    root = os.path.expanduser("~/.cache/huggingface/hub/models--" + name_or_path.replace("/", "--") + "/snapshots")
    snaps = sorted(glob.glob(os.path.join(root, "*")))
    if not snaps:
        raise FileNotFoundError(f"model {name_or_path} not found locally ({root})")
    return snaps[-1]


def load_state_dict(model_dir: str, device: str = "cpu") -> dict:
    from safetensors import safe_open
    sd = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt", device=device) as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)
    return sd


def load_qwen3(name_or_path: str, dtype=torch.bfloat16, device: str = "cpu") -> tuple[Qwen3LM, str]:
    d = resolve_model_dir(name_or_path)
    cfg = QCfg.from_hf_json(os.path.join(d, "config.json"))
    sd = load_state_dict(d, device)
    with torch.device("meta"):
        m = Qwen3LM(cfg)
    new = {"embed.weight": sd["model.embed_tokens.weight"], "norm.weight": sd["model.norm.weight"]}
    if not cfg.tie:
        new["lm_head.weight"] = sd["lm_head.weight"]
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        for n in Block.LINEARS:
            sub = "self_attn." if n.endswith("_proj") and n[0] in "qkvo" else "mlp."
            new[f"entries.{i}.{n}.weight"] = sd[p + sub + n + ".weight"]
        new[f"entries.{i}.ln1.weight"] = sd[p + "input_layernorm.weight"]
        new[f"entries.{i}.ln2.weight"] = sd[p + "post_attention_layernorm.weight"]
        new[f"entries.{i}.q_norm.weight"] = sd[p + "self_attn.q_norm.weight"]
        new[f"entries.{i}.k_norm.weight"] = sd[p + "self_attn.k_norm.weight"]
    new = {k: v.to(dtype) for k, v in new.items()}
    m.load_state_dict(new, assign=True)
    for i, e in enumerate(m.entries):
        e.t_end = i
    for p in m.parameters():
        p.requires_grad_(False)
    return m, d


def export_hf_state_dict(model: Qwen3LM) -> dict:
    """Plain (non-pruned, no adapters) model -> HF-named state dict (for saving a tuned teacher)."""
    sd = {"model.embed_tokens.weight": model.embed.weight.detach().cpu(),
          "model.norm.weight": model.norm.weight.detach().cpu()}
    if not model.cfg.tie:
        sd["lm_head.weight"] = model.lm_head.weight.detach().cpu()
    for i, b in enumerate(model.entries):
        p = f"model.layers.{i}."
        for n in Block.LINEARS:
            sub = "self_attn." if n[0] in "qkvo" and n.endswith("_proj") else "mlp."
            sd[p + sub + n + ".weight"] = getattr(b, n).weight.detach().cpu()
        sd[p + "input_layernorm.weight"] = b.ln1.weight.detach().cpu()
        sd[p + "post_attention_layernorm.weight"] = b.ln2.weight.detach().cpu()
        sd[p + "self_attn.q_norm.weight"] = b.q_norm.weight.detach().cpu()
        sd[p + "self_attn.k_norm.weight"] = b.k_norm.weight.detach().cpu()
    return {k: v.contiguous() for k, v in sd.items()}
