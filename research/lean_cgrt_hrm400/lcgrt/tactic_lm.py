"""Goal -> tactic model on traced Mathlib states (byte level, encoder-decoder).

    python -m lcgrt.tactic_lm --data mathlib-snap1 --source mathlib-full --preset tlm-hrm-S --out RUN

The encoder reads the main goal as Lean pretty-printed it (UTF-8 bytes, the last CTX bytes so
the target `⊢ ...` is always kept); the decoder writes the source tactic. Three encoders with
the same parameter count, mirroring the propositional gym:
  flat   a stack of transformer blocks
  hrm    one tied fast block iterated T times with input injection, cross-attending to slow
         slots that read the token stream every `slow_every` fast steps (two clocks)
  loop   one tied block iterated T times, no slow slots (single clock)
The decoder (2 causal layers with cross-attention) is identical for all three.

A fourth encoder, `trm`, follows the Tiny Recursive Model recipe (Jolicoeur-Martineau 2025, "Less
is More: Recursive Reasoning with Tiny Networks"): one 2-layer post-norm network updates a latent
z <- net(x + y + z) n times and then the answer state y <- net(y + z); T such recursions make a
supervision step, the first T-1 without gradient. Training is deep supervision with the carried,
detached (y, z) and ACT halting on teacher-forced exact match (`TacticLM.act_step`); the
weights evaluated are an EMA. At test time every supervision step runs. The decoder reads y.

If `--data` does not exist yet and `--source` is given, the first job assembles it as a
snapshot of the source trace (`mathlib.assemble`); every later job reads the same files.
Training reads the train split only; checkpoints, resume and the hardware guard follow
`lcgrt.train` (atomic hashed checkpoints, identity hash, exit 75 = pause, 76 = window).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import config, hw
from .errors import GymError
from .feedback import MODES as FEEDBACK_MODES, advance
from .mathlib import is_atom_row
from .train import EXIT_PAUSE, EXIT_WINDOW, load_verified, save, sha256

CTX, TGT = 384, 96                     # goal bytes kept (tail), tactic bytes incl. EOS
PAD, BOS, EOS, VOCAB = 256, 257, 258, 259
SKIP_KINDS = {"Lean.cdot"}             # a `· tac` block; its tactics have rows of their own


def norm_tactic(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def usable(row) -> bool:
    if (row.get("checker") != "certified" or not row["goals_before"] or row["kind"] in SKIP_KINDS
            or is_atom_row(row)):
        return False
    return 0 < len(norm_tactic(row["tactic"]).encode()) < TGT


def goal_bytes(row) -> bytes:
    return row["goals_before"][0].encode()[-CTX:]


# ----------------------------------------------------------------------------- data

def ensure_dataset(name, source):
    root = config.DATA_ROOT / name
    if not (root / "manifest.json").exists():
        if not source:
            raise GymError("missing_dataset", f"{root} has no manifest and no --source to assemble from")
        from .mathlib import assemble
        # Assemble into a private folder and rename it into place: two processes (a training
        # job and the HF uploader) may both find the dataset missing; the first rename wins.
        tmp = f"{name}.part{os.getpid()}"
        assemble(source, tmp, full=True, label=name)
        try:
            os.rename(config.DATA_ROOT / tmp, root)
        except OSError:
            if not (root / "manifest.json").exists():
                raise
            shutil.rmtree(config.DATA_ROOT / tmp)          # the other process won; keep its copy
    return json.loads((root / "manifest.json").read_text())


def pack(name, split):
    root = config.DATA_ROOT / name
    out = root / "packed_tlm" / f"{split}.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    enc, dec, el, dl, term = [], [], [], [], []
    with gzip.open(root / f"{split}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not usable(r):
                continue
            g, t = goal_bytes(r), norm_tactic(r["tactic"]).encode()
            e = np.zeros(CTX, np.uint8); e[:len(g)] = np.frombuffer(g, np.uint8)
            d = np.zeros(TGT, np.uint8); d[:len(t)] = np.frombuffer(t, np.uint8)
            enc.append(e); dec.append(d); el.append(len(g)); dl.append(len(t)); term.append(r.get("terminal", False))
    np.savez(out.with_suffix(".part.npz"), enc=np.stack(enc), dec=np.stack(dec), enc_len=np.array(el, np.int16),
             dec_len=np.array(dl, np.int16), terminal=np.array(term, bool))
    out.with_suffix(".part.npz").replace(out)
    return out


class Store:
    def __init__(self, path):
        z = np.load(path)
        self.enc, self.dec, self.el, self.dl, self.terminal = z["enc"], z["dec"], z["enc_len"], z["dec_len"], z["terminal"]
        self.n = len(self.el)

    def batch(self, idx, device):
        return make_batch(self.enc[idx], self.el[idx], device, self.dec[idx], self.dl[idx])


def make_batch(enc, el, device, dec=None, dl=None):
    B = len(el)
    e = torch.from_numpy(enc.astype(np.int64))
    pos = torch.arange(CTX)[None]
    emask = pos < torch.from_numpy(el.astype(np.int64))[:, None]
    e = e.masked_fill(~emask, PAD)
    b = {"enc": e.to(device), "enc_mask": emask.to(device)}
    if dec is not None:
        d = torch.from_numpy(dec.astype(np.int64))
        L = torch.from_numpy(dl.astype(np.int64))[:, None]
        tgt = torch.where(pos[:, :TGT] < L, d, torch.where(pos[:, :TGT] == L, EOS, PAD))
        b["dec_in"] = torch.cat([torch.full((B, 1), BOS), tgt[:, :-1]], 1).to(device)
        b["target"] = tgt.to(device)
    return b


# ----------------------------------------------------------------------------- model

@dataclass
class TLMConfig:
    arch: str = "hrm"          # hrm | loop | flat
    width: int = 320
    heads: int = 5
    layers: int = 4            # flat encoder depth
    fast_steps: int = 8        # hrm / loop iterations
    slow_every: int = 2
    slots: int = 16
    dec_width: int = 256
    dec_heads: int = 4
    dec_layers: int = 2
    mlp_ratio: int = 4
    feedback: str = "direct"  # direct | rlt | matched_damping; hrm/loop fast state only
    # trm only (left out of the identity hash of the other arms, so their runs keep their hashes)
    trm_n: int = 6             # latent updates per recursion
    trm_T: int = 3             # recursions per supervision step; the first T-1 without gradient
    trm_sup: int = 16          # supervision steps: the training maximum under ACT halting, all at test
    trm_explore: float = 0.1   # probability that a slot may not halt before a random step in [2, trm_sup]
    trm_ema: float = 0.999     # EMA of the weights; the EMA weights are validated and evaluated


TRM_FIELDS = ("trm_n", "trm_T", "trm_sup", "trm_explore", "trm_ema")
DATA_KEYS = ("enc", "enc_mask", "dec_in", "target")


def cfg_identity(cfg):
    d = asdict(cfg)
    if cfg.feedback == "direct":
        d.pop("feedback")     # existing checkpoints keep their training identities
    if cfg.arch != "trm":
        for k in TRM_FIELDS:
            d.pop(k)
    return d


PRESETS = {
    "tlm-flat-S": TLMConfig(arch="flat", width=256, heads=4, layers=4),  # 5.59M total
    "tlm-hrm-S": TLMConfig(arch="hrm", width=304, heads=8),        # 5.51M total
    "tlm-loop-S": TLMConfig(arch="loop", width=496, heads=8),      # 5.67M total
    # XS: the CPU-only profile (c3-cpu); about a quarter of S, matched within 1%.
    "tlm-flat-XS": TLMConfig(arch="flat", width=128, heads=4, layers=4, dec_width=128, dec_heads=4),  # 1.484M
    "tlm-hrm-XS": TLMConfig(arch="hrm", width=152, heads=4, dec_width=128, dec_heads=4),             # 1.473M
    "tlm-loop-XS": TLMConfig(arch="loop", width=240, heads=8, dec_width=128, dec_heads=4),           # 1.488M
    "tlm-tiny": TLMConfig(arch="hrm", width=64, heads=2, fast_steps=4, dec_width=64, dec_heads=2, dec_layers=1),
    # c4: TRM with the paper's n = 6, T = 3, 16 supervision steps; 2 layers at the decoder's width.
    "tlm-trm-4M": TLMConfig(arch="trm", width=256, heads=4, layers=2),                              # 4.01M
    # rapid gym: TRM at the XS budget (width 176, 44 per head), param-matched to flat-XS and loop-XS.
    "tlm-trm-XS": TLMConfig(arch="trm", width=176, heads=4, layers=2, dec_width=128, dec_heads=4),  # 1.491M
    "tlm-trm-tiny": TLMConfig(arch="trm", width=64, heads=2, layers=2, trm_n=2, trm_T=2, trm_sup=4,
                              dec_width=64, dec_heads=2, dec_layers=1),
}


class Attn(nn.Module):
    def __init__(self, d, heads, dk=None):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(dk or d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, mem, mask=None, causal=False):
        B, n, d = x.shape
        q = self.q(x).view(B, n, self.h, -1).transpose(1, 2)
        k, v = self.kv(mem).view(B, mem.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        return self.o(y.transpose(1, 2).reshape(B, n, d))


def mlp(d, r):
    return nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d))


class Block(nn.Module):
    """Pre-LN block: self-attention, optional cross-attention, MLP."""
    def __init__(self, d, heads, r, cross=False, dk=None):
        super().__init__()
        self.n1, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attn(d, heads)
        self.n2 = nn.LayerNorm(d) if cross else None
        self.cross = Attn(d, heads, dk) if cross else None
        self.ff = mlp(d, r)

    def forward(self, x, mask=None, mem=None, mem_mask=None, causal=False):
        h = self.n1(x)
        x = x + self.att(h, h, mask, causal)
        if self.cross is not None:
            x = x + self.cross(self.n2(x), mem, mem_mask)
        return x + self.ff(self.n3(x))


def _rms(x):
    return F.rms_norm(x, (x.shape[-1],), eps=1e-5)


class PostBlock(nn.Module):
    """Post-norm block (TRM): x = rms(x + attn(x)); x = rms(x + mlp(x)), RMS norm without weights."""
    def __init__(self, d, heads, r):
        super().__init__()
        self.att = Attn(d, heads)
        self.ff = mlp(d, r)

    def forward(self, x, mask=None):
        x = _rms(x + self.att(x, x, mask))
        return _rms(x + self.ff(x))


class TacticLM(nn.Module):
    def __init__(self, cfg: TLMConfig):
        super().__init__()
        if cfg.feedback not in FEEDBACK_MODES or (cfg.arch not in ("hrm", "loop") and cfg.feedback != "direct"):
            raise ValueError(f"feedback {cfg.feedback!r} is not supported for {cfg.arch!r}")
        self.cfg = cfg
        d, r = cfg.width, cfg.mlp_ratio
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(CTX, d) * 0.02)
        if cfg.arch == "flat":
            self.blocks = nn.ModuleList(Block(d, cfg.heads, r) for _ in range(cfg.layers))
        elif cfg.arch == "trm":
            self.trm_blocks = nn.ModuleList(PostBlock(d, cfg.heads, r) for _ in range(cfg.layers))
            self.register_buffer("y_init", torch.randn(d))          # fixed initial states, as in TRM
            self.register_buffer("z_init", torch.randn(d))
            self.q_head = nn.Linear(d, 1)                           # halting logit; starts at -5 (no halting)
            nn.init.zeros_(self.q_head.weight); nn.init.constant_(self.q_head.bias, -5.0)
        else:
            self.fast = Block(d, cfg.heads, r, cross=cfg.arch == "hrm")
            if cfg.arch == "hrm":
                self.slow = Block(d, cfg.heads, r, cross=True)
                self.slot_init = nn.Parameter(torch.randn(cfg.slots, d) * 0.02)
        self.enc_norm = nn.LayerNorm(d)
        D = cfg.dec_width
        self.mem_proj = nn.Linear(d, D) if d != D else nn.Identity()
        self.demb = nn.Embedding(VOCAB, D)
        self.dpos = nn.Parameter(torch.randn(TGT, D) * 0.02)
        self.dec = nn.ModuleList(Block(D, cfg.dec_heads, r, cross=True) for _ in range(cfg.dec_layers))
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, VOCAB))
        self.grad_ckpt = False      # recompute each block / recurrent step in backward (same math)

    def _run(self, mod, *args):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(mod, *args, use_reentrant=False)
        return mod(*args)

    def encode(self, enc, enc_mask, fast_steps=None, *, trace=None, initial_delta=None):
        """Optional detached traces/initial-state perturbation are for diagnostics only."""
        cfg = self.cfg
        if (trace is not None or initial_delta is not None) and cfg.arch not in ("hrm", "loop"):
            raise ValueError("feedback diagnostics require a loop or hrm encoder")
        x0 = self.emb(enc) + self.pos[: enc.shape[1]]
        km = enc_mask[:, None, None, :]
        if cfg.arch == "flat":
            x = x0
            for blk in self.blocks:
                x = self._run(blk, x, km)
            return self.mem_proj(self.enc_norm(x)), enc_mask
        if cfg.arch == "trm":                                    # inference: every supervision step
            y, z = self.trm_init(*enc.shape)
            for _ in range(fast_steps or cfg.trm_sup):
                y, z = self.deep_recursion(x0, y, z, km)
            return self.mem_proj(self.enc_norm(y)), enc_mask
        T = fast_steps or cfg.fast_steps
        if T < 1:
            raise ValueError("fast_steps must be positive")
        x = torch.zeros_like(x0)
        if initial_delta is not None:
            if initial_delta.shape != x.shape:
                raise ValueError("initial_delta must match the full token state")
            x = x + initial_delta
        s = self.slot_init[None].expand(x0.shape[0], -1, -1) if cfg.arch == "hrm" else None
        for t in range(T):
            previous = x
            endpoint = self._run(self.fast, x + x0, km, s)       # input injection every step
            x = advance(previous, endpoint, cfg.feedback, t, T)
            if s is not None and ((t + 1) % cfg.slow_every == 0 or t == T - 1):
                s = self._run(self.slow, s, None, x, km)          # slow clock reads the tokens
            if trace is not None:
                trace.append({"input": previous.detach().clone(), "state": x.detach().clone(),
                              "endpoint": endpoint.detach().clone(),
                              "slow": s.detach().clone() if s is not None else None})
        mem, mmask = x, enc_mask
        if s is not None:                                        # decoder also sees the slots
            mem = torch.cat([x, s], 1)
            mmask = torch.cat([enc_mask, torch.ones(s.shape[:2], dtype=torch.bool, device=s.device)], 1)
        return self.mem_proj(self.enc_norm(mem)), mmask

    # ---- trm
    def trm_init(self, B, L):
        return self.y_init.expand(B, L, -1), self.z_init.expand(B, L, -1)

    def _net(self, h, km):
        for blk in self.trm_blocks:
            h = blk(h, km)
        return h

    def _latent(self, x, y, z, km):
        """One recursion: n latent updates z <- net(x + y + z), then the answer update y <- net(y + z)."""
        for _ in range(self.cfg.trm_n):
            z = self._run(self._net, z + x + y, km)
        return self._run(self._net, y + z, km), z

    def deep_recursion(self, x, y, z, km):
        """T recursions; only the last one carries gradient."""
        with torch.no_grad():
            for _ in range(self.cfg.trm_T - 1):
                y, z = self._latent(x, y, z, km)
        return self._latent(x, y, z, km)

    def q_logit(self, y, mask):
        m = mask[..., None].to(y.dtype)
        return self.q_head((y * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1).float()

    def act_step(self, carry, b, rng):
        """One TRM training step over a batch of slots (the carry scheme of the TRM/HRM code).

        A slot that halted at the previous step takes this step's sample at its index and restarts
        from the initial (y, z); every other slot continues its own sample from the carried,
        detached (y, z). Loss: tactic cross-entropy + 0.5 x BCE(halting logit, teacher-forced exact
        match). A slot halts after trm_sup steps or when its halting logit is > 0, except that with
        probability trm_explore it may not halt before a random step in [2, trm_sup]. `carry=None`
        starts every slot fresh (also after a resume: the carry is not checkpointed)."""
        cfg, dev = self.cfg, b["enc"].device
        B, L = b["enc"].shape
        y0, z0 = self.trm_init(B, L)
        if carry is None:
            carry = {**{k: b[k] for k in DATA_KEYS}, "y": y0, "z": z0,
                     "steps": torch.zeros(B, dtype=torch.long, device=dev),
                     "halted": torch.ones(B, dtype=torch.bool, device=dev)}
        h = carry["halted"]
        data = {k: torch.where(h.view(-1, *([1] * (b[k].dim() - 1))), b[k], carry[k]) for k in DATA_KEYS}
        y = torch.where(h[:, None, None], y0, carry["y"])
        z = torch.where(h[:, None, None], z0, carry["z"])
        steps = torch.where(h, torch.zeros_like(carry["steps"]), carry["steps"]) + 1
        x = self.emb(data["enc"]) + self.pos[:L]
        y, z = self.deep_recursion(x, y, z, data["enc_mask"][:, None, None, :])
        logits = self.decode(data["dec_in"], self.mem_proj(self.enc_norm(y)), data["enc_mask"])
        q = self.q_logit(y, data["enc_mask"])
        tgt = data["target"]
        m = tgt != PAD
        with torch.no_grad():
            exact = ((logits.argmax(-1) == tgt) | ~m).all(-1)
            halted = steps >= cfg.trm_sup
            if cfg.trm_sup > 1:                                  # ACT only with more than one step
                explore = torch.from_numpy(rng.random(B) < cfg.trm_explore).to(dev)
                min_steps = torch.from_numpy(rng.integers(2, cfg.trm_sup + 1, B)).to(dev)
                halted |= (q > 0) & (~explore | (steps >= min_steps))
        loss = F.cross_entropy(logits[m], tgt[m]) + 0.5 * F.binary_cross_entropy_with_logits(q, exact.float())
        carry = {**data, "y": y.detach(), "z": z.detach(), "steps": steps, "halted": halted}
        info = {"fresh": int(h.sum()), "exact": float(exact.float().mean()), "halts": int(halted.sum()),
                "halt_steps": int(steps[halted].sum())}
        return loss, carry, info

    def decode(self, dec_in, mem, mmask):
        y = self.demb(dec_in) + self.dpos[: dec_in.shape[1]]
        mm = mmask[:, None, None, :]
        for blk in self.dec:
            y = blk(y, None, mem, mm, causal=True)
        return self.head(y).float()

    def forward(self, b, fast_steps=None):
        mem, mmask = self.encode(b["enc"], b["enc_mask"], fast_steps)
        return self.decode(b["dec_in"], mem, mmask)

    @torch.no_grad()
    def beam(self, enc, enc_mask, k=8, max_len=TGT, fast_steps=None, guard=None):
        """Top-k tactic byte strings per goal (beam search, length-unnormalised log-prob)."""
        B = enc.shape[0]
        if guard is not None:
            guard()
        mem, mmask = self.encode(enc, enc_mask, fast_steps)
        mem = mem.repeat_interleave(k, 0); mmask = mmask.repeat_interleave(k, 0)
        seq = torch.full((B * k, 1), BOS, device=enc.device)
        score = torch.zeros(B, k, device=enc.device); score[:, 1:] = -1e9
        done = torch.zeros(B * k, dtype=torch.bool, device=enc.device)
        for _ in range(max_len):
            if guard is not None:
                guard()
            lp = F.log_softmax(self.decode(seq, mem, mmask)[:, -1], -1)          # B*k, V
            lp[:, PAD] = -1e9; lp[:, BOS] = -1e9
            lp[done] = -1e9; lp[done, PAD] = 0.0                                  # finished beams stay
            cand = (score.view(-1, 1) + lp).view(B, -1)
            score, ix = cand.topk(k, -1)
            src = (ix // VOCAB) + torch.arange(B, device=enc.device)[:, None] * k
            tok = (ix % VOCAB).view(-1, 1)
            seq = torch.cat([seq[src.view(-1)], tok], 1)
            done = done[src.view(-1)] | (tok.view(-1) == EOS)
            if bool(done.all()):
                break
        out = []
        for i in range(B):
            row = []
            for j in range(k):
                t = seq[i * k + j, 1:].tolist()
                t = t[: t.index(EOS)] if EOS in t else None
                if t is not None and all(c < 256 for c in t):
                    row.append((bytes(t).decode("utf-8", "replace"), float(score[i, j])))
            out.append(row)
        return out


def count_parameters(model):
    enc = sum(p.numel() for n, p in model.named_parameters()
              if not n.startswith(("demb", "dpos", "dec.", "head", "mem_proj")))
    return {"total": sum(p.numel() for p in model.parameters()), "encoder": enc}


# ----------------------------------------------------------------------------- training

def apply_overrides(cfg, items):
    """`FIELD=VALUE` pairs over a preset (`--cfg`, the rapid gym's knobs), each typed as the
    preset's own field. The run's identity hash covers the result, since it hashes the config."""
    kw = {}
    for item in items or ():
        k, _, v = item.partition("=")
        if k not in TLMConfig.__dataclass_fields__ or not v:
            raise GymError("bad_argument", f"--cfg {item!r}: expected FIELD=VALUE with a TLMConfig field")
        kw[k] = type(getattr(cfg, k))(v)
    return replace(cfg, **kw) if kw else cfg


IDENTITY = ("data", "preset", "steps", "batch", "lr", "warmup", "weight_decay", "seed", "random_depth", "val_split")


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def nll(model, b, T=None):
    lg = model(b, T)
    return F.cross_entropy(lg.reshape(-1, VOCAB), b["target"].reshape(-1), ignore_index=PAD)


@torch.no_grad()
def validate(model, store, device, n=2048, seed=0, amp=None, batch_size=256):
    """Bits per tactic byte (incl. EOS) and greedy exact match, on a fixed sample."""
    model.eval()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(store.n, min(n, store.n), replace=False))
    tot = cnt = 0.0
    amp = device.startswith("cuda") if amp is None else amp
    for i in range(0, len(idx), batch_size):
        b = store.batch(idx[i:i + batch_size], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lg = model(b)
        m = b["target"] != PAD
        tot += float(F.cross_entropy(lg[m], b["target"][m], reduction="sum")); cnt += int(m.sum())
    model.train()
    return {"states": len(idx), "bits_per_byte": round(tot / cnt / math.log(2), 4)}


def continuation(parent_dir, identity):
    """Verify a fork's immutable parent; permit a new budget, batch and LR schedule.

    The optimizer is retained, and data sampling continues at the parent's global step.
    The new cosine schedule starts at local step zero. Existing runs have no added key.
    """
    directory = Path(parent_dir)
    which, parent = load_verified(directory, "cpu")
    if parent is None:
        raise GymError("missing_checkpoint", str(directory))
    for key in ("data", "preset", "seed", "weight_decay", "random_depth", "val_split", "model", "split_sha256"):
        if parent["identity"].get(key) != identity.get(key):
            raise GymError("config_mismatch", f"continuation parent differs at {key}")
    if "opt" not in parent or identity["model"]["arch"] not in ("loop", "hrm"):
        raise GymError("bad_argument", "continuation requires a loop/hrm optimizer checkpoint")
    lineage = {"checkpoint_sha256": sha256(directory / f"{which}.pt"),
               "parent_identity_hash": parent["identity_hash"], "parent_step": parent["step"],
               "data_step_offset": parent.get("data_step_offset", 0) + parent["step"],
               "schedule": "new cosine schedule; optimizer moments retained"}
    return parent, lineage


def train(a):
    if a.val_items < 1 or a.val_batch < 1:
        raise GymError("bad_argument", "--val-items and --val-batch must be positive")
    if a.stop_at is not None and not 0 < a.stop_at <= a.steps:
        raise GymError("bad_argument", "--stop-at must be within the fixed training budget")
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    if device.startswith("cuda") and a.cuda_memory_fraction:
        # Windows CUDA spills allocations past VRAM into system RAM instead of failing, which
        # slowed one run 100x for 12 h and blocked every other lease; a cap makes it an OOM.
        torch.cuda.set_per_process_memory_fraction(a.cuda_memory_fraction)
    cuda = device.startswith("cuda")
    amp = cuda and a.precision == "bf16"         # the CPU always runs fp32
    if cuda and a.precision == "fp32":           # full fp32 matmuls, as on the CPU (no TF32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    duty = a.gpu_duty if cuda else 1.0
    if not 0 < duty <= 1:
        raise GymError("bad_argument", f"--gpu-duty {a.gpu_duty} is not in (0, 1]")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    limits = hw.Limits(gpu_temp_c=a.max_gpu_temp, min_disk_gib=a.min_disk_gib, require_ac=not a.allow_battery,
                       use_gpu=device.startswith("cuda"))
    ok, why, snap = hw.check(limits, out)
    if not ok:
        print(json.dumps({"event": "pause", "reason": why, "snapshot": snap}), flush=True)
        return EXIT_PAUSE
    manifest = ensure_dataset(a.data, a.source)
    cfg = apply_overrides(PRESETS[a.preset], a.cfg)
    if cfg.arch == "trm" and a.random_depth:
        raise GymError("bad_argument", "--random-depth does not apply to the trm encoder (fixed n, T)")
    ident = {k: getattr(a, k) for k in IDENTITY} | {"model": cfg_identity(cfg), "split_sha256": manifest["split_sha256"]}
    if a.val_items != 2048:
        ident["val_items"] = a.val_items
    if a.val_batch != 256:
        ident["val_batch"] = a.val_batch
    parent = None
    if a.continue_from:
        parent, ident["continuation"] = continuation(a.continue_from, ident)
        ident["continuation"]["grad_ckpt"] = a.grad_ckpt
    data_offset = ident.get("continuation", {}).get("data_step_offset", 0)
    ident_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    torch.manual_seed(a.seed)
    model = TacticLM(cfg).to(device)
    model.grad_ckpt = a.grad_ckpt
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.arch == "trm" else None
    carry, fresh, halts, halt_steps = None, 0, 0, 0
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    first = 0
    which, ck = load_verified(out, device)
    if ck is not None:
        if ck["identity_hash"] != ident_hash:
            raise GymError("config_mismatch", f"checkpoint identity {ck['identity']} != {ident}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); first = ck["step"]
        if ema is not None:
            ema.load_state_dict(ck["ema"]); fresh = ck.get("fresh", 0)
    elif parent is not None:
        model.load_state_dict(parent["model"]); opt.load_state_dict(parent["opt"])
        which = "parent"
    del parent, ck
    tr = Store(pack(a.data, "train"))
    va = Store(pack(a.data, a.val_split))            # file-level validation split: monitoring and best-checkpoint choice
    params = count_parameters(model)
    (out / "config.json").write_text(json.dumps({"identity": ident, "identity_hash": ident_hash, "parameters": params,
                                                 "train_rows": tr.n, "device": device, "precision": "bf16" if amp else "fp32",
                                                 "gpu_duty": duty, "torch": torch.__version__,
                                                 "grad_ckpt": a.grad_ckpt,
                                                 **({"continue_from": str(Path(a.continue_from).resolve())} if a.continue_from else {})},
                                                indent=1) + "\n")
    # One start event per segment, so a run resumed on another device keeps a record of each.
    print(json.dumps({"event": "start", "resumed_from": which, "step": first, "params": params, "train_rows": tr.n,
                      "device": device, "precision": "bf16" if amp else "fp32", "gpu_duty": duty,
                      "data_step_offset": data_offset}), flush=True)
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")
    t_start = t_win = t_guard = time.time(); n_win = 0; code = 0
    duty_now = duty                       # lowered while the card is hot, back up to --gpu-duty as it cools
    best_p = out / "best.json"
    best_bpb = json.loads(best_p.read_text())["bits_per_byte"] if best_p.exists() else float("inf")

    def checkpoint(step):
        extra = {"ema": ema.state_dict(), "fresh": fresh} if ema is not None else {}
        if data_offset:
            extra["data_step_offset"] = data_offset
        return save(out, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "identity": ident,
                          "identity_hash": ident_hash, "config": asdict(cfg), "preset": a.preset, **extra})
    model.train()
    end = a.stop_at if a.stop_at is not None else a.steps
    for step in range(first, end):
        t_step = time.perf_counter()
        r = np.random.default_rng([a.seed, step + data_offset])   # pure function of (seed, global step)
        b = tr.batch(r.integers(0, tr.n, a.batch), device)
        T = None
        if a.random_depth and cfg.arch != "flat":
            T = int(np.random.default_rng([a.seed, step + data_offset, 7]).integers(a.random_depth[0], a.random_depth[1] + 1))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if ema is not None:
                loss, carry, info = model.act_step(carry, b, np.random.default_rng([a.seed, step, 11]))
                fresh += info["fresh"]; halts += info["halts"]; halt_steps += info["halt_steps"]
            else:
                loss = nll(model, b, T)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(p, 1 - cfg.trm_ema)
        if duty_now < 1:                           # keep the card at most `duty_now` busy: idle in proportion
            torch.cuda.synchronize()
            time.sleep((time.perf_counter() - t_step) * (1 - duty_now) / duty_now)
        n_win += 1
        done = step + 1
        if done % a.log_every == 0 or done == a.steps:
            if not torch.isfinite(loss):
                raise GymError("nonfinite_loss", f"step {done}")
            row = {"step": done, "loss": round(loss.item(), 4), "grad_norm": round(float(gn), 3), "lr": lr_at(step, a),
                   "steps_per_s": round(n_win / (time.time() - t_win), 3), "duty": round(duty_now, 3),
                   "unix": int(time.time())}
            if ema is not None:          # samples started so far; train exact match; mean steps at halting
                row |= {"fresh": fresh, "exact": round(info["exact"], 4),
                        "steps_at_halt": round(halt_steps / halts, 2) if halts else None}
                halts = halt_steps = 0
            evaluated = ema if ema is not None else model
            if done % a.eval_every == 0 or done == a.steps:
                row["val"] = validate(evaluated, va, device, n=a.val_items, amp=amp, batch_size=a.val_batch)
                if row["val"]["bits_per_byte"] < best_bpb:       # keep the best-on-val weights
                    best_bpb = row["val"]["bits_per_byte"]
                    torch.save({"model": evaluated.state_dict(), "config": asdict(cfg), "preset": a.preset, "step": done,
                                "identity_hash": ident_hash}, out / "best.part")
                    os.replace(out / "best.part", out / "best.pt")
                    best_p.write_text(json.dumps({"step": done, "bits_per_byte": best_bpb,
                                                  "sha256": sha256(out / "best.pt")}))
            log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
            t_win = time.time(); n_win = 0
        if done % a.save_every == 0 or done == end:
            checkpoint(done)
        # Guard every --guard-every steps, and at least every 15 s for a fast model on a hot card.
        if (done % a.guard_every == 0 or (cuda and time.time() - t_guard >= 15)) and done < a.steps:
            t_guard = time.time()
            ok, why, snap = hw.check(limits, out)
            if not ok:
                checkpoint(done)
                print(json.dumps({"event": "pause", "step": done, "reason": why}), flush=True)
                code = EXIT_PAUSE; break
            temp = (snap.get("gpu") or {}).get("temp_c")     # soft thermal limit: trade duty for temperature
            if cuda and temp is not None and a.soft_gpu_temp:     # fast down, slow up: settles just under it
                if temp >= a.soft_gpu_temp:
                    duty_now = max(0.1, duty_now / 2)
                elif temp < a.soft_gpu_temp - 1:
                    duty_now = min(duty, duty_now * 1.1)
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                checkpoint(done); code = EXIT_WINDOW; break
    if code == 0 and end < a.steps:
        print(json.dumps({"event": "segment_complete", "step": end, "total_step": end + data_offset}), flush=True)
    elif code == 0:
        final = {"event": "complete", "step": a.steps, "checkpoint_sha256": sha256(out / "latest.pt"),
                 "best": json.loads(best_p.read_text()) if best_p.exists() else None,
                 "wall_s": round(time.time() - t_start, 1)} | ({"fresh": fresh} if ema is not None else {})
        (out / "complete.json").write_text(json.dumps(final, indent=1) + "\n")
        print(json.dumps(final), flush=True)
    log.close()
    return code


def load_model(ckpt_dir, device, which="best"):
    """`best`: the best-on-val weights (hash-checked against best.json); else the final checkpoint."""
    bp = Path(ckpt_dir) / "best.json"
    if which == "best" and bp.exists():
        meta = json.loads(bp.read_text())
        if sha256(Path(ckpt_dir) / "best.pt") != meta["sha256"]:
            raise GymError("checkpoint_corrupt", f"{ckpt_dir}/best.pt")
        ck = torch.load(Path(ckpt_dir) / "best.pt", map_location=device, weights_only=False)
    else:
        _, ck = load_verified(Path(ckpt_dir), device)
    if ck is None:
        raise GymError("missing_checkpoint", str(ckpt_dir))
    model = TacticLM(TLMConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="mathlib-snap1")
    p.add_argument("--source", default=None, help="trace to snapshot if --data does not exist yet")
    p.add_argument("--preset", default="tlm-hrm-S", choices=sorted(PRESETS))
    p.add_argument("--cfg", nargs="*", default=[], metavar="FIELD=VALUE", help="override preset fields, e.g. trm_sup=4")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--continue-from", default=None, help="fork a verified loop/hrm optimizer checkpoint into a new run")
    p.add_argument("--stop-at", type=int, default=None, help="checkpoint at this local step without changing recipe identity")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--val-batch", type=int, default=256, help="validation microbatch; lower for bounded-memory pilots")
    p.add_argument("--val-items", type=int, default=2048, help="fixed validation sample size (part of run identity)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--random-depth", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    p.add_argument("--val-split", default="val")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--cuda-memory-fraction", type=float, default=0.55, help="0 disables the cap")
    p.add_argument("--precision", default="bf16", choices=("bf16", "fp32"),
                   help="on CUDA: bf16 autocast, or full fp32 without TF32 (the CPU recipe); the CPU always runs fp32")
    p.add_argument("--gpu-duty", type=float, default=1.0,
                   help="on CUDA: the share of wall time this job keeps the card busy (it idles after each step)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--guard-every", type=int, default=50)
    p.add_argument("--max-minutes", type=float, default=0)
    p.add_argument("--max-gpu-temp", type=float, default=85)
    p.add_argument("--soft-gpu-temp", type=float, default=82)
    p.add_argument("--min-disk-gib", type=float, default=10)
    p.add_argument("--allow-battery", action="store_true")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    a.random_depth = list(a.random_depth) if a.random_depth else None
    return train(a)


if __name__ == "__main__":
    sys.exit(main())
