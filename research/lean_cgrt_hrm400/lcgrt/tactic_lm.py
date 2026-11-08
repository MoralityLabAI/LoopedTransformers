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

If `--data` does not exist yet and `--source` is given, the first job assembles it as a
snapshot of the source trace (`mathlib.assemble`); every later job reads the same files.
Training reads the train split only; checkpoints, resume and the hardware guard follow
`lcgrt.train` (atomic hashed checkpoints, identity hash, exit 75 = pause, 76 = window).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import config, hw
from .errors import GymError
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


PRESETS = {
    "tlm-flat-S": TLMConfig(arch="flat", width=256, heads=4, layers=4),  # 5.59M total
    "tlm-hrm-S": TLMConfig(arch="hrm", width=304, heads=8),        # 5.51M total
    "tlm-loop-S": TLMConfig(arch="loop", width=496, heads=8),      # 5.67M total
    # XS: the CPU-only profile (c3-cpu); about a quarter of S, matched within 1%.
    "tlm-flat-XS": TLMConfig(arch="flat", width=128, heads=4, layers=4, dec_width=128, dec_heads=4),  # 1.484M
    "tlm-hrm-XS": TLMConfig(arch="hrm", width=152, heads=4, dec_width=128, dec_heads=4),             # 1.473M
    "tlm-loop-XS": TLMConfig(arch="loop", width=240, heads=8, dec_width=128, dec_heads=4),           # 1.488M
    "tlm-tiny": TLMConfig(arch="hrm", width=64, heads=2, fast_steps=4, dec_width=64, dec_heads=2, dec_layers=1),
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


class TacticLM(nn.Module):
    def __init__(self, cfg: TLMConfig):
        super().__init__()
        self.cfg = cfg
        d, r = cfg.width, cfg.mlp_ratio
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(CTX, d) * 0.02)
        if cfg.arch == "flat":
            self.blocks = nn.ModuleList(Block(d, cfg.heads, r) for _ in range(cfg.layers))
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

    def encode(self, enc, enc_mask, fast_steps=None):
        cfg = self.cfg
        x0 = self.emb(enc) + self.pos[: enc.shape[1]]
        km = enc_mask[:, None, None, :]
        if cfg.arch == "flat":
            x = x0
            for blk in self.blocks:
                x = self._run(blk, x, km)
            return self.mem_proj(self.enc_norm(x)), enc_mask
        T = fast_steps or cfg.fast_steps
        x = torch.zeros_like(x0)
        s = self.slot_init[None].expand(x0.shape[0], -1, -1) if cfg.arch == "hrm" else None
        for t in range(T):
            x = self._run(self.fast, x + x0, km, s)              # input injection every step
            if s is not None and ((t + 1) % cfg.slow_every == 0 or t == T - 1):
                s = self._run(self.slow, s, None, x, km)          # slow clock reads the tokens
        mem, mmask = x, enc_mask
        if s is not None:                                        # decoder also sees the slots
            mem = torch.cat([x, s], 1)
            mmask = torch.cat([enc_mask, torch.ones(s.shape[:2], dtype=torch.bool, device=s.device)], 1)
        return self.mem_proj(self.enc_norm(mem)), mmask

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
    def beam(self, enc, enc_mask, k=8, max_len=TGT, fast_steps=None):
        """Top-k tactic byte strings per goal (beam search, length-unnormalised log-prob)."""
        B = enc.shape[0]
        mem, mmask = self.encode(enc, enc_mask, fast_steps)
        mem = mem.repeat_interleave(k, 0); mmask = mmask.repeat_interleave(k, 0)
        seq = torch.full((B * k, 1), BOS, device=enc.device)
        score = torch.zeros(B, k, device=enc.device); score[:, 1:] = -1e9
        done = torch.zeros(B * k, dtype=torch.bool, device=enc.device)
        for _ in range(max_len):
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
def validate(model, store, device, n=2048, seed=0):
    """Bits per tactic byte (incl. EOS) and greedy exact match, on a fixed sample."""
    model.eval()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(store.n, min(n, store.n), replace=False))
    tot = cnt = 0.0
    for i in range(0, len(idx), 256):
        b = store.batch(idx[i:i + 256], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            lg = model(b)
        m = b["target"] != PAD
        tot += float(F.cross_entropy(lg[m], b["target"][m], reduction="sum")); cnt += int(m.sum())
    model.train()
    return {"states": len(idx), "bits_per_byte": round(tot / cnt / math.log(2), 4)}


def train(a):
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    if device.startswith("cuda") and a.cuda_memory_fraction:
        # Windows CUDA spills allocations past VRAM into system RAM instead of failing, which
        # slowed one run 100x for 12 h and blocked every other lease; a cap makes it an OOM.
        torch.cuda.set_per_process_memory_fraction(a.cuda_memory_fraction)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    limits = hw.Limits(gpu_temp_c=a.max_gpu_temp, min_disk_gib=a.min_disk_gib, require_ac=not a.allow_battery,
                       use_gpu=device.startswith("cuda"))
    ok, why, snap = hw.check(limits, out)
    if not ok:
        print(json.dumps({"event": "pause", "reason": why, "snapshot": snap}), flush=True)
        return EXIT_PAUSE
    manifest = ensure_dataset(a.data, a.source)
    cfg = PRESETS[a.preset]
    ident = {k: getattr(a, k) for k in IDENTITY} | {"model": asdict(cfg), "split_sha256": manifest["split_sha256"]}
    ident_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    torch.manual_seed(a.seed)
    model = TacticLM(cfg).to(device)
    model.grad_ckpt = a.grad_ckpt
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    first = 0
    which, ck = load_verified(out, device)
    if ck is not None:
        if ck["identity_hash"] != ident_hash:
            raise GymError("config_mismatch", f"checkpoint identity {ck['identity']} != {ident}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); first = ck["step"]
    tr = Store(pack(a.data, "train"))
    va = Store(pack(a.data, a.val_split))            # file-level validation split: monitoring and best-checkpoint choice
    params = count_parameters(model)
    (out / "config.json").write_text(json.dumps({"identity": ident, "identity_hash": ident_hash, "parameters": params,
                                                 "train_rows": tr.n, "device": device, "torch": torch.__version__},
                                                indent=1) + "\n")
    print(json.dumps({"event": "start", "resumed_from": which, "step": first, "params": params, "train_rows": tr.n}),
          flush=True)
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")
    t_start = t_win = time.time(); n_win = 0; throttle = 0.0; code = 0
    best_p = out / "best.json"
    best_bpb = json.loads(best_p.read_text())["bits_per_byte"] if best_p.exists() else float("inf")

    def checkpoint(step):
        return save(out, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "identity": ident,
                          "identity_hash": ident_hash, "config": asdict(cfg), "preset": a.preset})
    model.train()
    for step in range(first, a.steps):
        r = np.random.default_rng([a.seed, step])                 # data order: pure function of (seed, step)
        b = tr.batch(r.integers(0, tr.n, a.batch), device)
        T = None
        if a.random_depth and cfg.arch != "flat":
            T = int(np.random.default_rng([a.seed, step, 7]).integers(a.random_depth[0], a.random_depth[1] + 1))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss = nll(model, b, T)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        n_win += 1
        if throttle:
            time.sleep(throttle)
        done = step + 1
        if done % a.log_every == 0 or done == a.steps:
            if not torch.isfinite(loss):
                raise GymError("nonfinite_loss", f"step {done}")
            row = {"step": done, "loss": round(loss.item(), 4), "grad_norm": round(float(gn), 3), "lr": lr_at(step, a),
                   "steps_per_s": round(n_win / (time.time() - t_win), 3), "throttle_s": throttle,
                   "unix": int(time.time())}
            if done % a.eval_every == 0 or done == a.steps:
                row["val"] = validate(model, va, device)
                if row["val"]["bits_per_byte"] < best_bpb:       # keep the best-on-val weights
                    best_bpb = row["val"]["bits_per_byte"]
                    torch.save({"model": model.state_dict(), "config": asdict(cfg), "preset": a.preset, "step": done,
                                "identity_hash": ident_hash}, out / "best.part")
                    os.replace(out / "best.part", out / "best.pt")
                    best_p.write_text(json.dumps({"step": done, "bits_per_byte": best_bpb,
                                                  "sha256": sha256(out / "best.pt")}))
            log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
            t_win = time.time(); n_win = 0
        if done % a.save_every == 0 or done == a.steps:
            checkpoint(done)
        if done % a.guard_every == 0 and done < a.steps:
            ok, why, snap = hw.check(limits, out)
            if not ok:
                checkpoint(done)
                print(json.dumps({"event": "pause", "step": done, "reason": why}), flush=True)
                code = EXIT_PAUSE; break
            temp = (snap.get("gpu") or {}).get("temp_c")     # soft throttle, as in lcgrt.train
            if temp is not None and a.soft_gpu_temp:
                if temp >= a.soft_gpu_temp:
                    throttle = min(2.0, max(0.05, throttle * 2))
                elif temp < a.soft_gpu_temp - 3:
                    throttle = throttle / 2 if throttle > 0.05 else 0.0
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                checkpoint(done); code = EXIT_WINDOW; break
    if code == 0:
        final = {"event": "complete", "step": a.steps, "checkpoint_sha256": sha256(out / "latest.pt"),
                 "best": json.loads(best_p.read_text()) if best_p.exists() else None,
                 "wall_s": round(time.time() - t_start, 1)}
        (out / "complete.json").write_text(json.dumps(final, indent=1) + "\n")
        print(json.dumps(final), flush=True)
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
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--random-depth", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    p.add_argument("--val-split", default="val")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--cuda-memory-fraction", type=float, default=0.55, help="0 disables the cap")
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
