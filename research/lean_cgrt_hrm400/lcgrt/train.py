"""Train an HRM or flat policy on Lean-extracted proof states.

    python -m lcgrt.train --data lcgrt-v1 --preset S --steps 3000 --out D:/lean_cgrt/runs/S_s17

Resumable and unattended-safe:
  * checkpoints are written atomically, hashed, and rotated (latest -> lkg); resume picks
    the newest checkpoint whose hash matches its sidecar;
  * resume refuses a changed training identity (every hyperparameter that changes the
    result is in the identity hash);
  * the data order is a pure function of (seed, step), so a resumed run sees the same
    samples as an uninterrupted one;
  * a hardware guard (GPU temperature, GPU power cap, disk, RAM, AC power) saves and exits
    with code 75; `--max-minutes` saves and exits with 76 (both mean "resume me").
Only certified theorems are used. Training uses the train split only; `val` is monitored,
the final checkpoint is the one evaluated (no checkpoint selection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import config, hw
from .dataset import iter_records
from .packed import Packed, pack
from .errors import GymError
from .model import ProofNet, count_parameters, preset

EXIT_PAUSE, EXIT_WINDOW = 75, 76
IDENTITY_KEYS = ("data", "preset", "fast_steps", "slow_every", "graph", "steps", "batch", "token_budget", "lr",
                 "warmup", "weight_decay", "seed", "fail_aug", "w_atom", "w_halt", "width", "layers")


def load_split(name, split):
    path = config.DATA_ROOT / name / "packed" / f"{split}.npz"
    if not path.exists():
        pack(name, split)
    return Packed(path)


def losses(outs, b, w_atom, w_halt):
    """Deep supervision: every slow tick's readout is scored; ticks are stacked into one op."""
    opt, lab = b["optimal"], b["atom_label"]
    lg = torch.stack([o["logits"] for o in outs])                        # R,B,N
    pol = (torch.logsumexp(lg, -1) - torch.logsumexp(lg.masked_fill(~opt, -1e4), -1)).mean()
    known = (lab >= 0)[None].expand_as(lg)
    at = torch.stack([o["atom"] for o in outs])
    atom = torch.nn.functional.binary_cross_entropy_with_logits(at[known], lab[None].expand_as(lg)[known].float())
    hit = opt[None].expand_as(lg).gather(2, lg.argmax(-1, keepdim=True)).squeeze(2).float()   # R,B
    halt = torch.zeros((), device=lg.device)
    if w_halt and outs[0]["halt"] is not None:
        hl = torch.stack([o["halt"] for o in outs])
        halt = torch.nn.functional.binary_cross_entropy_with_logits(hl, hit)
    total = pol + w_atom * atom + w_halt * halt
    return total, {"policy": pol.detach(), "atom": atom.detach(), "halt": halt.detach(),
                   "top1_by_tick": hit.mean(1).detach()}


def scalars(parts):
    out = {}
    for k, v in parts.items():
        v = v.float().cpu()
        out[k] = [round(float(x), 4) for x in v] if v.ndim else round(float(v), 4)
    return out


@torch.no_grad()
def validate(model, store, device, n=4096, seed=0):
    """Top-1 policy accuracy per slow tick, overall and on states where the choice matters."""
    model.eval()
    rng = np.random.default_rng(seed)
    idx = np.arange(store.n_states) if store.n_states <= n else rng.choice(store.n_states, n, replace=False)
    hits = hard_hits = None; atom_ok = atom_n = 0
    hard = store.s_n_opt[idx] < store.s_n_legal[idx]
    for i in range(0, len(idx), 256):
        b = store.batch(idx[i:i + 256], rng, 0.0, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            outs = model(b)
        hm = torch.from_numpy(hard[i:i + 256]).to(device)
        h = [b["optimal"].gather(1, o["logits"].argmax(-1, keepdim=True)).squeeze(1) for o in outs]
        tot = [int(x.sum()) for x in h]; hh = [int((x & hm).sum()) for x in h]
        hits = tot if hits is None else [a + c for a, c in zip(hits, tot)]
        hard_hits = hh if hard_hits is None else [a + c for a, c in zip(hard_hits, hh)]
        lab = b["atom_label"]; known = lab >= 0
        atom_ok += int(((outs[-1]["atom"][known] > 0).long() == lab[known]).sum())
        atom_n += int(known.sum())
    model.train()
    nh = int(hard.sum())
    return {"states": len(idx), "top1_by_tick": [round(x / len(idx), 4) for x in hits],
            "choice_states": nh, "choice_top1_by_tick": [round(x / max(1, nh), 4) for x in hard_hits],
            "atom_acc": round(atom_ok / max(1, atom_n), 4)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def save(out: Path, state: dict):
    part = out / "latest.part"
    with open(part, "wb") as f:
        torch.save(state, f)
        f.flush(); os.fsync(f.fileno())
    digest = sha256(part)
    if (out / "latest.pt").exists() and (out / "latest.json").exists():
        os.replace(out / "latest.pt", out / "lkg.pt")
        os.replace(out / "latest.json", out / "lkg.json")
    os.replace(part, out / "latest.pt")
    side = out / "latest.json.part"
    side.write_text(json.dumps({"sha256": digest, "step": state["step"], "unix": int(time.time())}))
    os.replace(side, out / "latest.json")
    return digest


def load_verified(out: Path, device):
    for name in ("latest", "lkg"):
        pt, js = out / f"{name}.pt", out / f"{name}.json"
        if pt.exists() and js.exists():
            meta = json.loads(js.read_text())
            if sha256(pt) == meta["sha256"]:
                return name, torch.load(pt, map_location=device, weights_only=False)
            print(json.dumps({"event": "checkpoint_corrupt", "file": str(pt)}), flush=True)
    return None, None


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def identity(a, cfg):
    d = {k: getattr(a, k, None) for k in IDENTITY_KEYS}
    d.update(width=cfg.width, layers=cfg.layers, arch=cfg.arch, heads=cfg.heads, slots=cfg.slots)
    return d, hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def train(a):
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    if device.startswith("cuda") and a.cuda_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(a.cuda_memory_fraction)
    cfg = preset(a.preset, **({"fast_steps": a.fast_steps} if a.fast_steps else {}),
                 **({"slow_every": a.slow_every} if a.slow_every else {}), graph=not a.no_graph)
    ident, ident_hash = identity(a, cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    limits = hw.Limits(gpu_temp_c=a.max_gpu_temp, min_disk_gib=a.min_disk_gib, require_ac=not a.allow_battery)
    ok, why, snap = hw.check(limits, out)
    if not ok:
        print(json.dumps({"event": "pause", "reason": why, "snapshot": snap}), flush=True)
        return EXIT_PAUSE
    torch.manual_seed(a.seed)
    model = ProofNet(cfg).to(device)
    model.grad_ckpt = a.grad_ckpt
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    first = 0
    which, ck = load_verified(out, device)
    if ck is not None:
        if ck["identity_hash"] != ident_hash:
            raise GymError("config_mismatch", f"checkpoint identity {ck['identity']} != {ident}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        first = ck["step"]
        torch.set_rng_state(ck["torch_rng"].cpu())   # map_location may have moved it to CUDA
    params = count_parameters(model)
    t_load = time.time()
    train_store = load_split(a.data, "train")
    val_store = load_split(a.data, "val")
    (out / "config.json").write_text(json.dumps({
        "identity": ident, "identity_hash": ident_hash, "model": cfg.to_json(), "parameters": params,
        "train_theorems": train_store.n_theorems, "train_states": train_store.n_states,
        "val_states": val_store.n_states, "device": device,
        "torch": torch.__version__, "python": sys.version.split()[0]}, indent=1) + "\n")
    print(json.dumps({"event": "start", "resumed_from": which, "step": first, "params": params["total"],
                      "train_states": train_store.n_states, "load_s": round(time.time() - t_load, 1)}), flush=True)

    n_of = train_store.t_n_tok[train_store.s_thm]

    def make(step):
        r = np.random.default_rng([a.seed, step])
        idx = r.integers(0, train_store.n_states, a.batch)
        # Padded-token budget: rare very large graphs shorten the batch instead of
        # spiking VRAM. Deterministic in (seed, step), identical for every model.
        mx = np.maximum.accumulate(n_of[idx])
        keep = max(1, min(len(idx), int(((mx * np.arange(1, len(idx) + 1)) <= a.token_budget).sum())))
        return train_store.batch(idx[:keep], r, a.fail_aug, device)

    q: queue.Queue = queue.Queue(maxsize=3)
    stop = threading.Event()

    def producer():
        s = first
        while not stop.is_set() and s < a.steps:
            q.put((s, make(s))); s += 1
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")
    t_start = time.time(); t_win = time.time(); n_win = 0; tok_win = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    code = 0
    step = first

    def checkpoint(step):
        return save(out, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                          "identity": ident, "identity_hash": ident_hash, "config": cfg.to_json(),
                          "torch_rng": torch.get_rng_state()})
    while True:
        item = q.get()
        if item is None:
            break
        step, b = item
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            outs = model(b)
            loss, parts = losses(outs, b, a.w_atom, a.w_halt if cfg.arch == "hrm" else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % a.log_every == 0 and not torch.isfinite(loss):
            stop.set()
            raise GymError("nonfinite_loss", f"step {step}: {scalars(parts)}")
        n_win += 1; tok_win += int(b["tok_mask"].sum())
        done = step + 1
        if done % a.log_every == 0 or done == a.steps:
            dt = time.time() - t_win
            row = {"step": done, "loss": round(float(loss), 4), **scalars(parts),
                   "grad_norm": round(float(gn), 3), "lr": lr_at(step, a),
                   "steps_per_s": round(n_win / dt, 3), "tokens_per_s": round(tok_win / dt),
                   "samples_per_s": round(n_win * a.batch / dt, 1), "unix": int(time.time())}
            if device.startswith("cuda"):
                row["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
                torch.cuda.reset_peak_memory_stats()
            if done % a.eval_every == 0 or done == a.steps:
                row["val"] = validate(model, val_store, device)
                row["hw"] = hw.check(limits, out)[2]
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
            t_win = time.time(); n_win = 0; tok_win = 0
        if done % a.save_every == 0 or done == a.steps:
            checkpoint(done)
        if done % a.guard_every == 0 and done < a.steps:
            ok, why, snap = hw.check(limits, out)
            if not ok:
                checkpoint(done)
                print(json.dumps({"event": "pause", "step": done, "reason": why, "snapshot": snap}), flush=True)
                code = EXIT_PAUSE; break
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                checkpoint(done)
                print(json.dumps({"event": "window_end", "step": done}), flush=True)
                code = EXIT_WINDOW; break
    stop.set()
    if code == 0:
        digest = sha256(out / "latest.pt")
        final = {"event": "complete", "step": a.steps, "checkpoint_sha256": digest,
                 "wall_s": round(time.time() - t_start, 1)}
        (out / "complete.json").write_text(json.dumps(final, indent=1) + "\n")
        print(json.dumps(final), flush=True)
    return code


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="lcgrt-v1")
    p.add_argument("--preset", default="S")
    p.add_argument("--fast-steps", type=int, default=None)
    p.add_argument("--slow-every", type=int, default=None)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--token-budget", type=int, default=256 * 80)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--fail-aug", type=float, default=0.5)
    p.add_argument("--w-atom", type=float, default=0.5)
    p.add_argument("--w-halt", type=float, default=0.2)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--guard-every", type=int, default=50)
    p.add_argument("--max-minutes", type=float, default=0)
    p.add_argument("--max-gpu-temp", type=float, default=85)
    p.add_argument("--min-disk-gib", type=float, default=10)
    p.add_argument("--allow-battery", action="store_true")
    p.add_argument("--limit-theorems", type=int, default=None)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    a.no_graph = bool(a.no_graph)
    a.graph = not a.no_graph
    cfg = preset(a.preset)
    a.width, a.layers = cfg.width, cfg.layers
    return train(a)


if __name__ == "__main__":
    sys.exit(main())
