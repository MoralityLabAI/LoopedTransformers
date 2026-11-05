"""Measured throughput and peak VRAM per preset on the real training data (hardware receipt).

    python -m lcgrt.diag --out results/hw_diag.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import hw
from .model import ProofNet, count_parameters, flops_per_decision, preset
from .train import load_split, losses


def measure(name, store, steps, batch, budget, grad_ckpt, device="cuda", **over):
    cfg = preset(name, **over)
    model = ProofNet(cfg).to(device); model.grad_ckpt = grad_ckpt
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_of = store.t_n_tok[store.s_thm]
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    times, toks, samples = [], 0, 0
    for step in range(steps):
        r = np.random.default_rng([0, step])
        idx = r.integers(0, store.n_states, batch)
        mx = np.maximum.accumulate(n_of[idx])
        keep = max(1, int(((mx * np.arange(1, batch + 1)) <= budget).sum()))
        b = store.batch(idx[:keep], r, 0.5, device)
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = losses(model(b), b, 0.5, 0.2 if cfg.arch == "hrm" else 0.0)
        opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
        if step >= 5:
            times.append(time.perf_counter() - t); toks += int(b["tok_mask"].sum()); samples += keep
    dt = sum(times)
    out = {"preset": name, **over, "grad_ckpt": grad_ckpt, "params": count_parameters(model)["total"],
           "step_s": round(dt / len(times), 4), "samples_per_s": round(samples / dt, 1),
           "tokens_per_s": round(toks / dt), "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20),
           "reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20),
           "fwd_gflop_per_decision_at_42_tokens": round(flops_per_decision(cfg, 42) / 1e9, 3)}
    del model, opt; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lcgrt-v1")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="results/hw_diag.json")
    a = ap.parse_args()
    store = load_split(a.data, "train")
    rows = [{"hw_before": hw.check(hw.Limits())[2]}]
    for name, ck in [("S", False), ("flat-S", False), ("M", False), ("M", True), ("flat-M", False),
                     ("L", True), ("flat-L", False)]:
        try:
            rows.append(measure(name, store, a.steps, 256, 256 * 80, ck))
        except torch.OutOfMemoryError as e:
            rows.append({"preset": name, "grad_ckpt": ck, "oom": str(e)[:200]}); torch.cuda.empty_cache()
        print(json.dumps(rows[-1]), flush=True)
    rows.append({"hw_after": hw.check(hw.Limits())[2]})
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
