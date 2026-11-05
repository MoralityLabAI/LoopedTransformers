"""Exact parameter manifests and memory/time projections for scaled-up configurations.

    python -m lcgrt.scale --out results/scale_400m.json

The 400M candidates stack several blocks per clock (`fast_layers`, `slow_layers`), tied
across recurrent steps as in the local models. Counts come from instantiating the modules
on the meta device. Memory and throughput are projections calibrated on the RTX 5080
measurements in results/hw_diag.json; they are estimates, not measurements, and are
labelled as such in the output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from .model import FastBlock, ModelConfig, ProofNet, Readout, SlowBlock, count_parameters


class DeepHRM(nn.Module):
    """HRM with `fast_layers` blocks per fast step and `slow_layers` per slow tick (tied across steps)."""

    def __init__(self, cfg: ModelConfig, fast_layers: int, slow_layers: int):
        super().__init__()
        d = cfg.width
        self.feat = nn.Linear(12, d); self.ttype = nn.Embedding(5, d); self.readout = Readout(d)
        self.fast = nn.ModuleList(FastBlock(d, cfg.heads, cfg.mlp_ratio, cross=True) for _ in range(fast_layers))
        self.slow = nn.ModuleList(SlowBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(slow_layers))
        self.slot_init = nn.Parameter(torch.zeros(cfg.slots, d)); self.glob = nn.Linear(5, d)
        self.halt = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))


class DeepFlat(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.width
        self.feat = nn.Linear(12, d); self.ttype = nn.Embedding(5, d); self.readout = Readout(d)
        self.blocks = nn.ModuleList(FastBlock(d, cfg.heads, cfg.mlp_ratio, cross=False) for _ in range(cfg.layers))


def manifest(model):
    c = count_parameters(model)
    return {"total": c["total"], "by_module": c["by_module"]}


def calibrate(diag_rows):
    """Bytes of saved activation per (padded token x width x block application), from non-checkpointed runs."""
    out = []
    for r in diag_rows:
        if r.get("preset") in ("S", "M") and not r.get("grad_ckpt") and "peak_vram_mib" in r:
            d = {"S": 384, "M": 672}[r["preset"]]
            apps = 8 + 4            # fast steps + slow ticks
            weights = r["params"] * 16
            out.append((r["peak_vram_mib"] * 2 ** 20 - weights) / (256 * 80 * d * apps))
    return sum(out) / len(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="results/hw_diag.json")
    ap.add_argument("--out", default="results/scale_400m.json")
    # 48.1 TFLOPS: fp16 4096^3 matmul measured on this RTX 5080 Laptop (2026-11-02, shared card).
    ap.add_argument("--local-matmul-tflops", type=float, default=48.1)
    # Assumed, not measured: a dense bf16 4096^3 matmul on one H100 SXM.
    ap.add_argument("--h100-matmul-tflops", type=float, default=650.0)
    a = ap.parse_args(argv)
    diag = json.loads(Path(a.diag).read_text())
    rows = [r for r in diag if "preset" in r]
    bytes_per = calibrate(rows)
    cands = {}
    with torch.device("meta"):
        for name, (d, h, fl, sl) in {"HRM-400M-a": (1744, 16, 4, 4), "HRM-400M-b": (2008, 8, 3, 3),
                                      "HRM-400M-c (1+1 blocks)": (3376, 16, 1, 1)}.items():
            cfg = ModelConfig(width=d, heads=h)
            cands[name] = {"width": d, "heads": h, "fast_layers": fl, "slow_layers": sl,
                           "params": manifest(DeepHRM(cfg, fl, sl))}
        for name, (d, h, L) in {"flat-400M": (2304, 16, 6), "flat-400M-deep": (1648, 16, 12)}.items():
            cfg = ModelConfig(arch="flat", width=d, heads=h, layers=L)
            cands[name] = {"width": d, "heads": h, "layers": L, "params": manifest(DeepFlat(cfg))}
    # Activation memory per microbatch of 20,480 padded tokens (local token budget), bf16.
    tokens = 256 * 80
    for name, c in cands.items():
        d = c["width"]
        apps = (8 * c["fast_layers"] + 4 * c["slow_layers"]) if "fast_layers" in c else c["layers"]
        full = bytes_per * tokens * d * apps
        # Checkpoint every fast step: keep step boundaries, recompute one step's activations.
        per_step = bytes_per * tokens * d * (c.get("fast_layers", 1) + c.get("slow_layers", 0))
        ckpt = per_step + 2 * tokens * d * (8 + 4)
        state = c["params"]["total"] * 16         # bf16 weights + fp32 master + Adam m,v + grads
        c["estimate_gib"] = {"weights_optimizer": round(state / 2 ** 30, 2),
                             "activations_no_ckpt": round(full / 2 ** 30, 1),
                             "activations_ckpt_per_fast_step": round(ckpt / 2 ** 30, 1),
                             "total_ckpt": round((state + ckpt) / 2 ** 30, 1)}
    # Throughput: achieved training FLOP/s locally (fwd x3, x4 with recompute), from hw_diag.
    local = {}
    for r in rows:
        if "samples_per_s" not in r:
            continue
        fwd = r["fwd_gflop_per_decision_at_42_tokens"] * 1e9
        mult = 4 if r.get("grad_ckpt") else 3
        local[f"{r['preset']}{'+ckpt' if r.get('grad_ckpt') else ''}"] = round(r["samples_per_s"] * fwd * mult / 1e12, 2)
    best_local = max(local.values())
    util = best_local / a.local_matmul_tflops
    for name, c in cands.items():
        d = c["width"]; N = 42
        attn = 2 * N * 4 * d * d + 4 * N * N * d
        ff = 2 * N * 8 * d * d
        if "fast_layers" in c:
            cross = 2 * N * 2 * d * d + 2 * 8 * 2 * d * d + 4 * N * 8 * d
            slow = 2 * 8 * 4 * d * d * 2 + 4 * 8 * N * d + 2 * 8 * 8 * d * d
            fwd = 8 * c["fast_layers"] * (attn + cross + ff) + 4 * (c["slow_layers"] * slow + 2 * N * 3 * d * d)
        else:
            fwd = c["layers"] * (attn + ff) + 2 * N * 3 * d * d
        train_flop = fwd * 4                              # fwd + bwd + recompute
        sps = a.h100_matmul_tflops * util * 1e12 / train_flop
        c["projection"] = {"fwd_gflop_per_decision_at_42_tokens": round(fwd / 1e9, 2),
                           "h100_samples_per_s": round(sps, 1),
                           "h100_hours_per_million_samples": round(1e6 / sps / 3600, 2)}
    out = {"label": "ESTIMATE: exact parameter counts; memory and H100 time extrapolated from RTX 5080 measurements",
           "calibration_bytes_per_token_width_application": round(bytes_per, 2),
           "local_achieved_train_tflops": local, "local_matmul_ceiling_tflops": a.local_matmul_tflops,
           "utilisation_assumed": round(util, 3), "h100_matmul_ceiling_tflops_assumed": a.h100_matmul_tflops,
           "calibration_source": a.diag, "microbatch_padded_tokens": tokens, "candidates": cands}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"local_tflops": local, "util": out["utilisation_assumed"],
                      **{k: {"params": v["params"]["total"], **v["estimate_gib"], **v["projection"]}
                         for k, v in cands.items()}}, indent=1))


if __name__ == "__main__":
    main()
