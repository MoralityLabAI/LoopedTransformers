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
    a = ap.parse_args(argv)
    diag = json.loads(Path(a.diag).read_text())
    rows = [r for r in diag if "preset" in r]
    bytes_per = calibrate(rows)
    cands = {}
    with torch.device("meta"):
        for name, (d, h, fl, sl) in {"HRM-400M-a": (1760, 16, 4, 4), "HRM-400M-b": (2048, 16, 3, 3),
                                      "HRM-400M-c (1+1 blocks)": (3456, 24, 1, 1)}.items():
            cfg = ModelConfig(width=d, heads=h)
            cands[name] = {"width": d, "heads": h, "fast_layers": fl, "slow_layers": sl,
                           "params": manifest(DeepHRM(cfg, fl, sl))}
        for name, (d, h, L) in {"flat-400M": (2304, 16, 6), "flat-400M-deep": (1632, 16, 12)}.items():
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
    out = {"label": "ESTIMATE: exact parameter counts; memory extrapolated from RTX 5080 measurements",
           "calibration_bytes_per_token_width_application": round(bytes_per, 2),
           "calibration_source": a.diag, "microbatch_padded_tokens": tokens, "candidates": cands}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {"params": v["params"]["total"], **v["estimate_gib"]} for k, v in cands.items()}, indent=1))


if __name__ == "__main__":
    main()
