"""SPC-02 native LoopedChase phase-trace *control*, not an extrapolation claim.

A small untrained tied-layer transformer is run for 42 recurrent loops.
Its recorded answer-token state is analyzed with the same periodic-proxy and
alias sparsity diagnostics as cached checkpoint traces. No training or cloud
GPU is required. The checkpoint field is explicitly null in the receipt.
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np

from experiments.spectral_state_cost_v0.phase_stability import trace_spectral_diagnostics


def run_native_smoke(seed: int = 13012902, batch: int = 4,
                     n_nodes: int = 8, d_model: int = 64,
                     loops: int = 42, period: int = 21,
                     stride: int = 7, k: int = 2) -> dict[str, object]:
    if batch < 1 or n_nodes < 2 or d_model < 4 or loops < period or period % stride:
        raise ValueError("invalid bounded native smoke configuration")
    if k < 1 or k > stride or d_model % 4:
        raise ValueError("k or head dimension violates LoopedChase configuration")
    import torch
    from research_gym.scripts.loopscope_toy import LoopedChase, make_batch

    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    model = LoopedChase(n_nodes=n_nodes, d=d_model, layers=1, heads=4).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokens, hops = make_batch(n_nodes, batch, seed + 1, loops)
    with torch.no_grad():
        states = model(tokens, loops)   # [loops, batch, hidden_dim]
    if states.shape != (loops, batch, d_model):
        raise RuntimeError("native LoopedChase state contract changed")
    trajectory = states.detach().cpu().numpy().transpose(1, 0, 2)
    if not np.isfinite(trajectory).all():
        raise RuntimeError("nonfinite state in native smoke")
    diagnostics = [trace_spectral_diagnostics(trajectory[i], period, stride, k)
                   for i in range(batch)]
    relative = [float(x["periodic_proxy_relative_residual"]) for x in diagnostics]
    sparse = [float(x["spectral_tail_energy_fraction_after_k_terms"]) for x in diagnostics]
    return {
        "campaign": "SPC-02-native",
        "model": "research_gym.scripts.loopscope_toy.LoopedChase",
        "trained_checkpoint": None,
        "training_performed": False,
        "model_seed": seed,
        "batch_size": batch,
        "n_nodes": n_nodes,
        "d_model": d_model,
        "loop_count": loops,
        "period": period,
        "stride": stride,
        "max_modes_per_alias": k,
        "cpu": True,
        "torch_version": torch.__version__,
        "answer_position_state_shape": list(states.shape),
        "all_finite": True,
        "mean_relative_periodization_residual": float(np.mean(relative)),
        "mean_spectral_tail_after_k_terms": float(np.mean(sparse)),
        "per_item": diagnostics,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=13012902)
    p.add_argument("--items", type=int, default=4)
    p.add_argument("--loops", type=int, default=42)
    p.add_argument("--period", type=int, default=21)
    p.add_argument("--stride", type=int, default=7)
    p.add_argument("--k", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(run_native_smoke(seed=a.seed, batch=a.items, loops=a.loops,
                                      period=a.period, stride=a.stride, k=a.k),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
