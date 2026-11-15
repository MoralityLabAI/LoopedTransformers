"""SPC-03: held-out transient diagnostics for looped hidden-state trajectories.

Controls are defined before a model run. Predictions use a fixed first-half
fit and teacher-forced second-half evaluation. No target labels are used to
fit the transient models; exact spectral theorems require independent checks.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def _states(trace: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 12 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("trace must be a finite 2D depth-by-feature array, >=12 depths")
    return x


def dc_mask_identity(trace: np.ndarray, period: int) -> dict[str, float]:
    """Exact Pythagorean DC decomposition for any orthogonal period fit."""
    x = _states(trace)
    if not 1 <= period <= len(x):
        raise ValueError("period must be within observed depths")
    # P_n is the orthogonal projector onto vectors repeating mod period.
    def projection(z: np.ndarray) -> np.ndarray:
        pattern = np.array([z[s::period].mean(axis=0) for s in range(period)])
        return pattern[np.arange(len(z)) % period]
    mu = np.mean(x, axis=0, keepdims=True)
    z = x - mu
    residual_raw = np.linalg.norm(x - projection(x))
    residual_centered = np.linalg.norm(z - projection(z))
    raw_den = max(float(np.linalg.norm(x)), 1e-30)
    centered_den = max(float(np.linalg.norm(z)), 1e-30)
    raw = float(residual_raw / raw_den)
    centered = float(residual_centered / centered_den)
    energy_dc = float(len(x) * np.sum(mu ** 2))
    energy_centered = float(np.sum(z ** 2))
    energy_total = float(np.sum(x ** 2))
    exact_gap = abs(residual_raw ** 2 - residual_centered ** 2)
    identity_gap = abs(raw ** 2 - centered ** 2 * energy_centered / max(energy_total, 1e-30))
    if not np.isclose(energy_total, energy_dc + energy_centered, rtol=1e-10, atol=1e-10):
        raise AssertionError("orthogonal temporal-DC decomposition failed")
    if not np.isclose(residual_raw, residual_centered, rtol=1e-10, atol=1e-10):
        raise AssertionError("projector must preserve constant states")
    return {
        "raw_periodic_relative_residual": raw,
        "centered_periodic_relative_residual": centered,
        "dc_energy_fraction": energy_dc / max(energy_total, 1e-30),
        "dynamic_energy_fraction": energy_centered / max(energy_total, 1e-30),
        "absolute_residual_squared_gap": float(exact_gap),
        "dc_masking_identity_gap": float(identity_gap),
    }


def heldout_transient_models(trace: np.ndarray, train_fraction: float = 0.5) -> dict[str, float | int]:
    """Compare 0, persistence, scalar AR(1), scalar AR(2) on Δ state.

    All scalar coefficients are shared across features and fitted with train
    differences only. Held-out evaluation uses the actual previous differences,
    so metrics are one-step, *not* multi-step autonomous forecasts.
    """
    x = _states(trace)
    if not 0.4 <= train_fraction <= 0.7:
        raise ValueError("train fraction outside preregistered range")
    dx = np.diff(x, axis=0)
    N = len(dx)
    split = int(N * train_fraction)
    if not 5 <= split < N - 3:
        raise ValueError("insufficient held-out time points")
    # For predicting delta[t+1] from delta[t], t<split for training.
    t_train = np.arange(1, split)
    t_test = np.arange(split, N-1)
    X1 = dx[t_train].reshape(-1)
    Y = dx[t_train+1].reshape(-1)
    denom = float(X1 @ X1)
    ar1 = float((X1 @ Y) / denom) if denom > 1e-30 else 0.0
    A = np.stack((dx[t_train], dx[t_train-1]), axis=1).transpose(0,2,1).reshape(-1,2)
    coeff, *_ = np.linalg.lstsq(A, Y, rcond=None)
    ar2_a, ar2_b = map(float, coeff)
    ytest = dx[t_test+1]
    bases = {
        "zero": np.zeros_like(ytest),
        "persistence": dx[t_test],
        "scalar_ar1": ar1 * dx[t_test],
        "scalar_ar2": ar2_a * dx[t_test] + ar2_b * dx[t_test-1],
    }
    den = max(float(np.linalg.norm(ytest)), 1e-30)
    scores = {f"heldout_{name}_normalized_error": float(np.linalg.norm(ytest-pred)/den)
              for name, pred in bases.items()}
    # Forecaster is only useful when it beats the pre-registered zero baseline.
    scores.update({
        "train_steps": split + 1,
        "test_steps": len(t_test),
        "scalar_ar1_coefficient": ar1,
        "scalar_ar2_a": ar2_a,
        "scalar_ar2_b": ar2_b,
        "difference_initial_norm": float(np.linalg.norm(dx[0])),
        "difference_final_norm": float(np.linalg.norm(dx[-1])),
        "difference_decay_ratio_final_over_initial": float(np.linalg.norm(dx[-1]) / max(np.linalg.norm(dx[0]), 1e-30)),
    })
    return scores


def probe_trace(trace: np.ndarray, period: int, train_fraction: float = 0.5) -> dict[str,Any]:
    return {**dc_mask_identity(trace, period), **heldout_transient_models(trace, train_fraction)}


def native_sweep(seeds: tuple[int,...] = (13012902,13012903,13012904,13012905),
                 batch: int = 4, loops: int = 42, period: int = 21,
                 n_nodes: int = 8, d_model: int = 64) -> dict[str,Any]:
    """Repeat on actual repo LoopedChase, distinguishing untrained control."""
    import torch
    from research_gym.scripts.loopscope_toy import LoopedChase, make_batch
    if not seeds or batch < 1 or loops < 12 or period > loops or d_model % 4:
        raise ValueError("invalid configuration")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        model=LoopedChase(n_nodes=n_nodes,d=d_model,layers=1,heads=4).eval()
        for p in model.parameters():p.requires_grad_(False)
        tokens,_=make_batch(n_nodes,batch,seed+1,loops)
        with torch.no_grad():
            states=model(tokens,loops).detach().cpu().numpy().transpose(1,0,2)
        for item,x in enumerate(states):
            rows.append({"seed":seed,"item":item,**probe_trace(x,period)})
    fields=[key for key in rows[0] if key not in ("seed","item","train_steps","test_steps")]
    mean={name:float(np.mean([r[name] for r in rows])) for name in fields}
    return {"campaign":"SPC-03","model":"LoopedChase","trained_checkpoint":None,
            "training_performed":False,"dtype":"float32","device":"cpu",
            "seed_set":list(seeds),"batch_per_seed":batch,"loops":loops,
            "period":period,"train_fraction":0.5,"summary_mean":mean,"rows":rows}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default=None)
    p.add_argument("--seeds",type=int,nargs="+",default=[13012902,13012903,13012904,13012905])
    p.add_argument("--batch",type=int,default=4)
    a=p.parse_args()
    receipt=native_sweep(tuple(a.seeds),batch=a.batch)
    s=json.dumps(receipt,indent=2,sort_keys=True)
    if a.output:
        from pathlib import Path
        Path(a.output).write_text(s)
    print(s)

if __name__ == "__main__":main()
