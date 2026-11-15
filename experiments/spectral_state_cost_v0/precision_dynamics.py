"""SPC-03: precision-controlled native recurrent contraction and held-out forecast.

The exact same float32-initialized weights and fixed prompts are evaluated in
float32 and float64. Scalar increment forecasts train on loop-depth pairs
4..9 and are evaluated on held-out pairs 10..15. The first input-injection
jump is excluded by construction. These fixed windows are preregistered.
No trained checkpoints, capability claims, or interventions are implied.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from experiments.spectral_state_cost_v0.dynamic_controls import dc_mask_identity

# delta index i = h_{i+2} - h_{i+1} for zero-based h; fit on t=3..8,
# evaluate on t=9..14. Each pair predicts delta[t+1] from delta[t].
TRAIN_PAIRS = tuple(range(3, 9))
TEST_PAIRS = tuple(range(9, 15))


def heldout_scalar_damping(states: np.ndarray,
                           train_pairs: tuple[int,...] = TRAIN_PAIRS,
                           test_pairs: tuple[int,...] = TEST_PAIRS) -> dict[str, float]:
    z = np.asarray(states, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] <= max((*train_pairs,*test_pairs)) + 2:
        raise ValueError("insufficient 2D temporal states")
    dx = np.diff(z, axis=0)
    if max(train_pairs) >= min(test_pairs):
        raise ValueError("train and test periods must be disjoint and ordered")
    tr_x = dx[list(train_pairs)]
    tr_y = dx[[t+1 for t in train_pairs]]
    fit_denom = float(np.vdot(tr_x, tr_x).real)
    lam = float(np.vdot(tr_x, tr_y).real / fit_denom) if fit_denom > 1e-30 else 0.0
    te_x = dx[list(test_pairs)]
    te_y = dx[[t+1 for t in test_pairs]]
    denom = max(float(np.linalg.norm(te_y)), 1e-30)
    resid = float(np.linalg.norm(te_y-lam*te_x) / denom)
    pers = float(np.linalg.norm(te_y-te_x) / denom)
    norms = np.linalg.norm(dx, axis=1)
    ratios = norms[4:15]/np.maximum(norms[3:14],1e-30)
    # Retain coherence of sequential state changes at the held-out positions.
    numer = np.sum(te_x*te_y, axis=1)
    cos = numer / np.maximum(np.linalg.norm(te_x,axis=1)*np.linalg.norm(te_y,axis=1), 1e-30)
    return {
        "train_lambda": lam,
        "test_ar1_normalized_error": resid,
        "test_zero_normalized_error": 1.0,
        "test_persistence_normalized_error": pers,
        "test_cosine_mean": float(np.mean(cos)),
        "early_mean_sequential_norm_ratio": float(np.mean(ratios)),
        "initial_delta_norm": float(norms[0]),
        "delta_norm_at_15": float(norms[14]),
        "delta_norm_at_20": float(norms[19]),
        "delta_norm_at_30": float(norms[29]),
        "delta_norm_at_41": float(norms[40]),
        "late_delta_median": float(np.median(norms[24:41])),
    }


def run_precision_sweep(seeds: tuple[int,...] = (13012902,13012903,13012904,13012905),
                        batch: int = 4, loops: int = 42, period: int = 21,
                        d_model: int = 64, n_nodes: int = 8) -> dict[str,object]:
    import torch
    from research_gym.scripts.loopscope_toy import LoopedChase, make_batch
    if not seeds or batch < 1 or loops < 42 or d_model < 4 or d_model % 4:
        raise ValueError("invalid bounded sweep")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        f32=LoopedChase(n_nodes=n_nodes,d=d_model,layers=1,heads=4).eval()
        f64=copy.deepcopy(f32).double().eval()
        tokens,_=make_batch(n_nodes,batch,seed+1,loops)
        with torch.no_grad():
            a=f32(tokens,loops).detach().cpu().numpy().transpose(1,0,2)
            b=f64(tokens,loops).detach().cpu().numpy().transpose(1,0,2)
        for i in range(batch):
            r32=heldout_scalar_damping(a[i]);r64=heldout_scalar_damping(b[i])
            dc32=dc_mask_identity(a[i],period)
            dc64=dc_mask_identity(b[i],period)
            rows.append({"seed":seed,"item":i,"float32":r32,"float64":r64,
                         "dc32":dc32,"dc64":dc64,
                         "relative_state_discrepancy":float(np.linalg.norm(a[i]-b[i]) / max(np.linalg.norm(b[i]),1e-30))})
    keys = tuple(rows[0]["float64"].keys())
    def summary(field: str) -> dict[str,float]:
        return {k: float(np.mean([r[field][k] for r in rows])) for k in keys}
    return {
        "campaign":"SPC-03","model":"LoopedChase","seed_set":list(seeds),
        "batch_per_seed":batch,"total_trajectories":len(rows),"loops":loops,
        "period":period,"trained_checkpoint":None,"training_performed":False,
        "model_init_dtype":"float32","replay_dtypes":["float32","float64"],
        "shared_initial_weight_tensor":True,"identical_prompt_tokens":True,
        "train_pairs_zero_based":list(TRAIN_PAIRS),"test_pairs_zero_based":list(TEST_PAIRS),
        "float32_summary":summary("float32"),"float64_summary":summary("float64"),
        "mean_relative_float32_float64_state_error":float(np.mean([r['relative_state_discrepancy'] for r in rows])),
        "dc32_summary":{k:float(np.mean([r['dc32'][k] for r in rows])) for k in rows[0]['dc32']},
        "dc64_summary":{k:float(np.mean([r['dc64'][k] for r in rows])) for k in rows[0]['dc64']},
        "rows":rows,
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[13012902,13012903,13012904,13012905])
    p.add_argument('--batch',type=int,default=4)
    a=p.parse_args()
    result=run_precision_sweep(seeds=tuple(a.seeds),batch=a.batch)
    s=json.dumps(result,sort_keys=True,indent=2)
    if a.out:a.out.write_text(s)
    print(s)

if __name__=='__main__':main()
