"""One benchmark cell = (task, training regime, seed):  train -> compute/retain -> reuse -> controller.

    python -m crrbench.run_cell --task storyworld --regime random_exit --seed 0 --out results/full
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import controller as C
from . import reuse as R
from .collect import collect_split
from .evalctl import evaluate_groups
from .metrics import compute_retain_summary, first_correct
from .tasks import make_task, real_path
from .train import REGIMES, train_model

CONFIGS = {
    "full": dict(K_train=8, K_eval=32, d=128, heads=4, batch=256, lr=1e-3, steps=1500,
                 train_lengths=[2, 3, 4, 6, 8], groups={"id": [4, 8], "interior": [5, 7], "extrap": [10, 12]},
                 n_ctrl_per_len=800, n_test_per_len=800, log_depths=[1, 2, 4, 8, 16, 32],
                 delays=[0, 1, 2, 4, 8], suffix_len=4, resume_M=8, probe_steps=400,
                 ctrl_hidden=128, ctrl_lr=1e-3, ctrl_iters=600, ctrl_rollout=2048, ctrl_epochs=6, ctrl_mb=4096,
                 ctrl_target_every=20, n_boot=200),
    "quick": dict(K_train=8, K_eval=16, d=64, heads=4, batch=64, lr=2e-3, steps=60,
                  train_lengths=[2, 3, 4], groups={"id": [3, 4], "interior": [5], "extrap": [6]},
                  n_ctrl_per_len=100, n_test_per_len=100, log_depths=[1, 2, 4, 8, 16],
                  delays=[0, 1, 2, 4, 8], suffix_len=4, resume_M=4, probe_steps=40,
                  ctrl_hidden=32, ctrl_lr=1e-3, ctrl_iters=20, ctrl_rollout=256, ctrl_epochs=2, ctrl_mb=512,
                  ctrl_target_every=5, n_boot=20),
}


def make_split(model, task, rng, lengths, n_per_len, K, device):
    return collect_split(model, task, rng, lengths, n_per_len, K, device=device)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float):
        return None if o != o else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run_cell(task_name, regime, seed, out_dir, cfg_name="full", device="cuda", storyworld_dir=None, log=print,
             cfg_over=None, train_only=False):
    cfg = dict(CONFIGS[cfg_name])
    cfg.update(cfg_over or {})
    cell = f"{task_name}__{regime}__s{seed}"
    out_dir = real_path(out_dir)
    os.makedirs(os.path.join(out_dir, "cells"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    t_start = time.time()
    task = make_task(task_name, storyworld_dir=storyworld_dir)
    torch.set_float32_matmul_precision("high")
    K = cfg["K_eval"]
    tcfg = dict(cfg, K=cfg["K_train"])
    mpath = os.path.join(out_dir, "models", cell + ".pt")
    from .model import LoopedTransformer
    if os.path.exists(mpath):
        model = LoopedTransformer(task.vocab, task.n_classes, d=cfg["d"], heads=cfg["heads"]).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device)); model.eval()
        log(f"[{cell}] loaded model")
    else:
        model = train_model(task, regime, seed, tcfg, device=device, log=log)
        torch.save(model.state_dict(), mpath)
    t_train = time.time() - t_start
    if train_only:
        return None

    # ---- data splits (never seen during training: fresh rng streams) ---------------
    rng_c = np.random.default_rng(10_000 + seed)
    rng_t = np.random.default_rng(20_000 + seed)
    ctrl = make_split(model, task, rng_c, cfg["train_lengths"], cfg["n_ctrl_per_len"], K, device)
    test_lengths = sorted({n for v in cfg["groups"].values() for n in v})
    test = make_split(model, task, rng_t, test_lengths, cfg["n_test_per_len"], K, device)
    groups = {g: np.where(np.isin(test["N"], lens))[0] for g, lens in cfg["groups"].items()}
    groups["heldout"] = np.where(np.isin(test["N"], cfg["groups"]["interior"] + cfg["groups"]["extrap"]))[0]
    tok_len = test["N"] + task.prefix

    res = {"cell": cell, "task": task_name, "regime": regime, "seed": seed, "cfg": cfg, "n_params": model.n_params(),
           "majority": {int(n): task.majority_acc(np.random.default_rng(1), n) for n in test_lengths},
           "train_wall_s": t_train}

    # ---- compute / retain ----------------------------------------------------------
    cr = {}
    for g, rows in groups.items():
        s = compute_retain_summary(test["correct"][rows], depths=cfg["log_depths"])
        cr[g] = s
    res["compute_retain"] = cr
    res["compute_retain_by_len"] = {int(n): compute_retain_summary(test["correct"][test["N"] == n], depths=cfg["log_depths"])
                                    for n in test_lengths}

    # ---- logged predictions + hidden states at configured depths -------------------
    ld = [k for k in cfg["log_depths"] if k <= K]
    sel = np.concatenate([np.where(test["N"] == n)[0][:200] for n in test_lengths])
    np.savez_compressed(os.path.join(out_dir, "logs", cell + ".npz"), depths=np.array(ld), rows_N=test["N"][sel],
                        y=test["y"][sel], preds=test["preds"][sel][:, [k - 1 for k in ld]],
                        z=test["z"][sel][:, [k - 1 for k in ld]])

    # ---- reuse: delayed state freezing ------------------------------------------------
    t0 = time.time()
    kst_c = first_correct(ctrl["correct"])
    kst_t = first_correct(test["correct"])
    maxd = max(cfg["delays"])
    S_c = kst_c <= K - maxd
    S_t = kst_t <= K - maxd
    reuse = {"n_train_solved": int(S_c.sum()), "frac_train_solved": float(S_c.mean()),
             "delays": cfg["delays"], "probe": {g: {} for g in groups}, "native": {g: {} for g in groups},
             "fresh": {g: {} for g in groups}, "n": {g: int(S_t[groups[g]].sum()) for g in groups}}
    probe_acc = {g: np.full((len(cfg["delays"]), len(groups[g])), np.nan) for g in groups}
    if S_c.sum() >= 50:
        rng_p = np.random.default_rng(30_000 + seed)
        for di, dly in enumerate(cfg["delays"]):
            snap_c = np.where(S_c, kst_c + dly, 1)
            _, pooled_c = R.freeze_states(model, task, ctrl, snap_c, device=device)
            probe, mu, sd = R.fit_probe(task, pooled_c[S_c], ctrl["y"][S_c], ctrl["ctx"][S_c], rng_p,
                                        suffix_len=cfg["suffix_len"], steps=cfg["probe_steps"], device=device, seed=seed)
            snap_t = np.where(S_t, kst_t + dly, 1)
            full_t, pooled_t = R.freeze_states(model, task, test, snap_t, device=device)
            acc_i = R.eval_probe(task, probe, mu, sd, pooled_t, test["y"], test["ctx"], np.random.default_rng(40_000 + seed),
                                 suffix_len=cfg["suffix_len"], device=device)
            fd = np.clip(kst_t + dly + cfg["resume_M"], 1, K)
            cr_, cf_ = R.native_resume(model, task, test, full_t, np.random.default_rng(50_000 + seed),
                                       suffix_len=cfg["suffix_len"], M=cfg["resume_M"], Kfresh=K, device=device, fresh_depth=fd)
            for g, rows in groups.items():
                m = S_t[rows]
                reuse["probe"].setdefault(g, {})[dly] = float(acc_i[rows][m].mean()) if m.any() else None
                reuse["native"].setdefault(g, {})[dly] = [float(cr_[rows][m][:, j].mean()) for j in range(cr_.shape[1])] if m.any() else None
                reuse["fresh"].setdefault(g, {})[dly] = float(cf_[rows][m].mean()) if m.any() else None
                probe_acc[g][di, :] = np.where(m, acc_i[rows], np.nan)
        # paired within-cell bootstrap of the delay effect (probe reuse, d - 0)
        reuse["probe_delta_ci"] = {}
        rb = np.random.default_rng(seed)
        for g in groups:
            pa = probe_acc[g]
            ok = ~np.isnan(pa[0])
            if ok.sum() < 20:
                continue
            reuse["probe_delta_ci"][g] = {}
            for di, dly in enumerate(cfg["delays"]):
                dif = pa[di, ok] - pa[0, ok]
                bs = [dif[rb.integers(0, ok.sum(), ok.sum())].mean() for _ in range(cfg["n_boot"])]
                reuse["probe_delta_ci"][g][dly] = [float(dif.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res["reuse"] = reuse
    res["reuse_wall_s"] = time.time() - t0

    # ---- controller -----------------------------------------------------------------
    t0 = time.time()
    Ttr = C.TrajTensors(ctrl, device)
    Tte = C.TrajTensors(test, device)
    q = C.train_controller(Ttr, cfg, device=device, seed=seed, log=log)
    res["controller"] = evaluate_groups(q, Tte, tok_len, cfg["d"], groups, device=device, n_boot=cfg["n_boot"], seed=seed)
    # also record generalisation of the controller to the data it was trained on (sanity)
    res["ctrl_wall_s"] = time.time() - t0
    res["wall_s"] = time.time() - t_start
    with open(os.path.join(out_dir, "cells", cell + ".json"), "w") as f:
        json.dump(jsonable(res), f)
    log(f"[{cell}] done in {res['wall_s']:.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/full")
    ap.add_argument("--config", default="full", choices=list(CONFIGS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--storyworld-dir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train-only", action="store_true")
    a = ap.parse_args()
    over = {"steps": a.steps} if a.steps else None
    run_cell(a.task, a.regime, a.seed, a.out, a.config, a.device, a.storyworld_dir, cfg_over=over, train_only=a.train_only,
             log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
