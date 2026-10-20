"""Toy path step 8: successful hidden-state reuse (native resume, crrbench2.reuse protocol) and which signals/features
at freeze time predict it.

Protocol (same as the controller bench): freeze the full state at loop k_i = (first loop from which the answer stays
correct) + delay, append 4 fresh symbols, keep running the SAME frozen core for M=8 more loops, read the automaton state
after the suffix.  `success` = the resumed readout is correct at the last of the M loops.  The from-scratch run on
(string + suffix) with the same total number of loops gives the reference accuracy.
Associations are selected on ana_train and evaluated on held-out test_id / test_ood.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import toy
from recsae.analysis import Logit, first_stable, subsample_rows
from recsae.io import safe, SPLITS, fig_path, load_split, res_dir, sae_prefix
from recsae.sae import TopKSAE
from recsae.stats import auroc, auroc_cols, bootstrap_ci
from crrbench2.reuse import freeze_states, native_resume
from t3_toy_analyze import conf_feats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELAYS = (0, 2, 4, 8)          # overridable with --delays
M_RES, SUF = 8, 4


def collect_reuse(model, task, ck, sp, sae, name, rng_seed=0):
    lengths, npl, off = SPLITS[sp]
    rng = np.random.default_rng(1_000_000 * (ck["seed"] + 1) + off)           # identical stream to the saved split
    split = toy.make_split(task, rng, lengths, npl)
    d = load_split(name, sp)
    K = d["correct"].shape[1]
    fs = first_stable(d["correct"])
    kstar = np.where(fs <= K, fs, ck["cfg"]["K"])                            # never-stable -> train depth
    rows = []
    for delay in DELAYS:
        snap = np.clip(kstar + delay, 1, K)
        full, pooled = freeze_states(model, task, split, snap, device="cpu")
        r2 = np.random.default_rng(rng_seed + delay)
        cr, cf = native_resume(model, task, split, full, r2, suffix_len=SUF, M=M_RES, Kfresh=K, device="cpu",
                               fresh_depth=np.clip(snap + M_RES, 1, K))
        x_last = pooled[:, : pooled.shape[1] // 2].float()
        with torch.no_grad():
            z = sae.encode(x_last)
            xh = sae.decode(z)
        recon = ((x_last - xh).pow(2).sum(-1) / ((x_last - sae.mu).pow(2).sum(-1) + 1e-8)).numpy()
        lg = d["logits"][np.arange(len(snap)), snap - 1]
        mp, mg, en = conf_feats(lg[:, None])
        rows.append({"delay": delay, "snap": snap, "resume_ok": cr[:, -1], "resume_any": cr.any(1), "fresh_ok": cf,
                     "z": z.numpy(), "x": x_last.numpy(), "recon": recon, "maxp": mp[:, 0], "margin": mg[:, 0], "ent": en[:, 0],
                     "now_correct": d["correct"][np.arange(len(snap)), snap - 1], "N": d["N"]})
    return rows


def stack(rows, key):
    return np.concatenate([r[key] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--delays", default="0,2,4,8")
    a = ap.parse_args()
    global DELAYS
    DELAYS = tuple(int(x) for x in a.delays.split(","))
    torch.set_num_threads(a.threads)
    tag = a.tag or safe(a.name)
    model, task, ck = toy.load_toy(a.name)
    sae = TopKSAE.load(sae_prefix(a.name, "all"))
    n = sae.cfg.n_latents
    R = {"name": a.name, "M": M_RES, "suffix": SUF, "delays": list(DELAYS)}
    data = {sp: collect_reuse(model, task, ck, sp, sae, a.name) for sp in ("ana_train", "test_id", "test_ood")}
    for sp, rows in data.items():
        R[f"rate_{sp}"] = {int(r["delay"]): {"resume": float(r["resume_ok"].mean()), "fresh_same_total_loops": float(r["fresh_ok"].mean()),
                                              "now_correct": float(r["now_correct"].mean())} for r in rows}
        print(sp, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in R[f"rate_{sp}"].items()})

    def feats(rows, kind):
        N = stack(rows, "N")
        base = np.stack([stack(rows, "maxp"), stack(rows, "margin"), stack(rows, "ent"), stack(rows, "snap"), N, stack(rows, "now_correct")], 1)
        if kind == "base":
            return base
        if kind == "base+recon":
            return np.concatenate([base, stack(rows, "recon")[:, None]], 1)
        if kind == "base+code":
            return np.concatenate([base, stack(rows, "z")], 1)
        if kind == "base+raw":
            return np.concatenate([base, stack(rows, "x")], 1)

    tr = data["ana_train"]
    y_tr = stack(tr, "resume_ok")
    ii = subsample_rows(len(y_tr), 60000, np.random.default_rng(0))
    fits = {k: Logit(l2=3e-2, iters=60).fit(feats(tr, k)[ii], y_tr[ii]) for k in ("base", "base+recon", "base+code", "base+raw")}
    pred = {}
    for sp in ("test_id", "test_ood"):
        rows = data[sp]
        y = stack(rows, "resume_ok")
        sc = {k: fits[k].score(feats(rows, k)) for k in fits}
        res = {k: {"auc": auroc(v, y)} for k, v in sc.items()}
        for k in res:
            res[k]["ci"] = list(bootstrap_ci(lambda i, v=sc[k]: auroc(v[i], y[i]), len(y), B=a.boot, seed=0))
            if k != "base":
                f = lambda i, k=k: auroc(sc[k][i], y[i]) - auroc(sc["base"][i], y[i])
                lo, hi = bootstrap_ci(f, len(y), B=a.boot, seed=1)
                res[k]["delta_vs_base"] = [f(np.arange(len(y))), lo, hi]
        pred[sp] = {"n": int(len(y)), "pos_rate": float(y.mean()), "probes": res}
        print(f"[{sp}] resume success {y.mean():.3f}: " + ", ".join(f"{k}={v['auc']:.3f}" for k, v in res.items()))
    R["prediction"] = pred
    # ---- per-feature associations: select on ana_train, replicate on test
    Ztr = stack(tr, "z")
    au_tr = auroc_cols(Ztr, y_tr)
    act = (Ztr > 0).sum(0)
    ok = act >= 30
    top = np.argsort(-np.where(ok, np.abs(np.nan_to_num(au_tr, nan=0.5) - 0.5), -1))[:30]
    rep = {}
    rng = np.random.default_rng(0)
    for sp in ("test_id", "test_ood"):
        Zte, yte = stack(data[sp], "z"), stack(data[sp], "resume_ok")
        au_te = auroc_cols(Zte, yte)
        okte = (Zte > 0).sum(0) >= 30
        signed = np.sign(au_tr[top] - 0.5) * (np.nan_to_num(au_te[top], nan=0.5) - 0.5)
        null = []
        for _ in range(200):
            pick = rng.choice(np.where(okte & ok)[0], size=len(top), replace=False)
            null.append(float(np.mean(np.sign(au_tr[pick] - 0.5) * (np.nan_to_num(au_te[pick], nan=0.5) - 0.5))))
        rep[sp] = {"top30_signed_dev_test": float(np.mean(signed)), "null_mean": float(np.mean(null)), "null_p95": float(np.percentile(null, 95)),
                   "sign_agreement": float((signed > 0).mean()), "top10": top[:10].tolist(), "auc_train_top10": au_tr[top[:10]].tolist(), "auc_test_top10": au_te[top[:10]].tolist()}
        print(f"[{sp}] reuse-feature replication: signed dev {rep[sp]['top30_signed_dev_test']:.4f} (null mean {rep[sp]['null_mean']:.4f}, p95 {rep[sp]['null_p95']:.4f}), sign agreement {rep[sp]['sign_agreement']:.2f}")
    R["feature_replication"] = rep
    json.dump(R, open(os.path.join(res_dir(), f"{tag}__reuse.json"), "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sp in zip(axs, ("test_id", "test_ood")):
        rt = R[f"rate_{sp}"]
        ax.plot(DELAYS, [rt[d]["resume"] for d in DELAYS], marker="o", label="resume from frozen state")
        ax.plot(DELAYS, [rt[d]["fresh_same_total_loops"] for d in DELAYS], marker="s", label="from scratch, same total loops")
        ax.plot(DELAYS, [rt[d]["now_correct"] for d in DELAYS], marker="^", ls=":", label="frozen readout correct")
        ax.set_xlabel("freeze delay after first-stable loop"); ax.set_ylabel("accuracy"); ax.set_title(sp); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f10_reuse"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
