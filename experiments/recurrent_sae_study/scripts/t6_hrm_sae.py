"""Does adding SAE-derived signals to the HRM-style Q-controller (crrbench2: CONTINUE / PRESERVE / HALT, price-conditioned)
improve its accuracy-vs-compute frontier?  (RQ7)

Variants (same Q-learning recipe, same seeds; only the observation changes):
  base       : controller's own inputs  [z, 6 scalars]  (exactly crrbench2)
  +sae       : + SAE scalars {recon error, Jaccard(t,t-1), code cosine, carried energy, novel-feature mass}
  +sae_shuf  : the SAME SAE scalars permuted across examples (within length, loop kept) -> capacity/marginal control
  +dyn       : + non-SAE state dynamics {relative state change, cosine to previous state}  (feature-matched control)
  +sae+dyn
Metric M = mean over the price grid of [accuracy - budget-matched fixed-depth accuracy] (sibling's own yardstick),
FLOP-charged including the controller and the SAE encoder.  Controllers train on ana_train only; reported on held-out
test_id / test_ood.  Paired bootstrap over examples.
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
from recsae import DATA, toy
from recsae.analysis import encode_loops, feature_frequency
from recsae.io import safe, fig_path, is_mini, load_split, res_dir, sae_prefix
from recsae.sae import TopKSAE
from recsae.stats import bootstrap_ci
import crrbench2.controller as C
from crrbench2.controller import TrajTensors
from crrbench2.evalctl import LAMBDA_GRID, kappa_of, matched_fixed_curve
from crrbench2.flops import controller_flops, loop_flops, readout_flops
from t3_toy_analyze import per_loop_signals

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = dict(ctrl_hidden=128, ctrl_lr=1e-3, ctrl_iters=300, ctrl_rollout=2048, ctrl_epochs=6, ctrl_mb=4096, ctrl_target_every=20)
SAEF = ["recon", "jacc", "codecos", "carry", "novel"]
DYN = ["dstate", "cosstate"]


class TrajX(TrajTensors):
    def __init__(self, split, extra, device="cpu"):
        super().__init__(split, device)
        self.extra = torch.as_tensor(extra, dtype=torch.float32, device=device)       # [n, K, F]
        self.base_dim = self.base_dim + self.extra.shape[-1]

    def base(self, idx, k):
        return torch.cat([super().base(idx, k), self.extra[idx, k]], -1)


def build_split(name, split, model):
    d = load_split(name, split)
    last = torch.from_numpy(d["last"][:, 1:].astype(np.float32))
    with torch.no_grad():
        z = model.ln_f(last).numpy()
    return d, {"logits": d["logits"].astype(np.float32), "z": z, "y": d["y"], "N": d["N"]}


def shuffle_within(extra, groups, seed):
    rng = np.random.default_rng(seed)
    out = extra.copy()
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        out[m] = extra[rng.permutation(m)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=CFG["ctrl_iters"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    CFG["ctrl_iters"] = a.iters
    tag = a.tag or safe(a.name)
    model, task, ck = toy.load_toy(a.name)
    d_model, K_train = ck["cfg"]["d"], ck["cfg"]["K"]
    sae = TopKSAE.load(sae_prefix(a.name, "all"))
    n = sae.cfg.n_latents
    freq_ref = feature_frequency(encode_loops(sae, load_split(a.name, "sae_train")["last"], list(range(1, K_train + 1)))[0], n)
    raw, splits, extras = {}, {}, {}
    for s in ("ana_train", "test_id", "test_ood"):
        d, sp = build_split(a.name, s, model)
        K = d["correct"].shape[1]
        sig, _ = per_loop_signals(d, sae, list(range(1, K + 1)), "cpu", freq_ref)
        raw[s], splits[s] = d, sp
        extras[s] = {k: sig[k].astype(np.float32) for k in SAEF + DYN}
    # standardise extras with ana_train statistics
    stats = {k: (extras["ana_train"][k].mean(), extras["ana_train"][k].std() + 1e-6) for k in SAEF + DYN}
    for s in extras:
        for k in extras[s]:
            mu, sd = stats[k]
            extras[s][k] = np.clip((extras[s][k] - mu) / sd, -5, 5)

    def make(s, variant):
        ex = {k: v for k, v in extras[s].items()}
        grp = (raw[s]["N"] * 100 + raw[s]["y"]).astype(int)
        if variant == "base":
            return TrajTensors(splits[s], device="cpu")
        feats = {"+sae": SAEF, "+sae_shuf": SAEF, "+dyn": DYN, "+sae+dyn": SAEF + DYN}[variant]
        X = np.stack([ex[k] for k in feats], -1)
        if variant == "+sae_shuf":
            X = shuffle_within(X, grp, 123)
        return TrajX(splits[s], X)

    variants = ["base", "+sae", "+sae_shuf", "+dyn", "+sae+dyn"]
    sae_fl = 2 * d_model * n + 2 * n * sae.cfg.k
    res = {"name": a.name, "variants": variants, "cfg": CFG, "lam_grid": list(LAMBDA_GRID), "per": {}}
    out = {s: {v: [] for v in variants} for s in ("test_id", "test_ood")}     # per seed: dict(lam -> arrays)
    for v in variants:
        Ttr = make("ana_train", v)
        for seed in range(a.seeds):
            q = C.train_controller(Ttr, CFG, device="cpu", seed=seed)
            for s in ("test_id", "test_ood"):
                T = make(s, v)
                tok = raw[s]["N"].astype(int)
                loopF = np.array([loop_flops(int(t), d_model) for t in tok], float)
                rF = readout_flops(d_model, T.C)
                ctrl_F = controller_flops(T.base_dim + C.LOCK_DIM + 1, q.hidden)
                extraF = 0.0 if v in ("base", "+dyn") else float(sae_fl)
                per_lam = []
                for lam in LAMBDA_GRID:
                    r = C.run_policy(q, T, lam, device="cpu")
                    cost_i = r["tau"] * (loopF + rF + ctrl_F + extraF)
                    per_lam.append({"lam": lam, "tau": r["tau"], "ok": r["correct"].astype(float), "cost": cost_i})
                out[s][v].append(per_lam)
            print(f"  trained {v} seed {seed}")
    # ---- evaluation: advantage over budget-matched fixed depth, averaged over price grid and seeds
    for s in ("test_id", "test_ood"):
        corr = raw[s]["correct"]
        tok = raw[s]["N"].astype(int)
        loopF = np.array([loop_flops(int(t), d_model) for t in tok], float)
        rF = readout_flops(d_model, task.n_classes)
        lamgrid = list(LAMBDA_GRID)

        def M(v, ix):
            vals = []
            for per_lam in out[s][v]:
                for pl in per_lam:
                    kap = kappa_of(pl["cost"][ix].mean(), loopF[ix].mean(), rF)
                    vals.append(pl["ok"][ix].mean() - matched_fixed_curve(corr[ix], kap).mean())
            return float(np.mean(vals))

        allix = np.arange(len(corr))
        res["per"][s] = {"M": {v: M(v, allix) for v in variants}}
        for v in variants:
            lo, hi = bootstrap_ci(lambda ix, v=v: M(v, ix), len(corr), B=a.boot // 4, seed=0)
            res["per"][s]["M"][v + "|ci"] = [lo, hi]
        for v in variants[1:]:
            f = lambda ix, v=v: M(v, ix) - M("base", ix)
            lo, hi = bootstrap_ci(f, len(corr), B=a.boot // 4, seed=1)
            res["per"][s][f"delta[{v} - base]"] = [f(allix), lo, hi]
        for v in variants:
            res["per"][s]["M_by_seed_" + v] = [float(np.mean([pl["ok"].mean() - matched_fixed_curve(corr, kappa_of(pl["cost"].mean(), loopF.mean(), rF)).mean() for pl in per_lam])) for per_lam in out[s][v]]
        # frontier points (seed-averaged) for plotting
        fr = {}
        for v in variants:
            pts = []
            for li, lam in enumerate(lamgrid):
                acc = np.mean([per_lam[li]["ok"].mean() for per_lam in out[s][v]])
                kap = np.mean([kappa_of(per_lam[li]["cost"].mean(), loopF.mean(), rF) for per_lam in out[s][v]])
                pts.append((float(kap), float(acc), float(lam)))
            fr[v] = pts
        res["per"][s]["frontier"] = fr
        res["per"][s]["fixed"] = [(k, float(corr[:, k - 1].mean())) for k in range(1, corr.shape[1] + 1)]
        print(f"[{s}] M: " + ", ".join(f"{v}={res['per'][s]['M'][v]:+.4f}" for v in variants))
        print("   " + "; ".join(f"{k}={v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]" for k, v in res["per"][s].items() if k.startswith("delta")))
    json.dump(res, open(os.path.join(res_dir(), f"{tag}__hrm_sae.json"), "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = dict(zip(variants, ["k", "tab:red", "tab:gray", "tab:blue", "tab:purple"]))
    for ax, s in zip(axs, ("test_id", "test_ood")):
        fx = np.array(res["per"][s]["fixed"])
        ax.plot(fx[:, 0], fx[:, 1], c="gray", ls=":", label="fixed depth")
        for v in variants:
            p = np.array(res["per"][s]["frontier"][v])
            ax.plot(p[:, 0], p[:, 1], c=cols[v], marker="o", ms=3, label=f"{v} (M={res['per'][s]['M'][v]:+.3f})")
        ax.set_xlim(1, 16); ax.set_xlabel("mean depth equivalent (FLOP-charged)"); ax.set_ylabel("accuracy"); ax.set_title(s); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f6_hrm_sae"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
