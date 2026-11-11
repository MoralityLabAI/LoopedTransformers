"""Toy path step 5: do SAE-derived signals improve a loop-stopping policy? (RQ4/RQ7)

Policies map per-loop signals -> P(stopping now is safe); stop at the first loop whose score >= tau.  Curves sweep tau.
  fixed        : always run t loops
  conf         : max-softmax threshold                       (standard early-exit baseline)
  dstate       : 1 - relative state change threshold         (fixed-point detector)
  logit(base)  : logistic regression on {maxp, margin, entropy, t}
  logit(+sae)  : ... + SAE scalars {recon err, Jaccard(t,t-1), code cosine, carried energy, novel-feature mass}
  logit(+dyn+sae)
  hrm(base)    : two-timescale GRU controller (slow module ticks every `stride` loops) on base signals
  hrm(+sae)    : same with SAE scalars and a learned projection of the sparse code
  oracle       : stops at the first loop that is correct (upper bound)
Training uses ana_train; ana_val chooses the operating points; test_id / test_ood are held out.
FLOP accounting charges SAE encoding and controller updates against the budget (crrbench flops conventions).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import recsae  # noqa: F401
from recsae import DATA, toy
from recsae.analysis import Logit, acc_at_budget, dense_codes, pareto_area, policy_curve, stop_from_scores, subsample_rows
from recsae.io import safe, fig_path, is_mini, load_split, res_dir, sae_prefix
from recsae.sae import TopKSAE
from recsae.stats import bootstrap_ci
from crrbench.flops import controller_flops, loop_flops, readout_flops
from t3_toy_analyze import per_loop_signals

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ["maxp", "margin", "ent"]
DYN = ["dstate", "cosstate"]
SAEF = ["recon", "jacc", "codecos", "carry", "novel"]


def stack(sig, names, K):
    return np.stack([sig[n] for n in names], -1).astype(np.float32)      # [N, K, F]


class HRMController(nn.Module):
    """Fast GRU updates every loop; slow GRU updates every `stride` loops from the fast state and conditions the stop head."""

    def __init__(self, d_in, n_code=0, hid=32, stride=4):
        super().__init__()
        self.stride = stride
        self.code_proj = nn.Linear(n_code, 16) if n_code else None
        din = d_in + (16 if n_code else 0) + 1
        self.fast = nn.GRUCell(din + hid, hid)
        self.slow = nn.GRUCell(hid, hid)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.hid = hid

    def forward(self, u, code=None):
        """u [B, K, F]; code [B, K, n] or None -> logits [B, K]."""
        B, K, _ = u.shape
        hf = u.new_zeros(B, self.hid)
        hs = u.new_zeros(B, self.hid)
        out = []
        for t in range(K):
            x = [u[:, t], (torch.full((B, 1), float(t + 1) / K))]
            if self.code_proj is not None:
                x.append(F.gelu(self.code_proj(code[:, t])))
            hf = self.fast(torch.cat(x + [hs], -1), hf)
            if (t + 1) % self.stride == 0:
                hs = self.slow(hf, hs)
            out.append(self.head(torch.cat([hf, hs], -1)).squeeze(-1))
        return torch.stack(out, 1)


def train_hrm(U, C, y, seed, n_code, steps=250, bs=512, lr=3e-3):
    torch.manual_seed(seed)
    m = HRMController(U.shape[-1], n_code)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    N = U.shape[0]
    g = torch.Generator().manual_seed(seed)
    U, y = torch.as_tensor(U), torch.as_tensor(y).float()
    C = torch.as_tensor(C) if C is not None else None
    for s in range(steps):
        i = torch.randint(0, N, (bs,), generator=g)
        lg = m(U[i], C[i] if C is not None else None)
        loss = F.binary_cross_entropy_with_logits(lg, y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def hrm_scores(m, U, C):
    out = []
    for i in range(0, len(U), 1000):
        out.append(torch.sigmoid(m(torch.as_tensor(U[i:i + 1000]), torch.as_tensor(C[i:i + 1000]) if C is not None else None)).numpy())
    return np.concatenate(out)


def topk_code_dense(idx, val, n):
    N, K, k = idx.shape
    out = np.zeros((N, K, n), np.float32)
    for j in range(K):
        out[:, j] = dense_codes(idx, val, n, j=j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--hrm_seeds", type=int, default=3)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--K_train", type=int, default=None, help="required for mini runs")
    ap.add_argument("--cmax", type=float, default=16.0, help="upper end of the budget range for the Pareto area")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tag = a.tag or safe(a.name)
    sae = TopKSAE.load(sae_prefix(a.name, "all"))
    n = sae.cfg.n_latents
    if is_mini(a.name):
        K_train = a.K_train
        d_model, T_tok = sae.cfg.d_in, 16
        n_classes = 8
    else:
        model, task, ck = toy.load_toy(a.name)
        K_train, d_model, T_tok, n_classes = ck["cfg"]["K"], ck["cfg"]["d"], 16, task.n_classes
    S = {s: load_split(a.name, s) for s in ("ana_train", "ana_val", "test_id", "test_ood")}
    K = S["test_id"]["correct"].shape[1]
    loops = list(range(1, K + 1))
    from recsae.analysis import encode_loops, feature_frequency
    freq_ref = feature_frequency(encode_loops(sae, load_split(a.name, "sae_train")["last"], list(range(1, K_train + 1)))[0], n)
    sigs, codes = {}, {}
    for s, d in S.items():
        sigs[s], codes[s] = per_loop_signals(d, sae, loops, "cpu", freq_ref)
    # ---- FLOP accounting (per example): loop cost + SAE encode (last token) + controller update
    loop_fl = loop_flops(T_tok, d_model)                       # one loop at T=16 tokens
    sae_fl = 2 * d_model * n + 2 * n * sae.cfg.k               # encode matmul + top-k bookkeeping
    ctrl_fl = controller_flops(len(BASE) + len(DYN) + len(SAEF) + 1)
    over = {"none": 0.0, "base": readout_flops(d_model, n_classes) / loop_fl,
            "sae": (readout_flops(d_model, n_classes) + sae_fl) / loop_fl, "ctrl": ctrl_fl / loop_fl}
    res = {"name": a.name, "K": K, "overhead_frac_of_loop": over, "policies": {}}

    def feats(split, names):
        return stack(sigs[split], names, K)

    y_tr = S["ana_train"]["correct"]
    policies = {}
    # ----- logistic hazard policies
    for pname, names in (("logit(base)", BASE), ("logit(+dyn)", BASE + DYN), ("logit(+sae)", BASE + SAEF), ("logit(+dyn+sae)", BASE + DYN + SAEF)):
        Xtr = feats("ana_train", names)
        tt = np.broadcast_to(np.arange(1, K + 1)[None, :, None], Xtr.shape[:2] + (1,)).astype(np.float32)
        Xtr = np.concatenate([Xtr, tt, np.log(tt)], -1)
        ii = subsample_rows(Xtr.shape[0] * K, 200_000, np.random.default_rng(0))
        lr = Logit(l2=1e-2).fit(Xtr.reshape(-1, Xtr.shape[-1])[ii], y_tr.reshape(-1)[ii])
        pol = {}
        for s in ("ana_val", "test_id", "test_ood"):
            X = feats(s, names)
            t_ = np.broadcast_to(np.arange(1, K + 1)[None, :, None], X.shape[:2] + (1,)).astype(np.float32)
            X = np.concatenate([X, t_, np.log(t_)], -1)
            pol[s] = 1 / (1 + np.exp(-lr.score(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], K)))
        policies[pname] = (pol, "sae" if "sae" in pname else "base")
    # ----- hand-built baselines
    policies["conf(maxp)"] = ({s: sigs[s]["maxp"] for s in ("ana_val", "test_id", "test_ood")}, "base")
    policies["dstate"] = ({s: 1 - sigs[s]["dstate"] for s in ("ana_val", "test_id", "test_ood")}, "none")
    # ----- HRM-style controllers (multiple seeds; sparse code only for the +sae variant)
    for pname, names, use_code in (("hrm(base)", BASE, False), ("hrm(+sae)", BASE + SAEF, True)):
        scs = {s: [] for s in ("ana_val", "test_id", "test_ood")}
        Utr = feats("ana_train", names)
        mu, sd = Utr.reshape(-1, Utr.shape[-1]).mean(0), Utr.reshape(-1, Utr.shape[-1]).std(0) + 1e-6
        Ctr = topk_code_dense(*codes["ana_train"], n) if use_code else None
        for seed in range(a.hrm_seeds):
            m = train_hrm((Utr - mu) / sd, Ctr, y_tr, seed, n if use_code else 0)
            for s in scs:
                U = (feats(s, names) - mu) / sd
                C = topk_code_dense(*codes[s], n) if use_code else None
                scs[s].append(hrm_scores(m, U, C))
        policies[pname] = ({s: np.mean(v, 0) for s, v in scs.items()}, "sae" if use_code or "sae" in pname else "base")
        policies[pname + "|seeds"] = ({s: v for s, v in scs.items()}, None)
    # ----- evaluation
    taus = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), 1 - np.logspace(-4, -0.3, 40)]))
    budgets = [3, 4, 6, 8, 12] if a.cmax >= 12 else [2, 3, 4, 5, 6]
    for s in ("test_id", "test_ood"):
        c = S[s]["correct"]
        N = c.shape[0]
        res["policies"][s] = {}
        fixed = np.array([[t, float(c[:, t - 1].mean()), 0.0] for t in range(1, int(a.cmax) + 1)])
        c_lo, c_hi = 2.0, a.cmax
        out = {"fixed": {"area": pareto_area(fixed, c_lo, c_hi), "curve": fixed.tolist(),
                         **{f"acc@{b}": float(c[:, b - 1].mean()) for b in budgets}}}
        orc = np.array([[(c.argmax(1) + 1).clip(max=K).mean() if c.any() else K, float(c.any(1).mean()), 0.0]])
        out["oracle(first-correct)"] = {"mean_loops": float(orc[0, 0]), "acc": float(orc[0, 1])}
        for pname, (pol, ovk) in policies.items():
            if pname.endswith("|seeds"):
                continue
            sc = pol[s]
            cur = policy_curve(sc, c, taus)
            cur[:, 0] = cur[:, 0] * (1 + over[ovk] + (over["ctrl"] if pname.startswith(("hrm", "logit")) else 0.0))
            ent = {"area": pareto_area(cur, c_lo, c_hi), "curve": cur.tolist()}
            # operating points: tau chosen on ana_val to hit the budget, applied to test
            for b in budgets:
                v_cur = policy_curve(policies[pname][0]["ana_val"], S["ana_val"]["correct"], taus)
                v_cur[:, 0] = v_cur[:, 0] * (1 + over[ovk] + (over["ctrl"] if pname.startswith(("hrm", "logit")) else 0.0))
                ok = np.where(v_cur[:, 0] <= b)[0]
                if len(ok):
                    best = ok[np.argmax(v_cur[ok, 1])]
                    st = stop_from_scores(sc, v_cur[best, 2])
                    ent[f"op@{b}"] = {"mean_loops_charged": float((st + 1).mean() * (1 + over[ovk] + (over['ctrl'] if pname.startswith(('hrm', 'logit')) else 0.0))),
                                      "acc": float(c[np.arange(N), st].mean()), "tau": float(v_cur[best, 2])}
            # bootstrap CI for area (examples resampled; curves recomputed on resample)
            def area_fn(i, sc=sc, c=c, ovk=ovk, pname=pname):
                cc = policy_curve(sc[i], c[i], taus[::5])
                cc[:, 0] = cc[:, 0] * (1 + over[ovk] + (over["ctrl"] if pname.startswith(("hrm", "logit")) else 0.0))
                return pareto_area(cc, c_lo, c_hi)
            lo, hi = bootstrap_ci(area_fn, N, B=a.boot // 3, seed=0)
            ent["area_ci"] = [lo, hi]
            out[pname] = ent
        # paired area differences vs logit(base) and vs conf
        for p_new, p_ref in (("logit(+sae)", "logit(base)"), ("logit(+dyn+sae)", "logit(base)"), ("hrm(+sae)", "hrm(base)"), ("logit(+dyn)", "logit(base)"), ("logit(+sae)", "conf(maxp)")):
            def diff_fn(i, p_new=p_new, p_ref=p_ref):
                f = lambda pn: (lambda cc: pareto_area(np.column_stack([cc[:, 0] * (1 + over[policies[pn][1]] + (over["ctrl"] if pn.startswith(("hrm", "logit")) else 0.0)), cc[:, 1], cc[:, 2]]), c_lo, c_hi))(policy_curve(policies[pn][0][s][i], c[i], taus[::5]))
                return f(p_new) - f(p_ref)
            d0 = diff_fn(np.arange(N))
            lo, hi = bootstrap_ci(diff_fn, N, B=a.boot // 3, seed=1)
            out[f"delta_area[{p_new} - {p_ref}]"] = [float(d0), lo, hi]
        # seed variability for HRM controllers
        for pname in ("hrm(base)", "hrm(+sae)"):
            seeds = policies[pname + "|seeds"][0][s]
            areas = []
            for sc in seeds:
                cc = policy_curve(sc, c, taus)
                ovk = "sae" if pname == "hrm(+sae)" else "base"
                cc[:, 0] = cc[:, 0] * (1 + over[ovk] + over["ctrl"])
                areas.append(pareto_area(cc, c_lo, c_hi))
            out[pname + "|area_by_seed"] = areas
        res["policies"][s] = out
        print(f"[{s}] areas: " + ", ".join(f"{k}={v['area']:.3f}" for k, v in out.items() if isinstance(v, dict) and 'area' in v))
        print("   deltas: " + "; ".join(f"{k}={v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]" for k, v in out.items() if k.startswith("delta_area")))
    json.dump(res, open(os.path.join(res_dir(), f"{tag}__controller.json"), "w"), indent=1)
    plot(res, tag)


def plot(res, tag):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = {"fixed": "k", "conf(maxp)": "tab:gray", "dstate": "tab:brown", "logit(base)": "tab:blue", "logit(+sae)": "tab:red",
            "logit(+dyn+sae)": "tab:orange", "hrm(base)": "tab:cyan", "hrm(+sae)": "tab:purple"}
    for ax, s in zip(axs, ("test_id", "test_ood")):
        for p, c in cols.items():
            e = res["policies"][s].get(p)
            if not e:
                continue
            cur = np.array(e["curve"])
            o = np.argsort(cur[:, 0])
            ax.plot(cur[o, 0], np.maximum.accumulate(cur[o, 1]), c=c, label=f"{p} ({e['area']:.3f})", lw=1.5 if p != "fixed" else 2)
        ax.set_xlim(1, 16); ax.set_xlabel("mean loops (FLOP-charged)"); ax.set_ylabel("accuracy"); ax.set_title(s); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f5_controller"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
