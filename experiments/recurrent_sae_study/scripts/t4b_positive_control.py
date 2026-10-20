"""Toy path step 4b: positive-control ablations + persistence of edits.

Ablate, per example, the m ACTIVE features that write the example's *currently predicted* class the most (largest
direct-logit weight toward that class).  This is the 'does editing features move the readout at all' control for the
event-selected sets in t4.  Arms: target / count-matched random active features / energy-matched random direction.
Reported at the edited loop (immediate) and at K=32 (persistence), plus a persistence curve over later loops.
Held-out test_id / test_ood only.
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
from recsae.io import safe, SPLITS, fig_path, res_dir, sae_prefix
from recsae.sae import TopKSAE
from t4_toy_intervene import paired_ci

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PredClassEditor:
    def __init__(self, sae, model, G, t0, m, mode, seed=0, ref_norm=None):
        self.sae, self.model, self.G, self.t0, self.m, self.mode = sae, model, G, t0, m, mode
        self.g = torch.Generator().manual_seed(seed)
        self.ref_norm = ref_norm
        self.removed_norm = None

    @torch.no_grad()
    def __call__(self, k, h):
        if k != self.t0:
            return h
        x = h[:, -1]
        z = self.sae.encode(x)
        pred = (((x * self.model.ln_f.weight) @ self.model.head.weight.t()) + self.model.head.bias).argmax(-1)
        score = self.G[:, pred].t()                                  # [B, n] contribution of each unit feature to the predicted class
        active = z > 0
        if self.mode == "target":
            s = score.masked_fill(~active, float("-inf"))
            top = s.topk(min(self.m, s.shape[1]), dim=1).indices
            sel = torch.zeros_like(active).scatter_(1, top, True) & active & (score > 0)
        elif self.mode == "rand_feat":
            # same number of features as the target arm would remove, chosen at random among the remaining active ones
            s = score.masked_fill(~active, float("-inf"))
            top = s.topk(min(self.m, s.shape[1]), dim=1).indices
            tsel = torch.zeros_like(active).scatter_(1, top, True) & active & (score > 0)
            n_i = tsel.sum(1)
            cand = active & ~tsel
            r = torch.rand(z.shape, generator=self.g) * cand.float() - (~cand).float()
            sel = (r.argsort(1, descending=True).argsort(1) < n_i[:, None]) & cand
        delta = (z * sel.float()) @ self.sae.W_dec * float(self.sae.scale)
        self.removed_norm = delta.norm(dim=1)
        self.n_removed = sel.sum(1)
        xn = x - delta
        xn = xn * (x.norm(dim=1, keepdim=True) / xn.norm(dim=1, keepdim=True).clamp_min(1e-6))
        h = h.clone()
        h[:, -1] = xn
        return h


class RandDir:
    def __init__(self, t0, ref, seed=0):
        self.t0, self.ref, self.g = t0, ref, torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def __call__(self, k, h):
        if k != self.t0:
            return h
        x = h[:, -1]
        r = torch.randn(x.shape, generator=self.g)
        r = r / r.norm(dim=1, keepdim=True) * self.ref[:, None]
        xn = x - r
        xn = xn * (x.norm(dim=1, keepdim=True) / xn.norm(dim=1, keepdim=True).clamp_min(1e-6))
        h = h.clone(); h[:, -1] = xn
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--t0s", default="2,4,8,16,24")
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--n_eval", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tag = a.tag or safe(a.name)
    model, task, ck = toy.load_toy(a.name)
    sae = TopKSAE.load(sae_prefix(a.name, "all"))
    G = (float(sae.scale) * (sae.W_dec.detach() * model.ln_f.weight.detach()) @ model.head.weight.detach().t())   # [n, C]
    K = 32
    t0s, ms = [int(x) for x in a.t0s.split(",")], [int(x) for x in a.ms.split(",")]
    R = {"name": a.name, "rows": {}}
    for sp in ("test_id", "test_ood"):
        lengths, npl, off = SPLITS[sp]
        rng = np.random.default_rng(1_000_000 * (ck["seed"] + 1) + off)
        split = toy.make_split(task, rng, lengths, npl)
        xs = [x[: a.n_eval] for x in split["x"]]
        ys = np.concatenate([split["y"][o: o + a.n_eval] for o in np.cumsum([0] + [len(x) for x in split["x"]])[:-1]])

        def roll(hook_fn_factory):
            out = []
            for x in xs:
                xb = torch.from_numpy(x)
                out.append(toy.rollout(model, xb, K, hook=hook_fn_factory(xb))["logits"].numpy())
            return np.concatenate(out)
        clean = roll(lambda xb: None).argmax(-1) == ys[:, None]
        rows = []
        for t0 in t0s:
            for m in ms:
                res_t, res_rf, res_rd, rem, nrem = [], [], [], [], []
                for x in xs:
                    xb = torch.from_numpy(x)
                    e1 = PredClassEditor(sae, model, G, t0, m, "target")
                    res_t.append(toy.rollout(model, xb, K, hook=e1)["logits"].numpy()); rem.append(e1.removed_norm); nrem.append(e1.n_removed)
                    e2 = PredClassEditor(sae, model, G, t0, m, "rand_feat", seed=5)
                    res_rf.append(toy.rollout(model, xb, K, hook=e2)["logits"].numpy())
                    e3 = RandDir(t0, e1.removed_norm, seed=9)
                    res_rd.append(toy.rollout(model, xb, K, hook=e3)["logits"].numpy())
                ct = np.concatenate(res_t).argmax(-1) == ys[:, None]
                cf = np.concatenate(res_rf).argmax(-1) == ys[:, None]
                cr = np.concatenate(res_rd).argmax(-1) == ys[:, None]
                nr = torch.cat(nrem).numpy()
                row = {"t0": t0, "m": m, "mean_removed": float(nr.mean()), "frac_edited": float((nr > 0).mean())}
                for lab, j in (("at_t0", t0 - 1), ("t0+2", min(t0 + 1, K - 1)), ("t0+8", min(t0 + 7, K - 1)), ("final", K - 1)):
                    for nm, cc in (("target", ct), ("rand_feat", cf), ("rand_dir", cr)):
                        row[f"{nm}_d_{lab}"] = paired_ci(cc[:, j].astype(float) - clean[:, j].astype(float))
                row["persistence_target_minus_clean"] = (ct.astype(float) - clean.astype(float)).mean(0).tolist()
                row["persistence_randfeat_minus_clean"] = (cf.astype(float) - clean.astype(float)).mean(0).tolist()
                rows.append(row)
                if m == max(ms):
                    print(f"[{sp}] t0={t0} m={m}: edited={row['frac_edited']:.2f} d@t0 target={row['target_d_at_t0'][0]:+.3f} rf={row['rand_feat_d_at_t0'][0]:+.3f} rd={row['rand_dir_d_at_t0'][0]:+.3f}  "
                          f"| d@final target={row['target_d_final'][0]:+.3f} rf={row['rand_feat_d_final'][0]:+.3f} rd={row['rand_dir_d_final'][0]:+.3f}")
        R["rows"][sp] = rows
    json.dump(R, open(os.path.join(res_dir(), f"{tag}__poscontrol.json"), "w"), indent=1)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for r_, sp in enumerate(("test_id", "test_ood")):
        rows = R["rows"][sp]
        ax = axs[r_, 0]
        for t0, col in zip(t0s, plt.cm.viridis(np.linspace(0, 0.9, len(t0s)))):
            ax.plot(ms, [x["target_d_at_t0"][0] for x in rows if x["t0"] == t0], c=col, marker="o", label=f"t0={t0}")
            ax.plot(ms, [x["rand_feat_d_at_t0"][0] for x in rows if x["t0"] == t0], c=col, ls=":", marker="x")
        ax.set_title(f"{sp}: Δacc at edited loop (solid=class features, dotted=random)"); ax.set_xlabel("m features ablated"); ax.legend(fontsize=7)
        ax = axs[r_, 1]
        for t0, col in zip(t0s, plt.cm.viridis(np.linspace(0, 0.9, len(t0s)))):
            row = [x for x in rows if x["t0"] == t0 and x["m"] == max(ms)][0]
            ax.plot(np.arange(1, K + 1), row["persistence_target_minus_clean"], c=col, label=f"t0={t0}")
        ax.axhline(0, c="k", lw=0.5); ax.set_title(f"{sp}: Δacc vs clean by loop (m={max(ms)})"); ax.set_xlabel("loop")
    plt.tight_layout(); plt.savefig(fig_path(tag, "f9_poscontrol"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
