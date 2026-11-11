"""Toy path step 7: why do correct answers disappear?  Feature-level attribution of answer loss + feature atlas + CKA.

The readout is exactly linear in the last-token state (state norm is fixed by the loop's RMS-norm, ln_f is a rescale):
      logit_c = W_c . (w * x) + b_c       and     x = scale * sum_f z_f D_f + (mu + scale*b_pre) + residual
so the change in the margin between the old (correct) class c and the new (winning wrong) class w, between loop a (last
correct loop) and loop b, splits EXACTLY into per-feature terms plus a residual term:
      d(logit_c - logit_w) = sum_f dz_f * g_f  +  residual_part
      with g_f = scale * (W_c - W_w) . (w * D_f)
Only held-out test_id examples whose answer is correct at some loop <= K_train and wrong at loop 32 ('lost') are used;
the SAE and its frequency table come from sae_train.  Everything here is descriptive/attributional; the causal tests are t4.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import toy
from recsae.analysis import encode_loops, event_taxonomy, feature_frequency
from recsae.io import fig_path, load_split, res_dir, safe, sae_prefix
from recsae.sae import TopKSAE
from recsae.stats import linear_cka, subspace_overlap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--t_end", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tag = a.tag or safe(a.name)
    model, task, ck = toy.load_toy(a.name)
    K_train = ck["cfg"]["K"]
    sae = TopKSAE.load(sae_prefix(a.name, "all"))
    n = sae.cfg.n_latents
    scale = float(sae.scale)
    W, b = model.head.weight.detach(), model.head.bias.detach()
    w = model.ln_f.weight.detach()
    D = sae.W_dec.detach()
    G = scale * (D * w) @ W.t()                          # [n, C]: per-unit-activation contribution to each class logit
    R = {"name": a.name}
    freq_train = feature_frequency(encode_loops(sae, load_split(a.name, "sae_train")["last"], list(range(1, K_train + 1)))[0], n)
    # frequency at loops beyond training depth, also from sae_train (beyond-train-depth exposure of the dictionary)
    freq_late = feature_frequency(encode_loops(sae, load_split(a.name, "sae_train")["last"], list(range(K_train + 1, a.t_end + 1)))[0], n)
    out = {}
    for sp in ("test_id", "test_ood"):
        d = load_split(a.name, sp)
        c = d["correct"]
        y = d["y"]
        K = c.shape[1]
        tb = a.t_end
        # loop a: last correct loop <= K_train; require wrong at tb
        cand = c[:, :K_train]
        has = cand.any(1) & ~c[:, tb - 1]
        ta = np.where(has, K_train - 1 - np.argmax(cand[:, ::-1], axis=1), -1) + 1           # 1-indexed loop
        idx = np.where(has)[0]
        res = {"n_lost": int(len(idx)), "n": int(len(y))}
        if len(idx) < 30:
            out[sp] = res
            continue
        Xa = torch.from_numpy(d["last"][idx, ta[idx]].astype(np.float32))
        Xb = torch.from_numpy(d["last"][idx, tb].astype(np.float32))
        za, zb = sae.encode(Xa), sae.encode(Xb)
        wrong = torch.from_numpy(d["pred"][idx, tb - 1])
        cc = torch.from_numpy(y[idx])
        gd = G[:, cc].t() - G[:, wrong].t()                      # [N, n] per-feature margin weights (c minus w)
        dz = zb - za
        contrib = dz * gd                                         # [N, n]
        # residual (what the SAE does not explain) and baseline (mu / b_pre) parts
        ra = Xa - sae.decode(za)
        rb = Xb - sae.decode(zb)
        res_term = (((rb - ra) * w) * (W[cc] - W[wrong])).sum(1)
        total = (((Xb - Xa) * w) * (W[cc] - W[wrong])).sum(1)    # exact margin change (c - w) from the states
        explained = contrib.sum(1)
        # decomposition of the explained part
        support_lost = (contrib * ((dz < 0) & (gd > 0)).float()).sum(1)        # features that supported c were switched off
        rival_gain = (contrib * ((dz > 0) & (gd < 0)).float()).sum(1)          # features supporting the rival came on
        other = explained - support_lost - rival_gain                         # opposite-sign changes (partly offsetting)
        neg_part = lambda v: float(v.clamp(max=0).sum() / total.clamp(max=0).sum())
        res.update({
            "mean_margin_change_total": float(total.mean()),
            "share_support_lost": float(support_lost.sum() / total.sum()),
            "share_rival_gain": float(rival_gain.sum() / total.sum()),
            "share_offsetting_other": float(other.sum() / total.sum()),
            "share_sae_residual": float(res_term.sum() / total.sum()),
            "check_decomposition_error": float((total - explained - res_term).abs().mean()),
        })
        # novelty: how much of the rival-gain contribution comes from features that are rare within training depth
        rare = torch.from_numpy(freq_train < 0.01)
        late_only = torch.from_numpy((freq_train < 0.01) & (freq_late >= 0.01))
        mask_rg = ((dz > 0) & (gd < 0)).float()
        res["rival_gain_from_rare_train_features"] = float((contrib * mask_rg * rare[None].float()).sum() / (contrib * mask_rg).sum())
        res["rival_gain_from_late_only_features"] = float((contrib * mask_rg * late_only[None].float()).sum() / (contrib * mask_rg).sum())
        res["n_features_late_only"] = int(late_only.sum())
        # features with the largest total contribution to loss
        tot_f = contrib.sum(0)
        top = torch.argsort(tot_f)[:8]
        res["top_loss_features"] = [{"f": int(f), "total_contrib": float(tot_f[f]), "freq_trainK": float(freq_train[f]),
                                     "freq_late": float(freq_late[f]), "writes_class": int(G[f].argmax())} for f in top]
        out[sp] = res
        print(f"[{sp}] lost n={len(idx)}: support_lost {res['share_support_lost']:.2f} rival_gain {res['share_rival_gain']:.2f} "
              f"offset {res['share_offsetting_other']:.2f} residual {res['share_sae_residual']:.2f}; rival-gain from rare-in-training features "
              f"{res['rival_gain_from_rare_train_features']:.2f}, late-only {res['rival_gain_from_late_only_features']:.2f}")
    R["loss_attribution"] = out
    # ---------------- feature atlas: mean activation vs loop by trajectory type, with the class each feature writes
    d = load_split(a.name, "test_id")
    tax = event_taxonomy(d["correct"])
    loops = list(range(1, d["correct"].shape[1] + 1))
    idxc, valc = encode_loops(sae, d["last"], loops)
    Z = np.zeros((len(loops), d["last"].shape[0], n), np.float32)
    for j in range(len(loops)):
        np.put_along_axis(Z[j], idxc[:, j].astype(np.int64), valc[:, j].astype(np.float32), axis=1)
    ev = np.load(os.path.join(recsae.DATA, f"{safe(a.name)}__event_auc.npz"))
    pick = {"answer-correct": np.argsort(-(ev["now"] - 0.5))[:3], "about-to-lose": np.argsort(-(ev["harm"] - 0.5))[:3]}
    fig, axs = plt.subplots(2, 3, figsize=(14, 6))
    for r, (nm, fs) in enumerate(pick.items()):
        for c_, f in enumerate(fs):
            ax = axs[r, c_]
            for t, col in (("stable", "tab:green"), ("lost", "tab:red"), ("flicker", "tab:orange"), ("never", "gray")):
                m = tax == t
                if m.sum() > 10:
                    ax.plot(loops, Z[:, m, f].mean(1), c=col, label=t)
            ax.axvline(K_train, c="k", ls="--", lw=0.6)
            ax.set_title(f"{nm}: feature {int(f)}  writes class {int(G[f].argmax())}  (AUC {float(ev['now' if r == 0 else 'harm'][f]):.2f})", fontsize=9)
            ax.set_xlabel("loop"); ax.set_ylabel("mean activation")
    axs[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f7_atlas"), dpi=130); plt.close()
    # ---------------- CKA / subspace overlap between loops (representation-level stability)
    X = d["last"].astype(np.float32)
    sel = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
    cka = np.zeros((len(sel), len(sel)))
    sov = np.zeros_like(cka)
    for i, li in enumerate(sel):
        for j, lj in enumerate(sel):
            if j >= i:
                cka[i, j] = cka[j, i] = linear_cka(X[:, li], X[:, lj])
                sov[i, j] = sov[j, i] = subspace_overlap(X[:, li], X[:, lj], 8)
    R["cka_loops"] = {"loops": sel, "cka": cka.tolist(), "subspace_overlap_k8": sov.tolist()}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, M, ttl in ((axs[0], cka, "linear CKA between loops"), (axs[1], sov, "top-8 PCA subspace overlap")):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, fontsize=7); ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=7)
        ax.set_title(ttl); plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f8_loop_cka"), dpi=130); plt.close()
    json.dump(R, open(os.path.join(res_dir(), f"{tag}__attr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
