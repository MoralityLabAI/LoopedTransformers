"""Toy path step 3: SAE fidelity, feature stability, event features, predictive value (RQ1, RQ3, RQ4).

All feature/probe selection uses `ana_train` (+ `ana_val` for hyper-parameters); every number reported as a
result is computed on the held-out `test_id` / `test_ood` splits.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import DATA, ROOT, toy
from recsae.analysis import (Logit, adjacent_overlap, dense_codes, encode_loops, event_labels, event_taxonomy,
                             feature_frequency, first_stable, loop_recon, subsample_rows)
from recsae.io import fig_path, is_mini, load_split, res_dir, safe, sae_prefix
from recsae.sae import TopKSAE
from recsae.stats import auroc, auroc_cols, bootstrap_ci

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = 4                      # look-ahead horizon (loops) for retained / gain / harm targets
TMAX = 28                 # last loop t at which (example, t) rows are built (t+H <= K); reset in main()


def conf_feats(logits):
    """logits [N, K, C] -> max-prob, margin, entropy per (example, loop)."""
    z = logits - logits.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    ps = np.sort(p, -1)
    ent = -(p * np.log(p + 1e-12)).sum(-1)
    return ps[..., -1], ps[..., -1] - ps[..., -2], ent


def per_loop_signals(d, sae, loops, device="cpu", freq_ref=None):
    """Scalar signals per (example, loop slot) for loops (1-indexed loop numbers).  Returns dict of [N, nl] arrays
    plus sparse codes."""
    X = d["last"]
    N = X.shape[0]
    idx, val = encode_loops(sae, X, loops, device)
    n = sae.cfg.n_latents
    mp, mg, en = conf_feats(d["logits"])
    sig = {"maxp": mp[:, [l - 1 for l in loops]], "margin": mg[:, [l - 1 for l in loops]],
           "ent": en[:, [l - 1 for l in loops]]}
    # state dynamics
    xt = X[:, loops].astype(np.float32)
    xp = X[:, [l - 1 for l in loops]].astype(np.float32)
    sig["dstate"] = np.linalg.norm(xt - xp, axis=-1) / np.maximum(np.linalg.norm(xt, axis=-1), 1e-6)
    sig["cosstate"] = (xt * xp).sum(-1) / np.maximum(np.linalg.norm(xt, axis=-1) * np.linalg.norm(xp, axis=-1), 1e-6)
    # SAE recon error at each loop (normalised by state energy about SAE mean)
    err = np.zeros((N, len(loops)), np.float32)
    mu = sae.mu.cpu().numpy()
    for j, l in enumerate(loops):
        x = torch.from_numpy(X[:, l]).float()
        with torch.no_grad():
            xh = sae.decode(sae.encode(x))
        err[:, j] = ((x - xh).pow(2).sum(-1) / ((x - torch.from_numpy(mu)).pow(2).sum(-1) + 1e-8)).numpy()
    sig["recon"] = err
    # overlap with previous loop code
    idx_p, val_p = encode_loops(sae, X, [l - 1 for l in loops], device)
    jac = np.zeros((N, len(loops)), np.float32)
    cs = np.zeros_like(jac)
    cr = np.zeros_like(jac)
    nov = np.zeros_like(jac)
    for j in range(len(loops)):
        a, b = dense_codes(idx_p, val_p, n, j=j), dense_codes(idx, val, n, j=j)
        A, B = a > 0, b > 0
        inter, union = (A & B).sum(1), (A | B).sum(1)
        jac[:, j] = inter / np.maximum(union, 1)
        cs[:, j] = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
        cr[:, j] = np.minimum(a, b).sum(1) / np.maximum(a.sum(1), 1e-9)
        if freq_ref is not None:
            nov[:, j] = ((freq_ref[idx[:, j].astype(np.int64)] < 1e-3) * (val[:, j] > 0)).sum(1) / np.maximum(
                (val[:, j] > 0).sum(1), 1)
    sig.update({"jacc": jac, "codecos": cs, "carry": cr, "novel": nov})
    return sig, (idx, val)


def build_rows(d, sig, codes, loops, tmin=2, tmax=None):
    tmax = TMAX if tmax is None else tmax
    """Rows (example, t) for t in loops with t in [tmin, tmax]; returns feature dict and targets."""
    sel = [j for j, l in enumerate(loops) if tmin <= l <= tmax]
    ls = np.array([loops[j] for j in sel])
    ev = event_labels(d["correct"], H)               # columns c <-> loop c+1
    rows = {k: v[:, sel] for k, v in sig.items()}
    t_idx = ls - 1
    tg = {k: v[:, t_idx] for k, v in ev.items()}
    tt = np.broadcast_to(ls[None], rows["maxp"].shape)
    return rows, tg, tt, sel


def flat(a):
    return np.asarray(a).reshape(-1)


def feature_sets(rows, tt, kind):
    base = [flat(rows["maxp"]), flat(rows["margin"]), flat(rows["ent"]), flat(tt), np.log(flat(tt))]
    dyn = [flat(rows["dstate"]), flat(rows["cosstate"])]
    sae_sc = [flat(rows["recon"]), flat(rows["jacc"]), flat(rows["codecos"]), flat(rows["carry"]), flat(rows["novel"])]
    parts = {"base": base, "base+dyn": base + dyn, "base+sae": base + sae_sc, "base+dyn+sae": base + dyn + sae_sc}
    return np.stack(parts[kind], 1).astype(np.float32)


def bootstrap_auc(score, label, B=300, seed=0):
    n = len(score)
    lo, hi = bootstrap_ci(lambda i: auroc(score[i], label[i]), n, B=B, seed=seed)
    return lo, hi


def paired_delta_ci(s_a, s_b, label, B=300, seed=0):
    n = len(label)
    f = lambda i: auroc(s_a[i], label[i]) - auroc(s_b[i], label[i])
    lo, hi = bootstrap_ci(f, n, B=B, seed=seed)
    return f(np.arange(n)), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--K_train", type=int, default=None, help="required for mini runs")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global H, TMAX
    H = a.H
    tag = a.tag or safe(a.name)
    rng = np.random.default_rng(0)
    sae = TopKSAE.load(sae_prefix(a.name, "all"))
    sae_k = TopKSAE.load(sae_prefix(a.name, "trainK"))
    model = None
    if is_mini(a.name):
        K_train = a.K_train
    else:
        model, task, ck = toy.load_toy(a.name)
        K_train = ck["cfg"]["K"]
    n = sae.cfg.n_latents
    S = {s: load_split(a.name, s) for s in ("sae_train", "ana_train", "ana_val", "test_id", "test_ood")}
    K = S["test_id"]["correct"].shape[1]
    loops_all = list(range(1, K + 1))
    TMAX = K - H
    R = {"name": a.name, "K_train": K_train, "K_eval": K, "H": H, "n_latents": n, "k": sae.cfg.k}

    # ---------------------------------------------------------- 0. behaviour
    beh = {}
    for s in ("test_id", "test_ood"):
        c = S[s]["correct"]
        tax = event_taxonomy(c)
        fs = first_stable(c)
        beh[s] = {"acc_by_loop": c.mean(0).tolist(), "taxonomy": {k: float((tax == k).mean()) for k in
                                                               ("never", "stable", "lost", "flicker")},
                  "best_loop": int(c.mean(0).argmax() + 1), "best_acc": float(c.mean(0).max()),
                  "final_acc": float(c[:, -1].mean()),
                  "median_first_stable": float(np.median(fs[fs <= K])) if (fs <= K).any() else None}
    R["behaviour"] = beh
    print("behaviour:", json.dumps({s: {k: v for k, v in beh[s].items() if k != 'acc_by_loop'} for s in beh}))

    # ---------------------------------------------------------- 1. SAE fidelity per loop (RQ1/RQ3)
    fid = {}
    for sname, sv in (("all", sae), ("trainK", sae_k)):
        fid[sname] = {}
        for s in ("test_id", "test_ood"):
            r = loop_recon(sv, S[s]["last"], loops_all)
            fid[sname][s] = {"nmse_own": [x["nmse_own"] for x in r], "fvu_ref": [x["fvu_ref"] for x in r]}
    R["fidelity"] = fid

    # ---------------------------------------------------------- 2. feature stability across loops (RQ1)
    stab = {}
    freq_ref = feature_frequency(encode_loops(sae, S["sae_train"]["last"][:, :], list(range(1, K_train + 1)))[0], n)
    for s in ("test_id", "test_ood"):
        d = S[s]
        idx, val = encode_loops(sae, d["last"], loops_all)
        groups = (d["y"] * 1000 + d["N"]).astype(int)
        ov = adjacent_overlap(idx, val, n, groups=groups, seed=1)
        tax = event_taxonomy(d["correct"])
        stab[s] = {
            "jacc": [float(np.nanmean(o["jacc"])) for o in ov], "jacc_null": [float(np.nanmean(o["jacc_null"])) for o in ov],
            "cos": [float(np.nanmean(o["cos"])) for o in ov], "cos_null": [float(np.nanmean(o["cos_null"])) for o in ov],
            "carry": [float(np.nanmean(o["carry"])) for o in ov],
            "jacc_by_tax": {k: [float(np.nanmean(o["jacc"][tax == k])) if (tax == k).sum() > 5 else None for o in ov]
                            for k in ("stable", "lost", "never", "flicker")},
            "cos_by_tax": {k: [float(np.nanmean(o["cos"][tax == k])) if (tax == k).sum() > 5 else None for o in ov]
                           for k in ("stable", "lost", "never", "flicker")},
        }
        # per-feature persistence: P(active at t+1 | active at t) vs base rate (loops in [K_train, K])
        pers = []
        for j in range(K - 1):
            A = dense_codes(idx, val, n, j=j) > 0
            B = dense_codes(idx, val, n, j=j + 1) > 0
            pers.append(((A & B).sum(0) / np.maximum(A.sum(0), 1)).astype(np.float32))
        stab[s]["persistence_mean_by_loop"] = [float(np.mean(p[p > 0])) if (p > 0).any() else None for p in pers]
        # active-feature "lifetime": number of loops each feature is active, per example, then mean over features used
        life = np.zeros((idx.shape[0], n), np.int16)
        for j in range(K):
            ix = idx[:, j].astype(np.int64)
            life[np.arange(idx.shape[0])[:, None], ix] += (val[:, j] > 0)
        used = life > 0
        stab[s]["lifetime_hist"] = np.bincount(life[used].astype(int), minlength=K + 1).tolist()
        stab[s]["n_distinct_features_per_example"] = float(used.sum(1).mean())
        stab[s]["settle_loop"] = None
    R["stability"] = stab

    # ---------------------------------------------------------- 3. event features (RQ3), selection on ana_train
    sig_tr, codes_tr = per_loop_signals(S["ana_train"], sae, loops_all, a.device, freq_ref)
    rows_tr, tg_tr, tt_tr, sel = build_rows(S["ana_train"], sig_tr, codes_tr, loops_all)
    loops_sel = [loops_all[j] for j in sel]
    events = {}
    # class written by each feature (direct logit attribution through ln_f and head)
    if model is not None:
        W_dec = sae.W_dec.detach().cpu() * float(sae.scale)
        lnw = model.ln_f.weight.detach().cpu()
        dla = ((W_dec * lnw) @ model.head.weight.detach().cpu().t()).numpy()          # [n, C]
    else:
        dla = np.zeros((n, 1))
    R["feature_class_write"] = {"dla_max_class": dla.argmax(1).tolist(), "dla_margin": ((np.sort(dla, 1)[:, -1] - np.sort(dla, 1)[:, -2]).tolist() if dla.shape[1] > 1 else None)}
    idx_tr, val_tr = codes_tr
    for ev_name in ("now", "harm", "gain"):
        # per-loop-bin AUROCs for every feature on ana_train; average across bins weighted by n
        accum = np.zeros(n)
        wsum = np.zeros(n)
        for jj, l in enumerate(loops_sel):
            lab = tg_tr[ev_name][:, jj]
            cond = np.ones(len(lab), bool)
            if ev_name == "harm":
                cond = tg_tr["now"][:, jj]
            if ev_name == "gain":
                cond = ~tg_tr["now"][:, jj]
            if lab[cond].sum() < 20 or (~lab[cond]).sum() < 20:
                continue
            Z = dense_codes(idx_tr, val_tr, n, rows=np.where(cond)[0], j=sel[jj])
            au = auroc_cols(Z, lab[cond])
            act = (Z > 0).sum(0)
            wt = np.minimum(act, 500).astype(float) * (act >= 10)
            accum += np.nan_to_num(au - 0.5) * wt
            wsum += wt
        score = np.where(wsum > 0, accum / np.maximum(wsum, 1e-9) + 0.5, 0.5)
        events[ev_name] = {"auc_train": score}
    R["events_train_top"] = {k: np.argsort(-np.abs(v["auc_train"] - 0.5))[:10].tolist() for k, v in events.items()}

    # held-out replication: top-m features chosen on ana_train, evaluated on test_id/test_ood (same loop-binned AUC)
    freq_all = feature_frequency(idx_tr, n)
    rep = {}
    for s in ("test_id", "test_ood"):
        sig_te, codes_te = per_loop_signals(S[s], sae, loops_all, a.device, freq_ref)
        rows_te, tg_te, tt_te, sel_te = build_rows(S[s], sig_te, codes_te, loops_all)
        idx_te, val_te = codes_te
        rep[s] = {}
        for ev_name, e in events.items():
            auc_tr = e["auc_train"]
            top = np.argsort(-np.abs(auc_tr - 0.5))[:50]
            accum = np.zeros(n)
            wsum = np.zeros(n)
            for jj, l in enumerate(loops_sel):
                lab = tg_te[ev_name][:, jj]
                cond = np.ones(len(lab), bool)
                if ev_name == "harm":
                    cond = tg_te["now"][:, jj]
                if ev_name == "gain":
                    cond = ~tg_te["now"][:, jj]
                if lab[cond].sum() < 20 or (~lab[cond]).sum() < 20:
                    continue
                Z = dense_codes(idx_te, val_te, n, rows=np.where(cond)[0], j=sel_te[jj])
                au = auroc_cols(Z, lab[cond])
                act = (Z > 0).sum(0)
                wt = np.minimum(act, 500).astype(float) * (act >= 10)
                accum += np.nan_to_num(au - 0.5) * wt
                wsum += wt
            auc_te = np.where(wsum > 0, accum / np.maximum(wsum, 1e-9) + 0.5, 0.5)
            same_sign = np.sign(auc_te[top] - 0.5) == np.sign(auc_tr[top] - 0.5)
            ok = (wsum > 0)
            # null: random features with similar training frequency
            null_dev = []
            for _ in range(200):
                pick = rng.choice(np.where(ok)[0], size=len(top), replace=False)
                null_dev.append(float(np.mean(np.sign(auc_tr[top] - 0.5) * (auc_te[pick] - 0.5))))
            rep[s][ev_name] = {
                "top50_mean_train_dev": float(np.mean(np.abs(auc_tr[top] - 0.5))),
                "top50_mean_test_signed_dev": float(np.mean(np.sign(auc_tr[top] - 0.5) * (auc_te[top] - 0.5))),
                "top50_sign_agreement": float(same_sign[ok[top]].mean()) if ok[top].any() else None,
                "null_signed_dev_mean": float(np.mean(null_dev)), "null_signed_dev_p95": float(np.percentile(null_dev, 95)),
                "top10": top[:10].tolist(), "auc_train_top10": auc_tr[top[:10]].tolist(), "auc_test_top10": auc_te[top[:10]].tolist(),
                "dla_class_top10": dla[top[:10]].argmax(1).tolist(),
            }
        if s == "test_id":
            sig_te_id, codes_te_id = sig_te, codes_te
    R["event_features"] = rep

    # ---------------------------------------------------------- 4. predictive value of signals (RQ4)
    pred = {}
    _, tg_a, tt_a, _ = build_rows(S["ana_train"], sig_tr, codes_tr, loops_all)
    sets = ["base", "base+dyn", "base+sae", "base+dyn+sae"]
    Xtr = {k: feature_sets(rows_tr, tt_tr, k) for k in sets}
    # sparse-code and raw-state probes (controls)
    def code_feats(rows_idx, val_idx, sel_slots, nrows):
        out = []
        for j in sel_slots:
            out.append(dense_codes(rows_idx, val_idx, n, j=j))
        return np.stack(out, 1).reshape(-1, n)
    for tgt in ("stay", "harm", "gain"):
        pred[tgt] = {}
        y_tr = flat(tg_a[tgt])
        cond_tr = np.ones_like(y_tr, bool)
        if tgt == "harm":
            cond_tr = flat(tg_a["now"])
        if tgt == "gain":
            cond_tr = ~flat(tg_a["now"])
        ridx = np.where(cond_tr)[0]
        ridx = ridx[subsample_rows(len(ridx), 150_000, rng)]
        fits = {k: Logit(l2=1e-2).fit(Xtr[k][ridx], y_tr[ridx]) for k in sets}
        # control probes on code and raw state
        Zc = code_feats(idx_tr, val_tr, sel, len(y_tr))
        fits["base+code"] = Logit(l2=1e-1, iters=40).fit(np.concatenate([Xtr["base"][ridx], Zc[ridx]], 1), y_tr[ridx])
        Xr = S["ana_train"]["last"][:, loops_sel].astype(np.float32).reshape(-1, S["ana_train"]["last"].shape[-1])
        fits["base+raw"] = Logit(l2=1e-1, iters=40).fit(np.concatenate([Xtr["base"][ridx], Xr[ridx]], 1), y_tr[ridx])
        for s in ("test_id", "test_ood"):
            sig_te, codes_te = per_loop_signals(S[s], sae, loops_all, a.device, freq_ref)
            rows_te, tg_te, tt_te, sel_te = build_rows(S[s], sig_te, codes_te, loops_all)
            y = flat(tg_te[tgt])
            cond = np.ones_like(y, bool)
            if tgt == "harm":
                cond = flat(tg_te["now"])
            if tgt == "gain":
                cond = ~flat(tg_te["now"])
            ii = np.where(cond)[0]
            scores = {}
            for k in sets:
                scores[k] = fits[k].score(feature_sets(rows_te, tt_te, k)[ii])
            Zte = code_feats(codes_te[0], codes_te[1], sel_te, len(y))
            scores["base+code"] = fits["base+code"].score(np.concatenate([feature_sets(rows_te, tt_te, "base")[ii], Zte[ii]], 1))
            Xrte = S[s]["last"][:, [loops_all[j] for j in sel_te]].astype(np.float32).reshape(-1, S[s]["last"].shape[-1])
            scores["base+raw"] = fits["base+raw"].score(np.concatenate([feature_sets(rows_te, tt_te, "base")[ii], Xrte[ii]], 1))
            # single-signal AUCs (sign-free)
            single = {}
            for nm in ("maxp", "margin", "ent", "dstate", "cosstate", "recon", "jacc", "codecos", "carry", "novel"):
                single[nm] = auroc(flat(rows_te[nm])[ii], y[ii])
            yy = y[ii]
            res = {k: {"auc": auroc(v, yy)} for k, v in scores.items()}
            for k in res:
                lo, hi = bootstrap_auc(scores[k], yy, B=a.boot)
                res[k]["ci"] = [lo, hi]
            for k in res:
                if k != "base":
                    dlt, lo, hi = paired_delta_ci(scores[k], scores["base"], yy, B=a.boot)
                    res[k]["delta_vs_base"] = [dlt, lo, hi]
            pred[tgt][s] = {"n": int(len(yy)), "pos_rate": float(yy.mean()), "probes": res, "single_signal_auc": single}
            print(f"[{tgt}/{s}] n={len(yy)} pos={yy.mean():.3f} base={res['base']['auc']:.3f} "
                  f"+sae={res['base+sae']['auc']:.3f} +dyn={res['base+dyn']['auc']:.3f} +code={res['base+code']['auc']:.3f} "
                  f"+raw={res['base+raw']['auc']:.3f}")
    R["prediction"] = pred

    # ---------------------------------------------------------- save + figures
    json.dump(R, open(os.path.join(res_dir(), f"{tag}__analysis.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(os.path.join(DATA, f"{safe(a.name)}__event_auc.npz"), **{k: v["auc_train"] for k, v in events.items()})
    make_figures(R, tag, K_train)
    print("done", tag)


def make_figures(R, tag, K_train):
    K = R["K_eval"]
    L = np.arange(1, K + 1)
    # Fig 1: accuracy by loop + taxonomy
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for s, c in (("test_id", "tab:blue"), ("test_ood", "tab:red")):
        ax[0].plot(L, R["behaviour"][s]["acc_by_loop"], c=c, label=s)
    ax[0].axvline(K_train, c="gray", ls="--", lw=1)
    ax[0].set_xlabel("loop"); ax[0].set_ylabel("accuracy"); ax[0].set_title("answer accuracy by loop"); ax[0].legend()
    tax = ["stable", "lost", "flicker", "never"]
    w = 0.35
    for i, s in enumerate(("test_id", "test_ood")):
        ax[1].bar(np.arange(4) + i * w, [R["behaviour"][s]["taxonomy"][t] for t in tax], w, label=s)
    ax[1].set_xticks(np.arange(4) + w / 2); ax[1].set_xticklabels(tax); ax[1].set_title("trajectory types"); ax[1].legend()
    for sname, ls in (("all", "-"), ("trainK", "--")):
        for s, c in (("test_id", "tab:blue"), ("test_ood", "tab:red")):
            ax[2].plot(L, R["fidelity"][sname][s]["nmse_own"], c=c, ls=ls, label=f"{sname}/{s}")
    ax[2].axvline(K_train, c="gray", ls="--", lw=1); ax[2].set_yscale("log")
    ax[2].set_xlabel("loop"); ax[2].set_ylabel("NMSE"); ax[2].set_title("SAE reconstruction error by loop"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f1_behaviour_recon"), dpi=130); plt.close()
    # Fig 2: stability
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    t = np.arange(1, K)
    for s, c in (("test_id", "tab:blue"), ("test_ood", "tab:red")):
        st = R["stability"][s]
        ax[0].plot(t, st["jacc"], c=c, label=f"{s} Jaccard"); ax[0].plot(t, st["jacc_null"], c=c, ls=":", label=f"{s} label-matched null")
        ax[1].plot(t, st["cos"], c=c); ax[1].plot(t, st["cos_null"], c=c, ls=":")
    for k, c in zip(("stable", "lost", "never", "flicker"), ("tab:green", "tab:red", "gray", "tab:orange")):
        v = R["stability"]["test_id"]["jacc_by_tax"][k]
        ax[2].plot(t, [np.nan if x is None else x for x in v], c=c, label=k)
    for a_ in ax:
        a_.axvline(K_train, c="gray", ls="--", lw=1); a_.set_xlabel("loop t (overlap of t vs t+1)")
    ax[0].set_ylabel("active-set Jaccard"); ax[0].legend(fontsize=7); ax[0].set_title("feature-set overlap")
    ax[1].set_title("code cosine"); ax[2].set_title("Jaccard by trajectory type (test_id)"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fig_path(tag, "f2_stability"), dpi=130); plt.close()
    # Fig 3: predictive value
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    order = ["base", "base+dyn", "base+sae", "base+dyn+sae", "base+code", "base+raw"]
    for ax_, tgt in zip(axs, ("stay", "harm", "gain")):
        for i, (s, c) in enumerate((("test_id", "tab:blue"), ("test_ood", "tab:red"))):
            pr = R["prediction"][tgt][s]["probes"]
            aucs = [pr[k]["auc"] for k in order]
            err = [[pr[k]["auc"] - pr[k]["ci"][0] for k in order], [pr[k]["ci"][1] - pr[k]["auc"] for k in order]]
            ax_.errorbar(np.arange(len(order)) + 0.15 * i, aucs, yerr=err, fmt="o", c=c, label=s, capsize=2)
        ax_.set_xticks(np.arange(len(order))); ax_.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax_.set_title(f"AUROC predicting '{tgt}' (H={R['H']})"); ax_.axhline(0.5, c="gray", lw=0.5)
    axs[0].legend(); plt.tight_layout(); plt.savefig(fig_path(tag, "f3_prediction"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
