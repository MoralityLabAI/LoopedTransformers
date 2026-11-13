"""LOOPSCOPE-01 Q9 (amendment 9): why does F1/F8 predict length extrapolation?

M: state memory = median_items (||dm/dh_1|| ||h_1||) / (||dm/dh_8|| ||h_8||), m = own-top log-odds margin at loop 8.
C: loop-index (t = 2..8) ridge-probe accuracy from the normalized cursor state (chance 1/7).
Clock ablation: after every loop, u <- u - P P^T (u - mu) on the cursor (P = top-6 loop-index class-mean directions),
depth-16 items; vs 4 random 6-d subspaces.

    run:   all 24 arms -> <out>/<arm>.json
    score: registered rules -> <out>/q9_report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.loopscope01.scripts import loopscope_toy as L  # noqa: E402
from mechinterp.loopscope01.scripts import loopscope_toy_q7 as Q  # noqa: E402
from mechinterp.loopscope01.scripts.loopscope_toy_tca import spearman  # noqa: E402

OUT = Path("mechinterp/loopscope01/results/q9")
Q8_OUT = Q.EXP / "loopscope_toy_q8"
TRAIN_SEED, TEST_SEED, DEEP_SEED = 931008, 930008, 930016
N_TRAIN, N_TEST = 512, 256
LOOPS = list(range(2, 9))  # loop indices for the clock probe
RANK, N_RAND, GAIN = 6, 4, 0.10


def own_margin(logits):
    top = logits.argmax(-1)
    zc = logits.gather(1, top[:, None])[:, 0]
    return zc - torch.logsumexp(logits.scatter(1, top[:, None], float("-inf")), 1)


def state_memory(model, tokens, T=8):
    """Per-item (||g_1|| ||h_1||) / (||g_T|| ||h_T||), g_t = d(own margin at loop T)/d h_t, full state."""
    e = Q.embed(model, tokens).detach().requires_grad_(True)  # parameters are frozen; give the graph a leaf
    h = torch.zeros_like(e)
    hs = []
    for _ in range(T):
        h = Q.step(model, h, e)
        h.retain_grad()
        hs.append(h)
    own_margin(model.readout(hs[-1][:, -1])).sum().backward()
    n = lambda x: x.flatten(1).norm(dim=1)  # noqa: E731
    r = n(hs[0].grad) * n(hs[0]) / (n(hs[-1].grad) * n(hs[-1]))
    return r.detach().cpu().numpy()


def cursor_units(model, tokens, T=8):
    with torch.no_grad():
        hs = Q.trajectory(model, Q.embed(model, tokens), T)
    return [h[:, -1] for h in hs]


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def clock_probe(U_tr, U_te):
    """U_*: list over loops 1..8 of [N, d] unit cursor states. Returns test accuracy for t in LOOPS."""
    Xtr = torch.cat([U_tr[t - 1] for t in LOOPS]).cpu()
    Ytr = torch.cat([torch.full((len(U_tr[0]),), i) for i, _ in enumerate(LOOPS)])
    Xte = torch.cat([U_te[t - 1] for t in LOOPS]).cpu()
    Yte = torch.cat([torch.full((len(U_te[0]),), i) for i, _ in enumerate(LOOPS)])
    return float((L.ridge_probe(Xtr, Ytr, Xte, len(LOOPS)) == Yte).float().mean())


def clock_subspace(U_tr, k=RANK):
    means = torch.stack([U_tr[t - 1].mean(0) for t in LOOPS])
    mu = torch.cat([U_tr[t - 1] for t in LOOPS]).mean(0)
    return mu, torch.linalg.svd(means - means.mean(0), full_matrices=False)[2][:k].T


def run_ablated(model, tokens, T, P, mu):
    """Readout predictions per loop with the cursor's P-component mean-ablated after every loop (P None: clean)."""
    with torch.no_grad():
        e = Q.embed(model, tokens)
        h = torch.zeros_like(e)
        preds = []
        for _ in range(T):
            h = Q.step(model, h, e)
            if P is not None:
                c = h[:, -1]
                nrm = c.norm(dim=-1, keepdim=True)
                u = c / nrm
                h = h.clone()
                h[:, -1] = nrm * (u - ((u - mu) @ P) @ P.T)
            preds.append(model.readout(h[:, -1]).argmax(-1).cpu())
    return torch.stack(preds)


def accs(preds, hops):
    a = (preds == hops.T).float().mean(1).numpy()
    return {"acc8": float(a[7]), "deep": float(a[8:].mean()), "by_loop": a.tolist()}


def lock_in(acc_by_loop, T=8):
    return next(t for t in range(1, T + 1) if all(a >= 0.9 for a in acc_by_loop[t - 1:T]))


def cmd_run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_tr, _ = L.make_batch(16, N_TRAIN, TRAIN_SEED, 16)
    tok_te, _ = L.make_batch(16, N_TEST, TEST_SEED, 16)
    tok_deep, hops_deep = L.make_batch(16, N_TEST, DEEP_SEED, 16)
    dev = args.device
    for s, arm, toy, tca in Q.arms():
        dst = out / f"{arm}.json"
        if dst.exists() and not args.force:
            print(f"[skip] {arm}")
            continue
        ck = toy / "ckpt" / f"{arm}.final.pt"
        st = torch.load(ck, map_location="cpu")
        model = L.build_model(L.model_args_from(st["cfg"]))
        model.load_state_dict(st["model"])
        model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        M = float(np.median(state_memory(model, tok_te.to(dev))))
        U_tr = [unit(x) for x in cursor_units(model, tok_tr.to(dev))]
        U_te = [unit(x) for x in cursor_units(model, tok_te.to(dev))]
        C = clock_probe(U_tr, U_te)
        mu, P = clock_subspace(U_tr)
        clean = accs(run_ablated(model, tok_deep.to(dev), 16, None, None), hops_deep)
        clock = accs(run_ablated(model, tok_deep.to(dev), 16, P, mu), hops_deep)
        rand = []
        for k in range(N_RAND):
            g = torch.Generator().manual_seed(k)
            R = torch.linalg.qr(torch.randn(P.shape[0], RANK, generator=g))[0].to(P)
            rand.append(accs(run_ablated(model, tok_deep.to(dev), 16, R, mu), hops_deep))
        rnd = {"acc8": float(np.mean([r["acc8"] for r in rand])), "deep": float(np.mean([r["deep"] for r in rand]))}
        q7 = json.loads((Q.EXP / "loopscope_toy_q7" / f"{arm}.json").read_text())
        q8 = json.loads((Q8_OUT / f"{arm}.json").read_text())
        m = json.loads((toy / "runs" / arm / "metrics.json").read_text())
        r = json.loads((tca / f"{arm}.json").read_text())
        res = {"arm": arm, "set": s, "lam": st["cfg"]["lam"], "seed": st["cfg"]["seed"], "checkpoint_sha256": L.sha256_file(ck),
               "heldout": m["summary"]["cap_heldout"], "f1_over_f8": r["depths"]["8"]["frag_first_over_last"],
               "M": M, "C": C, "lock_in": lock_in(q8["acc_by_loop"]), "kappa_late": q7["kappa_late"]["pca_rel"],
               "ablation": {"clean": clean, "clock": clock, "random": rnd,
                            "effect_deep": clock["deep"] - rnd["deep"], "effect_acc8": clock["acc8"] - rnd["acc8"]}}
        dst.write_text(json.dumps(res, indent=1))
        a = res["ablation"]
        print(f"[{arm}] held-out={res['heldout']:.3f} F1/F8={res['f1_over_f8']:.4f} M={M:.4f} C={C:.3f} L={res['lock_in']} "
              f"deep clean/clock/rand={clean['deep']:.3f}/{clock['deep']:.3f}/{rnd['deep']:.3f} "
              f"acc8 clean/clock/rand={clean['acc8']:.3f}/{clock['acc8']:.3f}/{rnd['acc8']:.3f}", flush=True)
    return 0


def cmd_score(args):
    out = Path(args.out)
    rows = [json.loads((out / f"{a}.json").read_text()) for _, a, _, _ in Q.arms()]
    assert len(rows) == 24, len(rows)
    held = np.array([r["heldout"] for r in rows])
    fails = held < Q.EXTRAP
    f18 = np.array([r["f1_over_f8"] for r in rows])
    M = np.array([r["M"] for r in rows])
    C = np.array([r["C"] for r in rows])
    Lk = np.array([r["lock_in"] for r in rows], float)
    kl = np.array([r["kappa_late"] for r in rows])
    rho_M, rho_C = spearman(M, f18), spearman(C, f18)
    a_M, a_C, a_f = Q.auc(M, fails), Q.auc(C, fails), Q.auc(f18, fails)
    p9a = "supported" if rho_M >= 0.8 else "not supported"
    p9b = "supported" if a_C >= 0.85 and rho_C >= 0.6 else "weak" if a_C >= 0.75 else "not supported"
    X = np.column_stack([np.ones(24), Lk, np.log(kl)])
    y = np.log(f18)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    p9c = "supported" if r2 >= 0.7 else "not supported"
    eff = np.array([r["ablation"]["effect_deep"] for r in rows])
    n_gain = int((eff[fails] > GAIN).sum())
    p9d = "supported" if n_gain >= 4 else "partial" if n_gain >= 2 else "not supported"
    rep = {"P9a": p9a, "P9b": p9b, "P9c": p9c, "P9d": p9d,
           "spearman_with_f1f8": {"M": rho_M, "C": rho_C, "lock_in": spearman(Lk, f18), "log_kappa_late": spearman(np.log(kl), f18)},
           "auc": {"F1/F8": a_f, "M": a_M, "C": a_C, "lock_in": Q.auc(Lk, fails), "kappa_late": Q.auc(kl, fails)},
           "ols": {"beta_const_L_logkappa": beta.tolist(), "r2": r2},
           "ablation": {"failures_gaining_gt_0.1": n_gain,
                        "effect_deep_fail": {r["arm"]: r["ablation"]["effect_deep"] for r in rows if r["heldout"] < Q.EXTRAP},
                        "effect_acc8_fail": {r["arm"]: r["ablation"]["effect_acc8"] for r in rows if r["heldout"] < Q.EXTRAP},
                        "effect_deep_extrap_median": float(np.median(eff[~fails])),
                        "effect_acc8_extrap_median": float(np.median([r["ablation"]["effect_acc8"] for r in rows if r["heldout"] >= Q.EXTRAP]))}}
    (out / "q9_report.json").write_text(json.dumps(rep, indent=1))
    md = ["# LOOPSCOPE-01 Q9: why does F1/F8 predict extrapolation? (amendment 9)", "",
          f"P9a (state memory): **{p9a}**. Spearman(M, F1/F8) = {rho_M:+.3f}; AUC(M) = {a_M:.3f}.",
          f"P9b (loop clock): **{p9b}**. AUC(C) = {a_C:.3f}; Spearman(C, F1/F8) = {rho_C:+.3f}.",
          f"P9c (log F1/F8 ~ L + log kappa_late): **{p9c}**. R^2 = {r2:.3f}; beta = {np.round(beta, 3).tolist()}.",
          f"P9d (clock ablation raises deep accuracy > +0.10 over random): **{p9d}**. {n_gain}/7 failures.", "",
          "AUC (higher = fails): " + ", ".join(f"{k} {v:.3f}" for k, v in rep["auc"].items()) + ".",
          "Spearman with F1/F8: " + ", ".join(f"{k} {v:+.3f}" for k, v in rep["spearman_with_f1f8"].items()) + ".",
          f"Ablation in extrapolators: median deep effect {rep['ablation']['effect_deep_extrap_median']:+.3f}, "
          f"median acc8 effect {rep['ablation']['effect_acc8_extrap_median']:+.3f}.", "",
          "| arm | held-out | F1/F8 | M | C | L | kappa_late | deep clean | deep clock | deep random | acc8 clock | acc8 random |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["f1_over_f8"]):
        a = r["ablation"]
        md.append(f"| {r['arm']} ({r['set']}) | {r['heldout']:.3f} | {r['f1_over_f8']:.4f} | {r['M']:.4f} | {r['C']:.3f} | "
                  f"{r['lock_in']} | {r['kappa_late']:.3f} | {a['clean']['deep']:.3f} | {a['clock']['deep']:.3f} | "
                  f"{a['random']['deep']:.3f} | {a['clock']['acc8']:.3f} | {a['random']['acc8']:.3f} |")
    (out / "q9_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    return {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
