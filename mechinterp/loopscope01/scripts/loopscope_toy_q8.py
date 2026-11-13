"""LOOPSCOPE-01 Q8 (amendment 8): do failing toy loopers leave the contracting region beyond the training depth?

Drift: r_t = share of ||u_t - mu_ref||^2 outside the top-16 principal subspace of depth-8 reference cursor states
(u = h^c/||h^c||, loops 1..8); dr = median_items(mean r_{9..16} - mean r_{1..8}) on depth-16 items.
Deep contraction: kappa_deep = median_items(mean gamma_t, t = 8..15), gamma_t as in Q7.

    run:   all 24 arms -> <out>/<arm>.json
    score: registered rules -> <out>/q8_report.{json,md}
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

OUT = Path("mechinterp/loopscope01/results/q8")
REF_SEED, DEEP_SEED, ITEMS, K = 930008, 930016, 256, 16
TRAIN_T, DEEP_T = 8, 16


def unit_rows(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def ref_subspace(U, k=K):
    """U [N, d] normalized reference states -> (mu, basis [d, k])."""
    mu = U.mean(0)
    return mu, torch.linalg.svd(U - mu, full_matrices=False)[2][:k].T


def residual_fraction(U, mu, P):
    """Share of ||U - mu||^2 outside span(P), per row."""
    X = U - mu
    tot = (X * X).sum(-1).clamp_min(1e-30)
    inside = ((X @ P) ** 2).sum(-1)
    return (1 - inside / tot).clamp_min(0)


def analyse(model, tok_ref, tok_deep, hops_deep):
    with torch.no_grad():
        e_ref = Q.embed(model, tok_ref)
        U_ref = torch.cat([unit_rows(h[:, -1]) for h in Q.trajectory(model, e_ref, TRAIN_T)])
        mu, P = ref_subspace(U_ref)
        e = Q.embed(model, tok_deep)
        hs = Q.trajectory(model, e, DEEP_T)
        # float32 rounding accumulates over 16 loops (exact to 1e-14 in float64), so check per-item relative error
        a, b = torch.stack([h[:, -1] for h in hs]), model(tok_deep, DEEP_T)
        fwd_err = float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max())
        assert fwd_err < 1e-3, f"explicit transition != forward (max relative error {fwd_err:.2e})"
        Hc = [h[:, -1] for h in hs]
        r = torch.stack([residual_fraction(unit_rows(x), mu, P) for x in Hc]).cpu().numpy()  # [16, B]
        pred = torch.stack([model.readout(x).argmax(-1) for x in Hc]).cpu()
        acc = (pred == hops_deep.T).float().mean(1).numpy()  # per loop t, hop t
    cn = [x.norm(dim=1) for x in Hc]
    Qp = [Q.pca_basis(x) for x in Hc]
    gam = []
    for t in range(1, DEEP_T):
        f = lambda x: Q.step(model, x, e)  # noqa: E731
        _, s = Q.restricted(f, hs[t - 1].detach(), Qp[t - 1], Qp[t])
        gam.append((s * cn[t - 1] / cn[t]).detach().cpu().numpy())
    gam = np.stack(gam)  # [15, B], row t-1 = transition t -> t+1
    dr = float(np.median(r[TRAIN_T:].mean(0) - r[:TRAIN_T].mean(0)))
    kd = float(np.median(gam[TRAIN_T - 1:].mean(0)))
    kl = float(np.median(gam[[4, 5, 6]].mean(0)))
    r_med, g_med = np.median(r, 1), np.median(gam, 1)
    thr = 2 * r_med[:TRAIN_T].max()
    first = lambda xs, cond, off=1: next((i + off for i, x in enumerate(xs) if cond(x)), None)  # noqa: E731
    return {"forward_max_rel_err": fwd_err, "drift_dr": dr, "kappa_deep": kd, "kappa_late_deep_items": kl,
            "r_median_by_loop": r_med.tolist(), "gamma_median_by_transition": g_med.tolist(),
            "acc_by_loop": acc.tolist(), "norm_median_by_loop": [float(x.median()) for x in cn],
            "first_acc_below_0.9": first(acc, lambda a: a < 0.9),
            "first_drift_2x": first(r_med[TRAIN_T:], lambda x: x > thr, TRAIN_T + 1),
            "first_gamma_ge1_after8": first(g_med[TRAIN_T - 1:], lambda g: g >= 1, TRAIN_T)}


def cmd_run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_ref, _ = L.make_batch(16, ITEMS, REF_SEED, 16)
    tok_deep, hops_deep = L.make_batch(16, ITEMS, DEEP_SEED, 16)
    for s, arm, toy, tca in Q.arms():
        dst = out / f"{arm}.json"
        if dst.exists() and not args.force:
            print(f"[skip] {arm}")
            continue
        ck = toy / "ckpt" / f"{arm}.final.pt"
        st = torch.load(ck, map_location="cpu")
        model = L.build_model(L.model_args_from(st["cfg"]))
        model.load_state_dict(st["model"])
        model.to(args.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        res = analyse(model, tok_ref.to(args.device), tok_deep.to(args.device), hops_deep)
        m = json.loads((toy / "runs" / arm / "metrics.json").read_text())
        r = json.loads((tca / f"{arm}.json").read_text())
        res.update({"arm": arm, "set": s, "lam": st["cfg"]["lam"], "seed": st["cfg"]["seed"],
                    "checkpoint_sha256": L.sha256_file(ck), "heldout": m["summary"]["cap_heldout"],
                    "cap16_metrics": m["capability_by_depth"]["16"], "f1_over_f8": r["depths"]["8"]["frag_first_over_last"]})
        dst.write_text(json.dumps(res, indent=1))
        print(f"[{arm}] held-out={res['heldout']:.3f} dr={res['drift_dr']:+.4f} kappa_deep={res['kappa_deep']:.3f} "
              f"acc16={res['acc_by_loop'][-1]:.3f} r={np.round(res['r_median_by_loop'], 3).tolist()} "
              f"g={np.round(res['gamma_median_by_transition'], 2).tolist()}", flush=True)
    return 0


def cmd_score(args):
    out = Path(args.out)
    rows = [json.loads((out / f"{a}.json").read_text()) for _, a, _, _ in Q.arms()]
    assert len(rows) == 24, len(rows)
    held = np.array([r["heldout"] for r in rows])
    fails = held < Q.EXTRAP
    dr = np.array([r["drift_dr"] for r in rows])
    kd = np.array([r["kappa_deep"] for r in rows])
    f18 = np.array([r["f1_over_f8"] for r in rows])
    a_dr, a_kd, a_f = Q.auc(dr, fails), Q.auc(kd, fails), Q.auc(f18, fails)
    p8a = "supported" if a_dr >= 0.90 else "weak" if a_dr >= 0.75 else "not supported"
    absok = (kd[fails] >= 1).sum() >= 5 and (kd[~fails] >= 1).sum() <= 3
    p8b = ("supported" if a_kd >= 0.90 and absok else "rank only" if a_kd >= 0.90
           else "weak" if a_kd >= 0.75 else "not supported")
    order = {}
    for r in rows:
        if r["heldout"] >= Q.EXTRAP:
            continue
        a, d, g = r["first_acc_below_0.9"], r["first_drift_2x"], r["first_gamma_ge1_after8"]
        rel = ("no drift" if d is None else "no accuracy loss" if a is None
               else "drift precedes" if d < a else "coincides" if d == a else "drift follows")
        order[r["arm"]] = {"first_acc_below_0.9": a, "first_drift_2x": d, "first_gamma_ge1_after8": g, "relation": rel}
    rep = {"P8a": p8a, "P8b": p8b, "auc": {"drift_dr": a_dr, "kappa_deep": a_kd, "F1/F8": a_f},
           "kappa_deep_ge1": {"fail": int((kd[fails] >= 1).sum()), "extrap": int((kd[~fails] >= 1).sum())},
           "spearman": {"dr_vs_heldout": spearman(dr, held), "kd_vs_heldout": spearman(kd, held),
                        "dr_vs_f1f8": spearman(dr, f18), "kd_vs_f1f8": spearman(kd, f18)},
           "order": order}
    (out / "q8_report.json").write_text(json.dumps(rep, indent=1))
    md = ["# LOOPSCOPE-01 Q8: drift and contraction beyond the training depth (amendment 8)", "",
          f"P8a (drift, primary): **{p8a}**. AUC(dr) = {a_dr:.3f} (F1/F8 reference {a_f:.3f}).",
          f"P8b (contraction lost): **{p8b}**. AUC(kappa_deep) = {a_kd:.3f}; kappa_deep >= 1 in "
          f"{rep['kappa_deep_ge1']['fail']}/7 failures, {rep['kappa_deep_ge1']['extrap']}/17 extrapolators.",
          "Spearman: " + ", ".join(f"{k} {v:+.3f}" for k, v in rep["spearman"].items()) + ".", "",
          "P8c (order, failing arms; loops 1-indexed): " + "; ".join(
              f"{k}: acc<0.9 at {v['first_acc_below_0.9']}, drift 2x at {v['first_drift_2x']}, gamma>=1 at "
              f"{v['first_gamma_ge1_after8']} ({v['relation']})" for k, v in order.items()) + ".", "",
          "| arm | held-out | F1/F8 | dr | kappa_deep | r by loop (1..16) | gamma by transition (1..15) | acc by loop 9..16 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["drift_dr"]):
        md.append(f"| {r['arm']} ({r['set']}) | {r['heldout']:.3f} | {r['f1_over_f8']:.4f} | {r['drift_dr']:+.4f} | "
                  f"{r['kappa_deep']:.3f} | {' '.join(f'{x:.2f}' for x in r['r_median_by_loop'])} | "
                  f"{' '.join(f'{x:.2f}' for x in r['gamma_median_by_transition'])} | "
                  f"{' '.join(f'{x:.2f}' for x in r['acc_by_loop'][8:])} |")
    (out / "q8_report.md").write_text("\n".join(md) + "\n")
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
