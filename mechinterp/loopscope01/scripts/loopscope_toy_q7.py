"""LOOPSCOPE-01 Q7 (amendment 7): contraction restricted to the computation-carrying subspace, on frozen toy loopers.

For the transition h_{t+1} = core(inject(h_t) + e), with E_t the top-16 principal subspace of the cursor state:
    gamma_t = sigma_max(Q_{t+1}^T J_t^{cc} Q_t) * ||h_t^c|| / ||h_{t+1}^c||      (relative error gain)
    kappa_late = median over items of mean_{t=5,6,7} gamma_t

    run:   all 24 arms -> <out>/<arm>.json
    score: registered rules -> <out>/q7_report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp, vjp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.loopscope01.scripts import loopscope_toy as L  # noqa: E402

EXP = Path("experiments/recurrence_monitorability_v0")
SETS = {"q4": (EXP / "loopscope_toy", EXP / "loopscope_toy_tca"), "q5": (EXP / "loopscope_toy_q5", EXP / "loopscope_toy_q5_tca")}
OUT = Path("mechinterp/loopscope01/results/q7")
ITEM_SEED, ITEMS, DEPTH, K, LATE = 930008, 256, 8, 16, (5, 6, 7)
Q5_MISSES = ("lam0.01_s4", "lam0.01_s8", "lam0_s9")
EXTRAP = 0.90
F18_AUC_BAR = 113 / 119  # F1/F8 over the 24 arms, fixed in amendment 7


def embed(model, tokens):
    n = model.n_nodes
    edges = tokens[:, :2 * n].reshape(tokens.shape[0], n, 2)
    h = torch.cat([model.tok(edges[..., 0]) + model.dst(edges[..., 1]), model.tok(tokens[:, 2 * n:])], 1)
    return h + model.pos(torch.arange(h.shape[1], device=tokens.device))


def block(blk, x):
    """Block.forward with explicit softmax attention (forward-mode AD safe)."""
    b, t, d = x.shape
    hd = d // blk.heads
    q, k, v = blk.qkv(blk.ln1(x)).view(b, t, 3, blk.heads, hd).permute(2, 0, 3, 1, 4)
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), -1) @ v
    x = x + blk.proj(a.transpose(1, 2).reshape(b, t, d))
    return x + blk.fc2(F.gelu(blk.fc1(blk.ln2(x))))


def step(model, h, e):
    h = model.inject(h) + e
    for blk in model.core:
        h = block(blk, h)
    return h


def trajectory(model, e, T):
    h, hs = torch.zeros_like(e), []
    for _ in range(T):
        h = step(model, h, e)
        hs.append(h)
    return hs


def bnorm(x):
    return x.flatten(1).norm(dim=1)


def _unit(x):
    return x / bnorm(x).clamp_min(1e-30).view(-1, *([1] * (x.dim() - 1)))


def sigma_full(f, h, iters=30, seed=0, cursor=False):
    """Per-item sigma_max of J (or its cursor->cursor block) by power iteration on J^T J."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    _, back = vjp(f, h)
    mask = torch.zeros_like(h)
    if cursor:
        mask[:, -1] = 1
    else:
        mask += 1
    v = _unit(torch.randn(h.shape, generator=g).to(h) * mask)
    for _ in range(iters):
        u = jvp(f, (h,), (v,))[1] * mask
        v = _unit(back(u)[0] * mask)
    return bnorm(jvp(f, (h,), (v,))[1] * mask)


def restricted(f, h, Qa, Qb):
    """Per-item matrix Qb^T J^{cc} Qa [B, kb, ka] and its sigma_max."""
    cols = []
    for j in range(Qa.shape[1]):
        tan = torch.zeros_like(h)
        tan[:, -1] = Qa[:, j]
        cols.append(jvp(f, (h,), (tan,))[1][:, -1] @ Qb)
    M = torch.stack(cols, 2)
    return M, torch.linalg.svdvals(M)[:, 0]


def pca_basis(Hc, k=K):
    return torch.linalg.svd(Hc - Hc.mean(0), full_matrices=False)[2][:k].T


def label_basis(Hc, labels, n_classes):
    mu = torch.stack([Hc[labels == c].mean(0) for c in range(n_classes) if (labels == c).any()])
    return torch.linalg.svd(mu - mu.mean(0), full_matrices=False)[2][:n_classes - 1].T


def analyse(model, tokens, hops, T=DEPTH, seed=0):
    e = embed(model, tokens)
    with torch.no_grad():
        hs = trajectory(model, e, T)
        ref = model(tokens, T)
        assert torch.allclose(torch.stack([h[:, -1] for h in hs]), ref, atol=1e-4, rtol=1e-4), "explicit transition != forward"
    Hc = [h[:, -1] for h in hs]
    cn = [x.norm(dim=1) for x in Hc]
    fn = [bnorm(h) for h in hs]
    Qp = [pca_basis(x) for x in Hc]
    Ql = [label_basis(x, hops[:, t].to(x.device), model.n_nodes) for t, x in enumerate(Hc)]
    prof = {k: [] for k in ("pca_rel", "pca_abs", "lab_rel", "full_rel", "cursor_rel")}
    per_item = {k: [] for k in prof}
    for t in range(1, T):  # transition h_t -> h_{t+1}, t = 1..T-1 (index t-1 -> t)
        f = lambda x: step(model, x, e)  # noqa: E731
        h = hs[t - 1].detach()
        _, sp = restricted(f, h, Qp[t - 1], Qp[t])
        _, sl = restricted(f, h, Ql[t - 1], Ql[t])
        sf = sigma_full(f, h, seed=seed + t)
        sc = sigma_full(f, h, seed=seed + 100 + t, cursor=True)
        vals = {"pca_abs": sp, "pca_rel": sp * cn[t - 1] / cn[t], "lab_rel": sl * cn[t - 1] / cn[t],
                "full_rel": sf * fn[t - 1] / fn[t], "cursor_rel": sc * cn[t - 1] / cn[t]}
        for k, v in vals.items():
            per_item[k].append(v.detach().cpu().numpy())
            prof[k].append(float(v.median()))
    per_item = {k: np.stack(v) for k, v in per_item.items()}  # [T-1, B]
    late = [t - 1 for t in LATE]
    kap = {k: float(np.median(v[late].mean(0))) for k, v in per_item.items()}
    # finite perturbation after loop 5 in E_5, rerun loops 6..8
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    fin = {}
    with torch.no_grad():
        base = model.readout(Hc[-1]).argmax(-1)
        for eps in (0.1, 0.3):
            devs, flips = [], []
            for _ in range(4):
                w = _unit(torch.randn(len(tokens), K, generator=g)).to(e)
                hp = hs[4].clone()
                hp[:, -1] += eps * cn[4][:, None] * (w @ Qp[4].T)
                for _ in range(5, T):
                    hp = step(model, hp, e)
                devs.append(((hp[:, -1] - Hc[-1]) @ Qp[-1]).norm(dim=1) / (eps * cn[-1]))
                flips.append((model.readout(hp[:, -1]).argmax(-1) != base).float())
            fin[f"eps{eps}"] = {"rel_dev_median": float(torch.stack(devs).mean(0).median()),
                                "flip_rate": float(torch.stack(flips).mean())}
        acc = float((base.cpu() == hops[:, T - 1]).float().mean())
    onset = {k: next((t for t in range(1, T) if all(p < 1 for p in v[t - 1:])), T) for k, v in prof.items()}
    return {"profile_median": prof, "kappa_late": kap, "onset": onset, "finite": fin, "acc": acc,
            "late_all_below_1": {k: all(v[t - 1] < 1 for t in LATE) for k, v in prof.items()}}


def arms():
    out = []
    for s, (toy, tca) in SETS.items():
        for p in sorted((toy / "ckpt").glob("lam*.final.pt")):
            out.append((s, p.name.replace(".final.pt", ""), toy, tca))
    return out


def cmd_run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokens, hops = L.make_batch(16, ITEMS, ITEM_SEED, 16)
    for s, arm, toy, tca in arms():
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
        res = analyse(model, tokens.to(args.device), hops)
        m = json.loads((toy / "runs" / arm / "metrics.json").read_text())
        r = json.loads((tca / f"{arm}.json").read_text())
        res.update({"arm": arm, "set": s, "lam": st["cfg"]["lam"], "seed": st["cfg"]["seed"],
                    "checkpoint_sha256": L.sha256_file(ck), "heldout": m["summary"]["cap_heldout"],
                    "acc_d8_metrics": m["capability_by_depth"]["8"],
                    "f1_over_f8": r["depths"]["8"]["frag_first_over_last"]})
        dst.write_text(json.dumps(res, indent=1))
        print(f"[{arm}] acc={res['acc']:.3f} kappa_late={res['kappa_late']['pca_rel']:.3f} "
              f"pca_rel={np.round(res['profile_median']['pca_rel'], 3).tolist()} "
              f"full={np.round(res['profile_median']['full_rel'], 3).tolist()} fin={res['finite']}", flush=True)
    return 0


def auc(stat, fails):
    """P(stat of a random non-extrapolator > stat of a random extrapolator), ties half."""
    pos, neg = stat[fails], stat[~fails]
    return float(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)))


def cmd_score(args):
    out = Path(args.out)
    rows = [json.loads((out / f"{a}.json").read_text()) for _, a, _, _ in arms()]
    assert len(rows) == 24, len(rows)
    fails = np.array([r["heldout"] < EXTRAP for r in rows])
    kl = np.array([r["kappa_late"]["pca_rel"] for r in rows])
    stats = {"kappa_late (pca, primary)": kl, "F1/F8": np.array([r["f1_over_f8"] for r in rows])}
    for k, lab in (("full_rel", "global full J late"), ("cursor_rel", "global cursor block late"),
                   ("lab_rel", "label subspace late (secondary)"), ("pca_abs", "pca absolute late")):
        stats[lab] = np.array([r["kappa_late"][k] for r in rows])
    stats["finite eps0.1 rel dev (secondary)"] = np.array([r["finite"]["eps0.1"]["rel_dev_median"] for r in rows])
    stats["finite eps0.3 rel dev (secondary)"] = np.array([r["finite"]["eps0.3"]["rel_dev_median"] for r in rows])
    stats["finite eps0.3 flip rate (secondary)"] = np.array([r["finite"]["eps0.3"]["flip_rate"] for r in rows])
    aucs = {k: auc(v, fails) for k, v in stats.items()}
    a = aucs["kappa_late (pca, primary)"]
    p7a = ("supported" if a >= 0.90 and a > F18_AUC_BAR else "matches F1/F8" if a >= 0.90
           else "weaker" if a >= 0.75 else "not supported")
    kmin_fail = kl[fails].min()
    miss = {r["arm"]: r["kappa_late"]["pca_rel"] for r in rows if r["arm"] in Q5_MISSES}
    p7b = "supported" if all(v < kmin_fail for v in miss.values()) else "not supported"
    below = np.array([r["late_all_below_1"]["pca_rel"] for r in rows])
    p7c = "supported" if below[~fails].all() and below[fails].sum() <= 3 else "not supported"
    gl = max(aucs["global full J late"], aucs["global cursor block late"])
    p7d = "supported" if a - gl >= 0.10 else "not supported"
    on = {r["arm"]: r["onset"]["pca_rel"] for r in rows}
    ex_other = [on[r["arm"]] for r in rows if r["heldout"] >= EXTRAP and r["arm"] not in Q5_MISSES]
    rep = {"aucs": aucs, "P7a": p7a, "P7b": p7b, "P7c": p7c, "P7d": p7d, "F1F8_bar": F18_AUC_BAR,
           "q5_misses_kappa": miss, "min_kappa_nonextrap": float(kmin_fail),
           "late_below1": {"extrap": f"{int(below[~fails].sum())}/{int((~fails).sum())}",
                           "nonextrap": f"{int(below[fails].sum())}/{int(fails.sum())}"},
           "onset": {"q5_misses": {k: on[k] for k in Q5_MISSES}, "other_extrap_median": float(np.median(ex_other)),
                     "nonextrap": {r["arm"]: on[r["arm"]] for r in rows if r["heldout"] < EXTRAP}}}
    (out / "q7_report.json").write_text(json.dumps(rep, indent=1))
    md = ["# LOOPSCOPE-01 Q7: contraction restricted to the computation-carrying subspace (amendment 7)", "",
          f"P7a (primary): **{p7a}**. AUC(kappa_late) = {a:.3f} vs F1/F8 bar {F18_AUC_BAR:.3f} (24 arms, 7 fail).",
          f"P7b (Q5 misses below every non-extrapolator): **{p7b}**. Misses {', '.join(f'{k} {v:.3f}' for k, v in miss.items())}; "
          f"min non-extrapolator {kmin_fail:.3f}.",
          f"P7c (late median profile < 1): **{p7c}**. Extrapolators {rep['late_below1']['extrap']}, "
          f"non-extrapolators {rep['late_below1']['nonextrap']}.",
          f"P7d (restricted beats global by >= 0.10): **{p7d}**. Global best AUC {gl:.3f}.",
          f"P7e (onset, descriptive): Q5 misses {rep['onset']['q5_misses']}; other extrapolators median "
          f"{rep['onset']['other_extrap_median']:.1f}; non-extrapolators {rep['onset']['nonextrap']}.", "",
          "AUC for non-extrapolation: " + "; ".join(f"{k} {v:.3f}" for k, v in aucs.items()) + ".", "",
          "| arm | held-out | F1/F8 | kappa_late | gamma_t pca rel (t=1..7) | full J rel late | label late | fin 0.1 | flip 0.3 | onset |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["kappa_late"]["pca_rel"]):
        md.append(f"| {r['arm']} ({r['set']}) | {r['heldout']:.3f} | {r['f1_over_f8']:.4f} | {r['kappa_late']['pca_rel']:.3f} | "
                  f"{' '.join(f'{x:.2f}' for x in r['profile_median']['pca_rel'])} | {r['kappa_late']['full_rel']:.3f} | "
                  f"{r['kappa_late']['lab_rel']:.3f} | {r['finite']['eps0.1']['rel_dev_median']:.3f} | "
                  f"{r['finite']['eps0.3']['flip_rate']:.3f} | {r['onset']['pca_rel']} |")
    (out / "q7_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    return {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
