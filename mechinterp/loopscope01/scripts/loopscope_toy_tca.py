"""LOOPSCOPE-01 Q4 follow-up (amendment 3): transported component attribution on the toy looped chase models.

For each core matrix W (q, k, v slices of qkv, attention proj, fc1, fc2) at loop t, with X its input and A the
gradient of the readout w.r.t. its output (through all later loops), an SVD component (s, u, v) has first-order
removal effect  delta = -s * sum_pos (X v)(A u); fragility F = ||X^T A||_F; matched read/write rotation nulls give
z_read = delta / (s ||X^T A u|| / sqrt(d_in)) and z_write = delta / (s ||A^T X v|| / sqrt(d_out)).

Readout is the log-odds margin z_c - logsumexp_{j != c} z_j (trained arms saturate log p).

Subcommands: run (one checkpoint -> <out>/<arm>.json + .npz), report (all -> report.md + plot).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.loopscope01.scripts import loopscope_toy as L  # noqa: E402

TOY_DIR = Path("experiments/recurrence_monitorability_v0/loopscope_toy")
DEFAULT_OUT = Path("experiments/recurrence_monitorability_v0/loopscope_toy_tca")
ITEM_SEED = {8: 920008, 12: 920012}
TOP = 16


def margin(logits, target):
    zc = logits.gather(1, target[:, None])[:, 0]
    other = logits.scatter(1, target[:, None], float("-inf"))
    return zc - torch.logsumexp(other, 1)


def matrices(model):
    """[(name, module, out_slice)] with name like 'L0.q'."""
    out = []
    for li, blk in enumerate(model.core):
        d = blk.proj.weight.shape[0]
        for j, nm in enumerate("qkv"):
            out.append((f"L{li}.{nm}", blk.qkv, slice(j * d, (j + 1) * d)))
        out.append((f"L{li}.proj", blk.proj, slice(None)))
        out.append((f"L{li}.fc1", blk.fc1, slice(None)))
        out.append((f"L{li}.fc2", blk.fc2, slice(None)))
    return out


def svd_components(model):
    comps = {}
    for name, mod, sl in matrices(model):
        W = mod.weight.detach()[sl].double()  # [out, in]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        comps[name] = (U.float(), S.float(), Vh.T.contiguous().float())  # U [out,r], V [in,r]
    return comps


class Capture:
    """Records each Linear's input and output-gradient per call (call index = loop index)."""

    def __init__(self, model):
        self.x, self.g, self.handles = {}, {}, []
        mods = {id(m): m for _, m, _ in matrices(model)}
        for m in mods.values():
            self.x[m], self.g[m] = [], {}
            self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, mod, inp, out):
        idx = len(self.x[mod])
        self.x[mod].append(inp[0].detach())

        def save(grad, mod=mod, idx=idx):
            self.g[mod][idx] = grad.detach()
        out.register_hook(save)

    def remove(self):
        for h in self.handles:
            h.remove()


def tca(model, comps, depth, items, seed, chunk=64):
    """Return dict of arrays: delta, zr, zw per item [M,T,B,r]; batch-level z; F [M,T,B]; clean margins."""
    n = model.n_nodes
    dev = next(model.parameters()).device
    tokens, hops = L.make_batch(n, items, seed, depth)
    tokens, hops = tokens.to(dev), hops.to(dev)
    target = hops[:, depth - 1]
    cap = Capture(model)
    states = model(tokens, depth)
    logits = model.readout(states[-1])
    y = margin(logits, target)
    y.sum().backward()
    cap.remove()
    mats = matrices(model)
    M, T = len(mats), depth
    r = min(c[1].numel() for c in comps.values())
    delta = np.zeros((M, T, items, r), np.float32)
    zr = np.zeros_like(delta)
    zw = np.zeros_like(delta)
    zr_batch = np.zeros((M, T, r), np.float32)
    frag = np.zeros((M, T, items), np.float32)
    for mi, (name, mod, sl) in enumerate(mats):
        U, S, V = (c[:, :r] if c.dim() == 2 else c[:r] for c in comps[name])
        din, dout = V.shape[0], U.shape[0]
        for t in range(T):
            X = cap.x[mod][t]
            A = cap.g[mod][t][..., sl]
            XtAu_sum = torch.zeros(din, r, device=dev, dtype=X.dtype)
            for b0 in range(0, items, chunk):
                Xb, Ab = X[b0:b0 + chunk], A[b0:b0 + chunk]
                XV, AU = Xb @ V, Ab @ U  # [b,P,r]
                d = -S * (XV * AU).sum(1)  # [b,r]
                XtAu = torch.einsum("bpi,bpr->bir", Xb, AU)
                XtAu_sum += XtAu.sum(0)
                sdr = S * XtAu.norm(dim=1) / math.sqrt(din)
                sdw = S * torch.einsum("bpo,bpr->bor", Ab, XV).norm(dim=1) / math.sqrt(dout)
                G, H = Xb @ Xb.transpose(1, 2), Ab @ Ab.transpose(1, 2)
                frag[mi, t, b0:b0 + chunk] = (G * H).sum((1, 2)).clamp_min(0).sqrt().cpu().numpy()
                delta[mi, t, b0:b0 + chunk] = d.cpu().numpy()
                zr[mi, t, b0:b0 + chunk] = (d / sdr.clamp_min(1e-12)).cpu().numpy()
                zw[mi, t, b0:b0 + chunk] = (d / sdw.clamp_min(1e-12)).cpu().numpy()
            dsum = delta[mi, t].sum(0)
            sd_b = (S * XtAu_sum.norm(dim=0) / math.sqrt(din)).cpu().numpy()
            zr_batch[mi, t] = dsum / np.maximum(sd_b, 1e-12)
    with torch.no_grad():
        acc = float((logits.argmax(1) == target).float().mean())
        logp = float(F.log_softmax(logits, 1).gather(1, target[:, None]).mean())
    return {"delta": delta, "zr": zr, "zw": zw, "zr_batch": zr_batch, "frag": frag,
            "clean_margin": y.detach().cpu().numpy(), "acc": acc, "mean_logp": logp,
            "names": [m[0] for m in mats], "tokens": tokens, "target": target}


@torch.no_grad()
def ablate(model, mats, comps, tokens, target, depth, mi, ci, loops, v_override=None):
    """Margin after removing component ci of matrix mi at the given loops (1-based).

    With v_override, subtracts s (X r) u for the given read direction r instead (matched read rotation)."""
    name, mod, sl = mats[mi]
    U, S, V = comps[name]
    u, s, v = U[:, ci], S[ci], (V[:, ci] if v_override is None else v_override)
    calls = {"i": 0}

    def hook(m, inp, out):
        calls["i"] += 1
        if calls["i"] not in loops:
            return None
        out = out.clone()
        out[..., sl] -= s * (inp[0] @ v)[..., None] * u
        return out
    h = mod.register_forward_hook(hook)
    try:
        st = model(tokens, depth)
        return margin(model.readout(st[-1]), target)
    finally:
        h.remove()


def _avg_rank(x):
    """Ranks with ties given their average rank."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    """Spearman rho with average ranks for ties; pairs with a NaN are dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(_avg_rank(a), _avg_rank(b))[0, 1])


def validate(model, comps, res, depth, n_top=4, n_rand=2):
    mats = matrices(model)
    tokens, target = res["tokens"], res["target"]
    clean = torch.as_tensor(res["clean_margin"]).to(tokens.device)
    md = res["delta"].mean(2)  # [M,T,r]
    rng = np.random.default_rng(0)
    rows = []
    for mi in range(len(mats)):
        top = np.argsort(-np.abs(md[mi]).sum(0))[:n_top].tolist()
        rest = [c for c in range(md.shape[2]) if c not in top]
        rand = rng.choice(rest, n_rand, replace=False).tolist()
        for ci in top + rand:
            for win in [[t] for t in range(1, depth + 1)] + [list(range(1, depth + 1))]:
                meas = float((ablate(model, mats, comps, tokens, target, depth, mi, ci, set(win)) - clean).mean())
                pred = float(md[mi, [t - 1 for t in win], ci].sum())
                rows.append({"mat": mats[mi][0], "comp": int(ci), "kind": "top" if ci in top else "random",
                             "window": "all" if len(win) > 1 else win[0], "pred": pred, "meas": meas})
    p = np.array([r["pred"] for r in rows])
    m = np.array([r["meas"] for r in rows])
    big = np.abs(m) >= 0.05
    rho = spearman(p, m)
    sign = float((np.sign(p[big]) == np.sign(m[big])).mean()) if big.any() else float("nan")
    slope = float(np.polyfit(p, m, 1)[0])
    return {"n": len(rows), "spearman": rho, "sign_agree_big": sign, "n_big": int(big.sum()), "slope": slope,
            "valid": bool(rho >= 0.6 and sign >= 0.8), "rows": rows}


def controls(model, comps, depth, items, seed, n_top=4, n_rot=4):
    """Amendment 4: raw single-loop removal of each matrix's top components vs matched read rotations."""
    mats = matrices(model)
    tokens, hops = L.make_batch(model.n_nodes, items, seed, depth)
    tokens, hops = tokens.to(next(model.parameters()).device), hops.to(next(model.parameters()).device)
    target = hops[:, depth - 1]
    with torch.no_grad():
        clean = margin(model.readout(model(tokens, depth)[-1]), target)
    res = tca(model, comps, depth, items, seed)
    md = res["delta"].mean(2)
    cells = []
    for mi, (name, _, _) in enumerate(mats):
        din = comps[name][2].shape[0]
        top = np.argsort(-np.abs(md[mi]).sum(0))[:n_top].tolist()
        for ci in top:
            rots = []
            for k in range(n_rot):
                r = torch.randn(din, generator=torch.Generator().manual_seed(1000 * mi + 10 * ci + k))
                rots.append((r / r.norm()).to(comps[name][2].device, comps[name][2].dtype))
            for t in range(1, depth + 1):
                raw = float((ablate(model, mats, comps, tokens, target, depth, mi, ci, {t}) - clean).mean())
                ctl = [float((ablate(model, mats, comps, tokens, target, depth, mi, ci, {t}, r) - clean).mean()) for r in rots]
                cells.append({"mat": name, "comp": int(ci), "loop": t, "raw": raw, "controls": ctl,
                              "outside": bool(raw < min(ctl) - 0.05 or raw > max(ctl) + 0.05)})
    by_loop = []
    for t in range(1, depth + 1):
        c = [x for x in cells if x["loop"] == t]
        raw = np.mean([x["raw"] for x in c])
        ctl = np.mean([np.mean(x["controls"]) for x in c])
        by_loop.append({"loop": t, "raw": float(raw), "control": float(ctl), "specificity": float(raw - ctl),
                        "share_outside": float(np.mean([x["outside"] for x in c]))})
    return {"by_loop": by_loop, "cells": cells, "n_top": n_top, "n_rot": n_rot}


def cmd_control(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    dst = out / f"{args.arm}.controls.json"
    if dst.exists() and not args.force:
        print(f"[skip] {args.arm} controls")
        return 0
    st = torch.load(Path(args.toy) / "ckpt" / f"{args.arm}.final.pt", map_location="cpu")
    model = L.build_model(L.model_args_from(st["cfg"]))
    model.load_state_dict(st["model"])
    model.to(args.device).eval()
    comps = svd_components(model)
    res = controls(model, comps, 8, args.items, ITEM_SEED[8])
    res.update({"arm": args.arm, "lam": st["cfg"]["lam"], "seed": st["cfg"]["seed"]})
    dst.write_text(json.dumps(res))
    for b in res["by_loop"]:
        print(f"[{args.arm}] ctl loop {b['loop']}: raw {b['raw']:+.3f} control {b['control']:+.3f} "
              f"spec {b['specificity']:+.3f} outside {b['share_outside']:.2f}", flush=True)
    return 0


def summarise(res):
    """Per-loop profiles."""
    md = res["delta"].mean(2)
    topk = np.argsort(-np.abs(md).sum(1), axis=1)[:, :TOP]  # [M,TOP] per matrix
    M, T = md.shape[:2]
    zr_top = np.stack([res["zr"][mi][:, :, topk[mi]] for mi in range(M)])  # [M,T,B,TOP]
    zw_top = np.stack([res["zw"][mi][:, :, topk[mi]] for mi in range(M)])
    F_t = res["frag"].mean(2).sum(0)  # total fragility per loop
    return {
        "frag_by_loop": F_t.tolist(),
        "frag_by_mat_loop": res["frag"].mean(2).tolist(),
        "frag_first_over_last": float(F_t[0] / F_t[-1]),
        "zr_gt3_top16_by_loop": (np.abs(zr_top) > 3).mean((0, 2, 3)).tolist(),
        "zw_gt3_top16_by_loop": (np.abs(zw_top) > 3).mean((0, 2, 3)).tolist(),
        "zr_gt3_all_by_loop": (np.abs(res["zr"]) > 3).mean((0, 2, 3)).tolist(),
        "zr_gt3_top16_by_mat": (np.abs(zr_top) > 3).mean((1, 2, 3)).tolist(),
        "zr_batch_gt3_top16_by_loop": (np.abs(np.stack([res["zr_batch"][mi][:, topk[mi]] for mi in range(M)])) > 3).mean((0, 2)).tolist(),
        "absdelta_by_loop": np.abs(md).sum((0, 2)).tolist(),
        "names": res["names"], "acc": res["acc"], "mean_logp": res["mean_logp"],
        "clean_margin_mean": float(res["clean_margin"].mean()),
    }


def cmd_run(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = args.arm
    dst = out / f"{arm}.json"
    if dst.exists() and not args.force:
        print(f"[skip] {arm}")
        return 0
    st = torch.load(Path(args.toy) / "ckpt" / f"{arm}.final.pt", map_location="cpu")
    cfg = st["cfg"]
    model = L.build_model(L.model_args_from(cfg))
    model.load_state_dict(st["model"])
    model.to(args.device).eval()  # parameters keep requires_grad so output-gradient hooks can register; never stepped
    comps = svd_components(model)
    metrics = json.loads((Path(args.toy) / "runs" / arm / "metrics.json").read_text())
    rec = {"arm": arm, "lam": cfg["lam"], "seed": cfg["seed"], "checkpoint_sha256": L.sha256_file(Path(args.toy) / "ckpt" / f"{arm}.final.pt"),
           "cap_by_depth": metrics["capability_by_depth"], "items": args.items, "readout": "log-odds margin", "depths": {}}
    arrays = {}
    for depth in (8, 12):
        res = tca(model, comps, depth, args.items, ITEM_SEED[depth])
        s = summarise(res)
        if depth == 8:
            s["validation"] = validate(model, comps, res, depth)
            v = s["validation"]
            print(f"[{arm}] d8 valid={v['valid']} rho={v['spearman']:.3f} sign={v['sign_agree_big']:.3f} (n_big {v['n_big']}) slope={v['slope']:.2f}", flush=True)
        rec["depths"][str(depth)] = s
        arrays[f"d{depth}_meandelta"] = res["delta"].mean(2)
        arrays[f"d{depth}_frag"] = res["frag"].mean(2)
        arrays[f"d{depth}_zr_meanabs"] = np.abs(res["zr"]).mean(2)
        print(f"[{arm}] d{depth} acc={s['acc']:.3f} F1/FT={s['frag_first_over_last']:.3f} "
              f"F={np.round(s['frag_by_loop'], 3).tolist()} zr>3 top16={np.round(s['zr_gt3_top16_by_loop'], 3).tolist()}", flush=True)
    np.savez_compressed(out / f"{arm}.npz", **arrays)
    dst.write_text(json.dumps(rec))
    return 0


def cmd_report(args):
    out = Path(args.out)
    recs = [json.loads(p.read_text()) for p in sorted(out.glob("lam*.json")) if not p.name.endswith(".controls.json")]
    recs.sort(key=lambda r: (r["lam"], r["seed"]))
    ctls = {r["arm"]: json.loads((out / f"{r['arm']}.controls.json").read_text())
            for r in recs if (out / f"{r['arm']}.controls.json").exists()}
    lines = ["# LOOPSCOPE-01 Q4 TCA (amendment 3)", "",
             "Readout: log-odds margin of the answer after the read loop. 256 items per depth. Fragility F = sum over the 12 core",
             "matrices of mean per-item ||X^T A||_F. z shares are per item over each matrix's top-16 components.", "",
             "## Validity (depth 8, 648 measured single-component ablations per arm)", "",
             "| arm | cap@8 | cap@12 | Spearman | sign (|meas|>=0.05) | n big | slope | valid |", "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        v = r["depths"]["8"]["validation"]
        lines.append(f"| {r['arm']} | {r['cap_by_depth']['8']:.2f} | {r['cap_by_depth']['12']:.2f} | {v['spearman']:.3f} | "
                     f"{v['sign_agree_big']:.3f} | {v['n_big']} | {v['slope']:.2f} | {v['valid']} |")
    for depth in ("8", "12"):
        lines += ["", f"## Depth {depth}: fragility by loop (normalised to the last loop) and z_read share", "",
                  "| arm | F1/F_last | F by loop / F_last | zr>3 top-16 by loop | max/min zr share |", "|---|---|---|---|---|"]
        for r in recs:
            s = r["depths"][depth]
            Fl = np.array(s["frag_by_loop"])
            z = np.array(s["zr_gt3_top16_by_loop"])
            ratio = float(z.max() / max(z.min(), 1e-9))
            lines.append(f"| {r['arm']} | {s['frag_first_over_last']:.4f} | {' '.join(f'{x:.3f}' for x in Fl / Fl[-1])} | "
                         f"{' '.join(f'{x:.2f}' for x in z)} | {ratio:.2f} |")
    lines += ["", "## Predictions (valid arms with lam > 0, depth 8)", ""]
    pos = [r for r in recs if r["lam"] > 0 and r["depths"]["8"]["validation"]["valid"]]
    if pos:
        p1 = [r["depths"]["8"]["frag_first_over_last"] >= 1 / 3 for r in pos]
        p2 = []
        for r in pos:
            z = np.array(r["depths"]["8"]["zr_gt3_top16_by_loop"])
            p2.append(float(z.max() / max(z.min(), 1e-9)) <= 2)
        lines.append(f"- P1 (F_1/F_8 >= 1/3): {sum(p1)}/{len(p1)} arms.")
        lines.append(f"- P2 (max/min z_read share <= 2): {sum(p2)}/{len(p2)} arms.")
    else:
        lines.append("- no valid lam > 0 arm")
    lines += ["", "## Measured controls (amendment 4): single-loop removal of each matrix's top-4 components, depth 8", "",
              "Mean margin change over 12 matrices x 4 components; control = mean of 4 matched read rotations.", "",
              "| arm | raw by loop 1-8 | control by loop 1-8 | share outside controls by loop |", "|---|---|---|---|"]
    for r in recs:
        c = ctls.get(r["arm"])
        if not c:
            continue
        bl = c["by_loop"]
        raw_s = " ".join(f"{b['raw']:+.2f}" for b in bl)
        ctl_s = " ".join(f"{b['control']:+.2f}" for b in bl)
        out_s = " ".join(f"{b['share_outside']:.2f}" for b in bl)
        lines.append(f"| {r['arm']} | {raw_s} | {ctl_s} | {out_s} |")
    lines += ["", "## Per-arm summary (post hoc; not registered)", "",
              "| arm | held-out acc (Q4) | probe 1-16 (Q4) | F1/F8 | F1 | TCA valid | mean abs raw | mean abs control |",
              "|---|---|---|---|---|---|---|---|"]
    rows = []
    for r in recs:
        m = json.loads((Path(args.toy) / "runs" / r["arm"] / "metrics.json").read_text())["summary"]
        s8 = r["depths"]["8"]
        c = ctls.get(r["arm"])
        raw = float(np.mean([abs(b["raw"]) for b in c["by_loop"]])) if c else float("nan")
        ctl = float(np.mean([abs(b["control"]) for b in c["by_loop"]])) if c else float("nan")
        rows.append((m["cap_heldout"], m["probe_all"], s8["frag_first_over_last"], s8["frag_by_loop"][0]))
        lines.append(f"| {r['arm']} | {m['cap_heldout']:.3f} | {m['probe_all']:.3f} | {s8['frag_first_over_last']:.4f} | "
                     f"{s8['frag_by_loop'][0]:.2f} | {s8['validation']['valid']} | {raw:.3f} | {ctl:.3f} |")
    if len(rows) >= 4:
        a_ = np.array(rows)
        lines += ["", f"Spearman over {len(rows)} arms: F1/F8 vs held-out acc {spearman(a_[:, 2], a_[:, 0]):+.3f}, "
                  f"vs probe {spearman(a_[:, 2], a_[:, 1]):+.3f}; F1 vs held-out acc {spearman(a_[:, 3], a_[:, 0]):+.3f}, "
                  f"vs probe {spearman(a_[:, 3], a_[:, 1]):+.3f}. Ties (many arms at 1.000) make these coarse."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for r in recs:
            s = r["depths"]["8"]
            Fl = np.array(s["frag_by_loop"])
            c = {0.0: "C3", 0.01: "C0", 0.1: "C1", 1.0: "C2"}[r["lam"]]
            ax[0].plot(range(1, len(Fl) + 1), Fl / Fl[-1], color=c, alpha=0.8, label=r["arm"])
            ax[1].plot(range(1, len(Fl) + 1), s["zr_gt3_top16_by_loop"], color=c, alpha=0.8, label=r["arm"])
        ax[0].set(yscale="log", xlabel="loop", ylabel="fragility / last loop", title="depth 8: fragility profile")
        ax[1].set(xlabel="loop", ylabel="share |z_read|>3 (top-16)", title="depth 8: read specificity")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "tca_profiles.png", dpi=120)
    except ImportError:
        print("matplotlib absent; no plot")
    return 0


Q5_THRESHOLD = 0.05  # amendment 5: predict extrapolation iff F1/F8 < 0.05
Q5_EXTRAP = 0.90     # an arm extrapolates iff held-out (depths 9-16) accuracy >= 0.90


def cmd_q5(args):
    """Amendment 5: prospective test of F1/F8 (depth-8 items only) as a predictor of length extrapolation."""
    toy, out = Path(args.toy), Path(args.out)
    rows, invalid = [], []
    for p in sorted(out.glob("lam*.json")):
        if p.name.endswith(".controls.json"):
            continue
        r = json.loads(p.read_text())
        m = json.loads((toy / "runs" / r["arm"] / "metrics.json").read_text())
        rc = json.loads((toy / "runs" / r["arm"] / "receipt.json").read_text())
        acc8 = float(m["capability_by_depth"]["8"])
        row = {"arm": r["arm"], "lam": r["lam"], "seed": r["seed"], "acc_d8": acc8,
               "f1_over_f8": r["depths"]["8"]["frag_first_over_last"], "tca_valid": r["depths"]["8"]["validation"]["valid"],
               "heldout": m["summary"]["cap_heldout"], "probe_all": m["summary"]["probe_all"],
               "final_loss": rc["final_loss_last50_mean"]}
        row["pred_extrap"] = row["f1_over_f8"] < Q5_THRESHOLD
        row["extrap"] = row["heldout"] >= Q5_EXTRAP
        (invalid if acc8 < 0.90 else rows).append(row)
    n = len(rows)
    correct = sum(r["pred_extrap"] == r["extrap"] for r in rows)
    pos = sum(r["extrap"] for r in rows)
    informative = 2 <= pos <= n - 2
    if not informative:
        decision = "uninformative (fewer than 2 arms in a class)"
    elif correct >= math.ceil(0.9 * n):
        decision = "supported"
    elif correct >= math.ceil(0.75 * n):
        decision = "weak"
    else:
        decision = "not supported"
    def rho(key):
        return spearman(np.array([r[key] for r in rows]), np.array([r["heldout"] for r in rows])) if n >= 4 else float("nan")
    rep = {"decision": decision, "n_valid": n, "correct": correct, "n_extrap": pos, "threshold": Q5_THRESHOLD,
           "spearman_heldout": {"f1_over_f8": rho("f1_over_f8"), "final_loss": rho("final_loss"), "probe_all": rho("probe_all")},
           "rows": rows, "training_failed": invalid}
    (out / "q5_report.json").write_text(json.dumps(rep, indent=2))
    md = ["# LOOPSCOPE-01 Q5: does early-loop contraction predict length extrapolation? (amendment 5)", "",
          f"Decision: **{decision}**. {correct}/{n} valid arms classified correctly; {pos} of {n} extrapolate.", "",
          f"Rule: predict extrapolation iff F1/F8 < {Q5_THRESHOLD} (depth-8 items only); extrapolation = held-out accuracy >= {Q5_EXTRAP}.",
          "Supported if >= 90% correct with >= 2 arms in each class; weak if >= 75%; else not supported.", "",
          "| arm | acc d8 | F1/F8 | predicted | held-out acc | extrapolates | probe 1-16 | final loss | TCA valid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['arm']} | {r['acc_d8']:.3f} | {r['f1_over_f8']:.4f} | {r['pred_extrap']} | {r['heldout']:.3f} | "
                  f"{r['extrap']} | {r['probe_all']:.3f} | {r['final_loss']:.2e} | {r['tca_valid']} |")
    s = rep["spearman_heldout"]
    md += ["", f"Spearman with held-out accuracy: F1/F8 {s['f1_over_f8']:+.3f}; final training loss {s['final_loss']:+.3f}; "
           f"probe (needs hop labels) {s['probe_all']:+.3f}.", f"training_failed: {invalid or 'none'}"]
    (out / "q5_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--items", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--device", default="cpu")
    r.add_argument("--toy", default=str(TOY_DIR))
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--force", action="store_true")
    c = sub.add_parser("control")
    c.add_argument("--arm", required=True)
    c.add_argument("--items", type=int, default=256)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--device", default="cpu")
    c.add_argument("--toy", default=str(TOY_DIR))
    c.add_argument("--out", default=str(DEFAULT_OUT))
    c.add_argument("--force", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--toy", default=str(TOY_DIR))
    q = sub.add_parser("q5")
    q.add_argument("--out", default=str(DEFAULT_OUT))
    q.add_argument("--toy", default=str(TOY_DIR))
    a = ap.parse_args(argv)
    return {"run": cmd_run, "control": cmd_control, "report": cmd_report, "q5": cmd_q5}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
