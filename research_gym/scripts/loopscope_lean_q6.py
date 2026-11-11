"""LOOPSCOPE-01 Q6 (amendment 6): early-loop contraction F1/F8 on Lean-CGRT Gate-0 loopers vs test-time-loop gain.

    predict: F1/F8 per checkpoint (CPU, read-only, label-free)  -> <out>/predictor.json
    score:   read eval outcomes, apply the registered rule       -> <out>/q6_report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research_gym.scripts import loopscope_lean as LL  # noqa: E402

OUT = LL.REPO / "experiments" / "recurrence_monitorability_v0" / "loopscope_lean_q6"
SLOOP = [17, 29, 43, 53, 61, 71, 83, 97, 101, 113]
SRD = [43, 53, 61, 71, 83, 97, 101, 113]
CKPTS = [f"g0_Sloop_s{s}" for s in SLOOP] + [f"g0_Srd_s{s}" for s in SRD]
T = 8


def margin_own(logits, legal):
    """Log-odds margin of the top logit vs logsumexp of the other legal ones; NaN where < 2 legal."""
    top = logits.argmax(-1)
    zc = logits.gather(1, top[:, None])[:, 0]
    other = logits.masked_fill(~legal, float("-inf")).scatter(1, top[:, None], float("-inf"))
    m = zc - torch.logsumexp(other, 1)
    return torch.where(legal.sum(1) >= 2, m, torch.full_like(m, float("nan")))


def fast_fragility(model, examples, batch=8, device="cpu"):
    """Per fast step t = 1..T: sum over fast-clock matrices of mean per-state ||X^T A||_F (margin readout)."""
    from lcgrt.encode import collate
    for p in model.parameters():
        p.requires_grad_(False)
    if model.cfg.slow:
        model.slot_init.requires_grad_(True)
    tca = LL.TCA(model, K=4)
    n_kept = 0
    for i0 in range(0, len(examples), batch):
        exs = examples[i0:i0 + batch]
        b = collate(exs, device)
        b["x"] = b["x"].to(next(model.parameters()).dtype).requires_grad_(True)
        tca.reset()
        logits = LL.forward_states(model, b, T)
        y = margin_own(logits[-1], b["legal"])
        keep = ~torch.isnan(y)
        n_kept += int(keep.sum())
        y[keep].sum().backward()
        b["x"].grad = None
    tca.remove()
    F = np.zeros(T)
    for (name, c), rs in tca.rows.items():
        if tca.mods[name][1] != "fast":
            continue
        F[c] += float(torch.cat([r["F"] for r in rs]).mean())
    return F, n_kept


def cmd_predict(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "predictor.json"
    pred = json.loads(dst.read_text()) if dst.exists() else {}
    exs = LL.load_examples("lcgrt-mirror", "val", 64, 128, 0)
    for ck in CKPTS:
        if ck in pred:
            continue
        model = LL.load_model(ck, "cpu")
        F, n = fast_fragility(model, exs)
        pred[ck] = {"F_by_step": F.tolist(), "f1_over_f8": float(F[0] / F[-1]), "n_states": n,
                    "family": "Sloop" if "Sloop" in ck else "Srd"}
        dst.write_text(json.dumps(pred, indent=1))
        print(f"[q6] {ck} F1/F8={F[0] / F[-1]:.4f} F={np.round(F, 3).tolist()} n={n}", flush=True)
    return 0


def rate(summary, key="b4"):
    a = summary["all"]
    v = a[key]
    return float(v) / a["n"] if v > 1 else float(v)


def outcome(ck, split, policy, tdir):
    e = json.loads((LL.RUN_C1 / f"{tdir}_{ck}" / "eval.json").read_text())
    return rate(e["runs"][f"{split}/{policy}"]["summary"])


def perm_p(x, y, sign, n=10000, seed=0):
    from research_gym.scripts.loopscope_toy_tca import spearman
    rho = spearman(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(n)])
    p = float((np.sum(null <= rho) + 1) / (n + 1)) if sign < 0 else float((np.sum(null >= rho) + 1) / (n + 1))
    return rho, p


def cmd_score(args):
    from research_gym.scripts.loopscope_toy_tca import spearman
    out = Path(args.out)
    pred = json.loads((out / "predictor.json").read_text())
    assert all(ck in pred for ck in CKPTS), "predictor incomplete"
    rows = []
    for ck in CKPTS:
        r = {"ckpt": ck, **{k: pred[ck][k] for k in ("family", "f1_over_f8", "n_states")}}
        for split in ("test_mirror_deep", "test_mirror_id"):
            for pol in ("best", "greedy"):
                s8, s32 = outcome(ck, split, pol, "eval"), outcome(ck, split, pol, "evalT32")
                r[f"{split}/{pol}"] = {"T8": s8, "T32": s32, "gain": s32 - s8}
        rows.append(r)
    x = np.array([r["f1_over_f8"] for r in rows])
    gain = np.array([r["test_mirror_deep/best"]["gain"] for r in rows])
    rho_neg, p_neg = perm_p(x, gain, -1)
    _, p_pos = perm_p(x, gain, +1)
    if np.ptp(gain) == 0:  # constant outcome (e.g. every checkpoint at ceiling): rho undefined, no test
        rho_neg, p_neg, p_pos = float("nan"), float("nan"), float("nan")
        decision = "uninformative (primary outcome constant across checkpoints)"
    elif rho_neg <= -0.4 and p_neg < 0.05:
        decision = "supported"
    elif rho_neg >= 0.4 and p_pos < 0.05:
        decision = "reversed"
    else:
        decision = "null"
    fam = {}
    for f in ("Sloop", "Srd"):
        m = np.array([r["family"] == f for r in rows])
        fam[f] = spearman(x[m], gain[m])
    if decision in ("supported", "reversed") and all(np.sign(v) != np.sign(rho_neg) or abs(v) < 0.2 for v in fam.values()):
        decision += " (family confound: holds in neither family)"
    sec = {}
    for key, fn in [("deep_best_T8", lambda r: r["test_mirror_deep/best"]["T8"]),
                    ("id_best_gain", lambda r: r["test_mirror_id/best"]["gain"]),
                    ("deep_greedy_gain", lambda r: r["test_mirror_deep/greedy"]["gain"]),
                    ("id_greedy_gain", lambda r: r["test_mirror_id/greedy"]["gain"])]:
        sec[key] = spearman(x, np.array([fn(r) for r in rows]))
    # Exploratory, added after the primary outcome was found constant (not registered): tighter search budgets.
    expl = {}
    for bud in ("b1", "b2"):
        for split in ("test_mirror_deep", "test_mirror_id"):
            g = []
            for r in rows:
                e8 = json.loads((LL.RUN_C1 / f"eval_{r['ckpt']}" / "eval.json").read_text())["runs"][f"{split}/best"]["summary"]
                e32 = json.loads((LL.RUN_C1 / f"evalT32_{r['ckpt']}" / "eval.json").read_text())["runs"][f"{split}/best"]["summary"]
                g.append(rate(e32, bud) - rate(e8, bud))
            g = np.array(g)
            if np.ptp(g) == 0:
                expl[f"{split}/best/{bud}"] = {"rho": None, "p_two_sided": None, "gain_range": [float(g.min()), float(g.max())]}
                continue
            rho, p1 = perm_p(x, g, -1 if spearman(x, g) < 0 else +1)
            expl[f"{split}/best/{bud}"] = {"rho": rho, "p_two_sided": min(1.0, 2 * p1), "gain_range": [float(g.min()), float(g.max())]}
    rep = {"decision": decision, "rho": rho_neg, "p_one_sided_neg": p_neg, "p_one_sided_pos": p_pos,
           "within_family_rho": fam, "secondary_rho": sec, "exploratory_tighter_budgets": expl, "rows": rows}
    (out / "q6_report.json").write_text(json.dumps(rep, indent=1))
    md = ["# LOOPSCOPE-01 Q6: early-loop contraction vs test-time-loop gain on Lean-CGRT Gate-0 loopers (amendment 6)", "",
          f"Decision: **{decision}**. Spearman(F1/F8, deep/best gain T32-T8) = {rho_neg:+.3f}; one-sided p (negative) "
          f"{p_neg:.4f}, (positive) {p_pos:.4f}; n = {len(rows)}.",
          f"Within family: S-loop {fam['Sloop']:+.3f} (n=10), S-rd {fam['Srd']:+.3f} (n=8).",
          "Secondary Spearman with F1/F8 (nan = outcome constant): " + ", ".join(f"{k} {v:+.3f}" for k, v in sec.items()) + ".", "",
          "Exploratory (added after seeing the ceiling; not registered), best policy, gain T32-T8 at tighter budgets: "
          + "; ".join(f"{k}: rho {'n/a' if v['rho'] is None else format(v['rho'], '+.3f')}, two-sided p "
                      f"{'n/a' if v['p_two_sided'] is None else format(v['p_two_sided'], '.3f')}, gain range "
                      f"[{v['gain_range'][0]:+.3f}, {v['gain_range'][1]:+.3f}]" for k, v in expl.items()) + ".", "",
          "| ckpt | F1/F8 | deep best T8 | deep best T32 | gain | id best gain | deep greedy gain | states |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d, i, g = r["test_mirror_deep/best"], r["test_mirror_id/best"], r["test_mirror_deep/greedy"]
        md.append(f"| {r['ckpt']} | {r['f1_over_f8']:.4f} | {d['T8']:.3f} | {d['T32']:.3f} | {d['gain']:+.3f} | "
                  f"{i['gain']:+.3f} | {g['gain']:+.3f} | {r['n_states']} |")
    (out / "q6_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    return {"predict": cmd_predict, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
