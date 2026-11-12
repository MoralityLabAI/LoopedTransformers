"""LOOPSCOPE-01 Q10 (amendment 10): prospective depth-32 extrapolation, edge-clock causality, and label-free semantic echo.

Subcommands:
  dev24:    evaluate the 24 frozen Q4+Q5 arms on depth-32 items, edge/full clock probes, D_echo_own, and full-state clock ablation (P10c).
  eval-arm: evaluate one trained Q10 arm (--arm lam0_s10 --toy ... --tca ... --out ...).
  score:    score all 12 prospective Q10 arms on P10a, P10b, P10c -> q10_report.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research_gym.scripts import loopscope_toy as L  # noqa: E402
from research_gym.scripts import loopscope_toy_q7 as Q  # noqa: E402
from research_gym.scripts import loopscope_toy_q9 as Q9  # noqa: E402
from research_gym.scripts.loopscope_toy_tca import spearman  # noqa: E402

TOY_Q10 = Q.EXP / "loopscope_toy_q10"
TCA_Q10 = Q.EXP / "loopscope_toy_q10_tca"
OUT_DEV24 = Q.EXP / "loopscope_toy_q10_dev24"
OUT_Q10 = Q.EXP / "loopscope_toy_q10_eval"

TRAIN_SEED, TEST_SEED = 931008, 930008
DEEP32_SEED = 930032
ECHO_SEED = 940117
N_TRAIN, N_TEST, N_DEEP32 = 512, 256, 256
ECHO_ITEMS, ECHO_LOOPS, ECHO_HORIZON = 64, 32, 4
LOOPS = list(range(2, 9))
RANK, N_RAND, GAIN = 6, 4, 0.10
EXTRAP_CUT = 0.90
C_CUT = 0.63
ECHO_CUT = 0.05


def full_trajectory(model, tokens, T: int):
    """Returns list of length T of full token states [B, n_pos, d] after loops 1..T."""
    with torch.no_grad():
        return Q.trajectory(model, Q.embed(model, tokens), T)


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def probe_7way(X_tr_by_loop: list[torch.Tensor], X_te_by_loop: list[torch.Tensor]) -> float:
    """7-way ridge probe accuracy for loop index t in LOOPS (2..8)."""
    Xtr = torch.cat([X_tr_by_loop[t - 1] for t in LOOPS]).cpu()
    Ytr = torch.cat([torch.full((len(X_tr_by_loop[0]),), i, dtype=torch.long) for i, _ in enumerate(LOOPS)])
    Xte = torch.cat([X_te_by_loop[t - 1] for t in LOOPS]).cpu()
    Yte = torch.cat([torch.full((len(X_te_by_loop[0]),), i, dtype=torch.long) for i, _ in enumerate(LOOPS)])
    return float((L.ridge_probe(Xtr, Ytr, Xte, len(LOOPS)) == Yte).float().mean())


def clock_probes_all(hs_tr: list[torch.Tensor], hs_te: list[torch.Tensor]) -> dict[str, float]:
    """Compute C_cursor, C_edge (mean edge unit), and C_edge_full (concatenated per-edge units)."""
    cur_tr = [unit(h[:, -1]) for h in hs_tr]
    cur_te = [unit(h[:, -1]) for h in hs_te]
    edge_mean_tr = [unit(h[:, :-1].mean(dim=1)) for h in hs_tr]
    edge_mean_te = [unit(h[:, :-1].mean(dim=1)) for h in hs_te]
    edge_cat_tr = [unit(h[:, :-1]).flatten(1) for h in hs_tr]
    edge_cat_te = [unit(h[:, :-1]).flatten(1) for h in hs_te]
    return {
        "C_cursor": probe_7way(cur_tr, cur_te),
        "C_edge": probe_7way(edge_mean_tr, edge_mean_te),
        "C_edge_full": probe_7way(edge_cat_tr, edge_cat_te),
    }


def per_pos_clock_subspaces(hs_tr: list[torch.Tensor], k: int = RANK):
    """Fit per-position (p = 0..n_pos-1) pooled mean mu_p [d] and top-k loop-mean directions P_p [d, k]."""
    n_pos = hs_tr[0].shape[1]
    mus, Ps = [], []
    for pos in range(n_pos):
        u_loops = [unit(hs_tr[t - 1][:, pos]) for t in LOOPS]
        means = torch.stack([u.mean(0) for u in u_loops])
        mu = torch.cat(u_loops, dim=0).mean(0)
        _, _, Vh = torch.linalg.svd(means - means.mean(0), full_matrices=False)
        mus.append(mu)
        Ps.append(Vh[:k].T.contiguous())
    return torch.stack(mus, dim=0), torch.stack(Ps, dim=0)  # [n_pos, d], [n_pos, d, k]


def random_per_pos_subspaces(n_pos: int, d: int, k: int, seed: int, device, dtype):
    g = torch.Generator().manual_seed(seed)
    Rs = []
    for _ in range(n_pos):
        q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float32))
        Rs.append(q.to(device=device, dtype=dtype))
    return torch.stack(Rs, dim=0)


def run_full_ablated(model, tokens: torch.Tensor, T: int, Ps: torch.Tensor | None, mus: torch.Tensor | None):
    """Run T loops; if Ps [n_pos, d, k] is given, mean-ablate every token position's normalized state after each loop."""
    with torch.no_grad():
        e = Q.embed(model, tokens)
        h = torch.zeros_like(e)
        preds = []
        for _ in range(T):
            h = Q.step(model, h, e)
            if Ps is not None and mus is not None:
                nrm = h.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                u = h / nrm  # [B, n_pos, d]
                diff = u - mus.unsqueeze(0)  # [B, n_pos, d]
                proj = torch.einsum("bpd,pdk,pek->bpe", diff, Ps, Ps)
                h = nrm * (u - proj)
            preds.append(model.readout(h[:, -1]).argmax(-1).cpu())
        return torch.stack(preds, dim=0)  # [T, B]


def acc_windows_32(preds: torch.Tensor, hops: torch.Tensor) -> dict:
    """preds [32, B], hops [B, 32]. Returns accuracy windows."""
    a = (preds == hops.T).float().mean(1).numpy()
    return {
        "acc_1_8": float(a[:8].mean()),
        "acc8": float(a[7]),
        "acc_9_16": float(a[8:16].mean()),
        "acc_17_32": float(a[16:32].mean()),
        "acc_9_32": float(a[8:32].mean()),
        "by_loop": [float(x) for x in a],
    }


def own_echo_disagreement(model, n_items: int = ECHO_ITEMS, max_loops: int = ECHO_LOOPS,
                          horizon: int = ECHO_HORIZON, seed: int = ECHO_SEED, device: str = "cpu") -> dict:
    """100% label-free own-head continuation disagreement over max_loops.

    Gates pairs 1 <= t1 < t2 <= max_loops - horizon solely on pred(t1) == pred(t2).
    Items with zero own-gated pairs receive disagreement 1.0.
    """
    tokens, _ = L.make_batch(model.n_nodes, n_items, seed, max_loops)
    tokens = tokens.to(device)
    with torch.no_grad():
        hs = full_trajectory(model, tokens, max_loops)
        preds = torch.stack([model.readout(h[:, -1]).argmax(-1).cpu() for h in hs], dim=0).numpy()  # [T, B]
    max_start = max_loops - horizon  # 1-indexed visit t <= max_start => 0-indexed idx <= max_start - 1
    item_scores = []
    zero_pair_items = 0
    total_pairs = 0
    disagree_steps = 0
    for b in range(n_items):
        p = preds[:, b]
        pair_vals = []
        for i in range(max_start):
            for j in range(i + 1, max_start):
                if p[i] == p[j]:
                    mism = float(np.mean(p[i + 1:i + 1 + horizon] != p[j + 1:j + 1 + horizon]))
                    pair_vals.append(mism)
                    total_pairs += 1
                    disagree_steps += int(np.sum(p[i + 1:i + 1 + horizon] != p[j + 1:j + 1 + horizon]))
        if pair_vals:
            item_scores.append(float(np.mean(pair_vals)))
        else:
            zero_pair_items += 1
            item_scores.append(1.0)
    return {
        "D_echo_own": float(np.mean(item_scores)),
        "items": n_items,
        "zero_pair_items": zero_pair_items,
        "own_gated_pairs": total_pairs,
        "pooled_step_disagreement": float(disagree_steps / (total_pairs * horizon)) if total_pairs else 1.0,
    }


def evaluate_checkpoint(ck: Path, set_name: str, device: str, tca_json: Path | None = None) -> dict:
    st = torch.load(ck, map_location="cpu")
    cfg = st["cfg"]
    model = L.build_model(L.model_args_from(cfg))
    model.load_state_dict(st["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n = model.n_nodes
    tok_tr, _ = L.make_batch(n, N_TRAIN, TRAIN_SEED, 8)
    tok_te, _ = L.make_batch(n, N_TEST, TEST_SEED, 8)
    tok_d32, hops_d32 = L.make_batch(n, N_DEEP32, DEEP32_SEED, 32)

    M = float(np.median(Q9.state_memory(model, tok_te.to(device))))
    hs_tr = full_trajectory(model, tok_tr.to(device), 8)
    hs_te = full_trajectory(model, tok_te.to(device), 8)
    clocks = clock_probes_all(hs_tr, hs_te)
    mus, Ps = per_pos_clock_subspaces(hs_tr, k=RANK)

    clean = acc_windows_32(run_full_ablated(model, tok_d32.to(device), 32, None, None), hops_d32)
    full_clock = acc_windows_32(run_full_ablated(model, tok_d32.to(device), 32, Ps, mus), hops_d32)

    # Also cursor-only clock ablation to 32 loops for direct paired comparison
    Ps_cur_only = torch.zeros_like(Ps)
    Ps_cur_only[-1] = Ps[-1]
    cur_clock = acc_windows_32(run_full_ablated(model, tok_d32.to(device), 32, Ps_cur_only, mus), hops_d32)

    rand_runs = []
    for r_idx in range(N_RAND):
        Rs = random_per_pos_subspaces(Ps.shape[0], Ps.shape[1], RANK, 1000 + r_idx, Ps.device, Ps.dtype)
        rand_runs.append(acc_windows_32(run_full_ablated(model, tok_d32.to(device), 32, Rs, mus), hops_d32))
    rnd = {
        k: float(np.mean([r[k] for r in rand_runs]))
        for k in ("acc_1_8", "acc8", "acc_9_16", "acc_17_32", "acc_9_32")
    }

    echo_own = own_echo_disagreement(model, device=device)
    f1_f8 = None
    if tca_json is not None and tca_json.is_file():
        tca_data = json.loads(tca_json.read_text())
        f1_f8 = float(tca_data["depths"]["8"]["frag_first_over_last"])

    training_valid = bool(clean["acc8"] >= 0.90)
    extrap32 = bool(clean["acc_9_32"] >= EXTRAP_CUT)
    extrap16 = bool(clean["acc_9_16"] >= EXTRAP_CUT)

    return {
        "arm": f"lam{cfg['lam']:g}_s{cfg['seed']}",
        "set": set_name,
        "lam": cfg["lam"],
        "seed": cfg["seed"],
        "checkpoint": ck.name,
        "checkpoint_sha256": L.sha256_file(ck),
        "training_valid": training_valid,
        "extrap32": extrap32,
        "extrap16": extrap16,
        "f1_over_f8": f1_f8,
        "M": M,
        **clocks,
        **echo_own,
        "clean": clean,
        "ablation_full_clock": full_clock,
        "ablation_cursor_clock": cur_clock,
        "ablation_full_random": rnd,
        "effect_full_clock_9_32": float(full_clock["acc_9_32"] - rnd["acc_9_32"]),
        "effect_full_clock_9_16": float(full_clock["acc_9_16"] - rnd["acc_9_16"]),
        "effect_full_clock_acc8": float(full_clock["acc8"] - rnd["acc8"]),
        "effect_cursor_clock_9_32": float(cur_clock["acc_9_32"] - clean["acc_9_32"]),
    }


def score_rule(values: np.ndarray, fails: np.ndarray, cut: float) -> dict:
    preds_extrap = values < cut
    actual_extrap = ~fails
    correct = int((preds_extrap == actual_extrap).sum())
    n = int(len(values))
    acc = float(correct / n) if n else 0.0
    auc_val = float(Q.auc(values, fails)) if (fails.any() and (~fails).any()) else None
    if acc >= 0.90 and (auc_val is None or auc_val >= 0.90):
        verdict = "supported"
    elif acc >= 0.75 or (auc_val is not None and auc_val >= 0.80):
        verdict = "weak"
    else:
        verdict = "not supported"
    return {
        "threshold": cut,
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "auc_failure": auc_val,
        "false_positives_predicted_extrap_but_failed": int((preds_extrap & fails).sum()),
        "false_negatives_predicted_fail_but_extrap": int(((~preds_extrap) & actual_extrap).sum()),
        "verdict": verdict,
    }


def score_cohort(rows: list[dict], is_prospective: bool) -> dict:
    valid = [r for r in rows if r["training_valid"]]
    fails = np.array([not r["extrap32"] for r in valid], dtype=bool)
    c_cur = np.array([r["C_cursor"] for r in valid], dtype=float)
    c_edg = np.array([r["C_edge"] for r in valid], dtype=float)
    c_efl = np.array([r["C_edge_full"] for r in valid], dtype=float)
    d_own = np.array([r["D_echo_own"] for r in valid], dtype=float)
    acc32 = np.array([r["clean"]["acc_9_32"] for r in valid], dtype=float)

    p10a = score_rule(c_cur, fails, C_CUT)
    p10b = score_rule(d_own, fails, ECHO_CUT)

    f18_list = [r["f1_over_f8"] for r in valid if r.get("f1_over_f8") is not None]
    f18_rule = None
    if len(f18_list) == len(valid) and len(valid) > 0:
        f18 = np.array(f18_list, dtype=float)
        f18_rule = score_rule(f18, fails, 0.05)

    eff = np.array([r["effect_full_clock_9_32"] for r in valid], dtype=float)
    n_fail = int(fails.sum())
    n_gain = int((eff[fails] > GAIN).sum()) if n_fail else 0
    if n_fail == 0:
        p10c_verdict = "uninformative_no_failures"
    elif n_gain >= int(np.ceil(0.5 * n_fail)):
        p10c_verdict = "supported"
    elif n_gain >= 2:
        p10c_verdict = "partial"
    else:
        p10c_verdict = "not supported"

    return {
        "schema_version": "loopscope_q10.v1",
        "prospective": is_prospective,
        "total_arms": len(rows),
        "valid_arms": len(valid),
        "extrap32_count": int((~fails).sum()),
        "fail32_count": n_fail,
        "P10a_C_cursor": p10a,
        "P10b_D_echo_own": p10b,
        "F1_F8_comparison": f18_rule,
        "P10c_full_clock_ablation": {
            "failing_arms": n_fail,
            "failures_gaining_gt_0.10": n_gain,
            "verdict": p10c_verdict,
            "effect_9_32_on_failures": {r["arm"]: r["effect_full_clock_9_32"] for r in valid if not r["extrap32"]},
            "effect_acc8_on_failures": {r["arm"]: r["effect_full_clock_acc8"] for r in valid if not r["extrap32"]},
            "effect_9_32_extrap_median": float(np.median(eff[~fails])) if (~fails).any() else None,
        },
        "exploratory_aucs": {
            "C_cursor": float(Q.auc(c_cur, fails)) if (fails.any() and (~fails).any()) else None,
            "C_edge": float(Q.auc(c_edg, fails)) if (fails.any() and (~fails).any()) else None,
            "C_edge_full": float(Q.auc(c_efl, fails)) if (fails.any() and (~fails).any()) else None,
            "D_echo_own": float(Q.auc(d_own, fails)) if (fails.any() and (~fails).any()) else None,
        },
        "spearman_with_inaccuracy_9_32": {
            "C_cursor": float(spearman(c_cur, 1.0 - acc32)) if len(valid) > 1 else None,
            "C_edge": float(spearman(c_edg, 1.0 - acc32)) if len(valid) > 1 else None,
            "C_edge_full": float(spearman(c_efl, 1.0 - acc32)) if len(valid) > 1 else None,
            "D_echo_own": float(spearman(d_own, 1.0 - acc32)) if len(valid) > 1 else None,
        },
    }


def render_markdown(title: str, summary: dict, rows: list[dict]) -> str:
    p10a = summary["P10a_C_cursor"]
    p10b = summary["P10b_D_echo_own"]
    p10c = summary["P10c_full_clock_ablation"]
    lines = [
        f"# {title}",
        "",
        f"- **Evidence stage:** {'prospective fresh seeds (frozen before training)' if summary['prospective'] else 'retrospective Q4+Q5 development cohort'}",
        f"- **Valid arms:** {summary['valid_arms']}/{summary['total_arms']} "
        f"({summary['extrap32_count']} `extrap32`, {summary['fail32_count']} `fail32`).",
        f"- **P10a (`C_cursor < 0.63`):** **{p10a['verdict']}** — {p10a['correct']}/{p10a['n']} ({p10a['accuracy']:.3f}), "
        f"AUC = {p10a['auc_failure'] if p10a['auc_failure'] is not None else 'n/a'}.",
        f"- **P10b (`D_echo_own < 0.05`):** **{p10b['verdict']}** — {p10b['correct']}/{p10b['n']} ({p10b['accuracy']:.3f}), "
        f"AUC = {p10b['auc_failure'] if p10b['auc_failure'] is not None else 'n/a'}.",
    ]
    if summary.get("F1_F8_comparison"):
        f = summary["F1_F8_comparison"]
        lines.append(
            f"- **Comparison (`F1/F8 < 0.05`):** **{f['verdict']}** — {f['correct']}/{f['n']} ({f['accuracy']:.3f}), "
            f"AUC = {f['auc_failure'] if f['auc_failure'] is not None else 'n/a'}."
        )
    lines.extend([
        f"- **P10c (full-state edge+cursor clock ablation > +0.10 over random):** **{p10c['verdict']}** — "
        f"{p10c['failures_gaining_gt_0.10']}/{p10c['failing_arms']} failing arms.",
        "",
        "| arm | set | acc 9-16 | acc 17-32 | acc 9-32 | F1/F8 | C_cursor | C_edge | C_edge_full | D_echo_own | full-clock 9-32 | rand 9-32 | effect 9-32 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in sorted(rows, key=lambda x: x["C_cursor"]):
        f18_str = f"{r['f1_over_f8']:.4f}" if r.get("f1_over_f8") is not None else "—"
        lines.append(
            f"| {r['arm']} | {r['set']} | {r['clean']['acc_9_16']:.3f} | {r['clean']['acc_17_32']:.3f} | "
            f"{r['clean']['acc_9_32']:.3f} | {f18_str} | {r['C_cursor']:.3f} | {r['C_edge']:.3f} | "
            f"{r['C_edge_full']:.3f} | {r['D_echo_own']:.4f} | {r['ablation_full_clock']['acc_9_32']:.3f} | "
            f"{r['ablation_full_random']['acc_9_32']:.3f} | {r['effect_full_clock_9_32']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def cmd_dev24(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, arm, toy, tca in Q.arms():
        dst = out / f"{arm}.json"
        if dst.is_file() and not args.force:
            rec = json.loads(dst.read_text())
            if rec.get("training_valid") != bool(rec["clean"]["acc8"] >= 0.90):
                rec["training_valid"] = bool(rec["clean"]["acc8"] >= 0.90)
                dst.write_text(json.dumps(rec, indent=2) + "\n")
            rows.append(rec)
            print(f"[skip] {arm}", flush=True)
            continue
        ck = toy / "ckpt" / f"{arm}.final.pt"
        res = evaluate_checkpoint(ck, s, args.device, tca / f"{arm}.json")
        dst.write_text(json.dumps(res, indent=2) + "\n")
        rows.append(res)
        print(
            f"[{arm}] acc9-32={res['clean']['acc_9_32']:.3f} C_cur={res['C_cursor']:.3f} "
            f"C_edge={res['C_edge']:.3f}/{res['C_edge_full']:.3f} D_own={res['D_echo_own']:.4f} "
            f"eff_full={res['effect_full_clock_9_32']:+.3f}",
            flush=True,
        )
    summary = score_cohort(rows, is_prospective=False)
    (out / "dev24_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = render_markdown("LOOPSCOPE-01 Q10 on Q4+Q5 (P10c edge/full-state clock & calibration)", summary, rows)
    (out / "dev24_report.md").write_text(md)
    print(md)
    return 0


def cmd_eval_arm(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{args.arm}.json"
    if dst.is_file() and not args.force:
        print(f"[skip] {args.arm}", flush=True)
        return 0
    ck = Path(args.toy) / "ckpt" / f"{args.arm}.final.pt"
    tca_json = Path(args.tca) / f"{args.arm}.json" if args.tca else None
    res = evaluate_checkpoint(ck, "q10", args.device, tca_json)
    dst.write_text(json.dumps(res, indent=2) + "\n")
    print(
        f"[{args.arm}] acc9-32={res['clean']['acc_9_32']:.3f} extrap32={res['extrap32']} "
        f"C_cur={res['C_cursor']:.3f} D_own={res['D_echo_own']:.4f} eff_full={res['effect_full_clock_9_32']:+.3f}",
        flush=True,
    )
    return 0


def cmd_score(args):
    out = Path(args.out)
    expected_arms = [f"lam{lam:g}_s{seed}" for seed in args.seeds for lam in args.lams]
    rows = []
    for arm in expected_arms:
        p = out / f"{arm}.json"
        if not p.is_file():
            raise SystemExit(f"Missing evaluated Q10 arm receipt: {p}")
        rows.append(json.loads(p.read_text()))
    summary = score_cohort(rows, is_prospective=True)
    (out / "q10_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = render_markdown("LOOPSCOPE-01 Q10: prospective fresh-seed depth-32 extrapolation (amendment 10)", summary, rows)
    (out / "q10_report.md").write_text(md)
    print(md)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dev24")
    d.add_argument("--out", default=str(OUT_DEV24))
    d.add_argument("--device", default="cuda")
    d.add_argument("--force", action="store_true")

    e = sub.add_parser("eval-arm")
    e.add_argument("--arm", required=True)
    e.add_argument("--toy", default=str(TOY_Q10))
    e.add_argument("--tca", default=str(TCA_Q10))
    e.add_argument("--out", default=str(OUT_Q10))
    e.add_argument("--device", default="cuda")
    e.add_argument("--force", action="store_true")

    s = sub.add_parser("score")
    s.add_argument("--out", default=str(OUT_Q10))
    s.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14, 15])
    s.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.01])

    args = ap.parse_args(argv)
    return {"dev24": cmd_dev24, "eval-arm": cmd_eval_arm, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
