"""LOOPSCOPE-01 Q4: tied looped transformer on pointer chase, per-loop supervision weight sweep.

Subcommands: train (one arm/seed), eval (one checkpoint), report (all receipts -> frontier + Q4 decision).

Model is depth-agnostic: the loop count never enters the computation, so the state after loop j is the
same whatever depth k >= j is being asked. Accuracy at depth k is therefore the head read at loop k.
Loss per item of depth k: CE at loop k (weight 1) + lam * sum_{j<k} CE at loop j (same head, same
answer position). lam = 0 is answer-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_OUT = Path("experiments/recurrence_monitorability_v0/loopscope_toy")
LAMS = (0.0, 0.01, 0.1, 1.0)
EXIT_POWER_CAP = 75
VALID_ACC = 0.90
EVAL_SEEDS = {"probe_train": 910001, "probe_test": 910002, "depth": 910100}


# ---------------------------------------------------------------- data

def make_batch(n_nodes: int, batch: int, seed: int, max_depth: int, depth=None):
    """Edges (src,dst) in random order, [depth token], then start. Returns tokens, hops [B,max_depth].

    With `depth` ([B] ints in 1..16) the prompt states the hop count as token n_nodes + depth, so a model with a
    fixed loop budget can schedule its hops freely."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.rand(batch, n_nodes, generator=g).argsort(dim=1)  # perm[b,i] = successor of i
    order = torch.rand(batch, n_nodes, generator=g).argsort(dim=1)  # edge order
    src = order
    dst = perm.gather(1, src)
    edges = torch.stack([src, dst], dim=2).reshape(batch, 2 * n_nodes)
    start = torch.randint(0, n_nodes, (batch,), generator=g)
    query = torch.full((batch,), n_nodes, dtype=torch.long)
    # The start node is the last token: its position is the cursor the loop updates and the readout reads.
    parts = [edges] + ([(n_nodes + depth.long())[:, None]] if depth is not None else []) + [start[:, None]]
    tokens = torch.cat(parts, dim=1)
    hops = []
    cur = start
    for _ in range(max_depth):
        cur = perm.gather(1, cur[:, None])[:, 0]
        hops.append(cur)
    return tokens, torch.stack(hops, dim=1)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class LoopedChase(nn.Module):
    def __init__(self, n_nodes=16, d=256, layers=2, heads=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.tok = nn.Embedding(n_nodes + 17, d)  # nodes, QUERY, depth tokens 1..16
        self.dst = nn.Embedding(n_nodes, d)  # each edge is one position: tok(src) + dst(dst)
        self.pos = nn.Embedding(n_nodes + 3, d)
        self.core = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.inject = nn.LayerNorm(d)  # normalises the carried state before the prompt is added back
        self.head = nn.Linear(d, n_nodes)

    def readout(self, h):
        return self.head(self.ln_f(h))

    def forward(self, tokens, n_loops: int):
        """Return answer-position states after each loop [n_loops, B, d]."""
        n = self.n_nodes
        edges = tokens[:, :2 * n].reshape(tokens.shape[0], n, 2)
        h = torch.cat([self.tok(edges[..., 0]) + self.dst(edges[..., 1]), self.tok(tokens[:, 2 * n:])], 1)
        e = h + self.pos(torch.arange(h.shape[1], device=tokens.device))
        h = torch.zeros_like(e)
        states = []
        for _ in range(n_loops):
            h = self.inject(h) + e  # input injection: the prompt is re-read every loop (as in Huginn)
            for blk in self.core:
                h = blk(h)
            states.append(h[:, -1])
        return torch.stack(states)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- helpers

def arm_name(lam: float, seed: int) -> str:
    return f"lam{lam:g}_s{seed}"


def autocast(device: str):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def power_limit_watts():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        return float(out.replace("W", "").strip())
    except Exception:
        return None


def build_model(args) -> LoopedChase:
    return LoopedChase(args.n_nodes, args.d, args.layers, args.heads)


def model_args_from(cfg: dict):
    return argparse.Namespace(n_nodes=cfg["n_nodes"], d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"])


# ---------------------------------------------------------------- train

def lr_at(step, steps, lr, warm):
    if step < warm:
        return lr * (step + 1) / warm
    p = (step - warm) / max(1, steps - warm)
    return lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def cmd_train(args) -> int:
    out = Path(args.out)
    name = arm_name(args.lam, args.seed)
    run_dir = out / "runs" / name
    ckpt_dir = out / "ckpt"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"{name}.final.pt"
    receipt_path = run_dir / "receipt.json"
    if final.exists() and receipt_path.exists():
        print(f"[skip] {name} already trained")
        return 0
    dev = args.device
    torch.manual_seed(args.seed)
    model = build_model(args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    cfg = {k: getattr(args, k) for k in
           ("n_nodes", "d", "layers", "heads", "steps", "batch", "lr", "max_train_depth", "loops", "curriculum_frac", "fixed_depth", "lam", "seed")}
    step, wall = 0, 0.0
    ck = ckpt_dir / f"{name}.ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, wall = st["step"], st["wall"]
        print(f"[resume] {name} at step {step}")
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    K = args.max_train_depth
    T = args.loops
    t0 = time.time()
    losses = []
    model.train()
    while step < args.steps:
        # same data stream in every arm for a given seed: batch seed depends on (seed, step) only
        # Depth curriculum: the maximum training depth ramps from 1 to K over the first curriculum_frac of steps.
        kmax = K if args.curriculum_frac <= 0 else 1 + int((K - 1) * min(1.0, step / (args.curriculum_frac * args.steps)))
        depth = torch.randint(1, kmax + 1, (args.batch,), generator=torch.Generator().manual_seed(
            args.seed * 7_000_033 + step))
        if args.fixed_depth:  # amendment 2: every item has depth K, no depth token, T = K loops
            depth = torch.full((args.batch,), K)
        tokens, hops = make_batch(args.n_nodes, args.batch, args.seed * 1_000_003 + step, T,
                                  None if args.fixed_depth else depth)
        tokens, hops, depth = tokens.to(dev), hops.to(dev), depth.to(dev)
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        with autocast(dev):
            states = model(tokens, T)
            logits = model.readout(states).float()  # [T,B,n]
        # Loop j is supervised (weight lam, j < T) with hop min(j, depth): one hop per loop, then hold the answer.
        # The final loop T is supervised with the answer (weight 1) in every arm.
        j = torch.arange(1, T + 1, device=dev)[:, None]
        target = hops.t().gather(0, (torch.minimum(j, depth[None]) - 1).expand(T, args.batch))
        ce = F.cross_entropy(logits.reshape(-1, args.n_nodes), target.reshape(-1), reduction="none").view(T, args.batch)
        w = torch.where(j == T, torch.ones_like(ce), args.lam * torch.ones_like(ce))
        loss = (w * ce).sum(0).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        losses.append(loss.item() if step % 50 == 0 else 0.0)
        if step % args.log_every == 0:
            print(f"{name} step {step}/{args.steps} loss {loss.item():.4f} {(time.time() - t0) / max(1, len(losses)) * 1000:.0f} ms/step", flush=True)
        if step % args.ckpt_every == 0 and step < args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "wall": wall + time.time() - t0}, ck)
            if dev.startswith("cuda"):
                pl = power_limit_watts()
                if pl is not None and pl < 40.0:
                    print(f"power cap {pl} W < 40 W: checkpointed at step {step}, exit 75", flush=True)
                    return EXIT_POWER_CAP
    wall += time.time() - t0
    torch.save({"model": model.state_dict(), "cfg": cfg}, final)
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.startswith("cuda") else None
    receipt = {
        "arm": name, "lam": args.lam, "seed": args.seed, "config": cfg, "params": count_params(model),
        "steps": args.steps, "wall_seconds": wall, "peak_vram_mib": peak, "device": dev,
        "final_loss_last50_mean": float(np.mean([x for x in losses[-200:] if x != 0.0] or [float("nan")])),
        "checkpoint": final.name, "checkpoint_sha256": sha256_file(final),
        "torch": torch.__version__,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    if ck.exists():
        ck.unlink()
    print(f"[done] {name} {wall:.0f}s params {receipt['params']}")
    return 0


# ---------------------------------------------------------------- eval

def ridge_probe(Xtr, Ytr, Xte, n_classes, alpha_rel=1e-3):
    """Closed-form ridge on one-hot targets, bias column, float64. Returns predicted classes."""
    Xtr = torch.cat([Xtr.double(), torch.ones(len(Xtr), 1, dtype=torch.float64)], 1)
    Xte = torch.cat([Xte.double(), torch.ones(len(Xte), 1, dtype=torch.float64)], 1)
    Y = F.one_hot(Ytr, n_classes).double()
    G = Xtr.T @ Xtr
    alpha = alpha_rel * G.diagonal().mean()
    W = torch.linalg.solve(G + alpha * torch.eye(G.shape[0], dtype=torch.float64), Xtr.T @ Y)
    return (Xte @ W).argmax(1)


@torch.no_grad()
def evaluate_model(model, cfg, device, eval_items=512, max_eval_depth=16, probe_train_items=4096):
    """Capability with the trained loop budget T and with 2T; probes and lens on items of random depth 1..K.

    probe_grid[i][j] = ridge-probe accuracy for hop j from the state after loop i, on items with depth >= j.
    `probe` bits are the aligned diagonal (loop j -> hop j); `probe_anywhere` is the best loop per hop."""
    model.eval()
    n, K, T = cfg["n_nodes"], max_eval_depth, cfg.get("loops", max_eval_depth)
    tr_depth = cfg["max_train_depth"]

    def run(tokens, loops):
        outs = []
        for i in range(0, len(tokens), 1024):
            with autocast(device):
                outs.append(model(tokens[i:i + 1024].to(device), loops).float().cpu())
        return torch.cat(outs, 1)

    def head(S):
        with autocast(device):
            return model.readout(S.to(device)).float().argmax(-1).cpu()

    fixed = cfg.get("fixed_depth", False)
    if fixed:  # no depth token: depth k is answered after k loops (the model never sees k)
        T = K
    cap_bits, cap2_bits = {}, {}
    for k in range(1, K + 1):
        if fixed:
            tokens, hops = make_batch(n, eval_items, EVAL_SEEDS["depth"] + k, K)
            st = run(tokens, 2 * k)
            cap_bits[k] = (head(st[k - 1]) == hops[:, k - 1]).int().tolist()
            cap2_bits[k] = (head(st[2 * k - 1]) == hops[:, k - 1]).int().tolist()
            continue
        tokens, hops = make_batch(n, eval_items, EVAL_SEEDS["depth"] + k, K, torch.full((eval_items,), k))
        st = run(tokens, 2 * T)
        cap_bits[k] = (head(st[T - 1]) == hops[:, k - 1]).int().tolist()
        cap2_bits[k] = (head(st[2 * T - 1]) == hops[:, k - 1]).int().tolist()

    def states_for(seed, count):
        depth = torch.randint(1, K + 1, (count,), generator=torch.Generator().manual_seed(seed + 17))
        if fixed:
            depth = torch.full((count,), K)
        tokens, hops = make_batch(n, count, seed, K, None if fixed else depth)
        return run(tokens, T), hops, depth

    S_tr, H_tr, D_tr = states_for(EVAL_SEEDS["probe_train"], probe_train_items)
    S_te, H_te, D_te = states_for(EVAL_SEEDS["probe_test"], 4 * eval_items)
    grid = np.zeros((T, K))
    probe_bits, lens_bits = {}, {}
    for j in range(1, K + 1):
        mtr, mte = D_tr >= j, D_te >= j
        for i in range(1, T + 1):
            pred = ridge_probe(S_tr[i - 1][mtr], H_tr[mtr, j - 1], S_te[i - 1][mte], n)
            hit = (pred == H_te[mte, j - 1]).int()
            grid[i - 1, j - 1] = float(hit.float().mean())
            if i == j:
                probe_bits[j] = hit.tolist()
        if j <= T:
            lens_bits[j] = (head(S_te[j - 1][mte]) == H_te[mte, j - 1]).int().tolist()
    m = lambda bits, ks: float(np.mean([np.mean(bits[k]) for k in ks if k in bits])) if ks else None
    tr, ho, allk = range(1, tr_depth + 1), range(tr_depth + 1, K + 1), range(1, K + 1)
    return {
        "eval_items": eval_items, "max_eval_depth": K, "loops": T, "train_depths": [1, tr_depth],
        "capability_by_depth": {k: float(np.mean(v)) for k, v in cap_bits.items()},
        "capability_2T_by_depth": {k: float(np.mean(v)) for k, v in cap2_bits.items()},
        "probe_by_loop": {k: float(np.mean(v)) for k, v in probe_bits.items()},
        "lens_by_loop": {k: float(np.mean(v)) for k, v in lens_bits.items()},
        "probe_grid": grid.round(4).tolist(),
        "summary": {
            "cap_train": m(cap_bits, tr), "cap_heldout": m(cap_bits, ho),
            "cap2T_train": m(cap2_bits, tr), "cap2T_heldout": m(cap2_bits, ho),
            "probe_all": m(probe_bits, allk), "probe_train": m(probe_bits, tr), "probe_heldout": m(probe_bits, ho),
            "probe_anywhere_all": float(grid.max(0).mean()),
            "lens_all": m(lens_bits, allk), "lens_train": m(lens_bits, tr), "lens_heldout": m(lens_bits, ho),
        },
        "bits": {"cap": cap_bits, "cap2T": cap2_bits, "probe": probe_bits, "lens": lens_bits},
    }


def cmd_eval(args) -> int:
    out = Path(args.out)
    name = arm_name(args.lam, args.seed)
    final = out / "ckpt" / f"{name}.final.pt"
    mpath = out / "runs" / name / "metrics.json"
    if mpath.exists() and not args.force:
        print(f"[skip] {name} already evaluated")
        return 0
    st = torch.load(final, map_location=args.device)
    cfg = st["cfg"]
    model = build_model(model_args_from(cfg)).to(args.device)
    model.load_state_dict(st["model"])
    res = evaluate_model(model, cfg, args.device, args.eval_items, args.max_eval_depth, args.probe_train_items)
    res.update({"arm": name, "lam": cfg["lam"], "seed": cfg["seed"], "checkpoint_sha256": sha256_file(final)})
    mpath.write_text(json.dumps(res))
    print(f"[eval] {name} {json.dumps(res['summary'])}")
    return 0


# ---------------------------------------------------------------- report

def arm_stat(runs, key_bits, ks, rng, resample_items=True):
    """One bootstrap draw of an arm mean: resample runs (seeds), then items within each run."""
    idx = rng.integers(0, len(runs), len(runs))
    vals = []
    for i in idx:
        bits = runs[i]["bits"][key_bits]
        per_k = []
        for k in ks:
            b = np.asarray(bits[str(k)])
            if resample_items:
                b = b[rng.integers(0, len(b), len(b))]
            per_k.append(b.mean())
        vals.append(np.mean(per_k))
    return float(np.mean(vals))


def point(runs, key_bits, ks):
    return float(np.mean([np.mean([np.mean(r["bits"][key_bits][str(k)]) for k in ks]) for r in runs]))


def bootstrap_diff(a_runs, b_runs, key_bits, ks, n_boot, seed):
    rng = np.random.default_rng(seed)
    d = np.array([arm_stat(a_runs, key_bits, ks, rng) - arm_stat(b_runs, key_bits, ks, rng) for _ in range(n_boot)])
    return point(a_runs, key_bits, ks) - point(b_runs, key_bits, ks), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cmd_report(args) -> int:
    out = Path(args.out)
    runs, invalid = {}, []
    for p in sorted((out / "runs").glob("*/metrics.json")):
        r = json.loads(p.read_text())
        # Amendment 2 validity: depth-8 (training depth) accuracy >= 0.90 in that seed; invalid runs are training_failed.
        r["acc_d8"] = float(r["capability_by_depth"][str(r["train_depths"][1])])
        if r["acc_d8"] < args.valid_acc:
            invalid.append({"arm": r["arm"], "lam": r["lam"], "seed": r["seed"], "acc_d8": r["acc_d8"]})
            continue
        runs.setdefault(r["lam"], []).append(r)
    if 0.0 not in runs:
        print(f"no valid lam=0 runs (invalid: {invalid}); Q4 decision: lam=0 not trainable at this budget", file=sys.stderr)
        return 2
    tr_hi = runs[0.0][0]["train_depths"][1]
    K = runs[0.0][0]["max_eval_depth"]
    ho = list(range(tr_hi + 1, K + 1))
    allk = list(range(1, K + 1))
    rows, diffs = [], {}
    for lam in sorted(runs):
        rs = runs[lam]
        s = lambda key: [r["summary"][key] for r in rs]
        rows.append({"lam": lam, "n_seeds": len(rs), "acc_d8": (float(np.mean([r["acc_d8"] for r in rs])), float(np.std([r["acc_d8"] for r in rs]))),
                     **{k: (float(np.mean(s(k))), float(np.std(s(k)))) for k in
                        ("cap_train", "cap_heldout", "cap2T_heldout", "probe_all", "probe_train", "probe_heldout", "probe_anywhere_all", "lens_all")}})
    for lam in sorted(runs):
        if lam == 0.0:
            continue
        cap = bootstrap_diff(runs[lam], runs[0.0], "cap", ho, args.n_boot, 1)
        prb = bootstrap_diff(runs[lam], runs[0.0], "probe", allk, args.n_boot, 2)
        diffs[lam] = {"cap_heldout_diff": cap, "probe_all_diff": prb}
    tradeoff = [l for l, d in diffs.items()
                if d["cap_heldout_diff"][0] < -0.05 and d["cap_heldout_diff"][2] < 0 and d["probe_all_diff"][0] > 0]
    none = all(d["cap_heldout_diff"][0] >= 0 and d["probe_all_diff"][0] >= 0 for d in diffs.values())
    if tradeoff:
        decision = f"trade-off (lam {tradeoff})"
    elif none and diffs:
        decision = "no trade-off"
    else:
        decision = "mixed"
    n_runs = sum(len(v) for v in runs.values()) + len(invalid)
    complete = n_runs >= 3 * len(LAMS)
    rep = {"decision": decision, "complete": complete, "training_failed": invalid, "held_out_depths": ho, "rows": rows,
           "diffs_vs_lam0": {str(k): {kk: list(vv) for kk, vv in v.items()} for k, v in diffs.items()},
           "rule": "trade-off if some lam>0 has held-out accuracy below lam=0 by >0.05 with seed+item bootstrap 95% CI excluding 0 "
                   "and probe decodability (mean over loops 1-K) above lam=0; no trade-off if every lam>0 is >= lam=0 on both; else mixed.",
           "n_boot": args.n_boot}
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    f = lambda t: f"{t[0]:.3f}+-{t[1]:.3f}"
    md = [f"# LOOPSCOPE-01 Q4 frontier ({'complete' if complete else 'INCOMPLETE'})", "",
          f"Decision: **{decision}**", "",
          f"Held-out depths {ho[0]}-{ho[-1]}. Probe = ridge probe on a separate split, mean over loops 1-{K}. Lens = model head at loop j. mean+-sd over seeds.", "",
          "Validity: depth-8 accuracy >= %.2f per seed (amendment 2). training_failed: %s" % (args.valid_acc, invalid or "none"), "",
          "| lam | valid seeds | acc d8 | acc d1-%d | acc held-out | probe 1-%d | probe d1-%d | probe held-out | lens 1-%d |" % (tr_hi, K, tr_hi, K),
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['lam']:g} | {r['n_seeds']} | {f(r['acc_d8'])} | {f(r['cap_train'])} | {f(r['cap_heldout'])} | {f(r['probe_all'])} | {f(r['probe_train'])} | {f(r['probe_heldout'])} | {f(r['lens_all'])} |")
    md += ["", "Differences vs lam=0 (point, 95% CI; bootstrap over seeds then items):", "",
           "| lam | held-out acc diff | probe diff |", "|---|---|---|"]
    for lam, d in diffs.items():
        c, p = d["cap_heldout_diff"], d["probe_all_diff"]
        md.append(f"| {lam:g} | {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] | {p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}] |")
    md += ["", rep["rule"]]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default=str(DEFAULT_OUT))
        p.add_argument("--device", default="cuda")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--lam", type=float, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--n-nodes", type=int, default=16)
    t.add_argument("--d", type=int, default=256)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--steps", type=int, default=10000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=300)
    t.add_argument("--max-train-depth", type=int, default=8)
    t.add_argument("--loops", type=int, default=16, help="fixed loop budget T for every item")
    t.add_argument("--fixed-depth", action="store_true", help="amendment 2: all items depth K, no depth token; use with --loops K")
    t.add_argument("--curriculum-frac", type=float, default=0.5, help="fraction of steps over which max depth ramps 1->K (0: off)")
    t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=500)
    e = sub.add_parser("eval")
    common(e)
    e.add_argument("--lam", type=float, required=True)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--eval-items", type=int, default=512)
    e.add_argument("--max-eval-depth", type=int, default=16)
    e.add_argument("--probe-train-items", type=int, default=4096)
    e.add_argument("--force", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--n-boot", type=int, default=5000)
    r.add_argument("--valid-acc", type=float, default=VALID_ACC, help="amendment 2 validity gate on training-depth accuracy")
    args = ap.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
