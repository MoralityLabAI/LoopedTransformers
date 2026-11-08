"""LOOPSCOPE-01 Q3: decodability and transported component attribution (TCA) on Lean-CGRT loopers.

Read-only on the lean worktree and on D:\\lean_cgrt. lcgrt is imported, not copied.

    python -m research_gym.scripts.loopscope_lean run --ckpt Sloop_s17b S_t8_s17b --T 8 16 32 \
        --device cpu --threads 4 --n-states 128 --n-theorems 64 --batch 8
    python -m research_gym.scripts.loopscope_lean report

Per iteration t of a ProofNet forward at T fast steps it records
  * decodability: the model's own readout head applied to the state after step t, agreement of
    its top tactic with the final top tactic at T, and with the gold next tactic;
  * TCA for every recurrent matrix W (fast clock: one call per step; slow clock of the two-clock
    HRM: one call per slow tick): X = input, A = d log p(final choice) / d output of W at that call.
    Fragility F = ||X^T A||_F; for the top-K SVD components W = sum s u v^T,
    Delta = -v^T X^T A u (unit component) and z for the read / write / both-rotated nulls.
Quantities are computed per state (batch item), then summarised over states.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

LEAN_PKG = Path(r"C:\Users\patri\Documents\Codex\LoopedTransformers-lean\research\lean_cgrt_hrm400")
if str(LEAN_PKG) not in sys.path:
    sys.path.append(str(LEAN_PKG))

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "recurrence_monitorability_v0" / "loopscope_lean"
RUN_C1 = Path(r"D:\lean_cgrt\runs\c1")
DATA_ROOT = Path(r"D:\lean_cgrt\data")
EPS = 1e-12


# --------------------------------------------------------------------------- loading
def load_model(ckpt, device="cpu", dtype=torch.float32):
    from lcgrt.model import ModelConfig, ProofNet
    from lcgrt.train import load_verified
    p = Path(ckpt)
    if not p.exists():
        p = RUN_C1 / str(ckpt)
    _, ck = load_verified(p, "cpu")
    if ck is None:
        raise FileNotFoundError(f"no verified checkpoint in {p}")
    m = ProofNet(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.to(device=device, dtype=dtype).eval()


def load_examples(data="lcgrt-v1", split="test_mirror", n_theorems=64, n_states=128, seed=0):
    """Gold-path proof states of a seeded reservoir of theorems. Each Example gets .gold_tok."""
    from lcgrt.dataset import iter_records
    from lcgrt.encode import examples_from_record, tactic_token
    import lcgrt.config as C
    C.DATA_ROOT = DATA_ROOT
    rng = random.Random(seed)
    keep = []
    for i, r in enumerate(iter_records(data, split)):
        if len(keep) < n_theorems:
            keep.append(r)
        else:
            j = rng.randint(0, i)
            if j < n_theorems:
                keep[j] = r
    exs = []
    for r in keep:
        for e, step in zip(examples_from_record(r, random.Random(0), fail_aug=0.0), r["path"]):
            e.gold_tok = tactic_token(e.graph, step["tactic"])
            exs.append(e)
    rng.shuffle(exs)
    return exs[:n_states]


# --------------------------------------------------------------------------- model structure
def tracked_linears(model):
    """name -> (module, clock). Recurrent matrices applied once per fast step / slow tick."""
    out = {}
    fast = model.fast
    for n, m in [("att.qkv", fast.att.qkv), ("att.o", fast.att.o), ("ff.0", fast.ff[0]), ("ff.2", fast.ff[2])]:
        out[f"fast.{n}"] = (m, "fast")
    if fast.cross is not None:
        for n in ("q", "kv", "o"):
            out[f"fast.cross.{n}"] = (getattr(fast.cross, n), "fast")
    if getattr(model, "slow", None) is not None:
        sl = model.slow
        for blk in ("self_att", "read"):
            for n in ("q", "kv", "o"):
                out[f"slow.{blk}.{n}"] = (getattr(getattr(sl, blk), n), "slow")
        out["slow.ff.0"] = (sl.ff[0], "slow")
        out["slow.ff.2"] = (sl.ff[2], "slow")
    return out


def svd_bases(mods, K):
    """name -> (U[:, :K] [d_out,K], S[:K], V [d_in,K]) with W = U diag(S) V^T (nn.Linear weight [d_out,d_in])."""
    out = {}
    for name, (m, _) in mods.items():
        U, S, Vh = torch.linalg.svd(m.weight.detach().float(), full_matrices=False)
        k = min(K, S.numel())
        out[name] = (U[:, :k].to(m.weight.dtype), S[:k].to(m.weight.dtype), Vh[:k].T.contiguous().to(m.weight.dtype))
    return out


# --------------------------------------------------------------------------- forward with per-step states
def forward_states(model, b, T, glob=None):
    """Replicates ProofNet.forward for arch=hrm, returning the fast state after every step and the
    readout logits per step (head applied to every step, not only slow ticks)."""
    from lcgrt.model import N_GLOBAL
    import torch.nn.functional as F
    from lcgrt.encode import N_EDGE_TYPES
    cfg = model.cfg
    assert cfg.arch == "hrm", "loopscope_lean analyses the looped (hrm) models"
    etype = model._etype(b)
    x0 = model.feat(b["x"]) + model.ttype(b["ttype"])
    onehot = F.one_hot(etype, N_EDGE_TYPES).to(x0.dtype)
    nonedge = etype == 0
    k = min(cfg.slow_every, T)
    B = x0.shape[0]
    if glob is None:
        glob = torch.zeros(B, N_GLOBAL, device=x0.device, dtype=x0.dtype)
    s = model.slot_init[None].expand(B, -1, -1) + model.glob(glob)[:, None] if cfg.slow else None
    x = torch.zeros_like(x0)
    bias = model.fast.att.make_bias(onehot, nonedge)
    logits = []
    for t in range(T):
        x = model.fast(x, x0, bias, s)
        if (t + 1) % k == 0 or t == T - 1:
            if cfg.slow:
                s = model.slow(s, x, b["tok_mask"])
        lg, _ = model.readout(x, b["main"], b["legal"])
        logits.append(lg)
    return logits


# --------------------------------------------------------------------------- TCA hooks
class TCA:
    def __init__(self, model, K=32):
        self.mods = tracked_linears(model)
        self.bases = svd_bases(self.mods, K)
        self.count = {n: 0 for n in self.mods}
        self.rows = {}               # (name, call) -> dict of tensors [B] / [B,K]
        self.handles = []
        for name, (m, _) in self.mods.items():
            self.handles.append(m.register_forward_hook(self._make(name)))

    def reset(self):
        self.count = {n: 0 for n in self.mods}

    def remove(self):
        for h in self.handles:
            h.remove()

    def _make(self, name):
        def hook(mod, inp, out):
            c = self.count[name]
            self.count[name] += 1
            if not out.requires_grad:
                return
            X = inp[0].detach()
            out.register_hook(lambda g, X=X, c=c: self._process(name, c, X, g.detach()))
        return hook

    def _process(self, name, c, X, A):
        U, S, V = self.bases[name]
        d_in, d_out = X.shape[-1], A.shape[-1]
        F2 = ((X @ X.transpose(1, 2)) * (A @ A.transpose(1, 2))).sum((1, 2)).clamp_min(0)
        Fr = F2.sqrt()
        Xv, Au = X @ V, A @ U                                   # [B,P,K]
        delta = -(Xv * Au).sum(1)                               # [B,K] unit component (v, u)
        sd_r = torch.einsum("bpi,bpk->bik", X, Au).norm(dim=1) / math.sqrt(d_in)
        sd_w = torch.einsum("bpo,bpk->bok", A, Xv).norm(dim=1) / math.sqrt(d_out)
        sd_b = (Fr / math.sqrt(d_in * d_out))[:, None].expand_as(delta)
        self.rows.setdefault((name, c), []).append(dict(
            F=Fr.cpu(), delta=delta.cpu(), zr=(delta / (sd_r + EPS)).cpu(),
            zw=(delta / (sd_w + EPS)).cpu(), zb=(delta / (sd_b + EPS)).cpu(), sd_r=sd_r.cpu(), sd_w=sd_w.cpu()))


# --------------------------------------------------------------------------- analysis
def analyze(model, examples, T, batch=8, K=32, device="cpu"):
    """Run decodability + TCA for all examples at T fast steps. Returns the result dict."""
    from lcgrt.encode import collate
    for p in model.parameters():
        p.requires_grad_(False)
    if model.cfg.slow:
        model.slot_init.requires_grad_(True)
    tca = TCA(model, K)
    k = min(model.cfg.slow_every, T)
    agree_final, agree_gold, in_opt, p_final, ent = [], [], [], [], []
    for i0 in range(0, len(examples), batch):
        exs = examples[i0:i0 + batch]
        b = collate(exs, device)
        b["x"] = b["x"].to(next(model.parameters()).dtype).requires_grad_(True)
        gold = torch.tensor([e.gold_tok for e in exs], device=device)
        tca.reset()
        logits = forward_states(model, b, T)
        final = logits[-1].argmax(-1)
        lp_final = torch.log_softmax(logits[-1], -1).gather(1, final[:, None]).squeeze(1)
        lp_final.sum().backward()
        with torch.no_grad():
            for t, lg in enumerate(logits):
                top = lg.argmax(-1)
                pr = torch.softmax(lg, -1)
                while len(agree_final) <= t:
                    agree_final.append([]); agree_gold.append([]); in_opt.append([]); p_final.append([]); ent.append([])
                agree_final[t].append((top == final).float().cpu())
                agree_gold[t].append((top == gold).float().cpu())
                in_opt[t].append(b["optimal"].gather(1, top[:, None]).squeeze(1).float().cpu())
                p_final[t].append(pr.gather(1, final[:, None]).squeeze(1).cpu())
                ent[t].append(-(pr * torch.log(pr.clamp_min(1e-30))).sum(-1).cpu())
        b["x"].grad = None
    tca.remove()
    cat = lambda xs: torch.cat(xs)
    decod = []
    for t in range(T):
        decod.append(dict(t=t + 1, slow_tick=bool((t + 1) % k == 0 or t == T - 1),
                          agree_final=cat(agree_final[t]).mean().item(), agree_gold=cat(agree_gold[t]).mean().item(),
                          top_in_optimal=cat(in_opt[t]).mean().item(), p_final_mean=cat(p_final[t]).mean().item(),
                          entropy_mean=cat(ent[t]).mean().item()))
    cells = []
    for (name, c), rs in sorted(tca.rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        clock = tca.mods[name][1]
        t_eq = c + 1 if clock == "fast" else min(k * (c + 1), T)
        R = {key: torch.cat([r[key] for r in rs]) for key in rs[0]}
        Kk = R["delta"].shape[1]
        S = tca.bases[name][1].cpu()
        top = min(16, Kk)
        cell = dict(module=name, clock=clock, call=c + 1, t=t_eq, n=int(R["F"].shape[0]), K=Kk,
                    F_mean=R["F"].mean().item(), F_median=R["F"].median().item(),
                    delta_unit_abs_mean=R["delta"].abs().mean().item(),
                    delta_scaled_abs_mean=(R["delta"].abs() * S[None]).mean().item(),
                    delta_unit_mean=R["delta"].mean(0).tolist())
        for z in ("zr", "zw", "zb"):
            cell[f"{z}_abs_mean"] = R[z].abs().mean().item()
            cell[f"{z}_frac_gt3"] = (R[z].abs() > 3).float().mean().item()
            cell[f"{z}_frac_gt3_top16"] = (R[z][:, :top].abs() > 3).float().mean().item()
        cells.append(cell)
    return dict(T=T, n_states=len(examples), decod=decod, tca=cells)


def run(args):
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    exs = load_examples(args.data, args.split, args.n_theorems, args.n_states, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ck in args.ckpt:
        model = load_model(ck, args.device)
        for T in args.T:
            res = analyze(model, exs, T, args.batch, args.K, args.device)
            res.update(ckpt=ck, model=model.cfg.to_json(), data=args.data, split=args.split, seed=args.seed)
            out = OUT_DIR / f"{Path(ck).name}_T{T}.json"
            out.write_text(json.dumps(res, indent=1))
            print(json.dumps({"wrote": str(out), "n_states": len(exs), "agree_final_t1": res["decod"][0]["agree_final"]}),
                  flush=True)


# --------------------------------------------------------------------------- report
def _load_results(d):
    return sorted((json.loads(p.read_text()) for p in Path(d).glob("*_T*.json")), key=lambda r: (r["ckpt"], r["T"]))


def late_share(res, key="zr_frac_gt3", clock="fast"):
    """Mean share of |z|>3 over cells at t<=8 and t>8 for one clock."""
    lo = [c[key] for c in res["tca"] if c["clock"] == clock and c["t"] <= 8]
    hi = [c[key] for c in res["tca"] if c["clock"] == clock and c["t"] > 8]
    return (float(np.mean(lo)) if lo else float("nan"), float(np.mean(hi)) if hi else float("nan"))


def fragility_by_t(res, clock="fast"):
    """t -> sum over modules of F_mean."""
    out = {}
    for c in res["tca"]:
        if c["clock"] == clock:
            out[c["t"]] = out.get(c["t"], 0.0) + c["F_mean"]
    return dict(sorted(out.items()))


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = Path(args.dir)
    rs = _load_results(d)
    if not rs:
        raise SystemExit(f"no results in {d}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for r in rs:
        lab = f"{r['ckpt']} T={r['T']}"
        t = [x["t"] for x in r["decod"]]
        ax[0].plot(t, [x["agree_final"] for x in r["decod"]], marker=".", label=lab)
        ax[1].plot(t, [x["top_in_optimal"] for x in r["decod"]], marker=".", label=lab)
        fr = fragility_by_t(r, "fast")
        ax[2].plot(list(fr), list(fr.values()), marker=".", label=lab)
    ax[0].set_title("own head: top agrees with final top"); ax[1].set_title("own head: top in optimal set")
    ax[2].set_title("fragility F (sum over fast-clock matrices)"); ax[2].set_yscale("log")
    for a in ax:
        a.set_xlabel("iteration t"); a.axvline(8, color="gray", ls=":")
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(d / "decodability_fragility.png", dpi=130); plt.close(fig)
    lines = ["# LOOPSCOPE-01 Q3 (Lean-CGRT) summary", "",
             "Exploratory, one seed per checkpoint, gold-path states of test_mirror. Own-head decodability uses the model's readout applied to every step (trained only at slow ticks, every 2nd step).", "",
             "## Decodability", "", "| ckpt | T | agree(final) t=1 | t=T/2 | t=T-1 | top in optimal t=T | p(final) t=T/2 |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        D, T = r["decod"], r["T"]
        lines.append(f"| {r['ckpt']} | {T} | {D[0]['agree_final']:.3f} | {D[T // 2 - 1]['agree_final']:.3f} | "
                     f"{D[max(T - 2, 0)]['agree_final']:.3f} | {D[-1]['top_in_optimal']:.3f} | {D[T // 2 - 1]['p_final_mean']:.3f} |")
    lines += ["", "## Fragility F (sum over matrices of mean F), fast clock", "", "| ckpt | T | t=1 | t=4 | t=8 | t=T/2 | t=T |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        fr = fragility_by_t(r, "fast"); T = r["T"]
        g = lambda t: f"{fr[t]:.3g}" if t in fr else "-"
        lines.append(f"| {r['ckpt']} | {T} | {g(1)} | {g(4)} | {g(8)} | {g(T // 2)} | {g(T)} |")
    lines += ["", "## Share of cells with |z_read|>3 (top-32 components), t<=8 vs t>8", "",
              "| ckpt | T | clock | share t<=8 | share t>8 | top-16 t<=8 | top-16 t>8 |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        for clock in sorted({c["clock"] for c in r["tca"]}):
            a, b = late_share(r, "zr_frac_gt3", clock)
            a16, b16 = late_share(r, "zr_frac_gt3_top16", clock)
            lines.append(f"| {r['ckpt']} | {r['T']} | {clock} | {a:.3f} | {b:.3f} | {a16:.3f} | {b16:.3f} |")
    lines += ["", "Null-rotation baseline for |z|>3 under a Gaussian null is about 0.003.", "",
              "![](decodability_fragility.png)", ""]
    (d / "report.md").write_text("\n".join(lines))
    print(f"wrote {d / 'report.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", nargs="+", default=["Sloop_s17b", "S_t8_s17b"])
    r.add_argument("--T", nargs="+", type=int, default=[8, 16, 32])
    r.add_argument("--data", default="lcgrt-v1")
    r.add_argument("--split", default="test_mirror")
    r.add_argument("--n-theorems", type=int, default=64)
    r.add_argument("--n-states", type=int, default=128)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--K", type=int, default=32)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--device", default="cpu")
    p = sub.add_parser("report")
    p.add_argument("--dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
