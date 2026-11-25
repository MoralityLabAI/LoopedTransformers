"""LBF-19 Track R: can a small recurrent cell LEARN the sound guarded prov_adj2 support transfer?

Development-only research direction (see LBF_19_LEARNED_TRANSFER_REGISTRATION.md).
Boolean support is the proof-relevant object; learned outputs are ranking/approximation signals,
EXCEPT the hard-sound Boolean hull `U_k` (exact Boolean ops, never learned).

Usage:
    python lbf19_learned_transfer.py --stage all            # full registered run
    python lbf19_learned_transfer.py --stage all --quick    # tiny smoke run
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lbf18_incremental_fresco as m18  # noqa: E402

ALPHABET = "abcdef"
C2I = {c: i for i, c in enumerate(ALPHABET)}
M_RULES = 10  # max oriented rules (5 unoriented x 2)
FP = 4
SEED_TRAIN, SEED_VAL, SEED_TEST = 19110001, 19110002, 19110003
BOOT_SEED = 19110099
TRAIN_L = (8, 12)
TEST_LS = (8, 10, 12, 16, 20, 24)
KTEST = 8
KTRAIN = 3
RECALL_TARGET = 0.99

# ----------------------------------------------------------------------------
# 1. Rule systems, canonical lineage, words
# ----------------------------------------------------------------------------
_PERMS = [dict(zip(ALPHABET, p)) for p in itertools.permutations(ALPHABET)]


def canonical_lineage(rules) -> tuple:
    """Min over alphabet permutations of sorted tuple of sorted (lhs, rhs) pairs."""
    best = None
    for pm in _PERMS:
        key = tuple(sorted(tuple(sorted(("".join(pm[c] for c in u), "".join(pm[c] for c in v)))) for u, v in rules))
        if best is None or key < best:
            best = key
    return best


def sample_system(rng: random.Random):
    n = rng.choice((3, 4, 5))
    rules, lhss = [], set()
    while len(rules) < n:
        ll = 2 if rng.random() < 0.6 else 3
        u = "".join(rng.choice(ALPHABET) for _ in range(ll))
        v = "".join(rng.choice(ALPHABET) for _ in range(ll))
        if u == v or u in lhss:
            continue
        lhss.add(u)
        rules.append((u, v))
    return tuple(rules)


def oriented(rules):
    return [(u, v) for u, v in rules] + [(v, u) for u, v in rules]


def make_splits(n_train=400, n_val=80, n_test=100):
    """Test -> val -> train with canonical-lineage disjointness."""
    seen: set = set()
    out = {}
    for name, seed, n in (("test", SEED_TEST, n_test), ("val", SEED_VAL, n_val), ("train", SEED_TRAIN, n_train)):
        rng = random.Random(seed)
        systems = []
        while len(systems) < n:
            r = sample_system(rng)
            key = canonical_lineage(r)
            if key in seen:
                continue
            seen.add(key)
            systems.append(r)
        out[name] = systems
    return out


def sample_word(rng: random.Random, rules, L: int) -> str:
    w = [rng.choice(ALPHABET) for _ in range(L)]
    if rng.random() < 0.5:
        ors = oriented(rules)
        for _ in range(rng.randint(1, 3)):
            lhs, _ = rng.choice(ors)
            p = rng.randrange(0, L - len(lhs) + 1)
            w[p:p + len(lhs)] = list(lhs)
    return "".join(w)


# ----------------------------------------------------------------------------
# 2. Exact target (repo's guarded transfer) and batches
# ----------------------------------------------------------------------------
def exact_support(rules, w0: str, K: int) -> np.ndarray:
    """Y[k] (k=0..K) of shape (L,6) bool, from symbolic_prov_adj2_guarded (fable_len_guard)."""
    L = len(w0)
    hist = m18.symbolic_prov_adj2_guarded(rules, w0, k_max=K, fp=FP, guard_mode="fable_len_guard")
    ys = []
    for base_a2, fibers in hist:
        u = base_a2.clone()
        for f in fibers:
            u |= f
        out = torch.zeros((L, 6), dtype=torch.bool)
        out[:-1] |= u.any(dim=2)
        out[1:] |= u.any(dim=1)
        ys.append(out.numpy())
    while len(ys) < K + 1:
        ys.append(ys[-1])
    return np.stack(ys)


@dataclass
class Batch:
    w0: torch.Tensor      # (B,L,6)
    Rl: torch.Tensor      # (B,M,3,7)
    Rr: torch.Tensor      # (B,M,3,7)
    rm: torch.Tensor      # (B,M)
    Y: torch.Tensor       # (B,K+1,L,6) float
    root: np.ndarray      # (B,)
    L: int


def encode_rules(rules):
    Rl = np.zeros((M_RULES, 3, 7), np.float32)
    Rr = np.zeros((M_RULES, 3, 7), np.float32)
    rm = np.zeros((M_RULES,), np.float32)
    Rl[:, :, 6] = 1
    Rr[:, :, 6] = 1
    for i, (u, v) in enumerate(oriented(rules)):
        rm[i] = 1
        for q in range(len(u)):
            Rl[i, q, 6] = 0
            Rl[i, q, C2I[u[q]]] = 1
            Rr[i, q, 6] = 0
            Rr[i, q, C2I[v[q]]] = 1
    return Rl, Rr, rm


def build_items(systems, words_per_sys, Ls_fn, K, seed, timing=None):
    """Return list of (root, rules, word, Y(K+1,L,6) bool)."""
    rng = random.Random(seed ^ 0x5EED)
    items = []
    for ri, rules in enumerate(systems):
        for L in Ls_fn(rng):
            for _ in range(words_per_sys):
                w = sample_word(rng, rules, L)
                t0 = time.perf_counter()
                Y = exact_support(rules, w, K)
                if timing is not None:
                    timing.setdefault(L, []).append(time.perf_counter() - t0)
                items.append((ri, rules, w, Y))
    return items


def to_batch(items) -> Batch:
    L = len(items[0][2])
    w0 = torch.zeros((len(items), L, 6))
    enc = [encode_rules(it[1]) for it in items]
    for b, it in enumerate(items):
        for p, c in enumerate(it[2]):
            w0[b, p, C2I[c]] = 1
    return Batch(
        w0=w0,
        Rl=torch.from_numpy(np.stack([e[0] for e in enc])),
        Rr=torch.from_numpy(np.stack([e[1] for e in enc])),
        rm=torch.from_numpy(np.stack([e[2] for e in enc])),
        Y=torch.from_numpy(np.stack([it[3] for it in items]).astype(np.float32)),
        root=np.array([it[0] for it in items]),
        L=L,
    )


def bucket_by_L(items):
    d = {}
    for it in items:
        d.setdefault(len(it[2]), []).append(it)
    return d


# ----------------------------------------------------------------------------
# 3. Rule gate / exact Boolean hull
# ----------------------------------------------------------------------------
def rule_gate(y: torch.Tensor, Rl, Rr, rm) -> torch.Tensor:
    """Soft-AND support gate. y (B,L,6) in [0,1] -> G (B,L,6): max over rules/offsets of
    prod_q <lhs_q, y[p+q]> * rhs_q.  With Boolean y this is exactly the pos1 rewrite hull step."""
    B, L, _ = y.shape
    yp = F.pad(y, (0, 0, 0, 2))
    match = torch.ones((B, Rl.shape[1], L), dtype=y.dtype)
    for q in range(3):
        fq = torch.einsum("bmc,blc->bml", Rl[:, :, q, :6], yp[:, q:q + L]) + Rl[:, :, q, 6:7]
        match = match * fq
    match = match * rm.unsqueeze(-1)
    G = torch.zeros_like(y)
    for q in range(3):
        mq = F.pad(match, (q, 0))[:, :, :L]  # mq[..,p'] = match[..,p'-q]
        cand = (mq.unsqueeze(-1) * Rr[:, :, q, :6].unsqueeze(2)).amax(dim=1)
        G = torch.maximum(G, cand)
    return G


def hull_sequence(w0, Rl, Rr, rm, K) -> torch.Tensor:
    """U[k] k=0..K Boolean (as float) exact pos1 hull. (K+1,B,L,6)."""
    with torch.no_grad():
        u = w0.clone()
        out = [u]
        for _ in range(K):
            u = torch.maximum(u, (rule_gate(u, Rl, Rr, rm) > 0).float())
            out.append(u)
        return torch.stack(out)


# ----------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d: int, extra_in: int = 0):
        super().__init__()
        self.d = d
        self.conv = nn.Conv1d(12 + d + extra_in, d, 5, padding=2)
        self.q = nn.Linear(d, d)
        self.v = nn.Linear(d, 3 * d)
        self.cand = nn.Linear(2 * d, d)
        self.gate = nn.Linear(2 * d, d)
        self.out = nn.Linear(d, 6)
        self.a_bias = nn.Parameter(torch.zeros(()))

    def forward(self, y, h, w0, e, rm, extra=None):
        B, L, _ = y.shape
        inp = [y, h, w0] + ([extra] if extra is not None else [])
        z = F.relu(self.conv(torch.cat(inp, -1).transpose(1, 2)).transpose(1, 2))
        a = torch.sigmoid(torch.einsum("bld,bmd->bml", self.q(z), e) / math.sqrt(self.d) + self.a_bias)
        a = a * rm.unsqueeze(-1)
        vals = self.v(e).view(B, -1, 3, self.d)
        msg = 0
        for q in range(3):
            aq = F.pad(a, (q, 0))[:, :, :L]
            msg = msg + torch.einsum("bml,bmd->bld", aq, vals[:, :, q])
        zm = torch.cat([z, msg], -1)
        g = torch.sigmoid(self.gate(zm))
        h2 = (1 - g) * h + g * torch.tanh(self.cand(zm))
        return h2, self.out(h2)


class TransferNet(nn.Module):
    """kind: rec | nonrec ; prior: zero-preserving gate (rec only); hs: in-loop hard-sound mask."""

    def __init__(self, kind="rec", prior=False, hs=False, d=40, n_blocks=1):
        super().__init__()
        self.kind, self.prior, self.hs, self.d = kind, prior, hs, d
        self.enc = nn.Sequential(nn.Linear(42, 64), nn.ReLU(), nn.Linear(64, d))
        self.blocks = nn.ModuleList([Block(d, extra_in=(1 if kind == "nonrec" else 0)) for _ in range(n_blocks)])
        with torch.no_grad():
            for b in self.blocks:
                b.out.bias.fill_(2.0 if prior else 0.0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, bt: Batch, K: int, U: torch.Tensor | None = None) -> torch.Tensor:
        """Returns scores (K,B,L,6) for k=1..K."""
        B, L, _ = bt.w0.shape
        e = self.enc(torch.cat([bt.Rl.flatten(2), bt.Rr.flatten(2)], -1)) * bt.rm.unsqueeze(-1)
        if self.hs and U is None:
            U = hull_sequence(bt.w0, bt.Rl, bt.Rr, bt.rm, K)
        outs = []
        if self.kind == "rec":
            blk = self.blocks[0]
            y, h = bt.w0, torch.zeros((B, L, self.d))
            for k in range(1, K + 1):
                h, ul = blk(y, h, bt.w0, e, bt.rm)
                u = torch.sigmoid(ul)
                if self.prior:
                    y = torch.maximum(y, rule_gate(y, bt.Rl, bt.Rr, bt.rm) * u)
                else:
                    y = u
                if self.hs:
                    y = y * U[k]
                outs.append(y)
        else:
            for k in range(1, K + 1):
                kc = torch.full((B, L, 1), k / 8.0)
                y, h = bt.w0, torch.zeros((B, L, self.d))
                for blk in self.blocks:
                    h, ul = blk(y, h, bt.w0, e, bt.rm, kc)
                    y = torch.sigmoid(ul)
                outs.append(y)
        return torch.stack(outs)


def match_width(target_params: int) -> int:
    best = None
    for d in range(24, 80):
        n = TransferNet("nonrec", d=d).n_params()
        if best is None or abs(n - target_params) < best[0]:
            best = (abs(n - target_params), d)
    return best[1]


def make_arm(name: str) -> TransferNet:
    if name == "rec_prior":
        return TransferNet("rec", prior=True)
    if name == "rec_noprior":
        return TransferNet("rec", prior=False)
    if name == "rec_prior_hs":
        return TransferNet("rec", prior=True, hs=True)
    if name == "rec_noprior_hs":
        return TransferNet("rec", prior=False, hs=True)
    if name == "nonrec_param":
        return TransferNet("nonrec", d=match_width(TransferNet("rec").n_params()))
    if name == "nonrec_compute":
        return TransferNet("nonrec", n_blocks=3)
    raise ValueError(name)


LEARNED_ARMS = ("rec_prior", "rec_noprior", "rec_prior_hs", "rec_noprior_hs", "nonrec_param", "nonrec_compute")


def train_arm(name: str, seed: int, train_buckets, steps=1500, bs=64, lr=3e-3, val_bt=None):
    torch.manual_seed(seed)
    rng = random.Random(seed * 7919 + 13)
    model = make_arm(name)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Ls = sorted(train_buckets)
    t0 = time.perf_counter()
    last = float("nan")
    for step in range(steps):
        L = rng.choice(Ls)
        its = [rng.choice(train_buckets[L]) for _ in range(bs)]
        bt = to_batch(its)
        pred = model(bt, KTRAIN)
        tgt = bt.Y[:, 1:KTRAIN + 1].transpose(0, 1)
        loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), tgt)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = float(loss.detach())
    return model, time.perf_counter() - t0, last


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------
def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    n_pos = int(label.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, l = score[order], label[order]
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def calibrate_tau(pos_scores: np.ndarray, target=RECALL_TARGET) -> float:
    """Largest tau with cell recall >= target on val positives."""
    s = np.sort(pos_scores)
    idx = int(math.floor((1 - target) * len(s)))
    return float(s[min(idx, len(s) - 1)])


@torch.no_grad()
def predict_all(model, bt: Batch, K: int, masked=False):
    sc = model(bt, K)
    if masked:
        sc = sc * hull_sequence(bt.w0, bt.Rl, bt.Rr, bt.rm, K)[1:]
    return sc


def collect_val_tau(model, val_buckets, masked=False) -> float:
    pos = []
    for L, its in val_buckets.items():
        bt = to_batch(its)
        sc = predict_all(model, bt, KTRAIN, masked)
        for k in range(KTRAIN):
            y = bt.Y[:, k + 1].bool()
            pos.append(sc[k][y].numpy())
    return calibrate_tau(np.concatenate(pos))


def hull_val_tau():
    return 1.0  # boolean hull: score in {0,1}


METRIC_KEYS = ("tp", "fn", "fp", "sfn", "spos", "ntp", "nfn", "ap", "aph", "wfn", "nw", "hullfn")


def eval_scores(sc: np.ndarray, bt: Batch, K: int, tau: float, U: np.ndarray):
    """sc (K,B,L,6) -> per (root,k) dict arrays for this L bucket. Returns dict root->k->metric dict (sums over words)."""
    Y = bt.Y.numpy() > 0.5
    Y0 = Y[:, 0]
    res = {}
    roots = np.unique(bt.root)
    for r in roots:
        idx = np.where(bt.root == r)[0]
        for k in range(1, K + 1):
            s = sc[k - 1][idx]
            y = Y[idx, k]
            u = U[k][idx] > 0.5
            pred = s >= tau
            sp = s > 0
            novel = y & ~Y0[idx]
            m = dict(
                tp=int((pred & y).sum()), fn=int((~pred & y).sum()), fp=int((pred & ~y).sum()),
                sfn=int((~sp & y).sum()), spos=int(sp.sum()),
                ntp=int((pred & novel).sum()), nfn=int((~pred & novel).sum()),
                ap=average_precision(s.ravel(), y.ravel().astype(np.int64)),
                aph=average_precision(s[u], y[u].astype(np.int64)),
                wfn=int(((~pred & y).reshape(len(idx), -1).any(1)).sum()), nw=len(idx),
                hullfn=int((y & ~u).sum()),
                npred=int(pred.sum()), ntrue=int(y.sum()),
            )
            res[(int(r), k)] = m
    return res


def evaluate_arm_on_test(score_fn, tau, test_buckets, K=KTEST):
    """score_fn(bt, K) -> (K,B,L,6) torch. Returns dict (root,L,k)->metrics."""
    out = {}
    for L, bt in test_buckets.items():
        sc = score_fn(bt, K).numpy()
        U = hull_sequence(bt.w0, bt.Rl, bt.Rr, bt.rm, K).numpy()
        for (r, k), m in eval_scores(sc, bt, K, tau, U).items():
            out[(r, L, k)] = m
    return out


def pool(metrics: dict, roots, Ls, ks):
    """Pool counts across selected (root,L,k) -> per-root dict of recall/prec/ap/aph arrays over roots."""
    per = {}
    for r in roots:
        tp = fn = fp = nw = wfn = ntp = nfn = sfn = 0
        aps, aphs = [], []
        for L in Ls:
            for k in ks:
                m = metrics[(r, L, k)]
                tp += m["tp"]; fn += m["fn"]; fp += m["fp"]; nw += m["nw"]; wfn += m["wfn"]
                ntp += m["ntp"]; nfn += m["nfn"]; sfn += m["sfn"]
                aps.append(m["ap"]); aphs.append(m["aph"])
        per[r] = dict(
            recall=tp / max(tp + fn, 1), prec=tp / max(tp + fp, 1), nrecall=ntp / max(ntp + nfn, 1),
            ap=float(np.nanmean(aps)), aph=float(np.nanmean(aphs)) if not np.all(np.isnan(aphs)) else float("nan"),
            wfn_rate=wfn / max(nw, 1), sfn=sfn, over=(tp + fp) / max(tp + fn, 1),
        )
    return per


def root_vec(per: dict, key: str, roots):
    return np.array([per[r][key] for r in roots], dtype=np.float64)


def boot_ci(diff: np.ndarray, n=10000, seed=BOOT_SEED):
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t_ci(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = T975.get(n - 1, 1.96) * float(x.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h


# ----------------------------------------------------------------------------
# 6. SVG figures (plain code)
# ----------------------------------------------------------------------------
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f"]


def svg_panels(path: Path, title: str, xlabel: str, xs: list, panels: dict, series_style=None):
    """panels: {panel_title: {series_name: [y...]}} -> multi-panel line chart SVG."""
    pw, ph, pad_l, pad_b, pad_t = 330, 260, 52, 44, 36
    W = len(panels) * pw + 20
    H = ph + pad_b + pad_t + 70
    names = []
    for ser in panels.values():
        for n in ser:
            if n not in names:
                names.append(n)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="10" y="18" font-size="14" font-weight="bold">{title}</text>']
    for pi, (pt, ser) in enumerate(panels.items()):
        ox = 10 + pi * pw + pad_l
        oy = pad_t + 10
        iw, ih = pw - pad_l - 14, ph - 10
        vals = [v for s in ser.values() for v in s if v == v]
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        lo, hi = min(lo, 0.0 if lo > 0.5 else lo), max(hi, 1.0 if hi > 0.5 else hi)
        if hi - lo < 1e-9:
            hi = lo + 1
        o.append(f'<text x="{ox}" y="{oy - 4}" font-weight="bold">{pt}</text>')
        o.append(f'<rect x="{ox}" y="{oy}" width="{iw}" height="{ih}" fill="none" stroke="#999"/>')
        for t in range(5):
            v = lo + (hi - lo) * t / 4
            yy = oy + ih - ih * t / 4
            o.append(f'<line x1="{ox}" x2="{ox + iw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#eee"/>')
            o.append(f'<text x="{ox - 4}" y="{yy + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        for i, x in enumerate(xs):
            xx = ox + iw * i / max(len(xs) - 1, 1)
            o.append(f'<text x="{xx:.1f}" y="{oy + ih + 14}" text-anchor="middle">{x}</text>')
        o.append(f'<text x="{ox + iw / 2}" y="{oy + ih + 30}" text-anchor="middle">{xlabel}</text>')
        for n, s in ser.items():
            col = PALETTE[names.index(n) % len(PALETTE)]
            pts = [(ox + iw * i / max(len(xs) - 1, 1), oy + ih - ih * (v - lo) / (hi - lo)) for i, v in enumerate(s) if v == v]
            if pts:
                o.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
                for a, b in pts:
                    o.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.5" fill="{col}"/>')
    for i, n in enumerate(names):
        col = PALETTE[i % len(PALETTE)]
        lx = 14 + (i % 4) * 190
        ly = ph + pad_b + pad_t + 22 + (i // 4) * 16
        o.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="4" fill="{col}"/><text x="{lx + 17}" y="{ly}">{n}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


# ----------------------------------------------------------------------------
# 7. Experiment driver
# ----------------------------------------------------------------------------
def run_experiment(out_dir: Path, quick=False, seeds=(0, 1, 2, 3, 4), steps=1500, log=print, threads=4):
    torch.set_num_threads(threads)
    n_tr, n_va, n_te = (40, 12, 12) if quick else (400, 80, 100)
    wps_test = 2
    if quick:
        steps, seeds = 40, (0, 1)
    splits = make_splits(n_tr, n_va, n_te)
    keys = {n: {canonical_lineage(r) for r in s} for n, s in splits.items()}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])
    t0 = time.perf_counter()
    tr_items = build_items(splits["train"], 8 if not quick else 3, lambda rng: [rng.randint(*TRAIN_L)], KTRAIN, SEED_TRAIN)
    va_items = build_items(splits["val"], 4 if not quick else 3, lambda rng: [rng.randint(*TRAIN_L)], KTRAIN, SEED_VAL)
    log(f"train/val oracle gen {time.perf_counter() - t0:.1f}s ({len(tr_items)}/{len(va_items)} words)")
    oracle_t: dict = {}
    t0 = time.perf_counter()
    te_items = build_items(splits["test"], wps_test, lambda rng: list(TEST_LS), KTEST, SEED_TEST, timing=oracle_t)
    log(f"test oracle gen {time.perf_counter() - t0:.1f}s ({len(te_items)} words)  [generated, not scored yet]")
    tr_b, va_b = bucket_by_L(tr_items), bucket_by_L(va_items)
    te_b = {L: to_batch(its) for L, its in sorted(bucket_by_L(te_items).items())}
    roots = list(range(n_te))

    # hull soundness (H6 i) on test and val
    hull_viol = {}
    for L, bt in te_b.items():
        U = hull_sequence(bt.w0, bt.Rl, bt.Rr, bt.rm, KTEST)
        Y = bt.Y.transpose(0, 1)
        hull_viol[L] = int(((Y > 0.5) & (U < 0.5)).sum())
    log(f"hull violations (Y not in U) per L: {hull_viol}")

    evals = {}  # (arm, seed) -> metrics
    meta = {}
    taus = {}
    for arm in LEARNED_ARMS:
        for seed in seeds:
            model, ttrain, final_loss = train_arm(arm, seed, tr_b, steps=steps)
            model.eval()
            variants = [(arm, False)]
            if not arm.endswith("_hs"):
                variants.append((arm + "_masked", True))
            for vname, masked in variants:
                tau = collect_val_tau(model, va_b, masked)
                ti = time.perf_counter()
                met = evaluate_arm_on_test(lambda bt, K, m=model, mk=masked: predict_all(m, bt, K, mk), tau, te_b)
                tinf = time.perf_counter() - ti
                evals[(vname, seed)] = met
                taus[(vname, seed)] = tau
                # single-word inference timing (K=8) per L
                single = {}
                for L, bt in te_b.items():
                    one = Batch(bt.w0[:1], bt.Rl[:1], bt.Rr[:1], bt.rm[:1], bt.Y[:1], bt.root[:1], bt.L)
                    ts = []
                    for _ in range(5):
                        a = time.perf_counter(); predict_all(model, one, KTEST, masked); ts.append(time.perf_counter() - a)
                    single[L] = 1000 * float(np.median(ts))
                meta[(vname, seed)] = dict(params=model.n_params(), train_s=ttrain, final_loss=final_loss, tau=tau,
                                           infer_ms_per_word_single_K8=single,
                                           infer_ms_per_word_batched_K8=1000 * tinf / len(te_items))
            log(f"{arm} seed {seed}: params={model.n_params()} train={ttrain:.1f}s loss={final_loss:.4f}")
    # hull baseline (deterministic, seedless)
    def hull_scores(bt, K):
        return hull_sequence(bt.w0, bt.Rl, bt.Rr, bt.rm, K)[1:]
    ti = time.perf_counter()
    evals[("pos1_hull", 0)] = evaluate_arm_on_test(hull_scores, 1.0, te_b)
    meta[("pos1_hull", 0)] = dict(params=0, train_s=0.0, tau=1.0, infer_ms_per_word_batched_K8=1000 * (time.perf_counter() - ti) / len(te_items))
    oracle_ms = {L: 1000 * float(np.mean(v)) for L, v in oracle_t.items()}
    return dict(splits=splits, evals=evals, meta=meta, roots=roots, seeds=list(seeds), oracle_ms=oracle_ms,
                hull_viol=hull_viol, n_words=len(te_items), steps=steps)


def summarize(res, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    roots, seeds, evals = res["roots"], res["seeds"], res["evals"]
    arms = sorted({a for a, _ in evals}, key=lambda a: (a == "pos1_hull", a))
    seedset = {a: sorted(s for aa, s in evals if aa == a) for a in arms}
    Ls_in, Ls_all = (8, 10, 12), TEST_LS
    subsets = {
        "indist_K3_L<=12": (Ls_in, (1, 2, 3)),
        "indist_K3only_L<=12": (Ls_in, (3,)),
        "extK_K8_L<=12": (Ls_in, (8,)),
        "extK_K4to8_L<=12": (Ls_in, (4, 5, 6, 7, 8)),
        "extL_L24_K3": ((24,), (3,)),
        "extL_L24_K8": ((24,), (8,)),
        "extLK_L16to24_K4to8": ((16, 20, 24), (4, 5, 6, 7, 8)),
    }
    # per (arm, seed, subset) -> per-root dict
    per = {(a, s, sn): pool(evals[(a, s)], roots, Ls, ks) for a in arms for s in seedset[a] for sn, (Ls, ks) in subsets.items()}

    def seed_avg_root(a, sn, key):
        return np.mean([root_vec(per[(a, s, sn)], key, roots) for s in seedset[a]], axis=0)

    summary = {"subsets": {}, "contrasts": {}, "curves": {}, "cost": {}, "hull_violations_test": res["hull_viol"],
               "oracle_ms_per_word_K8": res["oracle_ms"], "training_steps": res["steps"]}
    for sn in subsets:
        summary["subsets"][sn] = {}
        for a in arms:
            d = {}
            for key in ("recall", "prec", "nrecall", "ap", "aph", "wfn_rate", "over"):
                seed_means = np.array([np.nanmean(root_vec(per[(a, s, sn)], key, roots)) for s in seedset[a]])
                d[key] = dict(mean=float(np.nanmean(seed_means)), sd=float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0)
            d["strict_fn_total"] = int(sum(sum(per[(a, s, sn)][r]["sfn"] for r in roots) for s in seedset[a]))
            summary["subsets"][sn][a] = d

    pairs = [("rec_prior", "rec_noprior"), ("rec_prior_hs", "rec_noprior_hs"), ("rec_noprior", "nonrec_param"),
             ("rec_noprior", "nonrec_compute"), ("rec_prior", "nonrec_param"), ("rec_prior", "nonrec_compute"),
             ("rec_noprior_masked", "nonrec_param_masked"), ("rec_prior_hs", "rec_prior"), ("rec_prior_masked", "rec_prior")]
    for sn in ("indist_K3only_L<=12", "extK_K8_L<=12", "extL_L24_K3", "extL_L24_K8", "extLK_L16to24_K4to8"):
        for (a, b) in pairs:
            if a not in seedset or b not in seedset or not seedset[a] or not seedset[b]:
                continue
            for key in ("ap", "recall", "prec", "aph"):
                rd = seed_avg_root(a, sn, key) - seed_avg_root(b, sn, key)
                m, lo, hi = boot_ci(rd)
                common = sorted(set(seedset[a]) & set(seedset[b]))
                sd = np.array([np.nanmean(root_vec(per[(a, s, sn)], key, roots)) - np.nanmean(root_vec(per[(b, s, sn)], key, roots)) for s in common])
                sm, slo, shi = t_ci(sd)
                summary["contrasts"][f"{sn}|{a}-{b}|{key}"] = dict(
                    root_level=dict(mean=m, ci95=[lo, hi]), seed_level=dict(mean=sm, ci95=[slo, shi], n_seeds=len(common)))
    # in-hull AP lift over hull precision (H6 iii): learned aph vs hull prec in same subset
    for sn in ("indist_K3only_L<=12", "extK_K8_L<=12", "extL_L24_K3", "extL_L24_K8"):
        for a in ("rec_prior_hs", "rec_noprior_hs", "rec_prior", "rec_noprior", "nonrec_param", "nonrec_compute"):
            lift = seed_avg_root(a, sn, "aph") - root_vec(per[("pos1_hull", 0, sn)], "aph", roots)
            m, lo, hi = boot_ci(lift)
            summary["contrasts"][f"{sn}|{a}_in_hull_AP-hull_AP|aph"] = dict(root_level=dict(mean=m, ci95=[lo, hi]))
    # curves
    ks = list(range(1, 9))
    curves = {"K": {}, "L_K3": {}, "L_K8": {}}
    for a in arms:
        for key in ("recall", "prec", "ap"):
            curves["K"][f"{a}|{key}"] = [float(np.nanmean([np.nanmean(root_vec(pool(evals[(a, s)], roots, Ls_in, (k,)), key, roots)) for s in seedset[a]])) for k in ks]
            curves["L_K3"][f"{a}|{key}"] = [float(np.nanmean([np.nanmean(root_vec(pool(evals[(a, s)], roots, (L,), (3,)), key, roots)) for s in seedset[a]])) for L in Ls_all]
            curves["L_K8"][f"{a}|{key}"] = [float(np.nanmean([np.nanmean(root_vec(pool(evals[(a, s)], roots, (L,), (8,)), key, roots)) for s in seedset[a]])) for L in Ls_all]
    summary["curves"] = curves
    for a in arms:
        ms = [res["meta"][(a, s)] for s in seedset[a]]
        summary["cost"][a] = dict(
            params=ms[0]["params"], train_s_mean=float(np.mean([m["train_s"] for m in ms])),
            tau_mean=float(np.mean([m["tau"] for m in ms])),
            infer_ms_per_word_batched_K8=float(np.mean([m["infer_ms_per_word_batched_K8"] for m in ms])),
            infer_ms_per_word_single_K8={str(L): float(np.mean([m["infer_ms_per_word_single_K8"][L] for m in ms])) for L in Ls_all}
            if "infer_ms_per_word_single_K8" in ms[0] else None)
    # seed-level raw table
    summary["per_seed"] = {f"{a}|seed{s}": {sn: {k: float(np.nanmean(root_vec(per[(a, s, sn)], k, roots))) for k in ("recall", "prec", "ap", "aph")}
                                            for sn in subsets} for a in arms for s in seedset[a]}
    (out_dir / "learned_transfer_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    # figures
    main_arms = [a for a in arms if a in ("rec_prior", "rec_noprior", "nonrec_param", "nonrec_compute", "rec_prior_hs", "pos1_hull")]
    for fname, ck, xs, xl, ttl in (("fig_extrapolation_K.svg", "K", ks, "horizon K (L 8-12)", "Extrapolation in horizon K (train K<=3)"),
                                   ("fig_extrapolation_L_K3.svg", "L_K3", list(Ls_all), "word length L (K=3)", "Extrapolation in word length L at K=3 (train L 8-12)"),
                                   ("fig_extrapolation_L_K8.svg", "L_K8", list(Ls_all), "word length L (K=8)", "Joint extrapolation: L at K=8")):
        panels = {key: {a: curves[ck][f"{a}|{key}"] for a in main_arms} for key in ("recall", "prec", "ap")}
        svg_panels(out_dir / fname, ttl, xl, xs, panels)
    return summary


def main():
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "shard", "merge"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="shard: single training seed")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shards", default=str(HERE / "results" / "lbf19" / "learned_transfer" / "_shards"))
    ap.add_argument("--out", default=str(HERE / "results" / "lbf19" / "learned_transfer"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = Path(a.shards)
    log = lambda *x: print(*x, flush=True)
    if a.stage == "shard":
        shards.mkdir(parents=True, exist_ok=True)
        res = run_experiment(out, quick=a.quick, seeds=(a.seed,), threads=a.threads, log=log)
        with open(shards / f"shard_seed{a.seed}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        return
    if a.stage == "merge":
        parts = [pickle.load(open(p, "rb")) for p in sorted(shards.glob("shard_seed*.pkl"))]
        res = parts[0]
        for p in parts[1:]:
            for k, v in p["evals"].items():
                if k[0] != "pos1_hull":
                    res["evals"][k] = v
                    res["meta"][k] = p["meta"][k]
        res["seeds"] = sorted({s for (_, s) in res["evals"] if _ != "pos1_hull"})
    else:
        res = run_experiment(out, quick=a.quick, threads=a.threads, log=log)
    summ = summarize(res, out)
    print(json.dumps({k: summ["subsets"][k] for k in ("indist_K3only_L<=12", "extK_K8_L<=12", "extL_L24_K3")}, indent=1)[:3000])


if __name__ == "__main__":
    main()