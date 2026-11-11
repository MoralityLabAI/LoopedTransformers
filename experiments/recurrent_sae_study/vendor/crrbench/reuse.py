"""Reuse: continue reasoning from a saved / frozen hidden state.

Two readings of "reuse", both measured at a per-sample freeze loop  k*_i + delay :

 A. native resume (zero-shot): paste the frozen state into the first T positions of a
    longer sequence, append fresh embeddings for `s` new symbols, run the SAME frozen
    recurrent core for M more loops and read the answer.  Needs the core's own dynamics
    to accept a saved state; compared with a from-scratch run given the same total loops.
 B. probe resume: a small trained MLP receives the frozen pooled state [h_last, mean(h)]
    plus the `s` new symbols and must output the machine state after them.  This is an
    interface-free test of whether the information needed to continue is still present.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def freeze_states(model, task, split, snap_k, device="cuda", bs=500):
    """Full state at per-sample loop snap_k (1-indexed).  Returns list of tensors per length group
    (states, and pooled [last, mean] features for every sample in split order)."""
    pooled, full, off = [], [], 0
    for x in split["x"]:
        b = x.shape[0]
        k = torch.from_numpy(snap_k[off:off + b]).to(device)
        xt = torch.from_numpy(x).to(device)
        Kmax = int(k.max().item())
        _, snap = model.run(xt, Kmax, snap_at=k)
        pooled.append(torch.cat([snap[:, -1], snap.mean(1)], -1).cpu())
        full.append(snap.cpu())
        off += b
    return full, torch.cat(pooled)


@torch.no_grad()
def native_resume(model, task, split, full_states, rng, suffix_len=4, M=8, Kfresh=32, device="cuda",
                  fresh_depth=None):
    """Returns (acc_resume[M], acc_fresh) over the rows of split (resume & fresh use same suffixes).

    fresh_depth [n]: per-row number of loops granted to the from-scratch reference (default: M)."""
    n = len(split["y"])
    off, correct_r, correct_f = 0, [], []
    for x, H in zip(split["x"], full_states):
        b = x.shape[0]
        y, ctx = split["y"][off:off + b], split["ctx"][off:off + b]
        suf = task.sample_suffix(rng, b, suffix_len)
        target = task.extend(ctx, y, suf)
        st = torch.from_numpy(suf).to(device)
        h = torch.cat([H.to(device), model.embed(st)], 1)
        lg = model.run_from(h, M)                         # [M,b,C]
        correct_r.append((lg.argmax(-1).cpu().numpy() == target[None]).T)
        xf = torch.cat([torch.from_numpy(x).to(device), st], 1)
        lgf = model.run(xf, Kfresh)                       # [Kfresh,b,C]
        pf = lgf.argmax(-1).cpu().numpy()                 # [Kfresh,b]
        fd = np.clip(fresh_depth[off:off + b], 1, Kfresh) if fresh_depth is not None else np.full(b, M)
        correct_f.append(pf[fd - 1, np.arange(b)] == target)
        off += b
    cr = np.concatenate(correct_r, 0)  # [n,M]
    cf = np.concatenate(correct_f, 0)  # [n]
    return cr, cf


class ReuseProbe(nn.Module):
    def __init__(self, d_in, n_symbols, n_ctx, n_classes, suffix_len=4, hidden=512, emb=32):
        super().__init__()
        self.se = nn.Embedding(n_symbols, emb)
        self.ce = nn.Embedding(max(1, n_ctx), 16)
        self.net = nn.Sequential(nn.Linear(d_in + 16 + suffix_len * emb, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, z, ctx, suf):
        return self.net(torch.cat([z, self.ce(ctx), self.se(suf).flatten(1)], -1))


def fit_probe(task, z_tr, y_tr, ctx_tr, rng, suffix_len=4, steps=400, bs=512, device="cuda", lr=2e-3, seed=0):
    torch.manual_seed(seed)
    z_tr = z_tr.float().to(device)
    mu, sd = z_tr.mean(0, keepdim=True), z_tr.std(0, keepdim=True) + 1e-6
    probe = ReuseProbe(z_tr.shape[1], task.n_symbols, task.n_ctx, task.n_classes, suffix_len).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    n = z_tr.shape[0]
    ctx_t = torch.from_numpy(ctx_tr).to(device)
    for s in range(steps):
        idx = rng.integers(0, n, size=bs)
        suf = task.sample_suffix(rng, bs, suffix_len)
        tgt = torch.from_numpy(task.extend(ctx_tr[idx], y_tr[idx], suf)).to(device)
        it = torch.from_numpy(idx).to(device)
        loss = F.cross_entropy(probe((z_tr[it] - mu) / sd, ctx_t[it], torch.from_numpy(suf).to(device)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    return probe, mu, sd


@torch.no_grad()
def eval_probe(task, probe, mu, sd, z, y, ctx, rng, suffix_len=4, device="cuda", reps=2):
    z = ((z.float().to(device)) - mu) / sd
    out = []
    for _ in range(reps):
        suf = task.sample_suffix(rng, len(y), suffix_len)
        tgt = task.extend(ctx, y, suf)
        pred = probe(z, torch.from_numpy(ctx).to(device), torch.from_numpy(suf).to(device)).argmax(-1).cpu().numpy()
        out.append(pred == tgt)
    return np.mean(out, 0)  # per-sample accuracy in [0,1]
