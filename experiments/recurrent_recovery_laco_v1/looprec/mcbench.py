"""Multiple-choice knowledge benchmarks scored by choice log-likelihood (lm-eval-harness style), offline:
ARC-Easy, ARC-Challenge, PIQA from the local HF cache (fetched by the LaCo reproduction project).
Reports acc and acc_norm (log-lik / #characters of the choice)."""
from __future__ import annotations

import glob
import os
import random

import torch

from .evaluation import token_stats

H = os.path.expanduser("~/.cache/huggingface/hub")


def _first(pat):
    fs = sorted(glob.glob(os.path.join(H, pat)))
    if not fs:
        raise FileNotFoundError(pat)
    return fs[0]


def load_mc(n: int, seed: int = 0) -> dict[str, list[dict]]:
    import pandas as pd
    out = {}
    for name, pat in (("arc_easy", "datasets--allenai--ai2_arc/snapshots/*/ARC-Easy/test-*.parquet"),
                      ("arc_challenge", "datasets--allenai--ai2_arc/snapshots/*/ARC-Challenge/test-*.parquet")):
        df = pd.read_parquet(_first(pat))
        items = []
        for q, ch, key in zip(df["question"], df["choices"], df["answerKey"]):
            labels = list(ch["label"])
            if key not in labels:
                continue
            items.append(dict(ctx=f"Question: {q}\nAnswer:", choices=[" " + str(t) for t in ch["text"]],
                              label=labels.index(key)))
        out[name] = items
    df = pd.read_parquet(_first("datasets--baber--piqa/snapshots/*/piqa_validation.parquet"))
    out["piqa"] = [dict(ctx=f"Question: {g}\nAnswer:", choices=[" " + a, " " + b], label=int(l))
                   for g, a, b, l in zip(df["goal"], df["sol1"], df["sol2"], df["label"])]
    rng = random.Random(seed)
    for k in out:
        rng.shuffle(out[k])
        out[k] = out[k][:n]
    return out


def tokenise_mc(tt, items: list[dict]):
    """-> list of (ids, cont_start) per (item, choice), plus index arrays."""
    ctx = tt.encode_batch([it["ctx"] for it in items])
    seqs, owner, nchar = [], [], []
    for i, (it, c) in enumerate(zip(items, ctx)):
        conts = tt.encode_batch(it["choices"])
        for j, ct in enumerate(conts):
            seqs.append((c + ct, len(c)))
            owner.append((i, j))
            nchar.append(len(it["choices"][j]) - 1)
    return seqs, owner, nchar


@torch.no_grad()
def eval_mc(student, tt, mc: dict, loops, device, bs: int = 48):
    """-> {str(T): {bench: {acc, acc_norm, n}}}"""
    W = student.head_weight()
    res = {str(T): {} for T in loops}
    for name, items in mc.items():
        seqs, owner, nchar = tokenise_mc(tt, items)
        order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]))
        ll = {str(T): torch.zeros(len(seqs)) for T in loops}
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            n = max(len(seqs[k][0]) for k in idx)
            ids = torch.full((len(idx), n), tt.pad_id, dtype=torch.long)
            cm = torch.zeros(len(idx), n, dtype=torch.bool)
            for r, k in enumerate(idx):
                s, st = seqs[k]
                ids[r, :len(s)] = torch.tensor(s)
                cm[r, st:len(s)] = True
            ids, cm = ids.to(device), cm.to(device)
            for T in loops:
                hs, _ = student.forward_hidden(ids, n_loops=T)
                st_ = token_stats(hs, None, ids, cm, W)
                tot = torch.zeros(len(idx), device=device).index_add_(0, st_["rows"], -st_["ce"])
                for r, k in enumerate(idx):
                    ll[str(T)][k] = tot[r].item()
        for T in loops:
            n_it = len(items)
            sc = [[None] * len(it["choices"]) for it in items]
            scn = [[None] * len(it["choices"]) for it in items]
            for k, (i, j) in enumerate(owner):
                sc[i][j] = ll[str(T)][k].item()
                scn[i][j] = sc[i][j] / max(nchar[k], 1)
            acc = sum(max(range(len(s)), key=s.__getitem__) == it["label"] for s, it in zip(sc, items)) / n_it
            accn = sum(max(range(len(s)), key=s.__getitem__) == it["label"] for s, it in zip(scn, items)) / n_it
            res[str(T)][name] = {"acc": acc, "acc_norm": accn, "n": n_it}
    return res
