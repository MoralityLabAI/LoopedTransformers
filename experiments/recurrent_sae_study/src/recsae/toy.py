"""Toy (crrbench) looped-transformer adapter: load, per-loop states, intervened rollouts.

State convention
----------------
h_k  = full residual state after loop k (k = 1..K), shape [B, T, d]
x_k  = h_k[:, -1]   the readout-position vector (this is what the SAEs are trained on)
logit_k = head(ln_f(x_k))

Loop 0 (embeddings) is stored as k = 0 so that `states[:, 0]` is the input state.
"""
from __future__ import annotations

import os

import numpy as np
import torch

import recsae  # noqa: F401
from crrbench.model import LoopedTransformer
from crrbench.tasks import TASKS            # frozen v1 definitions (own checkpoints: dfa = 8 states, cyclic7)
from crrbench2.tasks import TASKS as TASKS2  # sibling's later definitions (dfa5, cyclic5, ...): used for 'sib:' checkpoints
from recsae import ROOT, DATA


SIB_K_TRAIN = {"full": 8, "quick": 8}      # crrbench CONFIGS[profile]["K_train"]


def ckpt_path(name: str) -> str:
    if name.startswith("sib:"):
        return name[4:]
    return os.path.join(DATA, "toy", "ckpt", name + ".pt")


def load_toy(name: str, device="cpu"):
    """name: own checkpoint name, or 'sib:<path to results/<profile>/models/<task>__<regime>__s<seed>.pt>' from the
    crrbench sibling session (bare state_dict; K_train read from the profile directory name)."""
    path = ckpt_path(name)
    ck = torch.load(path, map_location="cpu")
    if name.startswith("sib:"):
        base = os.path.basename(path)[:-3]
        tname, regime, sd = base.split("__")
        profile = os.path.basename(os.path.dirname(os.path.dirname(path)))
        d = ck["emb.weight"].shape[1]
        ck = {"state_dict": ck, "cfg": dict(K=SIB_K_TRAIN.get(profile, 16), d=d, heads=4), "task": tname,
              "regime": regime, "seed": int(sd[1:])}
    cfg = ck["cfg"]
    task = (TASKS2 if name.startswith("sib:") else TASKS)[ck["task"]]()
    m = LoopedTransformer(task.vocab, task.n_classes, d=cfg["d"], heads=cfg["heads"])
    m.load_state_dict(ck["state_dict"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device).eval(), task, ck


@torch.no_grad()
def rollout(model: LoopedTransformer, x: torch.Tensor, K: int, hook=None, start_state=None, start_loop=0):
    """Run loops start_loop+1..K.  Returns dict with
         last  [B, K+1, d]  last-token state after each loop (index 0 = embeddings, or start state)
         mean  [B, K+1, d]  token-mean state
         logits[B, K, C]    readout after loops 1..K (entries < start_loop are NaN)
    `hook(k, h) -> h'` may modify the full state right after loop k (k is 1-indexed) - used for interventions."""
    h = model.embed(x) if start_state is None else start_state
    B = h.shape[0]
    last = torch.zeros(B, K + 1, h.shape[-1], device=h.device)
    mean = torch.zeros_like(last)
    logits = torch.full((B, K, model.n_classes), float("nan"), device=h.device)
    last[:, start_loop], mean[:, start_loop] = h[:, -1], h.mean(1)
    for k in range(start_loop + 1, K + 1):
        h = model.step(h)
        if hook is not None:
            h = hook(k, h)
        last[:, k], mean[:, k] = h[:, -1], h.mean(1)
        logits[:, k - 1] = model.read(h)[0]
    return {"last": last, "mean": mean, "logits": logits, "h": h}


def make_split(task, rng: np.random.Generator, lengths, n_per_len):
    xs, ys, ctxs, Ns = [], [], [], []
    for N in lengths:
        x, y, ctx = task.sample(rng, n_per_len, N)
        xs.append(x), ys.append(y), ctxs.append(ctx), Ns.append(np.full(n_per_len, N))
    return {"x": xs, "y": np.concatenate(ys), "ctx": np.concatenate(ctxs), "N": np.concatenate(Ns)}


@torch.no_grad()
def collect(model, split, K: int, device="cpu", bs=1000):
    """Collect last/mean states and logits for a split.  Returns numpy dict (fp16 states)."""
    L, M, G = [], [], []
    for x in split["x"]:
        for i in range(0, x.shape[0], bs):
            r = rollout(model, torch.from_numpy(x[i:i + bs]).to(device), K)
            L.append(r["last"].half().cpu()), M.append(r["mean"].half().cpu()), G.append(r["logits"].cpu())
    out = {"last": torch.cat(L).numpy(), "mean": torch.cat(M).numpy(), "logits": torch.cat(G).numpy().astype(np.float32),
           "y": split["y"], "ctx": split["ctx"], "N": split["N"]}
    out["pred"] = out["logits"].argmax(-1)
    out["correct"] = out["pred"] == out["y"][:, None]
    return out
