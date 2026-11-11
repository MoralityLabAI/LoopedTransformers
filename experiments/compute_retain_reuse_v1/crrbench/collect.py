"""Run a trained model on a split and keep everything downstream needs."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def collect_split(model, task, rng, lengths, n_per_len, K, device="cuda", bs=500, keep_z=True):
    """Returns dict of numpy arrays, concatenated over `lengths`.

    x is padded only conceptually: each length group has its own T, so x is a list.
    logits [n,K,C] fp16, z [n,K,d] fp16 (normalised last-token vectors that feed the head).
    """
    xs, ys, ctxs, Ns, logits, zs = [], [], [], [], [], []
    for N in lengths:
        for s in range(0, n_per_len, bs):
            b = min(bs, n_per_len - s)
            x, y, ctx = task.sample(rng, b, N)
            lg, z = model.run(torch.from_numpy(x).to(device), K, keep_z=True)
            logits.append(lg.permute(1, 0, 2).half().cpu())
            if keep_z:
                zs.append(z.permute(1, 0, 2).half().cpu())
            xs.append(x), ys.append(y), ctxs.append(ctx), Ns.append(np.full(b, N))
    out = {"y": np.concatenate(ys), "ctx": np.concatenate(ctxs), "N": np.concatenate(Ns),
           "x": xs, "logits": torch.cat(logits).numpy()}
    out["preds"] = out["logits"].astype(np.float32).argmax(-1)
    out["correct"] = out["preds"] == out["y"][:, None]
    if keep_z:
        out["z"] = torch.cat(zs).numpy()
    return out
