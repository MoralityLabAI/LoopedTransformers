"""Training regimes for the looped model."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import LoopedTransformer

REGIMES = ("final", "random_intermediate", "random_exit")


def regime_loss(ce, regime, gen):
    """ce: [K,B] per-loop cross-entropy.

    final               : supervise only loop K.
    random_intermediate : each sample supervises ONE uniformly drawn loop j in 1..K.
    random_exit         : each sample draws an exit depth e (log-uniform over 1..K) and is
                          supervised at EVERY loop 1..e (mean CE over the prefix of loops).
    """
    K, B = ce.shape
    if regime == "final":
        return ce[-1].mean()
    if regime == "random_intermediate":
        j = torch.randint(0, K, (B,), device=ce.device, generator=gen)
        return ce[j, torch.arange(B, device=ce.device)].mean()
    if regime == "random_exit":
        u = torch.rand(B, device=ce.device, generator=gen)
        e = torch.exp(u * math.log(K)).ceil().clamp(1, K)
        w = (torch.arange(1, K + 1, device=ce.device)[:, None] <= e[None]).float() / e[None]
        return (ce * w).sum(0).mean()
    raise ValueError(regime)


def train_model(task, regime, seed, cfg, device="cuda", log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    K, bs = cfg["K"], cfg["batch"]
    model = LoopedTransformer(task.vocab, task.n_classes, d=cfg["d"], heads=cfg["heads"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.98))
    steps, warm = cfg["steps"], max(1, cfg["steps"] // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))) * 0.95 + 0.05))
    lens = list(cfg["train_lengths"])
    t0 = time.time()
    for step in range(steps):
        N = int(rng.choice(lens))
        x, y, _ = task.sample(rng, bs, N)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        logits = model.run(x, K)  # [K,B,C]
        ce = F.cross_entropy(logits.reshape(K * bs, -1), y.repeat(K), reduction="none").view(K, bs)
        loss = regime_loss(ce, regime, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
            with torch.no_grad():
                acc_last = (logits[-1].argmax(-1) == y).float().mean().item()
            log(f"  [{task.name}/{regime}/s{seed}] step {step:5d} loss {loss.item():.4f} acc@K {acc_last:.3f} ({time.time()-t0:.0f}s)")
    model.eval()
    return model
