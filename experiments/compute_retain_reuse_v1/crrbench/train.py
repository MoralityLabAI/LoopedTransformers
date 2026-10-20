"""Training regimes for the looped model (optionally one CUDA graph per sequence length).

Every regime is expressed as per-(loop, sample) weights w[k,b] on the per-loop cross-entropy,
  loss = mean_b sum_k w[k,b] * CE[k,b],
which lets the whole forward/backward/optimizer step be captured as a CUDA graph (a big win on
a GPU that is shared with other jobs: one launch per step instead of thousands of tiny kernels).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import LoopedTransformer

REGIMES = ("final", "random_intermediate", "random_exit")


def regime_weights(regime, K, B, device, gen=None):
    """w[k,b] (k = 0..K-1 is loop k+1).

    final               : all weight on loop K.
    random_intermediate : each sample supervises ONE uniformly drawn loop j in 1..K.
    random_exit         : each sample draws an exit depth e (log-uniform over 1..K) and is
                          supervised at EVERY loop 1..e (mean CE over that prefix of loops).
    """
    ar = torch.arange(1, K + 1, device=device)[:, None]
    if regime == "final":
        return (ar == K).float().expand(K, B).contiguous()
    if regime == "random_intermediate":
        j = torch.randint(1, K + 1, (B,), device=device, generator=gen)
        return (ar == j[None]).float()
    if regime == "random_exit":
        u = torch.rand(B, device=device, generator=gen)
        e = torch.exp(u * math.log(K)).ceil().clamp(1, K)
        return (ar <= e[None]).float() / e[None]
    raise ValueError(regime)


def _loss(model, x, y, w, K):
    logits = model.run(x, K)                                  # [K,B,C]
    ce = F.cross_entropy(logits.reshape(K * x.shape[0], -1), y.repeat(K), reduction="none").view(K, -1)
    return (ce * w).sum(0).mean(), logits[-1]


def train_model(task, regime, seed, cfg, device="cuda", log=print, graph=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    K, bs = cfg["K"], cfg["batch"]
    model = LoopedTransformer(task.vocab, task.n_classes, d=cfg["d"], heads=cfg["heads"]).to(device)
    use_graph = graph and device == "cuda"
    lr_t = torch.tensor(cfg["lr"], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_t if use_graph else cfg["lr"], weight_decay=0.01,
                            betas=(0.9, 0.98), capturable=use_graph)
    params = [p for p in model.parameters()]
    steps, warm = cfg["steps"], max(1, cfg["steps"] // 20)
    lr_at = lambda s: cfg["lr"] * min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))) * 0.95 + 0.05)
    lens = list(cfg["train_lengths"])
    t0 = time.time()

    graphs = {}
    if use_graph:
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        side = torch.cuda.Stream()
        for N in lens:
            sx = torch.zeros(bs, N + task.prefix, dtype=torch.long, device=device)
            sy = torch.zeros(bs, dtype=torch.long, device=device)
            sw = torch.zeros(K, bs, device=device)
            lr_t.fill_(0.0)

            def one_step():
                opt.zero_grad(set_to_none=True)
                loss, last = _loss(model, sx, sy, sw, K)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
                opt.step()
                return loss, last

            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    one_step()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(g):
                loss, last = one_step()
            graphs[N] = (g, sx, sy, sw, loss, last)
        # undo the junk warm-up updates (lr was 0 so only optimizer statistics moved)
        model.load_state_dict(init_state)
        for st in opt.state.values():
            for v in st.values():
                v.zero_()
        torch.cuda.synchronize()

    for step in range(steps):
        N = int(rng.choice(lens))
        x, y, _ = task.sample(rng, bs, N)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        w = regime_weights(regime, K, bs, device, gen)
        if use_graph:
            g, sx, sy, sw, loss, last = graphs[N]
            sx.copy_(x); sy.copy_(y); sw.copy_(w)
            lr_t.fill_(lr_at(step))
            g.replay()
        else:
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            loss, last = _loss(model, x, y, w, K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
            acc_last = (last.argmax(-1) == (sy if use_graph else y)).float().mean().item()
            log(f"  [{task.name}/{regime}/s{seed}] step {step:5d} loss {loss.item():.4f} acc@K {acc_last:.3f} ({time.time()-t0:.0f}s)")
    model.eval()
    return model
