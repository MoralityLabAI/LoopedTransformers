"""TopK sparse autoencoders (Gao et al. 2024) with a dead-latent auxiliary loss.

Inputs are centred and rescaled by dataset statistics that are stored inside the module, so one SAE
can be applied to states from other loops / models and still report errors in the original units.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int
    n_latents: int
    k: int
    seed: int = 0
    steps: int = 3000
    batch: int = 4096
    lr: float = 1e-3
    aux_k: int = 64
    aux_coef: float = 1.0 / 32
    dead_after: int = 200          # steps without firing before a latent counts as dead
    train_tag: str = ""            # free text: which states / split this SAE was trained on


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d, n = cfg.d_in, cfg.n_latents
        g = torch.Generator().manual_seed(cfg.seed)
        dec = torch.randn(n, d, generator=g)
        dec = dec / dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(dec)
        self.W_enc = nn.Parameter(dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("scale", torch.ones(()))

    # ---- data normalisation ------------------------------------------------
    @torch.no_grad()
    def fit_norm(self, X: torch.Tensor):
        self.mu.copy_(X.mean(0))
        self.scale.copy_((X - self.mu).norm(dim=1).mean() / math.sqrt(X.shape[1]) + 1e-8)

    def norm_in(self, x):
        return (x - self.mu) / self.scale

    def norm_out(self, xn):
        return xn * self.scale + self.mu

    # ---- core --------------------------------------------------------------
    def pre_acts(self, xn):
        return (xn - self.b_pre) @ self.W_enc + self.b_enc

    def encode(self, x, normed: bool = False):
        """x -> sparse code z [.., n] (k non-zeros, ReLU'd)."""
        xn = x if normed else self.norm_in(x)
        pre = self.pre_acts(xn)
        v, i = pre.topk(self.cfg.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, i, F.relu(v))
        return z

    def decode_n(self, z):
        return z @ self.W_dec + self.b_pre

    def decode(self, z):
        return self.norm_out(self.decode_n(z))

    def forward(self, x):
        return self.decode(self.encode(x))

    @torch.no_grad()
    def renorm_decoder(self):
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def project_decoder_grad(self):
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data
        self.W_dec.grad -= (self.W_dec.grad * w).sum(1, keepdim=True) * w

    # ---- persistence -------------------------------------------------------
    def save(self, path_prefix: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save(self.state_dict(), path_prefix + ".pt")
        json.dump({**asdict(self.cfg), **(extra or {})}, open(path_prefix + ".json", "w"), indent=2)

    @classmethod
    def load(cls, path_prefix: str, device="cpu") -> "TopKSAE":
        c = json.load(open(path_prefix + ".json"))
        fields = SAEConfig.__dataclass_fields__
        sae = cls(SAEConfig(**{k: v for k, v in c.items() if k in fields}))
        sae.load_state_dict(torch.load(path_prefix + ".pt", map_location="cpu"))
        return sae.to(device).eval()


def train_sae(X: torch.Tensor, cfg: SAEConfig, X_val: torch.Tensor | None = None, device="cpu", log=print):
    """X: [N, d] float tensor (raw units).  Returns (sae, history)."""
    torch.manual_seed(cfg.seed)
    sae = TopKSAE(cfg).to(device)
    X = X.to(device)
    sae.fit_norm(X)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    warm, N = max(1, cfg.steps // 20), X.shape[0]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps)))))
    last_fire = torch.zeros(cfg.n_latents, device=device)
    hist, t0 = [], time.time()
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for step in range(cfg.steps):
        idx = torch.randint(0, N, (cfg.batch,), generator=g).to(device)
        xn = sae.norm_in(X[idx])
        pre = sae.pre_acts(xn)
        v, i = pre.topk(cfg.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, i, F.relu(v))
        xh = sae.decode_n(z)
        err = xn - xh
        denom = xn.pow(2).sum(-1).mean()
        loss_main = err.pow(2).sum(-1).mean() / denom
        fired = (z > 0).any(0)
        last_fire = torch.where(fired, torch.zeros_like(last_fire), last_fire + 1)
        dead = last_fire > cfg.dead_after
        loss_aux = torch.zeros((), device=device)
        if dead.any():
            kk = min(cfg.aux_k, int(dead.sum()))
            pre_d = pre.masked_fill(~dead, float("-inf"))
            vd, idd = pre_d.topk(kk, dim=-1)
            zd = torch.zeros_like(pre).scatter_(-1, idd, F.relu(vd))
            eh = zd @ sae.W_dec
            loss_aux = (err.detach() - eh).pow(2).sum(-1).mean() / (err.detach().pow(2).sum(-1).mean() + 1e-8)
        loss = loss_main + cfg.aux_coef * loss_aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_grad()
        opt.step()
        sched.step()
        sae.renorm_decoder()
        if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps - 1:
            rec = {"step": step, "fvu_batch": float(loss_main),
                   "dead_frac": float(dead.float().mean()), "aux": float(loss_aux), "t": time.time() - t0}
            if X_val is not None:
                rec["val_nmse"] = float(recon_stats(sae, X_val, device=device)["nmse"])
            hist.append(rec)
            log("  sae " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    sae.eval()
    return sae, hist


@torch.no_grad()
def recon_stats(sae: TopKSAE, X: torch.Tensor, device="cpu", chunk=65536, ref_mu: torch.Tensor | None = None):
    """NMSE = sum ||x - x_hat||^2 / sum ||x - mean||^2 (mean = ref_mu or the evaluated set's own mean).
    Also returns per-row squared errors and mean L0."""
    sae.eval()
    errs, l0 = [], []
    mu = ref_mu if ref_mu is not None else X.mean(0)
    den = (X - mu).pow(2).sum().item()
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk].to(device)
        z = sae.encode(x)
        xh = sae.decode(z)
        errs.append((x - xh).pow(2).sum(-1).cpu())
        l0.append((z > 0).float().sum(-1).cpu())
    e = torch.cat(errs)
    return {"nmse": e.sum().item() / max(den, 1e-12), "row_err": e, "l0": torch.cat(l0).mean().item()}


@torch.no_grad()
def encode_all(sae: TopKSAE, X: torch.Tensor, device="cpu", chunk=65536) -> torch.Tensor:
    out = []
    for i in range(0, X.shape[0], chunk):
        out.append(sae.encode(X[i:i + chunk].to(device)).cpu())
    return torch.cat(out)
