"""Two-pass math-informed latent-workspace learning gym (local pilot)."""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate(batch: int, size: int, device: str = "cpu", seed: int | None = None):
    """Contractive random Bellman systems; all labels computed from exact dense solves."""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    p = torch.rand(batch, size, size, generator=gen)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    gamma = torch.empty(batch, 1, 1).uniform_(0.45, 0.90, generator=gen)
    m = gamma * p
    b = torch.randn(batch, size, generator=gen) / math.sqrt(size)
    target = torch.linalg.solve(torch.eye(size).unsqueeze(0) - m, b.unsqueeze(-1)).squeeze(-1)
    return m.to(device), b.to(device), target.to(device), gamma.squeeze(-1).squeeze(-1).to(device)


def residual(m, b, z):
    return b + torch.bmm(m, z.unsqueeze(-1)).squeeze(-1) - z


def residual_certificate(m, b, z):
    """Infinity-norm bound based on induced matrix infinity norm."""
    q = m.abs().sum(-1).amax(-1)
    r = residual(m, b, z).abs().amax(-1)
    return r / (1.0 - q).clamp_min(1e-8)


class TwoPassSkill(nn.Module):
    """Shared-weight token-mixing block with persistent latent workspace."""
    def __init__(self, size: int = 8, width: int = 48, heads: int = 4, visits: int = 2):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.size, self.width, self.visits = size, width, visits
        self.embed = nn.Linear(4, width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.readout = nn.Linear(width, 1)
        self.latent_gate = nn.Linear(width, width)

    def forward(self, m, b, visits: int | None = None, reset_second: bool = False):
        count = self.visits if visits is None else visits
        z = torch.zeros_like(b)
        hidden = torch.zeros(*b.shape, self.width, device=b.device, dtype=b.dtype)
        states = []
        diag = m.diagonal(dim1=-2, dim2=-1)
        row_norm = m.abs().sum(-1)
        for t in range(count):
            r = residual(m, b, z)
            tokens = torch.stack((b, z, r, diag + row_norm), dim=-1)
            x = self.embed(tokens)
            if reset_second and t == 1:
                hidden = torch.zeros_like(hidden)
            q = self.norm(hidden + x)
            a, _ = self.attn(q, q, q, need_weights=False)
            hidden = self.norm(q + a)
            hidden = self.norm(hidden + self.ff(hidden))
            correction = self.readout(hidden).squeeze(-1)
            # The gate preserves exact fixed points in real arithmetic.
            z = z + r + residual(m, b, z).abs().amax(-1, keepdim=True) * correction
            states.append(z)
        return z, states


def paths(run_dir):
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root, root / "last.pt", root / "metrics.jsonl"


def train(args):
    seed_all(args.seed)
    root, checkpoint, metrics = paths(args.run_dir)
    model = TwoPassSkill(size=args.size, width=args.width, visits=args.visits).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    start = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if (state["size"], state["width"], state["visits"], state["seed"]) != (args.size, args.width, args.visits, args.seed):
            raise ValueError("checkpoint configuration differs from requested run")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if args.device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    def save(step):
        tmp = checkpoint.with_suffix(".tmp")
        torch.save({"step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "size": args.size, "width": args.width, "visits": args.visits, "seed": args.seed,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None}, tmp)
        os.replace(tmp, checkpoint)
    try:
        for step in range(start, args.steps):
            model.train()
            m, b, y, _ = generate(args.batch, args.size, args.device)
            out, trajectory = model(m, b)
            loss = F.mse_loss(out, y)
            for index, state in enumerate(trajectory):
                loss = loss + 0.15 / (index + 1) * F.mse_loss(state, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step + 1) % args.log_every == 0 or (step + 1) == args.steps:
                row = {"step": step+1, "train_loss": float(loss.item()),
                       "visits": args.visits, "seed": args.seed}
                with metrics.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if (step + 1) % args.save_every == 0:
                save(step + 1)
    except KeyboardInterrupt:
        save(step + 1 if "step" in locals() else start)
        return
    save(args.steps)


@torch.no_grad()
def evaluate(args):
    seed_all(args.seed)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = TwoPassSkill(state["size"], state["width"], visits=state["visits"]).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    m, b, y, _ = generate(args.examples, state["size"], args.device, seed=args.test_seed)
    z, states = model(m, b)
    baseline = torch.zeros_like(b)
    for _ in range(state["visits"]):
        baseline = baseline + residual(m, b, baseline)
    metrics = {
        "trained_steps": state["step"], "model_mse": float(F.mse_loss(z, y)),
        "solver_mse": float(F.mse_loss(baseline, y)),
        "max_error_p90": float(torch.quantile((z-y).abs().amax(-1), .9)),
        "certificate_p90": float(torch.quantile(residual_certificate(m,b,z), .9)),
        "hidden_reset_mse": float(F.mse_loss(model(m,b,reset_second=True)[0], y)),
        "pass_mse": [float(F.mse_loss(v,y)) for v in states],
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def export(args):
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": data["model"], "size": data["size"], "width": data["width"],
                "visits": data["visits"], "step": data["step"]}, out)
    print(out)


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--run-dir", default="two_pass_runs")
    train_p.add_argument("--steps", type=int, default=400)
    train_p.add_argument("--batch", type=int, default=64)
    train_p.add_argument("--size", type=int, default=8)
    train_p.add_argument("--width", type=int, default=48)
    train_p.add_argument("--visits", type=int, default=2)
    train_p.add_argument("--seed", type=int, default=211)
    train_p.add_argument("--lr", type=float, default=0.0005)
    train_p.add_argument("--save-every", type=int, default=100)
    train_p.add_argument("--log-every", type=int, default=50)
    train_p.add_argument("--device", default="cpu")
    train_p.add_argument("--resume", action="store_true")
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--device", default="cpu")
    eval_p.add_argument("--examples", type=int, default=256)
    eval_p.add_argument("--seed", type=int, default=999)
    eval_p.add_argument("--test-seed", type=int, default=91977)
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--out", default="two_pass_runs/module.pt")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    {"train": train, "eval": evaluate, "export": export}[a.cmd](a)
