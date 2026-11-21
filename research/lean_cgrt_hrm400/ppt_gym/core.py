"""Small, inspectable PPT-A/PPT-B recurrent specialists for checked Lean tactic pools.

This reference implementation is CPU-first and intentionally limited to ranking
an existing, frozen candidate menu. Lean remains the only proof authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

ROLES = ("GOAL", "CONTEXT", "LEMMA", "PLAN", "FRONTIER", "ERROR", "VALUE", "HALT")
PAD = 256


@dataclass(frozen=True)
class Config:
    plan: Literal["a", "b"] = "a"
    width: int = 48
    heads: int = 4
    slots: int = 8
    max_bytes: int = 160
    mlp_ratio: int = 2

    def validate(self):
        if self.plan not in ("a", "b") or self.slots != len(ROLES):
            raise ValueError("use plan a|b and exactly eight typed slots")
        if not (16 <= self.width <= 768 and self.width % self.heads == 0):
            raise ValueError("width must be 16..768 and divisible by heads")
        if not (32 <= self.max_bytes <= 384) or not (1 <= self.mlp_ratio <= 4):
            raise ValueError("invalid text or MLP budget")
        return self


class ByteEncoder(nn.Module):
    """Byte-level local convolution and masked pooling; no external tokenizer."""

    def __init__(self, d: int, max_bytes: int):
        super().__init__()
        self.max_bytes = max_bytes
        self.embed = nn.Embedding(257, d, padding_idx=PAD)
        self.conv = nn.Conv1d(d, d, 3, padding=1)
        self.norm = nn.LayerNorm(d)

    def forward(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("empty text batch")
        device = self.embed.weight.device
        ids = torch.full((len(texts), self.max_bytes), PAD, device=device, dtype=torch.long)
        lengths = []
        for i, text in enumerate(texts):
            b = text.encode("utf-8")[-self.max_bytes:]
            if not b:
                b = b" "
            ids[i, :len(b)] = torch.tensor(list(b), device=device)
            lengths.append(len(b))
        mask = (ids != PAD).unsqueeze(-1)
        x = self.embed(ids)
        h = F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        pooled = ((x + h) * mask).sum(1) / torch.tensor(lengths, device=device)[:, None]
        return self.norm(pooled)


def block(d: int, heads: int, ratio: int):
    return nn.TransformerEncoderLayer(
        d, heads, dim_feedforward=ratio * d, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    )


class MessageGate(nn.Module):
    """Previous-pass memory is gated at each block input, initially near zero."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, d)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -6.0)

    def forward(self, x: torch.Tensor, previous: torch.Tensor):
        strength = torch.sigmoid(self.gate(torch.cat([x, previous], -1)))
        return strength * self.proj(previous)


class PlasticProofTransformer(nn.Module):
    """Typed workspace with a tied Plan A cell or two tied Plan B middle blocks."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg.validate()
        d = cfg.width
        self.text = ByteEncoder(d, cfg.max_bytes)
        self.roles = nn.Parameter(torch.randn(cfg.slots, d) * 0.015)
        self.context = nn.Linear(2 * d + 1, d)
        self.work_norm = nn.LayerNorm(d)
        if cfg.plan == "a":
            self.recurrent = block(d, cfg.heads, cfg.mlp_ratio)
        else:
            self.prelude = block(d, cfg.heads, cfg.mlp_ratio)
            self.middle_a = block(d, cfg.heads, cfg.mlp_ratio)
            self.middle_b = block(d, cfg.heads, cfg.mlp_ratio)
            self.gate_a = MessageGate(d)
            self.gate_b = MessageGate(d)
            self.message_norm = nn.LayerNorm(d)
            self.coda = block(d, cfg.heads, cfg.mlp_ratio)
        self.feature = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh())
        self.policy = nn.Linear(d, 1)
        self.outcome = nn.Linear(d, 1)
        self.continue_head = nn.Linear(d, 1)

    def forward(
        self, goal: str, candidates: tuple[str, ...] | list[str],
        *, depth: int = 2, checks_remaining: int = 4,
        return_all: bool = False, message_mode: str = "normal",
    ) -> list[dict[str, torch.Tensor]]:
        if depth < 1 or depth > 16 or not 1 <= checks_remaining <= 8:
            raise ValueError("depth=1..16 and checks_remaining=1..8")
        if not candidates or len(candidates) > 64 or message_mode not in ("normal", "zero", "roll"):
            raise ValueError("candidate menu 1..64; message mode normal|zero|roll")
        goal_emb = self.text([goal])[0]
        cand_emb = self.text(list(candidates))
        context = self.context(torch.cat([
            goal_emb, cand_emb.mean(0), goal_emb.new_tensor([checks_remaining / 8.0]),
        ]))
        z = self.roles.unsqueeze(0) + context[None, None, :]
        z = z.clone()
        z[:, 0] = z[:, 0] + goal_emb
        z[:, 4] = z[:, 4] + cand_emb.mean(0)
        z = self.work_norm(z)
        if self.cfg.plan == "b":
            z = self.prelude(z)
            previous = torch.zeros_like(z)
        outputs = []
        for k in range(depth):
            if self.cfg.plan == "a":
                z = self.recurrent(z)
            else:
                msg = previous
                if message_mode == "zero":
                    msg = torch.zeros_like(msg)
                elif message_mode == "roll":
                    msg = msg.roll(1, dims=1)
                a = self.middle_a(z + self.gate_a(z, msg))
                b = self.middle_b(a + self.gate_b(a, msg))
                previous = self.message_norm((a + b) * 0.5)
                z = b
            if return_all or k == depth - 1:
                # Coda has its own cost; all-pass supervision calls it at every visit.
                read = self.coda(z) if self.cfg.plan == "b" else z
                anchor = read[:, 0, :].expand(len(candidates), -1)
                feat = self.feature(torch.cat((anchor, cand_emb, anchor * cand_emb), -1))
                outputs.append({
                    "logits": self.policy(feat).squeeze(-1),
                    "outcome": self.outcome(feat).squeeze(-1),
                    "features": feat,
                    "continue_logit": self.continue_head(read[:, 7, :]).flatten()[0],
                    "pass": torch.tensor(k + 1, device=feat.device),
                    "state": read,
                })
        return outputs

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture(self):
        return {**asdict(self.cfg), "parameter_count": self.parameter_count()}


def supervised_loss(outputs: list[dict], labels: tuple[int, ...]):
    """Binary tactic results and witness-ranking loss with censored labels excluded."""
    y = torch.tensor(labels, dtype=torch.float32, device=outputs[0]["logits"].device)
    known = y >= 0
    if not bool(known.any()):
        raise ValueError("a training case needs at least one definitive tactic verdict")
    loss = torch.zeros((), device=y.device)
    for out in outputs:
        outcome = F.binary_cross_entropy_with_logits(out["outcome"][known], y[known])
        logits = out["logits"][known]
        success = y[known] == 1
        ranking = (torch.logsumexp(logits, 0) - torch.logsumexp(logits[success], 0)
                   if bool(success.any()) else logits.sum() * 0.0)
        loss = loss + outcome + ranking
    return loss / len(outputs)


@torch.no_grad()
def select_candidate(logits: torch.Tensor, labels: tuple[int, ...]):
    """Return index and offline certification verdict. The latter is evaluator-only."""
    index = int(logits.argmax().item())
    return index, labels[index] == 1


class FastHead:
    """Independent, CPU-resident per-task logistic head updated only on support receipts."""

    def __init__(self, width: int, learning_rate: float = 1.0):
        self.weights = torch.zeros(width, dtype=torch.float32, device="cpu")
        self.lr = float(learning_rate)
        self.updates = 0

    def predict(self, features: torch.Tensor):
        h = F.normalize(features.detach().float().to("cpu"), dim=-1)
        return (h @ self.weights).to(features.device)

    def update(self, feature: torch.Tensor, verdict: int):
        if verdict not in (0, 1):
            raise ValueError("only definitive verifier outcomes may update fast weights")
        h = F.normalize(feature.detach().float().to("cpu").reshape(-1), dim=0)
        p = torch.sigmoid(h @ self.weights)
        self.weights.add_(self.lr * (float(verdict) - p) * h)
        self.weights.clamp_(-10.0, 10.0)
        if not torch.isfinite(self.weights).all():
            raise FloatingPointError("nonfinite online weights")
        self.updates += 1


class RetrievalControl:
    """Same support evidence, similarity readout only, no neural parameter update."""

    def __init__(self, width: int, limit: int = 32):
        self.width = width
        self.limit = limit
        self.items: list[tuple[torch.Tensor, int]] = []

    def update(self, feature: torch.Tensor, verdict: int):
        if verdict not in (0, 1):
            raise ValueError("retrieval accepts definitive support receipts only")
        h = F.normalize(feature.detach().float().cpu().reshape(-1), dim=0)
        self.items = (self.items + [(h, verdict)])[-self.limit:]

    def predict(self, features: torch.Tensor):
        if not self.items:
            return features.new_zeros((len(features),))
        h = F.normalize(features.detach().float().cpu(), dim=-1)
        keys = torch.stack([k for k, _ in self.items])
        values = torch.tensor([2 * y - 1 for _, y in self.items], dtype=torch.float32)
        sim = h @ keys.T
        return ((sim * values).sum(-1) / sim.abs().sum(-1).clamp_min(1e-6)).to(features.device)
