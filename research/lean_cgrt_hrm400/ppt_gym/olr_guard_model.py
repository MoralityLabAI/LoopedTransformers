"""Tiny trainable OLR guard-scoring reference and equally informed controls.

This is a developmental two-candidate policy for native Lean arithmetic guards.
A learned score never certifies a guard; Lean is still the sole proof authority.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GuardScorer(nn.Module):
    """Tied message-passing scorer over two candidate obligations.

    Input: (B, 2, 3) with separately visible numerals [lhs_a,lhs_b,rhs].
    No arithmetic relation, winning action, verifier label, or reference script
    is precomputed for the learned feature channel.

    mode=cross: previous-pass opposite candidate message is available;
    mode=isolated: message input is zeroed but *same parameters* are used.
    mode=direct: no recurrence (diagnostic lower-compute control).

    The evidence readout is initialized to 'unknown' at a new proof root.
    Status updates from a live verifier are a future separate experiment.
    """

    def __init__(self, width: int = 32, *, mode: str = "cross"):
        super().__init__()
        if width < 8 or width > 128 or mode not in ("cross", "isolated", "direct"):
            raise ValueError("invalid guard architecture")
        self.width, self.mode = width, mode
        self.embed = nn.Sequential(nn.Linear(3, width), nn.GELU(),
                                   nn.Linear(width, width), nn.LayerNorm(width))
        self.role = nn.Parameter(torch.zeros(1, 2, width))
        self.unknown_evidence = nn.Parameter(torch.zeros(1, 1, width))
        self.core = nn.Sequential(nn.Linear(4*width, 2*width), nn.Tanh(),
                                  nn.Linear(2*width, width))
        self.gate = nn.Linear(4*width, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(2*width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def forward(self, x: torch.Tensor, *, depth: int = 3,
                return_states: bool = False):
        if x.ndim != 3 or x.shape[1:] != (2, 3) or not 1 <= depth <= 16:
            raise ValueError("features must be B x 2 x 3 and depth 1..16")
        context = self.embed(x)
        slots = context + self.role
        states = []
        for _ in range(depth if self.mode != "direct" else 0):
            other = slots.flip(1) if self.mode == "cross" else torch.zeros_like(slots)
            evidence = self.unknown_evidence.expand_as(slots)
            packets = torch.cat((context, slots, other, evidence), -1)
            proposal = self.core(packets)
            gate = torch.sigmoid(self.gate(packets))
            slots = self.norm(slots + .25 * gate * proposal)
            states.append(slots)
        logits = self.head(torch.cat((context, slots), -1)).squeeze(-1)
        return (logits, states) if return_states else logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ranking_loss(logits: torch.Tensor, labels: torch.Tensor):
    if logits.ndim != 2 or logits.shape[-1] != 2 or labels.shape != logits.shape:
        raise ValueError("malformed paired tactic ranking batch")
    if not bool((labels.sum(-1) == 1).all()):
        raise ValueError("native training pairs must contain exactly one completed proof")
    return F.cross_entropy(logits, labels.argmax(-1))


def choose(model: GuardScorer, features, *, depth: int = 3) -> int:
    """Only a public feature packet, never a target label, is model-visible."""
    with torch.no_grad():
        batch = torch.as_tensor(features, dtype=torch.float32).reshape(1, 2, 3)
        return int(model(batch, depth=depth)[0].argmax().item())


def exact_arithmetic_reference(features) -> int:
    """Explicit host arithmetic oracle for a strict best-possible comparator.

    This control deliberately has access to the exact '+' semantics. It must
    never be presented as an OLR learned effect.
    """
    for i, row in enumerate(features):
        a, b, target = row
        if abs(40*a + 40*b - 80*target) < 1e-5:
            return i
    raise ValueError("no provable candidate in the exact arithmetic reference")
