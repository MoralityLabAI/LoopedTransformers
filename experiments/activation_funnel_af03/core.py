"""AF-03: typed, differentiable code interchange in the unchanged AF-02 cell.

This module imports AF-02's hash-pinned TypedLoop and original oracle. It does
not modify the frozen AF-02 implementation, checkpoints or scored outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from experiments.activation_funnel.model import LearnedCodec
from experiments.activation_funnel_af02 import gym
from experiments.activation_funnel_af02.model import NativeTeacher


class TokenCodec(nn.Module):
    """One separately shared encoder/decoder for atoms and for rules."""
    def __init__(self, token_code_width: int):
        super().__init__()
        if token_code_width not in (1, 2):
            raise ValueError('AF-03 registered widths are one or two per token')
        self.token_code_width = token_code_width
        self.code_width = 5 * token_code_width
        self.state_width = 80
        self.atom = LearnedCodec(width=16, code_width=token_code_width)
        self.rule = LearnedCodec(width=16, code_width=token_code_width)

    def encode(self, h):
        tokens = h.reshape(*h.shape[:-1], 5, 16)
        z = torch.cat([self.atom.encode(tokens[..., :3, :]),
                       self.rule.encode(tokens[..., 3:, :])], dim=-2)
        return z.reshape(*h.shape[:-1], self.code_width)

    def decode(self, z):
        tokens = z.reshape(*z.shape[:-1], 5, self.token_code_width)
        h = torch.cat([self.atom.decode(tokens[..., :3, :]),
                       self.rule.decode(tokens[..., 3:, :])], dim=-2)
        return h.reshape(*z.shape[:-1], 80)

    def forward(self, h):
        return self.decode(self.encode(h))


def patch_code(recipient: torch.Tensor, donor: torch.Tensor, first_branch: torch.Tensor,
               token_code_width: int, condition: str) -> torch.Tensor:
    """Intervene at an *actual carried code* boundary, before decoding."""
    if recipient.shape != donor.shape or recipient.ndim != 2:
        raise ValueError('patch requires equal B,C actual carried code tensors')
    if recipient.shape[1] != 5 * token_code_width or first_branch.shape != (len(recipient),):
        raise ValueError('bad token code or branch shape')
    if not torch.logical_or(first_branch == 0, first_branch == 1).all():
        raise ValueError('branch address must be zero or one')
    slot = torch.arange(5, device=recipient.device)[None, :]
    address = first_branch[:, None]
    if condition == 'addressed':
        mask = (slot == address) | (slot == address + 3)
    elif condition == 'other':
        mask = (slot == 1 - address) | (slot == 4 - address)
    elif condition == 'goal':
        mask = (slot == 2).expand(len(recipient), -1)
    elif condition == 'whole':
        mask = torch.ones((len(recipient), 5), dtype=torch.bool, device=recipient.device)
    elif condition == 'self':
        return recipient.clone()
    else:
        raise ValueError('unknown intervention condition')
    mask = mask.unsqueeze(-1).expand(-1, -1, token_code_width).reshape_as(recipient)
    return torch.where(mask, donor, recipient)


@dataclass
class Interchange:
    task_logits: torch.Tensor
    audit_logits: torch.Tensor
    direct_task_logits: torch.Tensor | None = None
    direct_audit_logits: torch.Tensor | None = None
    patched_code: torch.Tensor | None = None
    recipient_code: torch.Tensor | None = None
    donor_code: torch.Tensor | None = None


def paired(model: NativeTeacher, codec: TokenCodec, receiver: torch.Tensor,
           donor: torch.Tensor, first_branch: torch.Tensor, *, cut_event: int,
           condition: str = 'addressed', direct: bool = False) -> Interchange:
    """Cut at first completed preview; use recipient suffix as the continuation."""
    if receiver.shape != donor.shape or not 0 < cut_event < receiver.shape[1]:
        raise ValueError('invalid equal-length donor batch or cut')
    r = model.rollout(receiver, codec, stop_event=cut_event)
    d = model.rollout(donor, codec, stop_event=cut_event)
    code = patch_code(r.carried[:, -1], d.carried[:, -1], first_branch,
                      codec.token_code_width, condition)
    continuation = model.rollout(receiver, codec, initial=codec.decode(code),
                                 start_event=cut_event)
    if not direct:
        return Interchange(continuation.task_logits, continuation.audit_logits,
                           patched_code=code, recipient_code=r.carried[:, -1],
                           donor_code=d.carried[:, -1])
    hybrid = torch.cat([donor[:, :cut_event], receiver[:, cut_event:]], 1)
    with torch.no_grad():
        full = model.rollout(hybrid, codec)
    return Interchange(continuation.task_logits, continuation.audit_logits,
                       full.task_logits, full.audit_logits, code,
                       r.carried[:, -1], d.carried[:, -1])


def audited_cross_entropy(pred, target):
    return F.cross_entropy(pred.reshape(-1, 3), target.reshape(-1))


def zero_wait(depth: int):
    return [0., 0., 0., 0., 0., 0., 0., 0., depth / 8.]


def synthetic_extra_waits(case: gym.Case, extra_waits: int) -> torch.Tensor:
    """Tensor-only extrapolation; original host certificate is never relabelled."""
    if case.split != 'delay4' or case.schedule != 'PPWWWWG' or extra_waits not in (1, 4):
        raise ValueError('unregistered synthetic timing request')
    x = gym.feature_sequence(case)
    return torch.tensor(x[:-1] + [zero_wait(case.depth)] * extra_waits + x[-1:],
                        dtype=torch.float32)


def donor_roster(cases):
    """Every changed-prefix pair is certified by the AF-02 composed-history oracle."""
    pairs = gym.donor_pairs(cases)
    selected = {i: [] for i in range(len(cases))}
    for p in pairs:
        if p.changed_prefix_status:
            assert gym.audit_targets(cases[p.hybrid_index]) == p.audit_target
            assert gym.target_action(cases[p.hybrid_index]) == p.expected_action
            selected[p.receiver_index].append(p)
    if any(len(v) != 2 for v in selected.values()):
        raise ValueError('each receiver must have two changed-prefix donors')
    return selected
