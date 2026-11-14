"""AF-01 recurrent interface-memory models and carried-state codecs.

This is a small, newly trained GRU pilot. It does not substitute for the native
LR-01 TypedLoop or establish a transformer mechanism. Each public observation
invokes the same cell twice. Codecs mediate every carried state, including the
state consumed by the next observation and the final action/audit heads.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

INPUT_DIM = 9
WIDTH = 32
CODE_WIDTH = 4
LOOPS = 2
LEARNED_ARMS = ("task", "reconstruction", "causal", "joint")
CODEC_ARMS = LEARNED_ARMS + ("pca", "random")


@dataclass
class Trace:
    """Minimal visit-indexed adapter contract for a future native model."""
    task_logits: torch.Tensor
    audit_logits: torch.Tensor
    states: torch.Tensor  # batch, observation * visit, full-state width
    carried: torch.Tensor  # actual boundary code before decode; native state when no codec


class RecurrentTeacher(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, width=WIDTH, loops=LOOPS):
        super().__init__()
        self.input_dim, self.width, self.loops = input_dim, width, loops
        self.cell = nn.GRUCell(input_dim, width)
        self.task_head = nn.Linear(width, 3)
        self.audit_head = nn.Linear(width, 6)

    def rollout(self, x, codec=None, initial=None, start_event=0,
                stop_event=None, zero_before_goal=False, zero_between_calls=False):
        h = x.new_zeros((len(x), self.width)) if initial is None else initial
        states, codes = [], []
        stop_event = x.shape[1] if stop_event is None else stop_event
        for event in range(start_event, stop_event):
            if zero_before_goal and event == x.shape[1] - 1:
                h = torch.zeros_like(h)
            for _ in range(self.loops):
                if zero_between_calls:
                    h = torch.zeros_like(h)
                h = self.cell(x[:, event], h)
                if codec is not None:
                    z = codec.encode(h)
                    h = codec.decode(z)
                else:
                    z = h
                states.append(h)
                codes.append(z)
        if not states:
            raise ValueError("rollout must execute at least one observation")
        trace = torch.stack(states, dim=1)
        return Trace(self.task_head(h), self.audit_head(h).reshape(-1, 2, 3),
                     trace, torch.stack(codes, dim=1))


class LearnedCodec(nn.Module):
    def __init__(self, width=WIDTH, code_width=CODE_WIDTH):
        super().__init__()
        self.width, self.code_width = width, code_width
        self.encoder = nn.Linear(width, code_width)
        self.decoder = nn.Linear(code_width, width)

    def encode(self, h):
        return torch.tanh(self.encoder(h))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, h):
        return self.decode(self.encode(h))

    @torch.no_grad()
    def initialize_from_basis(self, mean, basis):
        # Identical initialization across learned objectives; no held-out fit.
        self.encoder.weight.copy_(basis.T * .5)
        self.encoder.bias.copy_(-(mean @ basis) * .5)
        self.decoder.weight.copy_(basis * 2.)
        self.decoder.bias.copy_(mean)


class FixedCodec(nn.Module):
    def __init__(self, mean, basis):
        super().__init__()
        self.width, self.code_width = len(mean), basis.shape[1]
        self.register_buffer("mean", mean.clone())
        self.register_buffer("basis", basis.clone())

    def encode(self, h):
        return (h - self.mean) @ self.basis

    def decode(self, z):
        return z @ self.basis.T + self.mean

    def forward(self, h):
        return self.decode(self.encode(h))


class CompactTransition(nn.Module):
    """Four-coordinate recurrence; decoder and source heads are fully charged."""
    def __init__(self, teacher, code_width=CODE_WIDTH):
        super().__init__()
        self.input_dim, self.width, self.loops = teacher.input_dim, code_width, teacher.loops
        self.full_width = teacher.width
        self.cell = nn.GRUCell(self.input_dim, code_width)
        self.decoder = nn.Linear(code_width, teacher.width)
        self.task_head = copy.deepcopy(teacher.task_head)
        self.audit_head = copy.deepcopy(teacher.audit_head)
        for p in list(self.task_head.parameters()) + list(self.audit_head.parameters()):
            p.requires_grad_(False)

    def rollout(self, x, initial=None, start_event=0, stop_event=None,
                zero_before_goal=False, zero_between_calls=False):
        z = x.new_zeros((len(x), self.width)) if initial is None else initial
        states, codes = [], []
        stop_event = x.shape[1] if stop_event is None else stop_event
        for event in range(start_event, stop_event):
            if zero_before_goal and event == x.shape[1] - 1:
                z = torch.zeros_like(z)
            for _ in range(self.loops):
                if zero_between_calls:
                    z = torch.zeros_like(z)
                z = self.cell(x[:, event], z)
                codes.append(z)
                states.append(self.decoder(z))
        if not states:
            raise ValueError("rollout must execute at least one observation")
        h = states[-1]
        return Trace(self.task_head(h), self.audit_head(h).reshape(-1, 2, 3),
                     torch.stack(states, 1), torch.stack(codes, 1))


def fit_basis(activation_bank, code_width=CODE_WIDTH, seed=0):
    bank = activation_bank.reshape(-1, activation_bank.shape[-1]).float()
    mean = bank.mean(0)
    _, _, vh = torch.linalg.svd(bank - mean, full_matrices=False)
    pca = vh[:code_width].T.contiguous()
    rng = torch.Generator().manual_seed(seed + 40000)
    random = torch.linalg.qr(torch.randn(len(mean), code_width, generator=rng)).Q
    return mean, pca, random


def audit_loss(logits, labels):
    return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))


def task_from_audits(audits):
    # Independent finite task target: prefer branch 0, then branch 1, else stop.
    return torch.where(audits[:, 0] == 1, 0, torch.where(audits[:, 1] == 1, 1, 2))


def composite_audits(x, audits, donor_indices):
    """Swap the first observed branch's history; keep recipient suffix fixed.

    The counterfactual changes evidence history. Original-host proof validity is
    scored separately. Donors must address the same first branch as recipients.
    """
    first = x[:, 0, 3].round().long()  # branch-1 indicator in the public vector
    donor_first = first[donor_indices]
    if not torch.equal(first, donor_first):
        raise ValueError("counterfactual donors must match first-probed branch")
    if not torch.equal(x[:, :, 8], x[:, :1, 8].expand(-1, x.shape[1])):
        raise ValueError("native observation depth must stay constant")
    if not torch.equal(x[:, 0, 8], x[donor_indices, 0, 8]):
        raise ValueError("counterfactual donors must match public depth")
    target = audits.clone()
    row = torch.arange(len(x), device=x.device)
    target[row, first] = audits[donor_indices, first]
    return target


def matched_donors(x, generator):
    """Permute only within public (first-branch, depth) strata."""
    groups = {}
    for i in range(len(x)):
        key = (float(x[i, 0, 3]), float(x[i, 0, 8]))
        groups.setdefault(key, []).append(i)
    result = torch.arange(len(x), device=x.device)
    for indices in groups.values():
        ix = torch.tensor(indices, dtype=torch.long, device=x.device)
        order = torch.randperm(len(indices), generator=generator, device=x.device)
        result[ix] = ix[order]
    return result


def paired_continuation(model, x, donor_indices, codec=None):
    if isinstance(model, CompactTransition):
        prefix = model.rollout(x, stop_event=1)
        return model.rollout(x, initial=prefix.carried[donor_indices, -1], start_event=1)
    prefix = model.rollout(x, codec=codec, stop_event=1)
    return model.rollout(x, codec=codec,
                         initial=prefix.states[donor_indices, -1], start_event=1)


def commutation_loss(teacher, codec, x, native_states):
    """One-step closure residual, measured in decoded source-state coordinates.

    V(z,x)=encode(U(decode(z),x)). Compare decode(V(encode(h),x))
    with decode(encode(U(h,x))); stop gradients through the target branch.
    """
    previous = x.new_zeros((len(x), teacher.width))
    errors = []
    for visit in range(native_states.shape[1]):
        event = visit // teacher.loops
        current = native_states[:, visit]
        actual = codec(teacher.cell(x[:, event], codec(previous)))
        expected = codec(current).detach()
        errors.append(F.mse_loss(actual, expected))
        previous = current
    return torch.stack(errors).mean()


def resources(model, codec=None, observations=3):
    """Dense matrix MAC estimate; excludes nonlinearities and framework overhead."""
    d, f, k = model.width, model.input_dim, model.loops * observations
    transition = k * 3 * d * (f + d)
    if isinstance(model, CompactTransition):
        # Decode each visit for the stored trace; head evaluation is final-only.
        decoding = k * d * model.full_width
        heads = model.full_width * 9
        state_dims = d
    else:
        decoding = 0 if codec is None else k * 2 * d * codec.code_width
        heads = d * 9
        state_dims = d if codec is None else codec.code_width
    parameters = sum(p.numel() for p in model.parameters())
    codec_parameters = 0 if codec is None else sum(p.numel() for p in codec.parameters())
    buffers = 0 if codec is None else sum(b.numel() for b in codec.buffers())
    return {"model_parameters": parameters, "codec_parameters": codec_parameters,
            "codec_buffer_floats": buffers, "total_stored_floats": parameters + codec_parameters + buffers,
            "carried_code_coordinates": state_dims, "coordinate_bits": 32,
            "carried_code_bits": 32 * state_dims, "decoded_working_state_coordinates":
            model.full_width if isinstance(model, CompactTransition) else d,
            "dense_matrix_macs_per_episode": transition + decoding + heads,
            "mac_scope": "matrix multiplies; every visit decoder and final task/audit heads; excludes nonlinearities, encoding public observations, verifier and framework overhead",
            "bottleneck_storage_scope": "compressed boundary code; full decoded working state is also required by codec arms"}
