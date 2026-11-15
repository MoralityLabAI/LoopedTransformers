"""AF-02 carried-state adapters for the pinned GRU and native TypedLoop.

The native cell is imported from the original source without modification.
Its public preview writer and readouts are specific to this experiment. All
external states use canonical order A0,A1,Goal,R0,R1; physical layouts affect
the cell's storage order and are reversed before returning a carried state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
NATIVE_TRAIN_SHA256 = "8c269d1e698c92b384b554b039eec5fc95b5b544520219ff20c3eb794f88cfa1"
NATIVE_DATA_SHA256 = "6c67e075fa8e5f162721d22c2e56078f40b8117d716951806afd2494dd08721f"
AF01_MODEL_SHA256 = "abc2ea1075726e9879940444c273230019cf330d68b6e606ae53612ac076fbfc"
INPUT_DIM, LOOPS = 9, 2
ARCHITECTURES = ("gru", "native")
ARMS = ("teacher", "task", "joint")
LEARNED_ARMS = ("task", "joint")


def _verify_source(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"pinned source hash mismatch: {path.name}")


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def native_cell_class():
    folder = ROOT / "experiments" / "lean_reasoner"
    _verify_source(folder / "train.py", NATIVE_TRAIN_SHA256)
    _verify_source(folder / "proof_data.py", NATIVE_DATA_SHA256)
    data = _load_module("_af02_pinned_lr01_proof_data", folder / "proof_data.py")
    previous = sys.modules.get("proof_data")
    sys.modules["proof_data"] = data
    try:
        module = _load_module("_af02_pinned_lr01_train", folder / "train.py")
    finally:
        if previous is None:
            sys.modules.pop("proof_data", None)
        else:
            sys.modules["proof_data"] = previous
    return module.TypedLoop


_verify_source(ROOT / "experiments" / "activation_funnel" / "model.py", AF01_MODEL_SHA256)
from ..activation_funnel.model import LearnedCodec, RecurrentTeacher as _AF01Teacher


@dataclass
class Predictions:
    task_logits: torch.Tensor
    audit_logits: torch.Tensor


@dataclass
class Trace(Predictions):
    before: torch.Tensor  # decoded carry immediately before writer/cell
    raw: torch.Tensor  # source update before this visit's codec
    states: torch.Tensor  # decoded carry consumed by the next visit
    carried: torch.Tensor  # actual code before decoding; raw state without codec


class _Rollout:
    def rollout(self, x, codec=None, initial=None, start_event=0, stop_event=None,
                zero_between_calls=False, layout=None):
        if x.ndim != 3 or x.shape[-1] != INPUT_DIM or not len(x):
            raise ValueError("rollout requires a nonempty true-length B,T,9 batch")
        stop_event = x.shape[1] if stop_event is None else stop_event
        if not 0 <= start_event < stop_event <= x.shape[1]:
            raise ValueError("rollout must execute a valid nonempty event interval")
        if start_event and initial is None:
            raise ValueError("continuation requires its decoded initial state")
        h = x.new_zeros((len(x), self.state_width)) if initial is None else initial
        if h.shape != (len(x), self.state_width):
            raise ValueError("decoded initial-state shape mismatch")
        before, raw, states, codes = [], [], [], []
        for event in range(start_event, stop_event):
            for _ in range(self.loops):
                if zero_between_calls:
                    h = torch.zeros_like(h)
                before.append(h)
                updated = self.one_step(h, x[:, event], layout=layout)
                raw.append(updated)
                if codec is None:
                    z, h = updated, updated
                else:
                    z = codec.encode(updated)
                    h = codec.decode(z)
                states.append(h)
                codes.append(z)
        task, audit = self.readout(h)
        return Trace(task, audit, torch.stack(before, 1), torch.stack(raw, 1),
                     torch.stack(states, 1), torch.stack(codes, 1))


class GRUTeacher(_Rollout, _AF01Teacher):
    architecture = "gru"

    def __init__(self):
        _AF01Teacher.__init__(self, input_dim=INPUT_DIM, width=32, loops=LOOPS)
        self.state_width = 32

    def one_step(self, h, x, layout=None):
        if layout is not None:
            raise ValueError("a GRU state has no graph layout")
        return self.cell(x, h)

    def readout(self, h):
        return self.task_head(h), self.audit_head(h).reshape(-1, 2, 3)


class NativeTeacher(_Rollout, nn.Module):
    architecture = "native"

    def __init__(self):
        super().__init__()
        self.width, self.state_width = 16, 80
        self.input_dim, self.loops = INPUT_DIM, LOOPS
        self.cell = native_cell_class()(self.width)
        self.atom_writer = nn.Linear(INPUT_DIM, self.width)
        self.rule_writer = nn.Linear(INPUT_DIM, self.width)
        self.task_head = nn.Linear(3 * self.width + 1, 1)
        self.audit_head = nn.Linear(3 * self.width, 3)
        self.stop_head = nn.Linear(3 * self.width, 1)
        self.register_buffer("prem", torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        self.register_buffer("head", torch.tensor([[0., 0., 1.], [0., 0., 1.]]))

    def unpack(self, h):
        tokens = h.reshape(*h.shape[:-1], 5, self.width)
        return tokens[..., :3, :], tokens[..., 3:, :]

    def one_step(self, h, x, layout=None):
        a, r = self.unpack(h)
        payload = x.clone()
        payload[:, 2:4] = 0.
        branch = x[:, :1] * x[:, 2:4]
        atom_routes = torch.cat([branch, 1. - branch.sum(-1, keepdim=True)], -1)
        a = a + atom_routes.unsqueeze(-1) * self.atom_writer(payload).unsqueeze(1)
        r = r + branch.unsqueeze(-1) * self.rule_writer(payload).unsqueeze(1)
        prem, head = self.prem.unsqueeze(0), self.head.unsqueeze(0)
        if layout is not None:
            atom_slots = tuple(layout.atom_slots)
            rule_slots = tuple(layout.rule_slots)
            if sorted(atom_slots) != [0, 1, 2] or sorted(rule_slots) != [0, 1]:
                raise ValueError("invalid logical-to-physical graph layout")
            # slots[logical] is physical; inverse order indexes canonical input.
            ai = sorted(range(3), key=atom_slots.__getitem__)
            ri = sorted(range(2), key=rule_slots.__getitem__)
            a, r = a[:, ai], r[:, ri]
            prem, head = prem[:, ri][:, :, ai], head[:, ri][:, :, ai]
        a, r = self.cell(a, r, prem, head)
        if layout is not None:
            a, r = a[:, atom_slots], r[:, rule_slots]
        return torch.cat([a, r], 1).reshape(len(h), self.state_width)

    def readout(self, h):
        a, r = self.unpack(h)
        goal = a[:, 2:3].expand(-1, 2, -1)
        features = torch.cat([r, goal, a[:, :2]], -1)
        ranks = h.new_tensor([0., 1.]).view(1, 2, 1).expand(len(h), -1, -1)
        branch = self.task_head(torch.cat([features, ranks], -1)).squeeze(-1)
        stop_features = torch.cat([a[:, 2], a[:, :2].mean(1), r.mean(1)], -1)
        task = torch.cat([branch, self.stop_head(stop_features)], -1)
        return task, self.audit_head(features)


class GRUCodec(LearnedCodec):
    architecture = "gru"

    def __init__(self):
        super().__init__(width=32, code_width=4)
        self.state_width = 32


class NativeCodec(nn.Module):
    architecture = "native"

    def __init__(self):
        super().__init__()
        self.width, self.state_width, self.code_width = 80, 80, 20
        self.token_width, self.token_code_width = 16, 4
        self.atom = LearnedCodec(width=16, code_width=4)
        self.rule = LearnedCodec(width=16, code_width=4)

    def encode(self, h):
        tokens = h.reshape(*h.shape[:-1], 5, 16)
        z = torch.cat([self.atom.encode(tokens[..., :3, :]),
                       self.rule.encode(tokens[..., 3:, :])], -2)
        return z.reshape(*h.shape[:-1], 20)

    def decode(self, z):
        tokens = z.reshape(*z.shape[:-1], 5, 4)
        h = torch.cat([self.atom.decode(tokens[..., :3, :]),
                       self.rule.decode(tokens[..., 3:, :])], -2)
        return h.reshape(*z.shape[:-1], 80)

    def forward(self, h):
        return self.decode(self.encode(h))


def make_model(architecture):
    if architecture == "gru":
        return GRUTeacher()
    if architecture == "native":
        return NativeTeacher()
    raise ValueError(f"unknown architecture: {architecture}")


def make_codec(architecture):
    if architecture == "gru":
        return GRUCodec()
    if architecture == "native":
        return NativeCodec()
    raise ValueError(f"unknown architecture: {architecture}")


def audit_loss(logits, target):
    return F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))


def task_from_audits(audits):
    return torch.where(audits[:, 0] == 1, 0, torch.where(audits[:, 1] == 1, 1, 2))


def matched_donors(cases, generator):
    from . import gym
    groups = {}
    for i, case in enumerate(cases):
        groups.setdefault(gym.donor_stratum(case), []).append(i)
    result = torch.arange(len(cases))
    for indices in groups.values():
        ix = torch.tensor(indices, dtype=torch.long)
        result[ix] = ix[torch.randperm(len(ix), generator=generator)]
    return result


def composite_audits(cases, audits, donor_indices):
    from . import gym
    if len(cases) != len(donor_indices):
        raise ValueError("donor-index length mismatch")
    target = audits.clone()
    for i, j in enumerate(donor_indices.tolist()):
        if not 0 <= j < len(cases) or gym.donor_stratum(cases[i]) != gym.donor_stratum(cases[j]):
            raise ValueError("donors must match first branch, public depth and complete schedule")
        branch = cases[i].probe_order[0]
        target[i, branch] = audits[j, branch]
    return target


def paired_continuation(model, x, cases, donor_indices, codec=None):
    """Resume from each donor's decoded state after its first complete preview."""
    from . import gym
    dummy = torch.zeros((len(cases), 2), dtype=torch.long)
    composite_audits(cases, dummy, donor_indices)  # enforce the same grammar guard
    cuts = [gym.first_probe_cut(c) for c in cases]
    prefix = model.rollout(x, codec, stop_event=max(cuts))
    task = x.new_empty((len(x), 3))
    audit = x.new_empty((len(x), 2, 3))
    for cut in sorted(set(cuts)):
        ix = torch.tensor([i for i, value in enumerate(cuts) if value == cut], dtype=torch.long)
        initial = prefix.states[donor_indices[ix], cut * model.loops - 1]
        resumed = model.rollout(x[ix], codec, initial=initial, start_event=cut)
        task = task.index_copy(0, ix, resumed.task_logits)
        audit = audit.index_copy(0, ix, resumed.audit_logits)
    return Predictions(task, audit)


def commutation_loss(model, codec, x, native_trace):
    """Decoded closure error: per-input visit means, then the input mean."""
    errors = []
    for visit in range(native_trace.raw.shape[1]):
        before = native_trace.before[:, visit]
        actual = codec(model.one_step(codec(before), x[:, visit // model.loops]))
        expected = codec(native_trace.raw[:, visit]).detach()
        errors.append((actual - expected).square().mean(-1))
    return torch.stack(errors, 1).mean(1).mean()


def resources(model, codec=None, events=3):
    visits = events * model.loops
    if model.architecture == "gru":
        per_visit = 3 * 32 * (9 + 32)
        heads, codec_macs = 32 * 9, (0 if codec is None else 2 * 32 * 4)
        writer_macs = 0
    else:
        # Rule/atom MLPs, two typed attention calls and dense incidence products.
        per_visit = (10 * 3 + 12 * 2) * 16 ** 2 + 6 * 3 * 2 * 16
        writer_macs = 2 * 9 * 16
        per_visit += writer_macs
        heads = 2 * (3 * 16 + 1) + 2 * 3 * 16 * 3 + 3 * 16
        codec_macs = 0 if codec is None else 5 * 2 * 16 * 4
    parameters = sum(p.numel() for p in model.parameters())
    buffers = sum(p.numel() for p in model.buffers())
    cp = 0 if codec is None else sum(p.numel() for p in codec.parameters())
    cb = 0 if codec is None else sum(p.numel() for p in codec.buffers())
    code = model.state_width if codec is None else codec.code_width
    return {"model_parameters": parameters, "model_buffer_floats": buffers,
            "codec_parameters": cp, "codec_buffer_floats": cb,
            "total_stored_floats": parameters + buffers + cp + cb,
            "source_state_coordinates": model.state_width,
            "carried_code_coordinates": code, "coordinate_bits": 32,
            "carried_code_bits": 32 * code,
            "decoded_working_state_coordinates": model.state_width,
            "real_events": events, "recurrent_visits": visits,
            "dense_matrix_macs_per_episode": visits * (per_visit + codec_macs) + heads,
            "writer_dense_macs_per_visit": writer_macs,
            "mac_scope": "dense matrices including writers, typed incidence, all real visits and final heads; excludes nonlinearities, indexing, framework, environment and verifier",
            "storage_scope": "boundary code plus decoded working state; no resident-memory reduction claim"}
