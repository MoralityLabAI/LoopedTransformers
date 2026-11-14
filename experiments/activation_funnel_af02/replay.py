"""Independent NumPy execution of the registered AF-02 inference equations.

This module imports no training, evaluator, model, or Torch implementation.
Named arrays are data. Every recurrent boundary and codec operation is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import erf


PROTOCOL_SHA256 = "f6c3a939a09183c265466d506de985ec1a92c76d2d57f50cdd5f3f984e224c41"
ATOL = RTOL = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(path, dtype=None):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, value in arrays.items():
        require(value.dtype.kind in "fiu" and np.isfinite(value).all(), f"invalid numeric export: {name}")
    return {name: value.astype(dtype) for name, value in arrays.items()} if dtype is not None else arrays


def assert_close(actual, expected, label, *, atol=ATOL, rtol=RTOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f"{label}: shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{label}: nonfinite values")
    require(np.allclose(actual, expected, atol=atol, rtol=rtol),
            f"{label}: numerical mismatch (max_abs={np.max(np.abs(actual - expected), initial=0):.8g})")


def linear(x, arrays, name):
    result = x @ arrays[name + ".weight"].T
    bias = arrays.get(name + ".bias")
    return result if bias is None else result + bias


def gelu(x):
    root_two = np.asarray(np.sqrt(2.), dtype=x.dtype)
    return (.5 * x * (1 + erf(x / root_two))).astype(x.dtype, copy=False)


def sigmoid(x):
    # Avoid overflow while preserving the declared logistic function.
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    out[~positive] = exp / (1 + exp)
    return out


def layer_norm(x, arrays, name):
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    return centered / np.sqrt(variance + 1e-5) * arrays[name + ".weight"] + arrays[name + ".bias"]


def _shapes(arrays, expected, label):
    require(set(arrays) == set(expected), f"{label}: named array roster mismatch")
    for name, shape in expected.items():
        require(arrays[name].shape == shape, f"{label}: wrong shape for {name}")


def source_shapes(architecture):
    if architecture == "gru":
        return {"cell.weight_ih": (96, 9), "cell.weight_hh": (96, 32),
                "cell.bias_ih": (96,), "cell.bias_hh": (96,),
                "task_head.weight": (3, 32), "task_head.bias": (3,),
                "audit_head.weight": (6, 32), "audit_head.bias": (6,)}
    require(architecture == "native", "unknown architecture")
    expected = {"prem": (2, 3), "head": (2, 3)}
    for name in ("atom_writer", "rule_writer"):
        expected.update({name + ".weight": (16, 9), name + ".bias": (16,)})
    for kind in ("atom", "rule"):
        prefix = f"cell.{kind}_update"
        expected.update({prefix + ".0.weight": (32, 48), prefix + ".0.bias": (32,),
                         prefix + ".2.weight": (16, 32), prefix + ".2.bias": (16,)})
    for name in ("q", "k", "v"):
        expected[f"cell.{name}.weight"] = (16, 16)
    for name in ("na", "nr"):
        expected.update({f"cell.{name}.weight": (16,), f"cell.{name}.bias": (16,)})
    for name, output, inputs in (("task_head", 1, 49), ("audit_head", 3, 48), ("stop_head", 1, 48)):
        expected.update({name + ".weight": (output, inputs), name + ".bias": (output,)})
    return expected


class NumpyCodec:
    def __init__(self, architecture, arrays):
        self.architecture, self.arrays = architecture, arrays
        width = 32 if architecture == "gru" else 16
        prefixes = ("",) if architecture == "gru" else ("atom.", "rule.")
        expected = {}
        for prefix in prefixes:
            expected.update({prefix + "encoder.weight": (4, width), prefix + "encoder.bias": (4,),
                             prefix + "decoder.weight": (width, 4), prefix + "decoder.bias": (width,)})
        _shapes(arrays, expected, "codec")
        self.code_width = 4 if architecture == "gru" else 20

    def encode(self, h):
        if self.architecture == "gru":
            return np.tanh(linear(h, self.arrays, "encoder"))
        shape = h.shape[:-1]
        tokens = h.reshape(*shape, 5, 16)
        atoms = np.tanh(linear(tokens[..., :3, :], self.arrays, "atom.encoder"))
        rules = np.tanh(linear(tokens[..., 3:, :], self.arrays, "rule.encoder"))
        return np.concatenate((atoms, rules), axis=-2).reshape(*shape, 20)

    def decode(self, code):
        if self.architecture == "gru":
            return linear(code, self.arrays, "decoder")
        shape = code.shape[:-1]
        tokens = code.reshape(*shape, 5, 4)
        atoms = linear(tokens[..., :3, :], self.arrays, "atom.decoder")
        rules = linear(tokens[..., 3:, :], self.arrays, "rule.decoder")
        return np.concatenate((atoms, rules), axis=-2).reshape(*shape, 80)

    def __call__(self, h):
        return self.decode(self.encode(h))


class NumpyModel:
    def __init__(self, architecture, arrays):
        self.architecture, self.arrays = architecture, arrays
        _shapes(arrays, source_shapes(architecture), "source")
        self.state_width = 32 if architecture == "gru" else 80
        self.dtype = arrays["task_head.weight"].dtype
        if architecture == "native":
            require(np.array_equal(arrays["prem"], [[1, 0, 0], [0, 1, 0]]), "native premise graph changed")
            require(np.array_equal(arrays["head"], [[0, 0, 1], [0, 0, 1]]), "native head graph changed")

    def _attention(self, atoms, rules, incidence):
        scores = linear(atoms, self.arrays, "cell.q") @ linear(rules, self.arrays, "cell.k").swapaxes(-1, -2) / 4.
        mask = incidence.T > 0
        scores = np.where(mask, scores, -1e4)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True) * mask
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
        return weights @ linear(rules, self.arrays, "cell.v")

    def step(self, h, x, layout=None, *, debug=False):
        h, x = np.asarray(h, dtype=self.dtype), np.asarray(x, dtype=self.dtype)
        single = h.ndim == 1
        if single:
            h, x = h[None], x[None]
        require(h.ndim == x.ndim == 2 and h.shape == (len(x), self.state_width) and x.shape[1] == 9,
                "invalid recurrent state/event shape")
        if self.architecture == "gru":
            require(layout is None, "GRU has no graph layout")
            gi = x @ self.arrays["cell.weight_ih"].T + self.arrays["cell.bias_ih"]
            gh = h @ self.arrays["cell.weight_hh"].T + self.arrays["cell.bias_hh"]
            ir, iz, inn = np.split(gi, 3, axis=-1)
            hr, hz, hn = np.split(gh, 3, axis=-1)
            reset, update = sigmoid(ir + hr), sigmoid(iz + hz)
            new = np.tanh(inn + reset * hn)
            result = (1 - update) * new + update * h
            details = {"written": h.copy()}
        else:
            tokens = h.reshape(-1, 5, 16)
            atoms, rules = tokens[:, :3].copy(), tokens[:, 3:].copy()
            payload = x.copy()
            payload[:, 2:4] = 0
            branches = x[:, :1] * x[:, 2:4]
            atom_mask = np.concatenate((branches, 1 - branches.sum(axis=-1, keepdims=True)), axis=-1)
            atoms += atom_mask[..., None] * linear(payload, self.arrays, "atom_writer")[:, None]
            rules += branches[..., None] * linear(payload, self.arrays, "rule_writer")[:, None]
            written = np.concatenate((atoms, rules), axis=1).reshape(-1, 80)
            atom_slots, rule_slots = _layout_slots(layout)
            pa, pr = np.empty_like(atoms), np.empty_like(rules)
            pa[:, atom_slots], pr[:, rule_slots] = atoms, rules
            prem, head = np.empty((2, 3), dtype=self.dtype), np.empty((2, 3), dtype=self.dtype)
            prem[np.ix_(rule_slots, atom_slots)] = self.arrays["prem"]
            head[np.ix_(rule_slots, atom_slots)] = self.arrays["head"]
            parents = (prem @ pa) / np.maximum(prem.sum(axis=-1, keepdims=True), 1.)
            rule_features = np.concatenate((pr, head @ pa, parents), axis=-1)
            rule_pre = pr + linear(gelu(linear(rule_features, self.arrays, "cell.rule_update.0")), self.arrays, "cell.rule_update.2")
            updated_rules = layer_norm(rule_pre, self.arrays, "cell.nr")
            incoming = self._attention(pa, updated_rules, head)
            outgoing = self._attention(pa, updated_rules, prem)
            atom_features = np.concatenate((pa, incoming, outgoing), axis=-1)
            atom_pre = pa + linear(gelu(linear(atom_features, self.arrays, "cell.atom_update.0")), self.arrays, "cell.atom_update.2")
            updated_atoms = layer_norm(atom_pre, self.arrays, "cell.na")
            result = np.concatenate((updated_atoms[:, atom_slots], updated_rules[:, rule_slots]), axis=1).reshape(-1, 80)
            details = {"written": written, "atom_pre_ln": atom_pre[:, atom_slots],
                       "rule_pre_ln": rule_pre[:, rule_slots]}
        if single:
            result = result[0]
            details = {key: value[0] for key, value in details.items()}
        return (result, details) if debug else result

    def readout(self, h):
        h = np.asarray(h, dtype=self.dtype)
        if self.architecture == "gru":
            return linear(h, self.arrays, "task_head"), linear(h, self.arrays, "audit_head").reshape(*h.shape[:-1], 2, 3)
        tokens = h.reshape(*h.shape[:-1], 5, 16)
        atoms, rules = tokens[..., :3, :], tokens[..., 3:, :]
        scores, audits = [], []
        for branch in (0, 1):
            features = np.concatenate((rules[..., branch, :], atoms[..., 2, :], atoms[..., branch, :]), axis=-1)
            rank = np.full((*features.shape[:-1], 1), branch, dtype=self.dtype)
            scores.append(linear(np.concatenate((features, rank), axis=-1), self.arrays, "task_head"))
            audits.append(linear(features, self.arrays, "audit_head"))
        stop = np.concatenate((atoms[..., 2, :], atoms[..., :2, :].mean(axis=-2), rules.mean(axis=-2)), axis=-1)
        scores.append(linear(stop, self.arrays, "stop_head"))
        return np.concatenate(scores, axis=-1), np.stack(audits, axis=-2)


def _layout_slots(layout):
    if layout is None:
        return np.arange(3), np.arange(2)
    atoms = layout["atom_slots"] if isinstance(layout, Mapping) else layout.atom_slots
    rules = layout["rule_slots"] if isinstance(layout, Mapping) else layout.rule_slots
    require(sorted(atoms) == [0, 1, 2] and sorted(rules) == [0, 1], "invalid graph permutation")
    return np.asarray(atoms, dtype=int), np.asarray(rules, dtype=int)


@dataclass
class ReplayTrace:
    before: np.ndarray
    raw: np.ndarray
    states: np.ndarray
    carried: np.ndarray
    task_logits: np.ndarray
    audit_logits: np.ndarray


def rollout(model, events, codec=None, *, initial=None, start_event=0,
            stop_event=None, zero_between_calls=False, layout=None):
    events = np.asarray(events, dtype=model.dtype)
    single = events.ndim == 2
    if single:
        events = events[None]
    require(events.ndim == 3 and events.shape[-1] == 9, "invalid event sequence")
    h = np.zeros((len(events), model.state_width), dtype=model.dtype) if initial is None else np.asarray(initial, dtype=model.dtype).reshape(len(events), model.state_width)
    before, raw, states, carried = [], [], [], []
    for event in range(start_event, events.shape[1] if stop_event is None else stop_event):
        for _ in range(2):
            if zero_between_calls:
                h = np.zeros_like(h)
            before.append(h.copy())
            update = model.step(h, events[:, event], layout)
            code = update if codec is None else codec.encode(update)
            h = update if codec is None else codec.decode(code)
            raw.append(update.copy())
            states.append(h.copy())
            carried.append(code.copy())
    require(bool(states), "replay must execute at least one event")
    task, audit = model.readout(h)
    arrays = [np.stack(values, axis=1) for values in (before, raw, states, carried)] + [task, audit]
    if single:
        arrays = [value[0] for value in arrays]
    return ReplayTrace(*arrays)


def patch_code(receiver, donor, architecture, condition, first_branch):
    receiver, donor = np.asarray(receiver), np.asarray(donor)
    require(receiver.shape == donor.shape, "patch donor/recipient shape mismatch")
    require(type(first_branch) is int and first_branch in (0, 1), "invalid patch branch")
    mask = np.zeros(receiver.shape[-1], dtype=bool)
    if condition == "whole":
        mask[:] = True
    elif condition == "self":
        require(architecture == "native", "self token control is native only")
    else:
        require(architecture == "native" and receiver.shape[-1] in (20, 80), "invalid native patch width")
        width = receiver.shape[-1] // 5
        if condition == "addressed_pair":
            tokens = (first_branch, 3 + first_branch)
        elif condition == "other_pair":
            tokens = (1 - first_branch, 4 - first_branch)
        elif condition == "goal_only":
            tokens = (2,)
        else:
            raise ValueError("unregistered patch condition")
        for token in tokens:
            mask[token * width:(token + 1) * width] = True
    return np.where(mask, donor, receiver), mask


def resource_counts(architecture, arm, events):
    require(architecture in ("gru", "native") and arm in ("teacher", "task", "joint"), "invalid resource arm")
    require(type(events) is int and events > 0, "invalid event count")
    native, compressed = architecture == "native", arm != "teacher"
    params, buffers, width = (5590, 12, 80) if native else (4425, 0, 32)
    codec_params = (296 if native else 292) if compressed else 0
    visit_macs, heads = (14688, 434) if native else (3936, 288)
    codec_macs = (640 if native else 256) if compressed else 0
    code = (20 if native else 4) if compressed else width
    return {"source_parameters": params, "source_buffer_floats": buffers,
            "codec_parameters": codec_params, "total_stored_floats": params + buffers + codec_params,
            "carried_code_coordinates": code, "carried_code_bits": 32 * code,
            "decoded_working_coordinates": width, "visits": 2 * events,
            "dense_matrix_macs": 2 * events * (visit_macs + codec_macs) + heads}
