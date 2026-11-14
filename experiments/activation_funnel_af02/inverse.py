"""Registered local perturbation recovery around retained native AF-02 states.

Torch supplies the frozen transition Jacobian. NumPy independently executes
the transition, directional finite differences, SVD recovery, and readouts.
The recorded reference state anchors this diagnostic; no inverse is fitted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import gym
from .replay import (NumpyModel, PROTOCOL_SHA256, assert_close, file_hash,
                     layer_norm, load_arrays, require)


SEEDS = (17, 29, 43)
HORIZONS = (1, 2)
AMPLITUDES = (1e-4, 1e-3)
RCOND, SECONDARY_RCOND = 1e-6, 1e-8
FLOOR, FD_STEP = 1e-12, 1e-5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def direction_seed(seed, input_hash):
    data = f"af02-inverse-v1:{seed}:{input_hash}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def tangent_basis(arrays):
    gains = [arrays["cell.na.weight"]] * 3 + [arrays["cell.nr.weight"]] * 2
    if min(float(np.min(np.abs(g))) for g in gains) < 1e-8:
        return None
    result = np.zeros((80, 75), dtype=np.float64)
    for token, gain in enumerate(gains):
        normal = 1 / np.asarray(gain, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        e0 = np.zeros(16)
        e0[0] = 1
        vector = e0 - normal
        norm2 = vector @ vector
        householder = np.eye(16) if norm2 < 1e-30 else np.eye(16) - 2 * np.outer(vector, vector) / norm2
        result[token * 16:(token + 1) * 16, token * 15:(token + 1) * 15] = householder[:, 1:]
    return result


def seeded_directions(seed, input_hash, basis):
    generator = np.random.Generator(np.random.PCG64(direction_seed(seed, input_hash)))
    directions = generator.standard_normal((4, 75))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions, directions @ basis.T


def iterate(model, state, event, horizon, *, debug=False):
    h, details = np.asarray(state).copy(), []
    for _ in range(horizon):
        if debug:
            h, detail = model.step(h, event, debug=True)
            details.append(detail)
        else:
            h = model.step(h, event)
    return (h, details) if debug else h


def finite_difference_jacobian(model, state, event, horizon):
    """Independent centered differences on every ambient coordinate."""
    directions = np.eye(80, dtype=np.float64)
    events = np.repeat(np.asarray(event)[None], 80, axis=0)
    plus = iterate(model, state[None] + FD_STEP * directions, events, horizon)
    minus = iterate(model, state[None] - FD_STEP * directions, events, horizon)
    return ((plus - minus) / (2 * FD_STEP)).T


def _error_metrics(delta, recovered, baseline, target, returned, model):
    displacement = float(np.linalg.norm(target - baseline))
    denominator = max(displacement, FLOOR)
    target_task, target_audit = model.readout(target)
    return_task, return_audit = model.readout(returned)
    zero_task, zero_audit = model.readout(baseline)
    return {"predecessor_relative_error": float(np.linalg.norm(recovered - delta) / max(float(np.linalg.norm(delta)), FLOOR)),
            "forward_return_relative_error": float(np.linalg.norm(returned - target) / denominator),
            "zero_predecessor_relative_error": 1.,
            "zero_forward_return_relative_error": displacement / denominator,
            "denominator_floor_used": displacement < FLOOR,
            "task_agreement": bool(np.argmax(target_task) == np.argmax(return_task)),
            "audit_exact_agreement": bool(np.array_equal(np.argmax(target_audit, -1), np.argmax(return_audit, -1))),
            "zero_task_agreement": bool(np.argmax(target_task) == np.argmax(zero_task)),
            "zero_audit_exact_agreement": bool(np.array_equal(np.argmax(target_audit, -1), np.argmax(zero_audit, -1)))}


def _gauge_arrays(model, state, event, horizon):
    _, details = iterate(model, state, event, horizon, debug=True)
    result = {}
    for kind, parameter in (("atom", "na"), ("rule", "nr")):
        pre = np.stack([d[kind + "_pre_ln"] for d in details])
        normal = layer_norm(pre, model.arrays, "cell." + parameter)
        shifted = layer_norm(pre + .125, model.arrays, "cell." + parameter)
        result[kind + "_pre_ln"] = pre
        result[kind + "_normalized"] = normal
        result[kind + "_shifted_normalized"] = shifted
    return result


def _expected_anchors():
    return [case for case in gym.canonical_cases("depth_context") if case.depth == 4 and case.schedule == "PPG"]


def run_inverse(model, source_npz, anchors, out_dir, seed):
    """Record one seed's fixed 18 anchors and 36 Jacobians; never fit weights."""
    import torch

    require(seed in SEEDS, "unregistered inverse seed")
    require(file_hash(Path(__file__).with_name("PROTOCOL.json")) == PROTOCOL_SHA256, "inverse protocol changed")
    source = load_arrays(source_npz)
    require(all(value.dtype == np.float32 for value in source.values()), "inverse requires frozen float32 exports")
    parameters = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    require(set(parameters) == set(source), "Torch/export inverse parameter roster")
    for name in parameters:
        require(np.array_equal(parameters[name], source[name]), f"Torch/export inverse parameter mismatch: {name}")
    require(len(anchors) == 18, "inverse requires the fixed eighteen anchors")
    expected = _expected_anchors()
    require([a["canonical_id"] for a in anchors] == [c.canonical_id for c in expected], "inverse canonical anchor roster")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    array_path, receipt_path = out / f"inverse_s{seed}.npz", out / f"inverse_s{seed}.json"
    require(not array_path.exists() and not receipt_path.exists(), "preserve previous inverse attempt")
    lifted = copy.deepcopy(model).double().eval()
    for parameter in lifted.parameters():
        parameter.requires_grad_(False)
    reference = NumpyModel("native", {name: value.astype(np.float64) for name, value in source.items()})
    basis = tangent_basis(reference.arrays)
    arrays, records = {}, []
    torch.set_num_threads(2)

    for anchor, case in zip(anchors, expected):
        input_hash = gym.public_input_hash(case)
        require(anchor["public_input_sha256"] == input_hash, "inverse anchor input binding")
        h32 = np.asarray(anchor["state"], dtype=np.float32)
        e32 = np.asarray(anchor["event"], dtype=np.float32)
        require(h32.shape == (80,) and e32.shape == (9,) and np.isfinite(h32).all(), "invalid inverse anchor tensors")
        require(np.array_equal(e32, np.asarray(gym.feature_sequence(case)[-1], dtype=np.float32)), "inverse anchor G event")
        h, event = h32.astype(np.float64), e32.astype(np.float64)
        x_torch = torch.from_numpy(event[None])

        for horizon in HORIZONS:
            prefix = f"record_{len(records):04d}"
            local = {"reference_float32": h32, "reference": h,
                     "event_float32": e32, "event": event}

            def torch_transition(state):
                state = state.reshape(1, 80)
                for _ in range(horizon):
                    state = lifted.one_step(state, x_torch)
                return state.reshape(80)

            tensor = torch.from_numpy(h.copy())
            jacobian = torch.autograd.functional.jacobian(torch_transition, tensor, vectorize=True).detach().numpy()
            baseline_torch = torch_transition(tensor).detach().numpy()
            baseline = iterate(reference, h, event, horizon)
            assert_close(baseline, baseline_torch, "inverse float64 source replay", atol=1e-10, rtol=1e-8)
            full_fd = finite_difference_jacobian(reference, h, event, horizon)
            assert_close(full_fd, jacobian, "full ambient Jacobian finite difference", atol=1e-7, rtol=1e-4)
            local.update(jacobian=jacobian, full_finite_difference_jacobian=full_fd, baseline=baseline_torch)

            # Capture actual pre-LN Torch tensors from the same lifted source.
            captured = {"atom": [], "rule": []}
            handles = []
            for kind, module in (("atom", lifted.cell.na), ("rule", lifted.cell.nr)):
                handles.append(module.register_forward_pre_hook(
                    lambda _module, inputs, kind=kind: captured[kind].append(inputs[0].detach().numpy()[0].copy())))
            try:
                with torch.no_grad():
                    torch_transition(tensor)
            finally:
                for handle in handles:
                    handle.remove()
            gauge = _gauge_arrays(reference, h, event, horizon)
            gauge_max = 0.
            for kind in ("atom", "rule"):
                actual_pre = np.stack(captured[kind])
                assert_close(gauge[kind + "_pre_ln"], actual_pre, "actual pre-LN replay", atol=1e-10, rtol=1e-8)
                gauge[kind + "_pre_ln"] = actual_pre
                error = float(np.max(np.abs(gauge[kind + "_normalized"] - gauge[kind + "_shifted_normalized"])))
                gauge_max = max(gauge_max, error)
            require(gauge_max <= 1e-10, "normalization gauge control failed")
            local.update(gauge)
            record = {"prefix": prefix, "seed": seed, "canonical_id": case.canonical_id,
                      "public_input_sha256": input_hash, "horizon": horizon,
                      "anchor_visit": 3, "source_trace": anchor.get("source_trace"),
                      "direction_seed": direction_seed(seed, input_hash),
                      "tangent_available": basis is not None, "gauge_max_absolute_error": gauge_max,
                      "full_finite_difference_max_absolute_error": float(np.max(np.abs(full_fd - jacobian))),
                      "engineering_passed": True, "probes": []}
            if basis is None:
                record.update(status="degenerate_gains", primary_rank=None, secondary_rank=None)
            else:
                tangent = basis.T @ jacobian @ basis
                u, singular, vh = np.linalg.svd(tangent, full_matrices=False)
                fd_singular = np.linalg.svd(basis.T @ full_fd @ basis, compute_uv=False)
                retained = singular > RCOND * singular[0]
                pseudo = (vh[retained].T / singular[retained]) @ u[:, retained].T if retained.any() else np.zeros((75, 75))
                directions, full_directions = seeded_directions(seed, input_hash, basis)
                fd = np.stack([(iterate(reference, h + FD_STEP * d, event, horizon) -
                                iterate(reference, h - FD_STEP * d, event, horizon)) / (2 * FD_STEP)
                               for d in full_directions])
                jvp = full_directions @ jacobian.T
                assert_close(fd, jvp, "inverse directional derivative", atol=1e-7, rtol=1e-4)
                probe_directions = np.concatenate((full_directions, (basis @ vh[-1])[None]))
                deltas, recovered, targets, returns = [], [], [], []
                target_tasks, target_audits, return_tasks, return_audits = [], [], [], []
                for amplitude in AMPLITUDES:
                    for index, direction in enumerate(probe_directions):
                        delta = amplitude * direction
                        target = iterate(reference, h + delta, event, horizon)
                        back = basis @ pseudo @ basis.T @ (target - baseline)
                        returned = iterate(reference, h + back, event, horizon)
                        deltas.append(delta)
                        recovered.append(back)
                        targets.append(target)
                        returns.append(returned)
                        tt, ta = reference.readout(target)
                        rt, ra = reference.readout(returned)
                        target_tasks.append(tt); target_audits.append(ta)
                        return_tasks.append(rt); return_audits.append(ra)
                        record["probes"].append({"kind": "seeded" if index < 4 else "least_singular",
                                                 "direction_index": index, "amplitude": amplitude,
                                                 **_error_metrics(delta, back, baseline, target, returned, reference)})
                local.update(q_in=basis, q_out=basis, tangent_jacobian=tangent,
                             svd_u=u, singular_values=singular, svd_vh=vh, pseudo_inverse=pseudo,
                             finite_difference_tangent_singular_values=fd_singular,
                             tangent_directions=directions, full_directions=full_directions,
                             finite_difference=fd, jacobian_direction=jvp,
                             delta=np.stack(deltas), recovered_delta=np.stack(recovered),
                             target_output=np.stack(targets), return_output=np.stack(returns),
                             target_task_logits=np.stack(target_tasks), target_audit_logits=np.stack(target_audits),
                             return_task_logits=np.stack(return_tasks), return_audit_logits=np.stack(return_audits))
                record.update(status="measured", primary_rank=int(retained.sum()),
                              secondary_rank=int((singular > SECONDARY_RCOND * singular[0]).sum()),
                              least_direction_discarded=bool(not retained[-1]),
                              finite_difference_primary_rank=int((fd_singular > RCOND * fd_singular[0]).sum()),
                              finite_difference_secondary_rank=int((fd_singular > SECONDARY_RCOND * fd_singular[0]).sum()),
                              finite_difference_max_absolute_error=float(np.max(np.abs(fd - jvp))))
            for name, value in local.items():
                arrays[f"{prefix}/{name}"] = value
            records.append(record)
    np.savez_compressed(array_path, **arrays)
    receipt = {"schema": "af02-local-inverse-v1", "seed": seed, "protocol_sha256": PROTOCOL_SHA256,
               "inverse_source_sha256": file_hash(__file__), "replay_source_sha256": file_hash(Path(__file__).with_name("replay.py")),
               "source_export": Path(source_npz).name, "source_export_sha256": file_hash(source_npz),
               "array_file": array_path.name, "array_sha256": file_hash(array_path),
               "jacobians": len(records), "degenerate_records": sum(not r["tangent_available"] for r in records),
               "engineering_passed": all(r["engineering_passed"] for r in records), "records": records,
               "claim_scope": "Local perturbation recovery around retained reference states; descriptive, with no fitted inverse."}
    write_json(receipt_path, receipt)
    return receipt


def verify_inverse(receipt_path, source_npz, anchors=None):
    """Replay retained inverse numerics without Torch or fitted parameters."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    require(receipt["protocol_sha256"] == PROTOCOL_SHA256, "inverse protocol binding")
    require(receipt["inverse_source_sha256"] == file_hash(__file__), "inverse source changed")
    require(receipt["replay_source_sha256"] == file_hash(Path(__file__).with_name("replay.py")), "inverse replay source changed")
    require(receipt["source_export_sha256"] == file_hash(source_npz), "inverse source export changed")
    require(receipt["source_export"] == Path(source_npz).name, "inverse source export identity")
    require(Path(receipt["array_file"]).name == receipt["array_file"], "inverse archive path")
    archive_path = receipt_path.parent / receipt["array_file"]
    require(file_hash(archive_path) == receipt["array_sha256"], "inverse array archive changed")
    arrays = load_arrays(archive_path)
    model = NumpyModel("native", load_arrays(source_npz, np.float64))
    basis = tangent_basis(model.arrays)
    expected = [(case, horizon) for case in _expected_anchors() for horizon in HORIZONS]
    require(len(receipt["records"]) == receipt["jacobians"] == 36, "inverse Jacobian roster")
    anchor_by_id = {} if anchors is None else {a["canonical_id"]: a for a in anchors}
    used = set()
    for index, (record, (case, horizon)) in enumerate(zip(receipt["records"], expected)):
        prefix = f"record_{index:04d}"
        require(record["prefix"] == prefix and record["canonical_id"] == case.canonical_id and record["horizon"] == horizon,
                "inverse context/horizon roster")
        require(record["seed"] == receipt["seed"] and record["anchor_visit"] == 3, "inverse seed/visit binding")
        require(record["public_input_sha256"] == gym.public_input_hash(case), "inverse input binding")
        require(record["direction_seed"] == direction_seed(receipt["seed"], gym.public_input_hash(case)), "inverse RNG seed")
        require(record["engineering_passed"] is True, "inverse record engineering gate")
        local = {name.split("/", 1)[1]: value for name, value in arrays.items() if name.startswith(prefix + "/")}
        used.update(prefix + "/" + name for name in local)
        h, event, jacobian = local["reference"], local["event"], local["jacobian"]
        require(h.shape == (80,) and event.shape == (9,) and jacobian.shape == (80, 80), "inverse tensor dimensions")
        require(local["reference_float32"].dtype == np.float32 and local["event_float32"].dtype == np.float32,
                "inverse operational anchor precision")
        require(np.array_equal(h, local["reference_float32"].astype(np.float64)), "inverse reference lift")
        require(np.array_equal(event, local["event_float32"].astype(np.float64)), "inverse event lift")
        require(np.array_equal(event, np.asarray(gym.feature_sequence(case)[-1], dtype=np.float64)), "inverse G event")
        if anchors is not None:
            require(case.canonical_id in anchor_by_id, "inverse anchor trace missing")
            require(np.array_equal(local["reference_float32"], anchor_by_id[case.canonical_id]["state"]), "inverse reference differs from retained source trace")
            require(record["source_trace"] == anchor_by_id[case.canonical_id].get("source_trace"), "inverse source trace reference changed")
        baseline = iterate(model, h, event, horizon)
        assert_close(local["baseline"], baseline, "inverse baseline replay", atol=1e-10, rtol=1e-8)
        full_fd = finite_difference_jacobian(model, h, event, horizon)
        assert_close(full_fd, jacobian, "full ambient Jacobian finite difference", atol=1e-7, rtol=1e-4)
        assert_close(local["full_finite_difference_jacobian"], full_fd, "retained full ambient finite difference", atol=1e-10, rtol=1e-8)
        require(abs(record["full_finite_difference_max_absolute_error"] - float(np.max(np.abs(full_fd - jacobian)))) <= 1e-12,
                "full finite-difference summary")
        gauge = _gauge_arrays(model, h, event, horizon)
        gauge_max = 0.
        for name, expected_array in gauge.items():
            assert_close(local[name], expected_array, "inverse gauge tensor replay", atol=1e-10, rtol=1e-8)
        for kind in ("atom", "rule"):
            gauge_max = max(gauge_max, float(np.max(np.abs(gauge[kind + "_normalized"] - gauge[kind + "_shifted_normalized"]))))
        require(gauge_max <= 1e-10 and abs(record["gauge_max_absolute_error"] - gauge_max) <= 1e-12, "inverse gauge control")
        require(record["tangent_available"] == (basis is not None), "inverse gain-degeneracy classification")
        if basis is None:
            require(record["status"] == "degenerate_gains" and not record["probes"] and
                    record["primary_rank"] is None and record["secondary_rank"] is None, "degenerate inverse record")
            continue
        require(record["status"] == "measured", "inverse measured status")
        assert_close(local["q_in"], basis, "inverse input tangent", atol=1e-12, rtol=1e-12)
        assert_close(local["q_out"], basis, "inverse output tangent", atol=1e-12, rtol=1e-12)
        tangent = basis.T @ jacobian @ basis
        assert_close(local["tangent_jacobian"], tangent, "inverse tangent Jacobian", atol=1e-10, rtol=1e-8)
        u, singular, vh = local["svd_u"], local["singular_values"], local["svd_vh"]
        require(singular.shape == (75,) and np.all(singular >= 0) and np.all(np.diff(singular) <= 0), "inverse singular-value ordering")
        assert_close(u.T @ u, np.eye(75), "inverse SVD left orthogonality", atol=1e-10, rtol=1e-8)
        assert_close(vh @ vh.T, np.eye(75), "inverse SVD right orthogonality", atol=1e-10, rtol=1e-8)
        assert_close((u * singular) @ vh, tangent, "inverse SVD reconstruction", atol=1e-10, rtol=1e-8)
        retained = singular > RCOND * singular[0]
        require(record["primary_rank"] == int(retained.sum()) and record["secondary_rank"] == int((singular > SECONDARY_RCOND * singular[0]).sum()), "inverse retained ranks")
        require(record["least_direction_discarded"] == bool(not retained[-1]), "inverse least direction classification")
        fd_singular = np.linalg.svd(basis.T @ full_fd @ basis, compute_uv=False)
        assert_close(local["finite_difference_tangent_singular_values"], fd_singular, "finite-difference singular values", atol=1e-10, rtol=1e-8)
        require(record["finite_difference_primary_rank"] == int((fd_singular > RCOND * fd_singular[0]).sum()) and
                record["finite_difference_secondary_rank"] == int((fd_singular > SECONDARY_RCOND * fd_singular[0]).sum()),
                "finite-difference rank cross-check")
        pseudo = (vh[retained].T / singular[retained]) @ u[:, retained].T if retained.any() else np.zeros((75, 75))
        assert_close(local["pseudo_inverse"], pseudo, "inverse pseudoinverse", atol=1e-9, rtol=1e-8)
        directions, full = seeded_directions(receipt["seed"], gym.public_input_hash(case), basis)
        assert_close(local["tangent_directions"], directions, "inverse seeded tangent directions", atol=1e-12, rtol=1e-12)
        assert_close(local["full_directions"], full, "inverse full directions", atol=1e-12, rtol=1e-12)
        fd = np.stack([(iterate(model, h + FD_STEP * d, event, horizon) - iterate(model, h - FD_STEP * d, event, horizon)) / (2 * FD_STEP) for d in full])
        jvp = full @ jacobian.T
        assert_close(fd, jvp, "inverse Jacobian finite difference", atol=1e-7, rtol=1e-4)
        assert_close(local["finite_difference"], fd, "retained inverse finite difference", atol=1e-10, rtol=1e-8)
        assert_close(local["jacobian_direction"], jvp, "retained inverse derivative", atol=1e-10, rtol=1e-8)
        require(abs(record["finite_difference_max_absolute_error"] - float(np.max(np.abs(fd - jvp)))) <= 1e-12,
                "inverse finite-difference summary")
        probes = [(amplitude, i, direction) for amplitude in AMPLITUDES
                  for i, direction in enumerate(np.concatenate((full, (basis @ vh[-1])[None])))]
        require(len(record["probes"]) == len(probes), "inverse perturbation roster")
        for p, (amplitude, direction_index, direction) in enumerate(probes):
            row = record["probes"][p]
            require(row["amplitude"] == amplitude and row["direction_index"] == direction_index and
                    row["kind"] == ("seeded" if direction_index < 4 else "least_singular"), "inverse perturbation identity")
            delta = amplitude * direction
            target = iterate(model, h + delta, event, horizon)
            back = basis @ pseudo @ basis.T @ (target - baseline)
            returned = iterate(model, h + back, event, horizon)
            for name, value in (("delta", delta), ("recovered_delta", back), ("target_output", target), ("return_output", returned)):
                assert_close(local[name][p], value, "inverse recovered tensor: " + name, atol=1e-10, rtol=1e-8)
            target_task, target_audit = model.readout(target)
            return_task, return_audit = model.readout(returned)
            for name, value in (("target_task_logits", target_task), ("target_audit_logits", target_audit),
                                ("return_task_logits", return_task), ("return_audit_logits", return_audit)):
                assert_close(local[name][p], value, "inverse readout replay: " + name, atol=1e-10, rtol=1e-8)
            for name, value in _error_metrics(delta, back, baseline, target, returned, model).items():
                require(row[name] == value if isinstance(value, bool) else abs(row[name] - value) <= 1e-8 * max(1., abs(value)),
                        "inverse metric replay: " + name)
        require(record["engineering_passed"] is True, "inverse record engineering gate")
    require(used == set(arrays), "unreferenced inverse arrays")
    require(receipt["degenerate_records"] == sum(not r["tangent_available"] for r in receipt["records"]), "inverse degeneracy total")
    require(receipt["engineering_passed"] is True, "inverse engineering gate")
    return {"seed": receipt["seed"], "jacobians": len(expected),
            "degenerate_records": receipt["degenerate_records"], "verified": True,
            "receipt_sha256": file_hash(receipt_path), "array_sha256": file_hash(archive_path)}


def anchors_from_evaluation(evaluation, seed):
    evaluation = Path(evaluation)
    manifest = json.loads((evaluation / "trace_manifest.json").read_text())
    records = manifest if isinstance(manifest, list) else manifest["records"]
    wanted = {case.canonical_id: case for case in _expected_anchors()}
    found = {}
    for record in records:
        if not (record["architecture"] == "native" and record["arm"] == "teacher" and record["seed"] == seed
                and record["split"] == "depth_context" and record["condition"] == "clean"):
            continue
        relative = record.get("relpath", record.get("file"))
        require(isinstance(relative, str), "inverse source trace path missing")
        with np.load(evaluation / relative, allow_pickle=False) as arrays:
            identifiers = arrays["canonical_ids"].tolist()
            for index, identifier in enumerate(identifiers):
                if identifier not in wanted:
                    continue
                require(identifier not in found, "duplicate inverse source anchor")
                case = wanted[identifier]
                require(arrays["inputs"].shape[1] == 3, "inverse source anchor schedule")
                require(np.array_equal(arrays["inputs"][index], np.asarray(gym.feature_sequence(case), dtype=np.float32)),
                        "inverse anchor full public input changed")
                found[identifier] = {"canonical_id": identifier,
                                     "public_input_sha256": gym.public_input_hash(case),
                                     "state": arrays["states"][index, 3].copy(),
                                     "event": arrays["inputs"][index, 2].copy(),
                                     "source_trace": {"file": relative, "index": index,
                                                      "sha256": file_hash(evaluation / relative)}}
    require(set(found) == set(wanted), "inverse source anchor coverage")
    return [found[case.canonical_id] for case in _expected_anchors()]


def main():
    import torch
    from .model import make_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for seed in SEEDS:
        source = args.checkpoints / f"native_teacher_s{seed}.npz"
        arrays = load_arrays(source)
        model = make_model("native")
        model.load_state_dict({name: torch.from_numpy(value) for name, value in arrays.items()})
        anchors = anchors_from_evaluation(args.evaluation, seed)
        receipt = run_inverse(model, source, anchors, args.out, seed)
        verified = verify_inverse(args.out / f"inverse_s{seed}.json", source, anchors)
        summaries.append(verified)
        print(json.dumps({"seed": seed, "jacobians": receipt["jacobians"], "engineering_passed": receipt["engineering_passed"]}), flush=True)
    summary = {"schema": "af02-inverse-summary-v1", "protocol_sha256": PROTOCOL_SHA256,
               "jacobians": sum(s["jacobians"] for s in summaries), "seeds": summaries,
               "engineering_passed": all(s["verified"] for s in summaries)}
    write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
