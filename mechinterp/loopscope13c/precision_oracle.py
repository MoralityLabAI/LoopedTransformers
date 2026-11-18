"""Source-only LGA-10C Hessian and native-arithmetic diagnostics.

This module performs tensor algebra and fresh derivatives on supplied models.
It creates no model, source, or RNG stream and imports no LGA-10B code. Native
checks run after candidate selection. The FP32 derivative oracle receives a
representable probe, while the physical delta is the FP64 subtraction of the
two stored FP32 states; those vectors can differ.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable

import torch


PAIR_DIMENSION = 96
CURVATURE_RTOL = 1e-8
DENSE_SKEW_RTOL = 1e-8
NATIVE_RTOL = 1e-6
NATIVE_ANGLE_ATOL = 2e-6
SOURCE_COSINE_LIMIT = 0.85
PAIRWISE_COSINE_LIMIT = 0.98
ROWSPACE_RTOL = 1e-9
ROWSPACE_ATOL = 1e-12


@dataclass
class NativePrecisionResult:
    receipt: dict
    raw: dict[str, torch.Tensor]


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.double()))


def _finite(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite torch tensor.")


def _pair(value: torch.Tensor, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    _finite(value, name=name)
    if value.shape not in ((96,), (2, 48)) or value.dtype != dtype:
        raise ValueError(f"{name} must have shape (96,) or (2,48) and dtype {dtype}.")
    return value.detach().reshape(96)


def _state(value: torch.Tensor, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    _finite(value, name=name)
    if value.shape != (1, 81, 48) or value.dtype != dtype:
        raise ValueError(f"{name} must have shape (1,81,48) and dtype {dtype}.")
    return value.detach()


def _goal_classes(goal: int, donor_class: int, original_class: int) -> None:
    if not 0 <= goal < 81 or not 0 <= donor_class < 10 or not 0 <= original_class < 10:
        raise ValueError("Goal and class indices are outside the physical model.")
    if donor_class == original_class:
        raise ValueError("A signed margin requires distinct donor and original classes.")


def dense_margin_hessian(
    hvp: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Measure all 96 coordinate columns of a supplied FP64 HVP oracle.

    The input coordinate vectors are CPU float64. A complete finite result
    is always returned, including a failed skew check, so its raw columns
    can be archived. ``receipt['passed']`` must gate use of the symmetrized
    result. Invalid or nonfinite individual columns raise explicitly.
    """
    started = time.perf_counter()
    coordinates = torch.eye(PAIR_DIMENSION, dtype=torch.float64)
    columns = []
    for coordinate in coordinates:
        column = _pair(hvp(coordinate), name="Hessian column", dtype=torch.float64)
        columns.append(column.clone())
    raw = torch.stack(columns, dim=1)
    skew = _norm(raw - raw.T)
    matrix_norm = _norm(raw)
    limit = DENSE_SKEW_RTOL * max(1.0, matrix_norm)
    symmetric = 0.5 * (raw + raw.T)
    receipt = {
        "passed": bool(skew <= limit),
        "hvp_count": PAIR_DIMENSION,
        "coordinate_order": "standard basis e_0 through e_95, columns H e_j",
        "raw_frobenius_norm": matrix_norm,
        "skew_frobenius_norm": skew,
        "relative_skew": skew / max(1.0, matrix_norm),
        "symmetry_limit": limit,
        "symmetry_rtol": DENSE_SKEW_RTOL,
        "seconds": time.perf_counter() - started,
        "dtype": "float64",
    }
    return symmetric, receipt, raw


def native_pair_jacobian(model, reached_h: torch.Tensor, goal: int):
    """Return the native-dtype ten-logit pair Jacobian and clean logits."""
    if reached_h.dtype not in (torch.float32, torch.float64):
        raise ValueError("Native derivative state must be float32 or float64.")
    _state(reached_h, name="reached_h", dtype=reached_h.dtype)
    if not 0 <= goal < 81:
        raise ValueError("Goal is outside the physical model.")
    predecessor = (goal - 1) % 81
    with torch.enable_grad():
        state = reached_h.detach().clone().requires_grad_(True)
        logits = model.head(model.block(model.block(state)))[0, goal]
        if logits.shape != (10,):
            raise ValueError("The native head must produce ten class logits.")
        rows = []
        for label in range(10):
            whole, = torch.autograd.grad(logits[label], state, retain_graph=label < 9)
            rows.append(whole[0, [predecessor, goal]].detach().reshape(96))
    jacobian = torch.stack(rows)
    _finite(jacobian, name="native Jacobian")
    _finite(logits, name="clean native logits")
    return jacobian, logits.detach()


def native_margin_hvp(model, reached_h: torch.Tensor, goal: int,
                      donor_class: int, original_class: int):
    """Construct a native-dtype HVP at the supplied clean reached state.

    Directions must already have the state's dtype. This deliberately
    exposes every cast to the caller and the raw precision receipt.
    """
    if reached_h.dtype not in (torch.float32, torch.float64):
        raise ValueError("Native derivative state must be float32 or float64.")
    base = _state(reached_h, name="reached_h", dtype=reached_h.dtype)
    _goal_classes(goal, donor_class, original_class)
    predecessor = (goal - 1) % 81

    def margin(edit):
        state = base.clone()
        state[0, predecessor] = state[0, predecessor] + edit[0]
        state[0, goal] = state[0, goal] + edit[1]
        logits = model.head(model.block(model.block(state)))[0, goal]
        return logits[donor_class] - logits[original_class]

    def hvp(direction):
        probe = _pair(direction, name="HVP direction", dtype=base.dtype)
        if probe.device != base.device:
            raise ValueError("HVP direction and native state must share a device.")
        zero = torch.zeros((2, 48), dtype=base.dtype, device=base.device)
        with torch.enable_grad():
            _, value = torch.autograd.functional.hvp(
                margin, zero, probe.reshape(2, 48), strict=False
            )
        return _pair(value, name="native HVP", dtype=base.dtype)

    return hvp


def _check_model_pair(model32, model64) -> dict:
    if model32.training or model64.training:
        raise ValueError("Both native oracle models must be in eval mode.")
    state32, state64 = model32.state_dict(), model64.state_dict()
    if state32.keys() != state64.keys():
        raise ValueError("The native and promoted model tensors differ.")
    floating_count = 0
    for key, value32 in state32.items():
        value64 = state64[key]
        if value32.is_floating_point():
            if value32.dtype != torch.float32 or value64.dtype != torch.float64:
                raise ValueError("Models must use native FP32 and promoted FP64 tensors.")
            _finite(value32, name=f"model32 tensor {key}")
            _finite(value64, name=f"model64 tensor {key}")
            equal = torch.equal(value32.double(), value64)
            floating_count += 1
        else:
            equal = torch.equal(value32, value64)
        if not equal:
            raise ValueError("model64 must contain exactly promoted model32 tensors.")
    return {"identical_when_promoted": True, "floating_tensor_count": floating_count}


def _native_geometry(head_projector, head, energy_gradient, jacobian):
    P = head_projector.detach().double()
    if P.shape != (48, 48):
        raise ValueError("head_projector must have shape (48,48).")
    _finite(P, name="head projector")
    projector_error = max(_norm(P - P.T), _norm(P @ P - P))
    if projector_error > 1e-8 or _norm(head @ P) > 1e-8 * max(1.0, _norm(head)):
        raise ValueError("head_projector must be the current-head orthogonal kernel projector.")
    pair_projector = torch.block_diag(P, P)
    energy_head = pair_projector @ energy_gradient
    energy_norm = _norm(energy_head)
    energy_active = energy_norm > max(ROWSPACE_ATOL, ROWSPACE_RTOL * _norm(energy_gradient))
    K = pair_projector.clone()
    if energy_active:
        K = K - torch.outer(energy_head, energy_head) / (energy_head @ energy_head)
    K = 0.5 * (K + K.T)
    restricted = jacobian @ K
    _, singular_values, vh = torch.linalg.svd(restricted, full_matrices=False)
    cutoff = max(ROWSPACE_ATOL, ROWSPACE_RTOL * float(singular_values[0]))
    rank = int((singular_values > cutoff).sum())
    row_basis = vh[:rank].T.contiguous()
    null_projector = K - row_basis @ row_basis.T
    null_projector = 0.5 * (null_projector + null_projector.T)
    diagnostics = {
        "energy_active": bool(energy_active),
        "native_jacobian_rank": rank,
        "native_jacobian_cutoff": cutoff,
        "native_k_dimension": int(round(float(torch.trace(K)))),
        "native_null_dimension": int(round(float(torch.trace(K)))) - rank,
        "head_projector_error": projector_error,
        "k_idempotence_error": _norm(K @ K - K),
        "null_projector_idempotence_error": _norm(null_projector @ null_projector - null_projector),
        "null_jacobian_residual": _norm(jacobian @ null_projector),
        "row_basis_in_k_error": _norm(K @ row_basis - row_basis),
    }
    return K, row_basis, null_projector, restricted, singular_values, diagnostics


@torch.no_grad()
def native_precision_check(
    model32, model64, h32: torch.Tensor, selected: torch.Tensor,
    shams: Iterable[torch.Tensor] | torch.Tensor, *, goal: int,
    donor_class: int, original_class: int, reference_h64: torch.Tensor,
    reference_j64: torch.Tensor, head_projector: torch.Tensor,
    target_curvature: float,
) -> NativePrecisionResult:
    """Check supplied edits using the actual native FP32 clean state.

    ``reference_h64`` and ``reference_j64`` belong to the promoted clean
    trajectory used for primary selection. They enter drift diagnostics.
    Arithmetic comparisons instead hold the native clean state fixed at
    ``h32.double()`` and the directional probe fixed at ``probe32.double()``.

    Native Q is ``physical_delta.double() @ HVP32(physical_delta.float()).double()``.
    Promoted physical Q uses a fresh HVP64 on the exact physical delta at
    the same lifted native state. Every diagnostic reduction uses FP64.
    The caller combines native_ready with the separate primary admission.
    """
    started = time.perf_counter()
    _goal_classes(goal, donor_class, original_class)
    base = _state(h32, name="h32", dtype=torch.float32)
    reference_h64 = _state(reference_h64, name="reference_h64", dtype=torch.float64)
    _finite(reference_j64, name="reference_j64")
    if reference_j64.shape not in ((10, 96), (10, 2, 48)) or reference_j64.dtype != torch.float64:
        raise ValueError("reference_j64 must be a float64 full ten-class pair Jacobian.")
    reference_j64 = reference_j64.detach().reshape(10, 96)
    if not math.isfinite(float(target_curvature)):
        raise ValueError("target_curvature must be finite.")
    promoted_weights = _check_model_pair(model32, model64)
    edits = [_pair(selected, name="selected", dtype=torch.float64)]
    edits.extend(_pair(sham, name="sham", dtype=torch.float64) for sham in shams)
    intended = torch.stack(edits)
    if any(value.device != base.device for value in
           (intended, reference_h64, reference_j64, head_projector)):
        raise ValueError("States, edits, Jacobians, and projector must share a device.")
    predecessor = (goal - 1) % 81
    pair_base = base[0, [predecessor, goal]].clone()
    intended32 = intended.float()
    pair_after = pair_base.unsqueeze(0) + intended32.reshape(-1, 2, 48)
    _finite(pair_after, name="native state-plus-edit result")
    actual = (pair_after.double() - pair_base.double().unsqueeze(0)).reshape(-1, 96)
    probe32 = actual.float()
    probe64 = probe32.double()
    lifted = base.double()
    count = len(edits)
    seconds = {}

    tick = time.perf_counter()
    jacobian32, logits32 = native_pair_jacobian(model32, base, goal)
    seconds["jacobian_native32"] = time.perf_counter() - tick
    tick = time.perf_counter()
    jacobian64, logits64 = native_pair_jacobian(model64, lifted, goal)
    seconds["jacobian_promoted_same_state"] = time.perf_counter() - tick
    oracle32 = native_margin_hvp(model32, base, goal, donor_class, original_class)
    oracle64 = native_margin_hvp(model64, lifted, goal, donor_class, original_class)
    tick = time.perf_counter()
    hvp32 = torch.stack([oracle32(probe) for probe in probe32])
    seconds["hvp_native32"] = time.perf_counter() - tick
    tick = time.perf_counter()
    hvp64_probe = torch.stack([oracle64(probe) for probe in probe64])
    seconds["hvp_promoted_probe"] = time.perf_counter() - tick
    tick = time.perf_counter()
    hvp64_actual = torch.stack([oracle64(delta) for delta in actual])
    seconds["hvp_promoted_exact"] = time.perf_counter() - tick

    J32 = jacobian32.double()
    H32 = hvp32.double()
    head = model32.head.weight.detach().double()
    energy_gradient = pair_base.double().reshape(96)
    K, R, N, restricted, singular_values, geometry = _native_geometry(
        head_projector, head, energy_gradient, J32
    )
    radius = _norm(intended[0])
    q_tolerance = CURVATURE_RTOL * max(1.0, abs(float(target_curvature)))
    limits = {
        "curvature": q_tolerance,
        "full_j": NATIVE_RTOL * max(1.0, _norm(J32 @ actual[0])),
        "k": NATIVE_RTOL * max(1.0, radius),
        "dose": NATIVE_RTOL * max(1.0, radius),
        "head": NATIVE_RTOL * max(1.0, float(torch.linalg.matrix_norm(head, ord=2)) * radius),
        "energy": NATIVE_RTOL * max(1.0, _norm(energy_gradient) * radius),
        "source_abs_cosine": SOURCE_COSINE_LIMIT + NATIVE_ANGLE_ATOL,
        "pairwise_signed_cosine": PAIRWISE_COSINE_LIMIT + NATIVE_ANGLE_ATOL,
    }
    dark = actual @ N.T
    dark_norms = torch.linalg.vector_norm(dark, dim=1)
    dark_units = dark / dark_norms.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(1)
    source_cosines = dark_units[1:] @ dark_units[0]
    gram = dark_units[1:] @ dark_units[1:].T
    q_native = (actual * H32).sum(dim=1)
    q64_actual = (actual * hvp64_actual).sum(dim=1)
    q32_same_probe = (probe64 * H32).sum(dim=1)
    q64_same_probe = (probe64 * hvp64_probe).sum(dim=1)
    native_parity = q_native - q_native[0]
    promoted_parity = q64_actual - q64_actual[0]
    paired_resolution = (native_parity - promoted_parity).abs()
    probe_resolution = (q32_same_probe - q64_same_probe).abs()
    hvp_errors = torch.linalg.vector_norm(H32 - hvp64_probe, dim=1)
    hvp_reference_norms = torch.linalg.vector_norm(hvp64_probe, dim=1)
    hvp_limits = NATIVE_RTOL * hvp_reference_norms.clamp_min(1.0)
    j_error, j_reference_norm = _norm(J32 - jacobian64), _norm(jacobian64)
    j_limit = NATIVE_RTOL * max(1.0, j_reference_norm)
    forward_error = float((logits32.double() - logits64).abs().max())
    forward_reference_norm = float(logits64.abs().max())
    forward_limit = NATIVE_RTOL * max(1.0, forward_reference_norm)
    common_derivative_passed = j_error <= j_limit and forward_error <= forward_limit
    k_errors = torch.linalg.vector_norm(actual @ K.T - actual, dim=1)
    j_errors = torch.linalg.vector_norm((actual - actual[0]) @ J32.T, dim=1)
    dose_errors = (torch.linalg.vector_norm(actual, dim=1) - radius).abs()
    head_errors = torch.linalg.vector_norm(actual.reshape(-1, 2, 48) @ head.T, dim=(1, 2))
    energy_errors = (actual @ energy_gradient).abs()
    cast_errors = torch.linalg.vector_norm(actual - probe64, dim=1)
    application_errors = torch.linalg.vector_norm(actual - intended, dim=1)
    arms = []
    for index in range(count):
        pairwise = -1.0
        if index > 0 and count > 2:
            mask = torch.arange(count - 1, device=base.device) != index - 1
            pairwise = float(gram[index - 1, mask].max())
        errors = {
            "curvature": abs(float(native_parity[index])),
            "full_j": float(j_errors[index]),
            "k": float(k_errors[index]),
            "dose": float(dose_errors[index]),
            "head": float(head_errors[index]),
            "energy": float(energy_errors[index]),
            "source_abs_cosine": abs(float(source_cosines[index - 1])) if index else 0.0,
            "pairwise_signed_cosine": pairwise,
        }
        physical = bool(dark_norms[0] > 0 and dark_norms[index] > 0) and all(
            math.isfinite(value) and value <= limits[key] for key, value in errors.items()
        )
        promoted = abs(float(promoted_parity[index])) <= q_tolerance
        resolution = float(paired_resolution[index]) <= q_tolerance and float(probe_resolution[index]) <= q_tolerance
        derivative = common_derivative_passed and float(hvp_errors[index]) <= float(hvp_limits[index])
        arms.append({
            "index": index, "label": "selected" if index == 0 else f"sham_{index - 1}",
            "native_physical": bool(physical),
            "promoted_curvature": bool(promoted),
            "curvature_resolution": bool(resolution),
            "derivative_agreement": bool(derivative),
            "native_ready": bool(physical and promoted and resolution and derivative),
            "errors": errors,
            "q_native": float(q_native[index]),
            "q64_actual": float(q64_actual[index]),
            "q32_same_probe": float(q32_same_probe[index]),
            "q64_same_probe": float(q64_same_probe[index]),
            "native_parity": float(native_parity[index]),
            "promoted_parity": float(promoted_parity[index]),
            "paired_resolution_error": float(paired_resolution[index]),
            "probe_resolution_error": float(probe_resolution[index]),
            "native_vs_promoted_physical_error": abs(float(q_native[index] - q64_actual[index])),
            "promoted_direction_cast_curvature_drift": abs(float(q64_same_probe[index] - q64_actual[index])),
            "hvp_error": float(hvp_errors[index]),
            "hvp_reference_norm": float(hvp_reference_norms[index]),
            "hvp_limit": float(hvp_limits[index]),
            "direction_cast_norm": float(cast_errors[index]),
            "application_rounding_norm": float(application_errors[index]),
            "dark_radius": float(dark_norms[index]),
        })

    state_drift = lifted - reference_h64
    jacobian_drift = jacobian64 - reference_j64
    receipt = {
        "schema": "lga10c_native_precision_v1",
        "native_ready": all(arm["native_ready"] for arm in arms),
        **{key: all(arm[key] for arm in arms) for key in
           ("native_physical", "promoted_curvature", "curvature_resolution", "derivative_agreement")},
        "goal": goal, "donor_class": donor_class, "original_class": original_class,
        "n_shams": count - 1, "target_curvature": float(target_curvature),
        "q_tolerance": q_tolerance, "physical_limits": limits,
        "curvature_estimator": "physical_delta64 dot HVP32(physical_delta64.float()).double()",
        "reference_curvature_estimator": "physical_delta64 dot HVP64_at_h32_double(physical_delta64)",
        "weight_promotion": promoted_weights, "geometry": geometry,
        "jacobian_agreement": {"passed": bool(j_error <= j_limit), "error": j_error,
                               "reference_norm": j_reference_norm, "limit": j_limit},
        "forward_agreement": {"passed": bool(forward_error <= forward_limit), "error": forward_error,
                              "reference_norm": forward_reference_norm, "limit": forward_limit},
        "state_drift": {"full_l2": _norm(state_drift),
                        "pair_l2": _norm(state_drift[0, [predecessor, goal]]),
                        "max_abs": float(state_drift.abs().max())},
        "jacobian_state_drift": {"frobenius_norm": _norm(jacobian_drift),
                                 "relative_to_max_one_reference": _norm(jacobian_drift) / max(1.0, _norm(reference_j64))},
        "arms": arms,
        "hvp_counts": {"native32": count, "promoted_exact": count, "promoted_probe": count},
        "jacobian_counts": {"native32": 10, "promoted_same_state": 10},
        "seconds": seconds,
        "oracle_input_keys": {
            "native_hvp_state": "native_h32", "native_hvp_direction": "probe32",
            "promoted_hvp_state": "native_h32_lifted",
            "promoted_probe_hvp_direction": "probe64",
            "promoted_exact_hvp_direction": "actual_edits_native_fp64",
        },
    }
    seconds["total"] = time.perf_counter() - started
    raw = {
        "intended_edits": intended,
        "intended_edits_fp32": intended32,
        "native_h32": base,
        "native_h32_lifted": lifted,
        "reference_h64": reference_h64,
        "native_pair_base_fp32": pair_base,
        "native_pair_after_fp32": pair_after,
        "actual_edits_native_fp64": actual,
        "probe32": probe32,
        "probe64": probe64,
        "direction_cast_residual": actual - probe64,
        "application_rounding_residual": actual - intended,
        "jacobian32": jacobian32,
        "jacobian64_same_state": jacobian64,
        "reference_j64": reference_j64,
        "clean_logits32": logits32,
        "clean_logits64_same_state": logits64,
        "hvp32": hvp32,
        "hvp64_same_probe": hvp64_probe,
        "hvp64_actual_delta": hvp64_actual,
        "native_head_weight": model32.head.weight.detach(),
        "head_projector": head_projector.detach().double(),
        "energy_gradient_native": energy_gradient,
        "k_native": K,
        "row_basis_native": R,
        "null_projector_native": N,
        "native_restricted_jacobian": restricted,
        "native_jacobian_singular_values": singular_values,
        "dark_native": dark,
        "dark_units_native": dark_units,
        "q_native": q_native,
        "q64_actual": q64_actual,
        "q32_same_probe": q32_same_probe,
        "q64_same_probe": q64_same_probe,
        "native_parity": native_parity,
        "promoted_parity": promoted_parity,
        "paired_resolution_errors": paired_resolution,
        "probe_resolution_errors": probe_resolution,
        "source_cosines": source_cosines,
        "pairwise_cosines": gram,
    }
    return NativePrecisionResult(receipt, {key: value.detach().clone() for key, value in raw.items()})
