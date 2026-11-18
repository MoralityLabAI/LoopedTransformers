"""LGA-10D geometry with one native source fiber and two derivative states.

This module accepts supplied models and reached states. It initializes no
model, input cohort or global random stream. The frozen LGA-10B proposal
arithmetic and LGA-10C physical admission code remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch

from mechinterp.loopscope11.pair_fiber import plan_pair, project_into_k
from mechinterp.loopscope12.full_jacobian import FullFutureFiber, measure_full_response
from mechinterp.loopscope13.curvature_match import native_margin_hvp
from mechinterp.loopscope13b.spectral_match import SpectralPlan, make_spectral_shams
from mechinterp.loopscope13c.precision_oracle import dense_margin_hessian
from mechinterp.loopscope13c.roundtrip_match import PhysicalGate, make_roundtrip_shams


CONSTRUCTION_CATEGORIES = ("reference_construction", "dense_primary", "external_validation")


def _state(value: torch.Tensor, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.shape != (1, 81, 48)
            or value.dtype != dtype or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite CPU {dtype} state with shape (1,81,48)")
    return value.detach()


def _model(model64) -> None:
    if model64.training:
        raise ValueError("The promoted geometry model must be in eval mode")
    for value in model64.state_dict().values():
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float64):
            raise ValueError("The promoted geometry model must use CPU float64 tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("The promoted geometry model contains a nonfinite tensor")


def build_common_native_parent(model64, observer, h32: torch.Tensor,
                               hdon32: torch.Tensor, mu: torch.Tensor,
                               std: torch.Tensor, *, goal: int,
                               original_class: int, donor_class: int,
                               teacher_seed: int, base_index: int,
                               head_projector: torch.Tensor) -> dict:
    """Lift both native states before computing the natural source displacement.

    The existing ``plan_pair`` source eligibility is retained, including its
    auxiliary random-K and signed-margin control construction. Its selected
    donor, natural radius and K fiber are subsequently shared by both arms.
    """
    _model(model64)
    recipient = _state(h32, name="h32", dtype=torch.float32)
    donor = _state(hdon32, name="hdon32", dtype=torch.float32)
    lifted_recipient, lifted_donor = recipient.double(), donor.double()
    parent = plan_pair(
        model64, observer, lifted_recipient, lifted_donor, mu, std,
        goal=goal, old_class=original_class, donor_class=donor_class,
        teacher_seed=teacher_seed, base_index=base_index,
        head_projector=head_projector)
    pair = [(goal - 1) % 81, goal]
    exact_difference = (lifted_donor - lifted_recipient)[0, pair]
    if not torch.equal(parent["fiber"].donor, exact_difference):
        raise ArithmeticError("Common native donor differs from the exact lifted-state subtraction")
    parent["native_source"] = {
        "schema": "LGA10D/common-native-source-v1",
        "source_state_dtype": "float32", "geometry_dtype": "float64",
        "subtraction_order": "hdon32.double() - h32.double()",
        "retained_auxiliary_parent_eligibility": True,
        "teacher_seed": teacher_seed, "base_index": base_index, "goal": goal,
        "original_class": original_class, "donor_class": donor_class,
    }
    parent["native_source_states"] = {"native_recipient_state": recipient.clone(),
                                     "native_donor_state": donor.clone()}
    return parent


@dataclass
class GeometryResult:
    eligible: bool
    reason: str | None
    receipt: dict
    raw: dict[str, torch.Tensor]
    hvp: Callable
    reference: SpectralPlan | None = None
    plan: SpectralPlan | None = None
    future: FullFutureFiber | None = None
    hessian: torch.Tensor | None = None
    hessian_raw: torch.Tensor | None = None
    dense_receipt: dict | None = None


def build_geometry(model64, derivative_state64: torch.Tensor,
                   application_state32: torch.Tensor, common_parent: dict,
                   head_projector: torch.Tensor, *, teacher_seed: int,
                   base_index: int, goal: int, original_class: int,
                   donor_class: int) -> GeometryResult:
    """Construct one arm using its derivative state and the shared native fiber.

    Each arm retains its own frozen spectral target and construction
    tolerance. The runner supplies the common native target to the later
    final validators. ``hvp(direction, category)`` remains available for fresh
    validation; its receipt counts and timing dictionaries update in place.

    Completed HVPs and attempted HVPs are counted separately. Computed
    spectral failures return their complete plan. A thrown numerical error
    returns a stage-labelled failure and the partial construction HVP trace.
    """
    _model(model64)
    derivative = _state(derivative_state64, name="derivative_state64", dtype=torch.float64)
    application = _state(application_state32, name="application_state32", dtype=torch.float32)
    if not 0 <= goal < 81 or not 0 <= original_class < 10 or not 0 <= donor_class < 10:
        raise ValueError("Goal or source class lies outside the physical model")
    if original_class == donor_class:
        raise ValueError("A signed margin requires different source classes")
    if (head_projector.shape != (48, 48) or head_projector.dtype != torch.float64
            or head_projector.device.type != "cpu" or not bool(torch.isfinite(head_projector).all())):
        raise ValueError("The common head projector must be finite CPU float64 with shape (48,48)")
    source = common_parent.get("native_source")
    if source is not None:
        identity = {"teacher_seed": teacher_seed, "base_index": base_index,
                    "goal": goal, "original_class": original_class, "donor_class": donor_class}
        if any(source.get(key) != value for key, value in identity.items()):
            raise ValueError("Geometry identity differs from the common native source")
        if not torch.equal(common_parent["native_source_states"]["native_recipient_state"], application):
            raise ValueError("Geometry application state differs from the common native source")
    started = time.perf_counter()
    counts = {key: 0 for key in CONSTRUCTION_CATEGORIES}
    attempts = {key: 0 for key in CONSTRUCTION_CATEGORIES}
    seconds = {key: 0. for key in CONSTRUCTION_CATEGORIES}
    traces: dict[str, dict[str, list]] = {}
    recording = True
    oracle = native_margin_hvp(model64, derivative, goal, donor_class, original_class)

    def counted_hvp(direction, category="external_validation"):
        counts.setdefault(category, 0)
        attempts[category] = attempts.get(category, 0) + 1
        seconds.setdefault(category, 0.)
        trace = traces.setdefault(category, {"directions": [], "values": []}) if recording else None
        if trace is not None:
            trace["directions"].append(direction.detach().double().reshape(96).clone())
        tick = time.perf_counter()
        try:
            value = oracle(direction)
            if trace is not None:
                trace["values"].append(value.detach().double().reshape(96).clone())
            counts[category] += 1
            return value
        finally:
            seconds[category] += time.perf_counter() - tick

    pair = [(goal - 1) % 81, goal]
    receipt = {
        "schema": "LGA10D/derivative-state-geometry-v1",
        "teacher_seed": teacher_seed, "base_index": base_index, "goal": goal,
        "original_class": original_class, "donor_class": donor_class,
        "derivative_dtype": "float64", "application_dtype": "float32",
        "derivative_state_equals_native_lift": bool(torch.equal(derivative, application.double())),
        "common_native_source": source is not None,
        "common_parent_eligible": bool(common_parent.get("eligible")),
        "common_parent_reason": common_parent.get("reason"),
        "hvp_counts": counts, "hvp_attempt_counts": attempts, "hvp_seconds": seconds,
        "construction_tolerance_source": "this arm's unchanged LGA-10B selected curvature",
        "stage": "common_source", "eligible": False, "reason": None,
    }
    raw = {"derivative_state64": derivative.clone(),
           "application_state32": application.clone(),
           "application_pair64": application[0, pair].double().reshape(96)}
    result = GeometryResult(False, None, receipt, raw, counted_hvp)

    def finish(reason):
        nonlocal recording
        recording = False
        traces.clear()
        result.eligible, result.reason = reason is None, reason
        receipt.update({"eligible": result.eligible, "reason": reason,
                        "elapsed_seconds": time.perf_counter() - started})
        return result

    try:
        fiber = common_parent["fiber"]
        selected = common_parent["directions"]["pair_energy_tangent"].detach().double().reshape(96)

        def project(value):
            return project_into_k(value.reshape(2, 48), fiber, head_projector).reshape(96)

        k = torch.stack([project(axis) for axis in torch.eye(96, dtype=torch.float64)], dim=1)
        raw.update({"selected": selected.clone(), "natural_pair": fiber.donor.reshape(96).clone(),
                    "energy_gradient": fiber.egrad.reshape(96).clone(), "k_projector": k})
        receipt["radius"] = float(common_parent["radius"])
        receipt["k_dimension"] = int(fiber.dimension)
        if not common_parent.get("eligible"):
            return finish("COMMON_NATIVE_SOURCE_FAILURE")
        receipt["stage"] = "future_jacobian"
        future = measure_full_response(model64, derivative, goal, common_parent, head_projector)
        result.future = future
        raw.update({"future_jacobian": future.full_jacobian.reshape(10, 96).clone(),
                    "restricted_jacobian": future.restricted_jacobian.clone(),
                    "row_basis": future.row_basis.clone()})
        receipt.update({"rank_future_j": int(future.rank),
                        "n_full_j_dark": int(future.future_null_dimension)})
        receipt["stage"] = "spectral_reference"
        reference = make_spectral_shams(
            selected, future.row_basis, project,
            lambda value: counted_hvp(value, "reference_construction"),
            teacher_seed=teacher_seed, base_index=base_index, goal=goal,
            expected_complement_dim=64)
        result.reference = reference
        receipt.update({"reference_eligible": bool(reference.eligible), "reference_reason": reference.reason})
        if "selected_curvature" not in reference.receipt:
            return finish(reference.reason or "REFERENCE_TARGET_UNAVAILABLE")
        own_target = float(reference.receipt["selected_curvature"])
        receipt["construction_target_curvature"] = own_target
        receipt["construction_curvature_tolerance"] = float(reference.receipt["curvature_tolerance"])
        receipt["stage"] = "dense_primary"
        dense, dense_receipt, dense_raw = dense_margin_hessian(
            lambda value: counted_hvp(value, "dense_primary"))
        result.hessian, result.hessian_raw, result.dense_receipt = dense, dense_raw, dense_receipt
        raw.update({"dense_hessian": dense, "dense_hessian_raw": dense_raw})
        if not dense_receipt["passed"]:
            return finish("DENSE_HESSIAN_SYMMETRY_FAILURE")
        receipt["stage"] = "roundtrip_selection"
        gate = PhysicalGate(
            selected=selected, hpair=raw["application_pair64"], hessian=dense,
            k_projector=k, jacobian=raw["future_jacobian"], head=model64.head.weight.detach(),
            egrad=raw["energy_gradient"], row_basis=future.row_basis,
            q0=reference.raw["q0"], target_curvature=own_target)
        plan = make_roundtrip_shams(reference, gate, teacher_seed=teacher_seed,
                                    base_index=base_index, goal=goal)
        result.plan = plan
        receipt["stage"] = "complete"
        return finish(plan.reason)
    except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
        receipt["exception"] = {"stage": receipt["stage"], "type": type(error).__name__, "message": str(error)}
        for category, trace in traces.items():
            for kind, values in trace.items():
                raw["partial_hvp__" + category + "__" + kind] = (
                    torch.stack(values) if values else torch.empty((0, 96), dtype=torch.float64))
        return finish("EXPLICIT_GEOMETRY_EXCEPTION")
