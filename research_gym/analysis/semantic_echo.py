"""CPU metrics for the Semantic Echo pilot.

Each trajectory belongs to one fixed permutation graph and one start node.
``predictions[t - 1]`` is the predicted node *after* loop ``t``. Structural
visit pairs are chosen from the exact oracle before inspecting predictions.
Only endpoint correctness gates a pair; future predictions never select pairs.

The outputs describe trajectories. Overlapping visits within a graph are
dependent observations, so pair counts are denominators, not sample sizes for
independent-observation confidence intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np

SCHEMA_VERSION = "semantic_echo.trajectory.v1"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _integer_vector(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not arr.size:
        if allow_empty:
            return np.empty(0, dtype=np.int64)
        raise ValueError(f"{name} must be nonempty")
    if arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers, excluding booleans")
    if arr.dtype.kind == "u" and np.any(arr > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains an integer outside int64 range")
    return arr.astype(np.int64, copy=True)


def validate_permutation(permutation: Any) -> np.ndarray:
    """Return an independent int64 permutation; reject partial maps/duplicates."""
    p = _integer_vector(permutation, "permutation")
    if not np.array_equal(np.sort(p), np.arange(len(p), dtype=np.int64)):
        raise ValueError("permutation must contain every integer in 0..N-1 exactly once")
    return p


def _start_node(start: Any, n_nodes: int) -> int:
    start = _integer(start, "start")
    if start >= n_nodes:
        raise ValueError("start must be in 0..N-1")
    return start


def _labels(predictions: Any, n_nodes: int) -> np.ndarray:
    pred = _integer_vector(predictions, "predictions")
    if np.any(pred < 0) or np.any(pred >= n_nodes):
        raise ValueError("predictions must be node labels in 0..N-1")
    return pred


def _cycle(p: np.ndarray, start: int) -> list[int]:
    nodes = [start]
    node = int(p[start])
    while node != start:
        nodes.append(node)
        node = int(p[node])
    return nodes


def cycle_nodes(permutation: Any, start: int) -> list[int]:
    """Cycle in successor order, with ``start`` at phase index zero."""
    p = validate_permutation(permutation)
    return _cycle(p, _start_node(start, len(p)))


def cycle_length(permutation: Any, start: int) -> int:
    return len(cycle_nodes(permutation, start))


def oracle_trajectory(permutation: Any, start: int, n_steps: int) -> np.ndarray:
    """Return pi(start), pi^2(start), ..., pi^n_steps(start)."""
    p = validate_permutation(permutation)
    current = _start_node(start, len(p))
    n_steps = _integer(n_steps, "n_steps")
    out = np.empty(n_steps, dtype=np.int64)
    for i in range(n_steps):
        current = int(p[current])
        out[i] = current
    return out


def normalized_phase_residual(phases: Any, cycle_size: int) -> dict[str, Any]:
    """m=1 phase-law residual for already normalized complex observations.

    The convention is z(s)=exp(2*pi*i*index(s)/ell), so an exact successor
    satisfies z_next=omega*z. Dividing by |1-omega|^2 makes a frozen unit
    phase score one at every informative cycle length. A fixed point has a
    zero normalization denominator and is explicitly uninformative.
    """
    ell = _integer(cycle_size, "cycle_size", 1)
    z = np.asarray(phases, dtype=np.complex128)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("phases must be a nonempty finite one-dimensional array")
    if not np.allclose(np.abs(z), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("phases must have unit magnitude")
    if ell == 1:
        return {"mode": 1, "informative": False, "reason": "fixed_point",
                "n_transitions": max(0, len(z) - 1), "normalizer": 0.0,
                "normalized_residuals": None, "mean_normalized_residual": None}
    omega = np.exp(2j * np.pi / ell)
    denominator = float(abs(1 - omega) ** 2)
    residuals = np.abs(z[1:] - omega * z[:-1]) ** 2 / denominator
    return {"mode": 1, "informative": True, "reason": None,
            "n_transitions": len(residuals), "normalizer": denominator,
            "normalized_residuals": residuals.tolist(),
            "mean_normalized_residual": float(residuals.mean()) if residuals.size else None}


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def phase_telescoping(phases: Any, cycle_size: int) -> dict[str, Any]:
    """Check the telescoping identity and its finite-horizon error bound.

    The certificate concerns the nearest root to omega^T*z0 and assumes z0
    is a cycle root. It does not certify that z0 has the task's correct label.
    """
    score = normalized_phase_residual(phases, cycle_size)
    z = np.asarray(phases, dtype=np.complex128)
    ell = int(cycle_size)
    count = len(z) - 1
    if ell == 1:
        return {"informative": False, "reason": "fixed_point", "n_transitions": count,
                "identity_error": None, "rms_bound": None,
                "certifies_phase_relative_to_initial": False}
    omega = np.exp(2j * np.pi / ell)
    residuals = z[1:] - omega * z[:-1]
    weighted = sum((omega ** (count - 1 - j)) * residuals[j] for j in range(count))
    deviation = z[-1] - omega ** count * z[0]
    separation = float(abs(1 - omega))
    mean = score["mean_normalized_residual"]
    bound = separation * count * np.sqrt(mean) if count else 0.0
    roots = np.exp(2j * np.pi * np.arange(ell) / ell)
    initial_on_root = bool(np.min(np.abs(roots - z[0])) < 1e-10)
    return {"informative": True, "reason": None, "n_transitions": count,
            "initial_on_cycle_root": initial_on_root,
            "final_deviation": float(abs(deviation)), "identity_error": float(abs(deviation - weighted)),
            "sum_absolute_residuals": float(np.abs(residuals).sum()), "rms_bound": float(bound),
            "half_root_separation": separation / 2,
            "certifies_phase_relative_to_initial": bool(initial_on_root and bound < separation / 2),
            "anchor_note": "The initial phase must separately match the task oracle."}


def _phase_labels(cycle: list[int], pred: np.ndarray) -> dict[str, Any]:
    ell = len(cycle)
    index = {node: i for i, node in enumerate(cycle)}
    valid = [int(node) in index for node in pred]
    off_cycle = [{"loop": i + 1, "label": int(node)}
                 for i, node in enumerate(pred) if not valid[i]]
    denominator = float(abs(1 - np.exp(2j * np.pi / ell)) ** 2) if ell > 1 else 0.0
    omega = np.exp(2j * np.pi / ell)
    rows = []
    values = []
    for i in range(len(pred) - 1):
        in_cycle = valid[i] and valid[i + 1]
        residual = None
        if ell > 1 and in_cycle:
            za = np.exp(2j * np.pi * index[int(pred[i])] / ell)
            zb = np.exp(2j * np.pi * index[int(pred[i + 1])] / ell)
            residual = float(abs(zb - omega * za) ** 2 / denominator)
            values.append(residual)
        rows.append({"from_loop": i + 1, "to_loop": i + 2,
                     "both_labels_in_start_cycle": in_cycle,
                     "normalized_residual": residual})
    n_valid = sum(row["both_labels_in_start_cycle"] for row in rows)
    return {"mode": 1, "cycle_nodes": cycle, "informative": ell > 1,
            "reason": "fixed_point" if ell == 1 else None,
            "normalizer": denominator, "n_transitions": len(rows),
            "n_valid_transitions": n_valid, "n_invalid_transitions": len(rows) - n_valid,
            "valid_transition_coverage": _ratio(n_valid, len(rows)),
            "n_scored_transitions": len(values),
            "off_cycle_prediction_count": len(off_cycle), "off_cycle_predictions": off_cycle,
            "mean_normalized_residual": float(np.mean(values)) if values else None,
            "raw_rows": rows}


def analyze_trajectory(
    permutation: Any,
    start: int,
    predictions: Any,
    *,
    horizon: int = 4,
    min_visit: int = 1,
    max_visit: int | None = None,
) -> dict[str, Any]:
    """Score one graph/start trajectory, returning JSON-safe scalars and rows.

    ``min_visit`` and ``max_visit`` are inclusive one-based endpoint bounds.
    An omitted upper bound uses every visit with a complete future horizon.
    An explicit upper bound can exceed the observed trace; available visits
    still stop at T-horizon. Empty denominators return ``None``.

    Continuation mismatch compares the two predicted futures at offsets
    1..horizon. Oracle continuation accuracy compares *both* futures to
    pi^r(s), where s is their shared exact endpoint state.
    """
    p = validate_permutation(permutation)
    start = _start_node(start, len(p))
    pred = _labels(predictions, len(p))
    horizon = _integer(horizon, "horizon", 1)
    min_visit = _integer(min_visit, "min_visit", 1)
    if max_visit is not None:
        max_visit = _integer(max_visit, "max_visit", 1)
        if max_visit < min_visit:
            raise ValueError("max_visit must be >= min_visit")
    oracle = oracle_trajectory(p, start, len(pred))
    correct = pred == oracle
    upper = min(len(pred) - horizon, max_visit if max_visit is not None else len(pred))
    structural = gated = disagree_pairs = mismatch_steps = oracle_hits = both_hits = 0
    first_witness = first_oracle_failure = None
    raw_rows = []
    for t in range(min_visit, upper + 1):
        for u in range(t + 1, upper + 1):
            if oracle[t - 1] != oracle[u - 1]:
                continue
            structural += 1
            gate = bool(correct[t - 1] and correct[u - 1])
            row = {"t": t, "u": u, "semantic_label": int(oracle[t - 1]),
                   "prediction_t": int(pred[t - 1]), "prediction_u": int(pred[u - 1]),
                   "endpoint_correct_t": bool(correct[t - 1]),
                   "endpoint_correct_u": bool(correct[u - 1]), "endpoint_gate": gate,
                   "expected_future": None, "future_predictions_t": None,
                   "future_predictions_u": None, "step_mismatches": None,
                   "oracle_correct_t": None, "oracle_correct_u": None,
                   "pair_any_disagreement": None}
            if gate:
                gated += 1
                expected = oracle_trajectory(p, int(oracle[t - 1]), horizon)
                ft, fu = pred[t:t + horizon], pred[u:u + horizon]
                mismatch = ft != fu
                hit_t, hit_u = ft == expected, fu == expected
                mismatch_steps += int(mismatch.sum())
                oracle_hits += int(hit_t.sum() + hit_u.sum())
                both_hits += int((hit_t & hit_u).sum())
                any_disagreement = bool(mismatch.any())
                disagree_pairs += int(any_disagreement)
                row.update(expected_future=expected.tolist(), future_predictions_t=ft.tolist(),
                           future_predictions_u=fu.tolist(), step_mismatches=mismatch.tolist(),
                           oracle_correct_t=hit_t.tolist(), oracle_correct_u=hit_u.tolist(),
                           pair_any_disagreement=any_disagreement)
                if first_witness is None and any_disagreement:
                    r = int(np.flatnonzero(mismatch)[0])
                    first_witness = {"kind": "continuation_disagreement", "t": t, "u": u,
                                     "offset": r + 1, "semantic_label": int(oracle[t - 1]),
                                     "expected": int(expected[r]),
                                     "prediction_from_t": int(ft[r]), "prediction_from_u": int(fu[r])}
                failed = ~(hit_t & hit_u)
                if first_oracle_failure is None and failed.any():
                    r = int(np.flatnonzero(failed)[0])
                    first_oracle_failure = {"t": t, "u": u, "offset": r + 1,
                                            "expected": int(expected[r]),
                                            "prediction_from_t": int(ft[r]),
                                            "prediction_from_u": int(fu[r])}
            raw_rows.append(row)
    steps = gated * horizon
    transitions = pred[1:] == p[pred[:-1]]
    return {
        "schema_version": SCHEMA_VERSION, "n_nodes": len(p), "start": start,
        "n_loops": len(pred), "cycle_length": len(_cycle(p, start)),
        "horizon": horizon, "min_visit": min_visit, "max_visit": max_visit,
        "last_available_visit": upper,
        "indexing": "predictions[t-1] is after loop t; oracle is pi^t(start), t>=1",
        "oracle_labels": oracle.tolist(), "predicted_labels": pred.tolist(),
        "per_loop_correct": correct.tolist(), "n_label_correct": int(correct.sum()),
        "n_labels": len(pred), "label_accuracy": float(correct.mean()),
        "transition": {"correct_count": int(transitions.sum()), "comparisons": len(transitions),
                       "accuracy": _ratio(int(transitions.sum()), len(transitions))},
        "phase": _phase_labels(_cycle(p, start), pred),
        "continuation": {
            "eligible_pairs": structural, "endpoint_gated_pairs": gated,
            "endpoint_gate_coverage": _ratio(gated, structural),
            "pair_any_disagreement_count": disagree_pairs,
            "pair_any_disagreement_rate": _ratio(disagree_pairs, gated),
            "per_step_mismatch_count": mismatch_steps, "per_step_comparisons": steps,
            "per_step_mismatch_rate": _ratio(mismatch_steps, steps),
            "oracle_correct_count": oracle_hits, "oracle_comparisons": 2 * steps,
            "oracle_continuation_accuracy": _ratio(oracle_hits, 2 * steps),
            "both_oracle_correct_count": both_hits, "both_oracle_comparisons": steps,
            "both_oracle_correct_rate": _ratio(both_hits, steps),
            "first_witness": first_witness, "first_oracle_failure": first_oracle_failure,
            "raw_rows": raw_rows,
        },
        "dependence_note": "Visit pairs overlap within one graph/start; pair counts are not independent sample sizes.",
    }


def aggregate_trajectories(
    results: Sequence[dict[str, Any]], *, unit: str = "item"
) -> dict[str, Any]:
    """Unweighted descriptive means over caller-specified sampling units.

    An item can be an iid draw of a permutation and start; repeated values of
    that draw remain distinct items. When starts share a sampled graph, the
    caller must group them before graph-level inferential resampling. Merely
    choosing a unit name makes no independence claim. Empty metric denominators
    are omitted from that metric's mean and counted explicitly in coverage.
    """
    if unit not in ("item", "graph", "trajectory"):
        raise ValueError("unit must be 'item', 'graph', or 'trajectory'")
    metrics = {
        "label_accuracy": [r["label_accuracy"] for r in results],
        "endpoint_gate_coverage": [r["continuation"]["endpoint_gate_coverage"] for r in results],
        "pair_any_disagreement_rate": [r["continuation"]["pair_any_disagreement_rate"] for r in results],
        "per_step_mismatch_rate": [r["continuation"]["per_step_mismatch_rate"] for r in results],
        "oracle_continuation_accuracy": [r["continuation"]["oracle_continuation_accuracy"] for r in results],
        "mean_normalized_phase_residual": [r["phase"]["mean_normalized_residual"] for r in results],
    }
    means = {}
    for key, values in metrics.items():
        valid = [value for value in values if value is not None]
        means[key] = {"mean": float(np.mean(valid)) if valid else None,
                      "units_with_metric": len(valid), "units_total": len(results)}
    return {"aggregation_unit": unit, "n_units": len(results), "unit_means": means,
            "eligible_pairs_total": sum(r["continuation"]["eligible_pairs"] for r in results),
            "endpoint_gated_pairs_total": sum(r["continuation"]["endpoint_gated_pairs"] for r in results),
            "uncertainty": "descriptive only; caller defines sampling units, never overlapping visit pairs"}


def aggregate_graphs(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper for one entry per sampled graph.

    Use aggregate_trajectories(unit='item') for iid permutation/start draws.
    Repeated starts sharing a sampled graph must be combined by the caller
    before graph-level inferential resampling.
    """
    summary = aggregate_trajectories(results, unit="graph")
    means = {key: {"mean": metric["mean"], "graphs_with_metric": metric["units_with_metric"],
                   "graphs_total": metric["units_total"]}
             for key, metric in summary["unit_means"].items()}
    return {"aggregation_unit": "graph", "n_graphs": summary["n_units"], "graph_means": means,
            "eligible_pairs_total": summary["eligible_pairs_total"],
            "endpoint_gated_pairs_total": summary["endpoint_gated_pairs_total"],
            "uncertainty": summary["uncertainty"]}
