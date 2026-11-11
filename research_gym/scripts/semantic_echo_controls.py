"""Run deterministic CPU controls for the Semantic Echo trajectory metrics.

Example:
    python -m research_gym.scripts.semantic_echo_controls --out /tmp/semantic-echo

This runner evaluates constructed dynamics and analytic formulas. Every report
has status CONTROL_ONLY. It does not load, train, or evaluate a neural model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from research_gym.analysis.semantic_echo import (
    analyze_trajectory,
    normalized_phase_residual,
    phase_telescoping,
)

SCHEMA_VERSION = "semantic_echo.controls.v1"
DEFAULT_SEED = 731


def _walk(permutation: list[int], start: int, count: int) -> list[int]:
    """Independent list oracle used to construct controls, not the metric's oracle."""
    values = []
    node = start
    for _ in range(count):
        node = permutation[node]
        values.append(node)
    return values


def _control(name: str, description: str, checks: dict[str, bool], **data: Any) -> dict[str, Any]:
    return {"name": name, "description": description,
            "checks": {key: bool(value) for key, value in checks.items()},
            "passed": all(checks.values()), **data}


def correct_reversible_control() -> dict[str, Any]:
    permutation = [2, 4, 1, 0, 3]
    n, alpha, count = len(permutation), 0.6, 24
    operator = np.zeros((n, n), dtype=np.float64)
    for source, destination in enumerate(permutation):
        operator[destination, source] = 1.0
    hidden = np.r_[np.eye(n)[0], [1.25, -0.75]]
    original = hidden.copy()
    predictions = []
    for _ in range(count):
        hidden = np.r_[operator @ hidden[:n], alpha * hidden[n:]]
        predictions.append(int(hidden[:n].argmax()))
    for _ in range(count):
        hidden = np.r_[operator.T @ hidden[:n], hidden[n:] / alpha]
    error = float(np.linalg.norm(hidden - original))
    score = analyze_trajectory(permutation, 0, predictions, horizon=4, max_visit=8)
    c = score["continuation"]
    return _control("correct_permutation_with_nuisance_contraction",
                    "An exact semantic permutation advances while independent nuisance coordinates contract.",
                    {"independent_oracle_match": predictions == _walk(permutation, 0, count),
                     "reverse_error_below_1e-12": error < 1e-12,
                     "has_gated_pairs": c["endpoint_gated_pairs"] > 0,
                     "continuation_oracle_perfect": c["oracle_continuation_accuracy"] == 1.0,
                     "no_pair_disagreement": c["pair_any_disagreement_count"] == 0,
                     "phase_residual_below_1e-24": score["phase"]["mean_normalized_residual"] < 1e-24},
                    alpha=alpha, exact_reverse_error=error, trajectory=score)


def deadline_control() -> dict[str, Any]:
    # The displayed answer is cached for a deadline at loop eight. All graphs
    # succeed at that endpoint, even though the displayed state never advances.
    permutations = [[1, 2, 3, 4, 0], [2, 4, 1, 0, 3], [4, 0, 1, 2, 3]]
    traces = []
    for permutation in permutations:
        target_at_eight = _walk(permutation, 0, 8)[-1]
        traces.append(analyze_trajectory(permutation, 0, [target_at_eight] * 20,
                                         horizon=4, max_visit=8))
    return _control("deadline_answer_cache",
                    "An augmented state can retain a hidden deadline while its readout repeats the loop-eight answer.",
                    {"every_loop8_answer_correct": all(t["per_loop_correct"][7] for t in traces),
                     "every_successor_transition_wrong": all(t["transition"]["accuracy"] == 0.0 for t in traces),
                     "same_future_readout_agrees": all(t["continuation"]["pair_any_disagreement_rate"] == 0.0 for t in traces),
                     "oracle_detects_wrong_continuation": all(t["continuation"]["oracle_continuation_accuracy"] < 1.0 for t in traces),
                     "frozen_phase_scores_one": all(abs(t["phase"]["mean_normalized_residual"] - 1.0) < 1e-12 for t in traces)},
                    trajectories=traces)


def hidden_clock_control() -> dict[str, Any]:
    permutation = [1, 2, 0]
    prediction = _walk(permutation, 0, 12)
    prediction[4] = 0  # a hidden clock branch changes the output at loop five
    score = analyze_trajectory(permutation, 0, prediction, horizon=2, max_visit=4)
    c = score["continuation"]
    return _control("same_semantic_endpoints_different_futures",
                    "Loops one and four expose the same correct node; a hidden-clock branch changes the following step.",
                    {"one_structural_pair": c["eligible_pairs"] == 1,
                     "one_gated_pair": c["endpoint_gated_pairs"] == 1,
                     "witness_at_first_future_step": c["first_witness"]["offset"] == 1,
                     "half_future_steps_disagree": c["per_step_mismatch_rate"] == 0.5,
                     "oracle_accuracy_three_quarters": c["oracle_continuation_accuracy"] == 0.75},
                    trajectory=score)


def clock_phase_control() -> dict[str, Any]:
    permutation = [1, 2, 3, 4, 0]
    reverse_permutation = [4, 0, 1, 2, 3]
    count, ell = 20, 5
    # A clock produces a legal phase progression with a uniformly wrong label
    # offset. Changing the graph leaves this generated trace unchanged.
    prediction = [(t + 1) % ell for t in range(1, count + 1)]
    phases = np.exp(2j * np.pi * (np.arange(1, count + 1) + 1) / ell)
    raw_phase = normalized_phase_residual(phases, ell)
    base = analyze_trajectory(permutation, 0, prediction, horizon=4, max_visit=8)
    changed = analyze_trajectory(reverse_permutation, 0, prediction, horizon=4, max_visit=8)
    return _control("clock_only_wrong_phase_anchor",
                    "An m=1 clock oscillator passes the phase-law residual while its labels ignore the graph.",
                    {"oscillator_phase_residual_small": raw_phase["mean_normalized_residual"] < 1e-24,
                     "label_phase_residual_small": base["phase"]["mean_normalized_residual"] < 1e-24,
                     "every_base_label_wrong": base["label_accuracy"] == 0.0,
                     "endpoint_coverage_zero": base["continuation"]["endpoint_gate_coverage"] == 0.0,
                     "counterfactual_graph_label_accuracy_below_half": changed["label_accuracy"] < 0.5,
                     "graph_change_witnesses_wrong_transition": changed["transition"]["accuracy"] == 0.0},
                    raw_phase=raw_phase, base_trajectory=base, counterfactual_trajectory=changed)


def phase_cancellation_control() -> dict[str, Any]:
    ell = 5
    omega = np.exp(2j * np.pi / ell)
    phases = np.asarray([1.0, omega ** 2, omega ** 2])
    residual = normalized_phase_residual(phases, ell)
    telescope = phase_telescoping(phases, ell)
    return _control("intermediate_phase_errors_cancel_at_endpoint",
                    "Two incorrect phase transitions cancel in the final phase; local residuals expose the path.",
                    {"endpoint_correct": telescope["final_deviation"] < 1e-12,
                     "local_residual_nonzero": residual["mean_normalized_residual"] > 0.5,
                     "telescoping_identity": telescope["identity_error"] < 1e-12,
                     "rms_bound_holds": telescope["final_deviation"] <= telescope["rms_bound"] + 1e-12,
                     "local_certificate_withheld": not telescope["certifies_phase_relative_to_initial"]},
                    phase=residual, telescoping=telescope)


def noisy_inverse_control(seed: int) -> dict[str, Any]:
    alpha, sigma = 0.8, 0.05
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, 20_000)
    sampled_base = float(np.mean(noise ** 2))
    rows = []
    for k in (0, 8, 16, 32, 64):
        gain = alpha ** (-k)
        # The maps diag(1,alpha) and diag(alpha,1) share singular values.
        # Isotropic terminal noise is inverted on the task's first coordinate.
        rows.append({"loops": k, "smallest_singular_value": alpha ** k,
                     "good_task_mse_analytic": sigma ** 2,
                     "fragile_task_mse_analytic": sigma ** 2 * gain ** 2,
                     "good_task_mse_sample": sampled_base,
                     "fragile_task_mse_sample": float(np.mean((noise * gain) ** 2)),
                     "good_bayes_error": 0.5 * math.erfc(1 / (sigma * math.sqrt(2))),
                     "fragile_bayes_error": 0.5 * math.erfc(alpha ** k / (sigma * math.sqrt(2)))})
    same_singular = np.allclose(np.linalg.svd(np.diag([1.0, alpha]), compute_uv=False),
                                np.linalg.svd(np.diag([alpha, 1.0]), compute_uv=False))
    return _control("same_spectrum_different_semantic_noise_gain",
                    "Invertible maps with equal singular values place the contracted coordinate on different task axes.",
                    {"same_singular_values": same_singular,
                     "analytic_noise_amplification": all(np.isclose(r["fragile_task_mse_sample"] / sampled_base,
                                                                        alpha ** (-2 * r["loops"])) for r in rows),
                     "fragile_bayes_error_increases": all(a["fragile_bayes_error"] < b["fragile_bayes_error"]
                                                          for a, b in zip(rows, rows[1:])),
                     "good_mse_constant": all(r["good_task_mse_analytic"] == sigma ** 2 for r in rows)},
                    alpha=alpha, sigma=sigma, terminal_noise_only=True, sample_count=len(noise), rows=rows)


def xor_control() -> dict[str, Any]:
    inputs = [(a, b) for a in (0, 1) for b in (0, 1)]
    outputs = [(a, a ^ b) for a, b in inputs]
    recovered = [(a, a ^ c) for a, c in outputs]
    targets = [a ^ b for a, b in inputs]
    before = sum(b == target for (_, b), target in zip(inputs, targets)) / 4
    after = sum(c == target for (_, c), target in zip(outputs, targets)) / 4
    return _control("xor_bijection_changes_readout_accessibility",
                    "The reversible map (a,b)->(a,a xor b) exposes the task in a fixed one-coordinate readout.",
                    {"bijection": len(set(outputs)) == 4,
                     "exact_inverse": recovered == inputs,
                     "same_total_information": len(set(inputs)) == len(set(outputs)),
                     "fixed_readout_improves": before == 0.5 and after == 1.0},
                    readout_family="read only the second bit", inputs=inputs, outputs=outputs,
                    task_labels=targets, fixed_readout_accuracy_before=before,
                    fixed_readout_accuracy_after=after, uniform_state_entropy_bits=2.0,
                    task_entropy_bits=1.0)


def run_controls(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    controls = [correct_reversible_control(), deadline_control(), hidden_clock_control(),
                clock_phase_control(), phase_cancellation_control(), noisy_inverse_control(seed), xor_control()]
    return {"schema_version": SCHEMA_VERSION, "status": "CONTROL_ONLY", "seed": seed,
            "all_controls_passed": all(control["passed"] for control in controls),
            "controls_passed": sum(control["passed"] for control in controls), "controls_total": len(controls),
            "scope": "Constructed trajectories and analytic controls; no neural checkpoint measurements.",
            "indexing": "predictions[t-1] is the post-loop label at t; t starts at 1",
            "dependence": "Repeated same-state pairs are dependent within each graph.",
            "controls": controls}


def report_markdown(report: dict[str, Any]) -> str:
    rows = ["# Semantic Echo CPU controls", "", "Status: **CONTROL_ONLY**", "",
            report["scope"], "", f"Schema: `{report['schema_version']}`. Seed: {report['seed']}.", "",
            f"Checks passed in {report['controls_passed']}/{report['controls_total']} constructed controls.", "",
            "| Control | Result | Meaning |", "|---|---|---|"]
    for control in report["controls"]:
        rows.append(f"| {control['name']} | {'PASS' if control['passed'] else 'FAIL'} | {control['description']} |")
    rows += ["", "## Metric conventions", "",
             "Predictions are indexed after loops 1 through T. Same-state pairs come from the exact permutation oracle. "
             "Both endpoints must have correct predictions, and selection never inspects their future predictions.", "",
             "The continuation report supplies structural pair counts, endpoint coverage, pair disagreement and "
             "per-step mismatch denominators. Oracle accuracy checks both predicted futures against the exact successors.", "",
             "Phase uses mode m=1 and the start cycle's explicit node order. Its residual is divided by "
             "|1-exp(2 pi i / cycle_length)|^2. Fixed points are uninformative; off-cycle predictions receive explicit invalid markers.", "",
             "A phase residual can pass with a uniformly wrong phase anchor. The clock control demonstrates why "
             "oracle labels and counterfactual graph changes accompany the residual.", "",
             "Graph-level summaries are descriptive. Overlapping visit pairs do not supply independent observations.", ""]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory for controls.json and report.md")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    report = run_controls(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "controls.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out / "report.md").write_text(report_markdown(report))
    print(f"CONTROL_ONLY: {report['controls_passed']}/{report['controls_total']} controls passed; {args.out}")
    return 0 if report["all_controls_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
