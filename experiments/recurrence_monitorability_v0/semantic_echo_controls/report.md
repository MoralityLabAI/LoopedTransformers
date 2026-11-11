# Semantic Echo CPU controls

Status: **CONTROL_ONLY**

Constructed trajectories and analytic controls; no neural checkpoint measurements.

Schema: `semantic_echo.controls.v1`. Seed: 731.

Checks passed in 7/7 constructed controls.

| Control | Result | Meaning |
|---|---|---|
| correct_permutation_with_nuisance_contraction | PASS | An exact semantic permutation advances while independent nuisance coordinates contract. |
| deadline_answer_cache | PASS | An augmented state can retain a hidden deadline while its readout repeats the loop-eight answer. |
| same_semantic_endpoints_different_futures | PASS | Loops one and four expose the same correct node; a hidden-clock branch changes the following step. |
| clock_only_wrong_phase_anchor | PASS | An m=1 clock oscillator passes the phase-law residual while its labels ignore the graph. |
| intermediate_phase_errors_cancel_at_endpoint | PASS | Two incorrect phase transitions cancel in the final phase; local residuals expose the path. |
| same_spectrum_different_semantic_noise_gain | PASS | Invertible maps with equal singular values place the contracted coordinate on different task axes. |
| xor_bijection_changes_readout_accessibility | PASS | The reversible map (a,b)->(a,a xor b) exposes the task in a fixed one-coordinate readout. |

## Metric conventions

Predictions are indexed after loops 1 through T. Same-state pairs come from the exact permutation oracle. Both endpoints must have correct predictions, and selection never inspects their future predictions.

The continuation report supplies structural pair counts, endpoint coverage, pair disagreement and per-step mismatch denominators. Oracle accuracy checks both predicted futures against the exact successors.

Phase uses mode m=1 and the start cycle's explicit node order. Its residual is divided by |1-exp(2 pi i / cycle_length)|^2. Fixed points are uninformative; off-cycle predictions receive explicit invalid markers.

A phase residual can pass with a uniformly wrong phase anchor. The clock control demonstrates why oracle labels and counterfactual graph changes accompany the residual.

Graph-level summaries are descriptive. Overlapping visit pairs do not supply independent observations.
