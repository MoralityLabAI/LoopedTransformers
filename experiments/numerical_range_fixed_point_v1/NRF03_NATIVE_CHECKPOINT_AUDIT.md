# NRF-03: observed amplification and dark-to-visible leakage on twelve saved checkpoints

Date: 2026-11-15. The actual archived tied ToyLoop weights from July 2026 were read from the private repository's twelve `.pt` files. Each binary SHA-256 was independently recomputed and matched against `surface_events.jsonl`. Raw trained weights stay in their existing checkpoint directory.

## Reproduction and exact/empirical boundary

```bash
python -m research_gym.analysis.nrf03_checkpoint_probe \
  --out experiments/numerical_range_fixed_point_v1/nrf03_local_replay.json
python -m unittest tests.test_nrf03_checkpoint_probe -v
```

The module verifies every SHA-256 **before** `torch.load(weights_only=True,map_location='cpu')` and refuses unexpected state-dict schemas. Its exact binary32 integer sums yield a valid upward-rounded global one-step Lipschitz bound for the real map with those parameter values:

```
F(h) = h + (W2 tanh(W1 h + b1) + b2)/4.
L <= 1 + (1/4)*sqrt(||W1||_1 ||W1||_infinity)*sqrt(||W2||_1 ||W2||_infinity).
```

The existing 12-sample outcome JSON `nrf03_measured_native.json` records an independent floating-point replay. The values of `global_one_step_norm_bound_float64` are computed via ordinary double summation, whereas the reproducible script's `global_one_step_upper` uses integer-exact row/column sums and upward six-decimal rounding. The user should compare these two implementations before trusting the last decimal.

The local Jacobian gain is obtained by differentiating the nonlinearity analytically at the deterministic zero-input trajectory, composing 16 or 32 residual Jacobians, then running 12 fixed-start power iterations. Finite differences perturb only coordinate zero by 0.001. For observation, the first eight coordinates are fixed in advance, and the dark direction lies in the remaining 120.

## Matched checkpoint outcomes

| Round depth | Seed | Full-cycle Jacobian at E2048 | At E4096 | Dark leakage E2048 -> E4096 |
|---|---|---:|---:|---:|
| 16 | 101 | 3.349 | 2.358 | 0.193 -> 0.161 |
| 16 | 103 | 3.111 | 2.141 | 0.371 -> 0.294 |
| 16 | 107 | 3.367 | 2.156 | 0.391 -> 0.316 |
| 32 | 101 | 12.149 | 6.265 | 0.480 -> 0.316 |
| 32 | 103 | 9.468 | 5.710 | 0.730 -> 0.544 |
| 32 | 107 | 10.504 | 5.976 | 0.776 -> 0.619 |

All six paired Jacobian estimates decrease during the second training interval; so do all six dark-leakage directional measurements. The median Jacobian drop is 37.8%. All twelve local Jacobian witnesses remain above one; they invalidate a *strict global full-cycle contraction* claim in the tested Euclidean norm on any neighborhood of the zero point, subject to numerical replay precision.

Previously measured kappa rises across all six paired comparisons while local Jacobian amplification falls. This is a descriptive coexistence of alignment growth and sensitivity reduction. It is not a causal attribution to kappa, nor proof of generalization.

The observed/dark split is only a fixed coordinate projection, and its nonzero dark-to-observed directional derivative obstructs an **exact** quotient by that coordinate fiber near the tested trajectory. It does not establish a uniform leakage bound on an entire state domain. MathCan NRF-03 gives the quantitative approximate-quotient theorem that would be required for a genuine observability certificate.

## Remaining experiment gates

Replicate the saved-weight analysis in native PyTorch, compare the JSON values within a registered tolerance, then measure output-task sensitivity with a semantically grounded decoder and matched random orthogonal projection controls. For an actual global observable quotient, certify a uniform fiber-derivative upper bound on a specified state domain and combine it with the graph's weighted contraction witness.

**Current infrastructure:** the private repo's NRF-02 Actions jobs have failed before runner setup on multiple attempts. This PR contains the reproducible local command; hosted model validation has not completed. No training or weights were changed.
