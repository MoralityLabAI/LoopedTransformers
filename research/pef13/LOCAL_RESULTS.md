# PEF-13 | Prospective lattice-phase calibration: negative primary result

**Local run:** 2026-11-15. Python 3.13.5, PyTorch 2.10.0+cpu, single-thread CPU, oneDNN AVX2. [Standalone registration commit `9483da0`](https://github.com/MoralityLabAI/LoopedTransformers/commit/9483da079c3707711b071efa9dbd04fa1a286b40) preceded all PEF-13 heldout scoring. Original PEF-12 and earlier outcomes were known before the new registration.

The original PEF-07 six 450-step models, the 768-probe/768-heldout source split, the PEF-09/10/11/12 sealed ledgers, and PEF-10 fixed h12 ridge readouts were reused without retraining. Every checkpoint and upstream ledger SHA256 was checked. Twenty-four PEF-12 geometry/readout conditions (6 models × 2 tangents × 2 angles) and four **new** lattice steps (0.0075, 0.0125, 0.035, 0.08 of the frozen principal-component scale) generated 96 previously unscored heldout comparisons. All 96 forecasts were sealed before any heldout state was constructed.

**Frozen PEF-13 forecast ledger SHA256:** `0ef0a408f6604a0464877df086e40da85929bc8fd60b05c4dd33c66916d55dc0`.

## Registered accuracy-forecast criterion: 0/3 coalition seeds

Across 16 prospective quantization conditions per seed, the pre-registered rule required a phase-aware accuracy mean absolute error (MAE) ≤0.035 and at least a 25% reduction over the uniform-phase predictor. The outcome was `criterion_not_met` with **0/3 coalition seeds** meeting both requirements.

| Seed | Task | Phase-aware MAE | Uniform-phase MAE | Phase-shuffled MAE | Clean-constant MAE | Registered pass |
|---:|---|---:|---:|---:|---:|---|
| 103 | Coalition | 0.019613 | 0.020386 | 0.018555 | 0.065592 | No |
| 131 | Coalition | 0.011963 | 0.014577 | 0.017904 | 0.058512 | No |
| 167 | Coalition | 0.015137 | 0.011889 | 0.014648 | 0.065104 | No |
| 103 | Threshold | 0.014160 | 0.011408 | 0.010091 | 0.052490 | No |
| 131 | Threshold | 0.009440 | 0.008476 | 0.007080 | 0.069092 | No |
| 167 | Threshold | 0.012777 | 0.017445 | 0.021566 | 0.061849 | Yes |

The two coalition improvements were only 3.8% and 17.9%, short of the 25% requirement. The third coalition seed became less accurate with exact probe phases. The **phase-shuffled** control often had similar MAE, limiting evidence that the phase-specific probe association transfers consistently to new context groups.

### Aggregate by tangent and angle

Mean absolute prediction error (fraction correct) averaged over the three coalition seeds and all four prospective steps:

| Tangent / angle | Phase-aware | Uniform-phase | Clean constant |
|---|---:|---:|---:|
| Random, 0.25° | 0.0251 | 0.0249 | 0.2105 |
| Pair 01, 0.25° | 0.0353 | 0.0359 | 0.0381 |
| Random, 4° | 0.0000 | 0.0000 | 0.0000 |
| Pair 01, 4° | 0.0020 | 0.0016 | 0.0037 |

The large clean-constant error for the 0.25° random tangent confirms that the prospective steps induce a real loss of decoded accuracy; quantization effects are negligible at 4° under this grid. Per-model and per-condition observations, error histograms for each of the sixteen focal masks and signed forecast errors appear in `pef13_results/summary.json` and `calibration.csv`.

## Independent audit

An independent **Python standard-library** auditor recomputed all 96 phase-aware forecasts via Python's round-to-even, all 96 uniform-phase forecasts, verified all original checkpoint byte SHA256s and compared every heldout correct count and focal-mask error sum against the saved report. It reproduced the registered `criterion_not_met` decision and all six model MAEs. The complete PEF-08 through PEF-13 test suite passed **43/43 tests**, including eight new PEF-13 source/algebra invariants. These checks were performed locally; hosted Python 3.12 CI remains a distinct unverified environment.

## POSTHOC decomposition: local linearity survives

The following diagnostic was run **after** sealing PEF-13's heldout scores. It was not used to tune or alter the primary forecasts or decisions. It replayed all 96 quantized *probe* cases under the original nonlinear recurrent suffix. It also computed PEF-13's identical linearization directly on the *heldout* inputs, to distinguish local-linear error from transferred-distribution error. Heldout Jacobians were seen **only during this posthoc investigation**.

| Mean absolute difference, across 48 conditions/task | Coalition | Threshold |
|---|---:|---:|
| Probe linear forecast vs nonlinear quantized probe score | 0.000217 | 0.000163 |
| Heldout linear oracle vs nonlinear quantized heldout score | 0.000054 | 0.000163 |
| Probe linear forecast vs heldout linear oracle | 0.015625 | 0.011963 |
| Probe linear forecast vs heldout nonlinear result | 0.015571 | 0.012126 |

The phase-aware linearization agrees with actual recurrent quantization remarkably well **on the same data** (less than 0.03 percentage points average error). The error in using that probe average to predict heldout accuracy is predominantly transfer across context-disjoint source groups. The triangle inequality separates these sources of error without implying that all context distributions have changed in a causal or structural way. The exact probe and heldout samples are deterministic finite cohorts; sampling variation contributes to transfer differences.

The conditional Taylor bound in `MATH.md` links per-input approximation error to directional curvature and quantizer step. The posthoc outcome is consistent with a small local curvature effect at these steps. Actual per-sample curvature maxima were **not** certified and the mathematical condition remains a bound under an explicit assumption.

## Limitations and next question

The precision intervention quantizes a single retained latent coordinate after h8 rank-one ablation. It is not a CPU/GPU low-bit kernel emulation. All models were trained on the same four-bit teacher family over unseen eight-bit contexts. The PEF-10 ridge readouts were fitted on probe contexts, so the probe quantization replay is **in-sample** relative to readout fitting. No novel operator, architecture or pretrained language-model transfer was tested.

A follow-up could preregister conditional-margin integration, context-bootstrap uncertainty and phase-averaged forecasting over multiple *new* source splits, with the model checkpoint weights kept fixed. Separate, new training seeds would be required before claiming architectural robustness.