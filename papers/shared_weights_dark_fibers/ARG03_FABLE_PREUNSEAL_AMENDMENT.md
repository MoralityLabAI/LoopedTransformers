# ARG-03-N / LOOPSCOPE-04: Fable pre-unseal development amendment

**Date:** 2026-11-15. **Source:** ARG03 frozen protocol and original development receipts; Fable independent review of the public VPD supplement. This note is a **post-hoc development audit and prospective amendment**. The frozen [ARG03_FROZEN_TEST_PROTOCOL.md](ARG03_FROZEN_TEST_PROTOCOL.md) remains unchanged. Its 168-case test.jsonl stays sealed and must not be run, graded, opened for model selection, or quietly redefined here.

## Confirmed development failure modes

The original 96-case development receipts were mirrored in compact form, with Git blob hashes, to the new pure-Python audit package at mechinterp/dark_fiber_gyms/arg03_preunseal.

### 1. Depth-dependent correct-program sets

| Depths | Both correct | Discordant | Phi |
| --- | ---: | ---: | ---: |
| 1 vs 2 | 1/96 | 66/96 | -0.4902 |
| 2 vs 4 | 35/96 | 15/96 | +0.6835 |
| 2 vs 8 | 13/96 | 58/96 | -0.2275 |
| 4 vs 8 | 20/96 | 45/96 | +0.0501 |

Of the 96 cases, 13 are correct at ALL 2/4/8 depths, while 72 are correct at ANY of them. The raw depth-1/2 success gain thus conceals large rank/program swaps. Under these development measurements, depth is behaving more like program-selection reassignment than monotone derivational refinement. The H1a paired McNemar p-value of 0.064 is the actual development comparison.

Top-1 future success is 7, 5, 6, 5 out of 96 at depths 1/2/4/8. Within baseline ranking-change summaries, 2/4/8 have only 2, 10, 5 distinct top-1 program IDs across cases; these are top-1 identifiers, not full top-12 lists. The per-case receipt lacks the 64-program ranking and target final operator. **No top-12 diversity or correct-program rank is imputed.**

### 2. H1c confounds prior transfer and recurrence

The in-sample fitted [0.35, 0.5, 2.0] positional temperature vector was fitted on development targets whose final operator was not R. The frozen test is all final R and includes more algebraic families. A calibration failure after the shift can arise from mismatched priors, target-program shape or model feedback. The existing H1c may test *parity* if the frozen calibration generalizes, while any degradation is insufficient by itself to establish closed-loop computation.

**Prospective dev-only response:** The new prepare_final_r.py creates an independently seeded final-R development cohort from the allowed calibration families (signed/product/sparse), split into 48 fit and 48 validation cases. Eight final-R operator programs are assigned to each split, so the program sets are disjoint between fit and validation. Model input public JSONL is separated from host-only source program IDs. The generator explicitly avoids synthesizing ARG03's old sealed test, excludes EvalMath / ARG02 and the prior ARG03 development signatures, and reports test-vs-new-dev collision status as UNKNOWN pending a custodian's blinded digest comparison. Never assert clean test overlap without that check.

The new calibration.py supplies a **12 stored parameter** per-position/per-operator additive bias baseline. Site-wise softmax has a gauge redundancy, so it has 9 identifiable degrees of freedom after row mean-centering. Fit on the new development-finalR-fit cohort using source programs; freeze before development-finalR-valid. Evaluate top-1 and source-program rank alongside future-correct@12 via the independent oracle, retaining the existing candidate-budget metric. Dev-valid grading must occur in an oracle-only process, not during model inference. Freeze a prospective evaluation rule before any later unsealing.

### 3. H2a random control confounded by propagated displacement

At t=1 the two patched arms were matched (mean swap/random 14.65/14.66); subsequent recurrent evolution broke amplitude parity:

| t | Mean swap/random full-state delta ratio | Mean readout L2 per state L2, swap | Random | Cases swap larger per unit |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.8492 | 4.4565 | 6.7534 | 4/96 |
| 4 | 1.4441 | 4.7670 | 6.1442 | 13/96 |
| 8 | 1.0999 | 6.6769 | 7.2430 | 22/96 |

These are descriptive ratios of means of per-case ratios. The original raw L2 H2a test is an observational difference under the respective post-loop displacement distributions. Its 0.00 pre-continuation leakage at t=1 is by construction: changed context positions are outside the current scored readout.

A **new, separately labelled time-local arm** is defined in arg03_preunseal/controls.py. At each evaluated depth, begin from clean h_t and inject random context-state direction with the same per-case full-state L2 norm as the structured h_t^swap - h_t^clean displacement, then compare successor full-vocabulary readout q(F_x(h_t)). Preserve the scored token coordinates; assert readout parity, intervention radius and model input parity at this NEW t-local fork. This changes the experimental estimand and must never be reported as a rerun of the original H2a.

A clean effect above this arm, at matched radius and with paired uncertainty, would support structure-specific delayed observability. Until then, count H2a's strong causal specificity gate as unsupported.

## LOOPSCOPE-04 refinements

### 3A: missing on-manifold control

Current paired random_orth controls are sphere/norm-calibrated, not necessarily drawn from another board's actual latent-state distribution. Add same-step *cross-board donor* fibers as an explicitly matched control on the same checkpoint, preserving recipient readout and per-cell norm under the original cylinder/sphere constraints. In separate arms test different-board donors and same-board different-fork donors. Source input mismatch and donor identity must remain explicit. Compare per-board response and lag-12 exact solve at fixed checkpoint; report any cases rejected by readout/norm gate.

### 3B: scope of history advantage

The 53,248-parameter ridge traces improve held-out orthogonal-state R2 on both checkpoints. For the 1,880,064-parameter transformer reconstructor, gaoxin492's snapshot has LOWER validation MSE (0.0683 vs 0.0710) and leads at m=2 (0.800 vs 0.7875). History's claimed elimination of the m=1 deficit is 5 additional boards of 80. For yagizdevre, unconditional mean-fill at m=2 equals the m=1 trace model at 0.600, suggesting a material periodic-reset component. Report checkpoint, model class, m frequency, and absolute board counts, with mean-fill and periodic reset as distinct baselines.

### 3C: eight-dimensional Gramian and heavy tails

The restricted Gramian uses 8 sampled directions per board inside a 501-dimensional kernel. The 119/120 high-versus-low result confirms a direction ordering WITHIN that sampled 8D subspace. It is not a claim about the full 501D kernel.

The source-hashed board receipts yield these gaoxin492 leading sampled eigenvalue values (n=40 each):

| Stratum | Mean lambda1 | Median lambda1 | Top 3 boards' share of sum |
| --- | ---: | ---: | ---: |
| WB-Prop | 404 | 32.18 | 88.3% |
| WB-Stall_partial | 9.68e6 | 49.34 | 87.7% |
| S4_stalled | 2.74e7 | 5.14e6 | 69.0% |

The ratio of mean WB-Stall lambda1 to mean WB-Prop lambda1 is about 45,900x, dominated by extreme boards. The robust median finding is a dramatic S4-stalled separation, not a universal stalled-vs-propagation gap. Report medians, tail share and stratified prediction of lagged visibility; add independent random subspace seeds and wider probe-count sweeps before inferring rank in the ambient kernel.

## Completed CPU verification and new development roster

The [pre-unseal audit run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43724575262) passed **9 unit tests** and generated a development-only report. Its tests independently exercise the historical depth flips, H2a displacement confound, sampled-Gramian robust statistics, source-hash and test-path guards, fresh-cohort split logic, synthetic bias fitting and score ranks, and the new per-depth random-state norm matcher.

The [fresh final-R source-only generation run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43724624163) passed, using pinned EvalMath source `06f6a0889853dcc62da6c74d7b3a2326102dcf48` and Compactification-Theory source `1ac36353b36581b6a634626bcd55542e53dae456`. It generated **48 development-finalR-fit** plus **48 development-finalR-valid** model-input cases. Distinct source programs occupy each half. The public-source and host-label artifacts are separate and carry a fresh seed (`2026111617`).

This completed only **CPU source generation and integrity checks**. It has not run Nanbeige, refitted the scalar temperature or twelve-bias operator calibration on model logits, graded future@12, performed the new H2a time-local fork, or checked overlap against the sealed original test. Those remain prospective gates before another human unseal review.

## Decision and independent checks

The user-approved frozen test is **not authorized** here. No sealed test, pretrained checkpoint, or new GPU evaluation was accessed. Complete the fresh final-R development + controls and freeze a new addendum to any future confirmatory protocol before a separate human unseal decision.

Source IDs: original supplement in [MoralityLabAI/VPD](https://github.com/MoralityLabAI/VPD/tree/main/output/supplements/loopscope04_arg03); frozen source branch research/causal-dark-fibers-arg03-20261115 at bf8fecad7d210306a38e909ef0366ceeefefd521.
