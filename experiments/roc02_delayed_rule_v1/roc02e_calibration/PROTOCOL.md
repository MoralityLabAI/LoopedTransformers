# ROC-02E: matched rule-versus-cue interventions on the newer ROC-02C observers

**Preregistered 17 November 2026, before any ROC-02E measurements**, at config commit `8fc97c97d9ed37a6a312933bc2040f5cc603d0c6`. This is an implementation-stage, calibration-only protocol. Parent: LoopedTransformers ROC-02C PR #144 at `004301995d44388bf7ceefccdc89c845c8452d00`. Original ROC-02 models and splits remain frozen. Source families 6 and 7 are sealed and excluded from all script inputs.

## Why run this?

The earlier ROC-02D PR #130 already found a negative readout-sensitivity result (11/12 model-seed directions) using six exposed calibration cases per model and old rank-8 adapters from PR #127. That study used gamma up to 2. ROC-02C PR #144 uses gamma <= 1, includes both ranks 7 and 8, and does not publish saved optimized U adapters. Its 24/24 reduced-B and task-MSE findings are confined to an already used calibration source family.

ROC-02E tests the new observer selections with the same physical intervention budget. Rank 7 is primary because centered eight-way one-hot labels have supervised cross-covariance rank at most 7; the eighth singular direction is noncanonical. ROC-02E is still a dev-family study. It will not constitute prospective source-generalization or the complete ROC-01 Amendment C type-classification experiment.

## Source interventions

Use **all 256 original calibration records** from families 4 and 5, which encode maps `f_4(q)=(3q+1) mod 8` and `f_5(q)=(3q+5) mod 8`. For each original record with correct target `y`, construct two separately valid task counterfactuals:

* **Rule edit:** replace only the initial source-A binding table by the other calibration-family table, preserving the delay, query and distractors. The exact new label is `(y+4) mod 8`.
* **Cue edit:** replace only the final source-A query value `q` with `(q+4) mod 8`, retaining the original binding table and every intermediate event. The exact new label is again `(y+4) mod 8`.

Require equality of the two counterfactual labels, verified independently with `symbolic_execute`, and retain original/counterfactual source hashes. This paired design compares a change to an early carried rule against an equally label-changing modification of the final visible cue. On cases with B/C distractors, a label-preserving distractor-table replacement supplies an extra negative control.

The legal full edits are discrete inputs whose physical midpoint shifts vary by record. Report them as authentic semantic interventions at their natural dose.

## Same-dose controls

From the original hidden midpoint `h`, let `h'_r` and `h'_q` be the midpoints induced by the **full legal** rule and cue edits. For each `e in {rule,cue}`, set `v_e=h'_e-h`. At registered radii `r in {0.05, 0.10, 0.20}`, compute:

1. **Soft input at equal physical dose:** interpolate the encoded input `x(t)=x+t(x'_e-x)` until the resulting midpoint satisfies `||h(x(t))-h||_2=r`. Find the first crossing on the fixed 64-cell grid, followed by 24 bisection steps. Abort the paired-radius comparison for an unreachable or inaccurate crossing and report it. Interpolated inputs have **no discrete symbolic label**.
2. **Input-induced hidden steering:** `h+r v_e/||v_e||`.
3. **Observer-allocated steering:** `h+r P_U v_e/||P_U v_e||` and `h+r (I-P_U)v_e/||(I-P_U)v_e||`, with invalid-zero flags. This directs the legal input change into the measured observed/dark subspaces.
4. **Matched random controls:** one SHA-keyed isotropic direction and one SHA-keyed direction within span(U), each injected at the same midpoint with norm `r`.

Finish exactly the original residual suffix and evaluate the **unchanged original model eight-logit readout**. Preserve initial physical norm, final hidden change, actual output delta, invalid and null flags for every arm. Readout ablation has a different dose and is recorded separately.

The newer ROC-02C PR #144 has no serialized U weights. First reconstruct the task-only and selected-gamma rank-7/8 observers by the exact 80-Adam/QR protocol, verify every original checkpoint and source hash, compare replayed per-model task MSE and normalized B against their published receipts, and stop if replay disagrees. Never substitute PR #127's old rank-8 adapter files.

## Endpoints and statistical unit

For original correct label `y`, valid counterfactual target `y'`, and original model logits `z`, define directional target-margin gain

`g(z';z,y,y') = (z'[y']-z'[y])-(z[y']-z[y])`.

At rank 7, depth 16 and physical radius 0.20, average over all calibration cases the gain for rule-induced observed steering for the task-only and the leakage-aware selected observer. The **single primary descriptive paired effect** is `mean(g_joint-g_task_only)` per model seed, then across six seeds. Report the six individual paired effects and the exact one-sided sign-randomization distribution over `2^6=64` seed-flip patterns. No case-level p-value may treat the 256 correlated calibration pairs as independent model replications.

Separately compare the joint observer against each equal-radius soft-input rule edit, equal-radius soft-input cue edit, input-induced full-space hidden steering, task-only observed steering and isotropic control. Report the per-case maximum comparator and its provenance; the original ROC-01 criterion requires exceeding the best appropriate input-level control on a new family test.

Report task-label argmax accuracy for the untouched input, the *full legal* counterfactuals and each directional perturbation. On each observer's frozen train-only ridge head, also record the directional correct-target margin gain, both raw and scaled by a declared train-only standard deviation of pairwise target-margin outputs. Readout scale must not silently turn shrinkage into an apparent mechanism effect.

Secondary analyses include depth 8, rank 8, the two smaller radii and label-preserving distractor edits. All are descriptive for this already exposed calibration family. Retain ROC-01 Amendment C's six-direction Holm family for any subsequent per-state A/B/C/H diagnostic tags. A learned causal differentiation claim additionally needs independent family holdout, 5/6 consistent seed directions, appropriate multiple-testing control, and actual rule-answer causal improvement beyond the best input control.

## Execution and audit gate

1. Source and checkpoint SHA checks must pass before readout replay. Validate all 256 original source labels and all legal counterfactuals using the exact interpreter.
2. Recompute and verify the chosen rank-7/8 observer bases against PR #144 source receipts. Record chosen gamma for each seed/rank and the fit output normalization.
3. Freeze every intervention vector, input interpolation bracket and metric definition. Report all radii and every ineligible case.
4. Compute calibration-only receipts with full per-case results, all 24 model/rank cells, depth/family strata, paired seed means and raw-to-standardized output differences.
5. Keep families 6/7 untouched. No final ROC-02 mechanism label or source-generalization headline is authorized by this stage.

This protocol supplements the original ROC-01 Amendment C and ROC-02 registration. Their original requirements for the complete rank grid, physical T-C type identification, Chernoff calculations and prospective sealed evaluation remain active.
