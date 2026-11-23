# Bounded Bellman corrections to the native guard planner

The prior guard-recovery pilot found that a supervised guard reader with a budgeted planner outperformed the unrestricted Bellman actor. This follow-up keeps the full 1103 supervised reader and its planning recurrence frozen. The recurrent actor can change an individual action value by at most 0.0375 utility. It starts with zero correction. This allows competition among close actions while preserving any baseline lead larger than 0.075. Every arm uses the public completion mask and a zero-valued stop option.

## Registered comparison

Use the same source tapes and the same 40 previously exposed development roots, with both worlds at six, seven, and eight charged native calls. Keep the root as the experimental unit. Fit the Bellman correction, matched empirical-suffix correction, and TRM correction with 600 batches of 32 observed decisions. The two recurrent variants share their initial weights and sample order. All heads receive the same typed guard, paid-feedback, frozen-prior, planner value, and planner gap fields. Fit a clipped affine correction from the same backed-up targets. Include the unchanged supervisor and a temperature-calibrated version. Learn the temperature by out-of-fold source guard likelihood, holding out both members of every source pair together. The correction targets and features also use those out-of-fold reader predictions. Shared closed expressions can recur across roots. Neither development outcomes nor oracle guard truth enter the corrections at inference.

Report root-macro utility, proofs, complete budgeted pairs, checks, and the number of decisions changed from the unchanged supervisor. Compare with every fitted and calibrated control and require the earlier +0.025 utility floor. Keep the unperturbed two-slot Bellman correction separate from a donor intervention. At the first decision where the correction changes one feasible tactic into another, swap the two residual addends while leaving the supervised values, paid prefix, and physical Lean outcomes fixed. The swap preserves the exact residual values and changes their recipients. Replay a selection of changed successful traces through Lean and independently compile every distinct successful winning script.

This is a diagnostic on an exposed development cohort under `FSI-01 REDESIGN_OR_PILOT_ONLY`. A positive result would still require a fresh frozen cohort before any confirmation claim.

For a separate headroom bound, allow a hindsight solver with the true guard outcomes to choose only actions that could win after a correction of the registered size. It charges the same calls and permits boundary ties, providing an upper bound for this correction radius. Keep it outside the trained-control comparison.

## Follow-up contrast objective

The absolute-value residual improved the frozen planner by less than the registered floor. The hindsight bound showed that more improvement was available at the same radius. This development-only follow-up keeps the same freeze, correction radius and action menu. Form pairs of logged source actions at the same observable decision, retaining a pair only when either action could win under the registered bound. Fit the difference between their corrected values to the difference between their backed-up values, with the same sampled pairs and updates for the empirical and TRM controls. Fit ridge on those feature differences and include both sign orientations. The temperature control and frozen planner stay fixed. Report this variant separately because its objective was chosen after seeing the first development result.

## Absolute-value results

| Frozen policy or fitted correction | Proofs / 240 | Complete pairs / 120 | Charged calls | Root-macro utility |
| --- | ---: | ---: | ---: | ---: |
| Frozen supervised planner | 188 | 94 | 1,572 | 0.619583 |
| Temperature calibration | 186 | 93 | 1,568 | 0.611667 |
| Ridge correction | 188 | 94 | 1,574 | 0.619375 |
| Empirical two-slot correction | 190 | 95 | 1,587 | 0.626354 |
| Bellman TRM correction | 188 | 94 | 1,572 | 0.619583 |
| Bellman two-slot correction | 193 | 96 | 1,581 | 0.639479 |
| Bellman residuals swapped at the changed decision | 188 | 94 | 1,572 | 0.619583 |
| Bounded hindsight | 214 | 107 | 1,542 | 0.731042 |

The bounded Bellman actor gained five proofs and 0.019896 utility over the unchanged planner. It gained three proofs and 0.013125 utility over its matched empirical correction. Both gains remained below the registered floor. The action-matched intervention returned the aggregate score to the unchanged planner's level. The hindsight diagnostic shows 21 additional proofs were available to a better chooser within the same score bound.

All five added proofs came at the six-call budget. The unchanged planner closed 44 of those 80 scenarios, the Bellman correction closed 49, and hindsight closed 58. At seven calls the planner and learned correction each closed 64, with 76 possible in hindsight. At eight calls all three closed 80. Root-macro utility favored the Bellman correction over the planner on four roots, tied on 30, and declined on six. The positive mean came from the size of the gains on the winning roots.

The Bellman correction changed at least one decision in 85 scenarios. The intervention reassigned the two addends and changed the next action in all 85, preserving their paid prefixes. Within that group the natural actor produced 74 proofs and the swapped actor produced 69. The unchanged cases make the absolute result's net gain local to the learned action correction in this finite pilot.

## Action-contrast results

| Fitted correction | Proofs / 240 | Complete pairs / 120 | Charged calls | Root-macro utility |
| --- | ---: | ---: | ---: | ---: |
| Empirical two-slot contrast | 188 | 94 | 1,574 | 0.619375 |
| Bellman TRM contrast | 188 | 94 | 1,572 | 0.619583 |
| Ridge contrast | 188 | 94 | 1,588 | 0.617917 |
| Bellman two-slot contrast | 188 | 94 | 1,572 | 0.619583 |
| Bellman contrast with residuals swapped | 188 | 94 | 1,572 | 0.619583 |

The source bank supplied 7,421 admissible action rows and 1,831 logged same-state contrasts. The contrast Bellman actor matched the frozen planner on utility at every root, including at six calls. Its correction changed decisions in 53 scenarios. The donor changed the intervened action in all 53, with 53 completed proofs both before and after.

The source temperature fit selected 2.0 for both runs. The absolute-fit SHA-256 is `19d082c9f6234917ac2dc829211d3c8dd9f119876752d5256d717f08e8c03354`. The contrast fit SHA-256 is `fd819f643decfc896b919c6439cf8ff0963858683bb5a7aa7d384b0d40e38c84`. The frozen planner reproduced every earlier completion-masked trajectory. Across both experiments, Lean independently compiled 157 and 158 distinct scripts and directly replayed three changed successful trajectories in each. All 110 native-enabled tests passed. Fitting, scoring, and testing ran on logical CPU 2 under the 2 GiB process-tree cap, with peak committed memory of 0.51 GiB.

[Absolute-value results and headroom audit](results/ppt_hybrid_rl/ppt-guard-residual-audited-1109.json.gz) · [Action-contrast results](results/ppt_hybrid_rl/ppt-guard-residual-pairwise-results-1109.json.gz)

```text
python -m configs.run_ppt_guard_residual fit BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS
python -m configs.run_ppt_guard_residual eval BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS
```

Use a separate checkpoint and result path and pass `--pairwise` during fitting for the contrast variant.
