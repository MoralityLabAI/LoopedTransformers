# Bellman corrections to the frozen guard belief

The 1109 action-value correction gained five verified proofs at six calls and left the seven- and eight-call proof totals unchanged. This experiment puts the recurrent correction upstream of the frozen planner. The same four relative weights describe which guard is false at all budgets. Paid Lean observations eliminate inconsistent worlds before the planner values the next action. The action menu, billed Lean operations, zero-valued stop option, and terminal proof reward stay fixed.

For each hypothesis about the false guard, the existing two-slot actor receives two witness tokens and the four-source frozen tactic context. Both witness tokens include the selected guard's full descriptor and the sibling's five scalar fields. Their orientation follows the hypothesis. The shared actor issues one additive correction in ±0.75 log-weight units per hypothesis. Its initial output is zero. This allows at most a 1.5 change in any pairwise log odds. The same four corrected weights are used throughout a proof attempt. The exact finite planner is batched only for training. Evaluation uses the existing scalar recurrence. It operates on public guard descriptors and paid observations.

## Registered comparison

Use the 48 original source root pairs with four-fold out-of-fold frozen guard predictions. Use the same 600 batches of 32 admissible source action rows and initial two-slot weights for Bellman Q targets, empirical suffix Q targets, and direct false-guard classification. The latter is trained with four-way cross-entropy on the native-checked false guard of each sampled source row. The source-only affine calibrator uses the same input, bound, sample order and classification loss. The TRM control uses the same Bellman loss, sample order and representation. The frozen supervised planner and its earlier source-only temperature fitting procedure are separate controls. Training uses AdamW at .002 and gradient clipping at 1.

Score both worlds in each of the 40 previously exposed development roots at 6, 7 and 8 calls. The primary endpoint is root-macro proof utility, with a charge of .025 per native call. The floor is +.025 versus every trained and calibrated control, including comparison with the previous bounded action correction on the same archived cases. Report all arms and the per-budget, paired-root changes. Absolute performance and root comparisons are descriptive because the test cohort shaped the architecture. This pilot remains `FSI-01 REDESIGN_OR_PILOT_ONLY` and has no prospectively calibrated resolution or sham envelope. Results above the floor would still require a new, sealed root cohort.

For BPD, take the first decision where the learned belief changes the frozen planner's choice. Swap the two sets of witness log-weight addends for that decision and restore the learned placement afterwards. A control permutes the two guards within each witness on the same decision. Both retain the four addends, the supervised prior, paid observations and Lean transition outcomes. Report the size of each permutation's logit change as well as action and proof effects. Their doses need not agree. Independently compile every distinct successful script and live replay selected successes and failures.

## Development results

| Policy | Six calls / 80 | Seven calls / 80 | Eight calls / 80 | Proofs / 240 | Complete pairs / 120 | Charged calls | Root-macro utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen supervised planner | 44 | 64 | 80 | 188 | 94 | 1,572 | 0.619583 |
| Temperature calibration | 42 | 64 | 80 | 186 | 93 | 1,568 | 0.611667 |
| Affine classification calibration | 42 | 62 | 80 | 184 | 92 | 1,562 | 0.603958 |
| Empirical two-slot belief | 46 | 62 | 80 | 188 | 94 | 1,566 | 0.620208 |
| Bellman TRM belief | 44 | 64 | 80 | 188 | 94 | 1,572 | 0.619583 |
| Supervised two-slot belief | 40 | 72 | 80 | 192 | 96 | 1,570 | 0.636458 |
| Bellman two-slot belief | 42 | 62 | 80 | 184 | 92 | 1,556 | 0.604583 |
| Bellman belief with witnesses swapped | 44 | 60 | 80 | 184 | 92 | 1,562 | 0.603958 |
| Bellman belief with guards swapped within each witness | 40 | 58 | 80 | 178 | 89 | 1,558 | 0.579375 |
| Archived Bellman action-value residual | 49 | 64 | 80 | 193 | 96 | 1,581 | 0.639479 |

The Bellman belief correction lost 0.015 utility against each of the frozen planner and matched TRM, 0.031875 against its supervised twin, and 0.034896 against the archived bounded action correction. It won on 14 roots, tied on 11, and lost on 15 against the frozen planner. Direct supervision shifted performance in the other direction: it gained eight proofs at seven calls and lost four at six. Every seven-call trajectory for the frozen planner and both two-slot belief variants began with a probe, so the seven-call differences came from which guard was probed and how its observation was used. The archived bounded action correction retained the lead and recorded the strongest six-call result.

False-guard top-1 accuracy was 22/80 frozen, 26/80 Bellman belief, and 30/80 for the supervised twin. Their mean negative log likelihoods were 1.487, 1.552, and 1.457, respectively. The affine calibrator attained 1.434 log loss with 184 proofs. Ranking individual guard hypotheses, calibrating their probabilities, and choosing a tactic under a call cap are different measured outcomes. The budget-specific tradeoff motivates testing whether a correction can distinguish the value of committing to a witness from the information value of probing while retaining the exact update from paid evidence.

The BPD intervention was eligible in 146 scenarios. The witness swap changed the next action in 142, with 28 proofs gained and 28 lost among the eligible cases. The within-witness control changed the action in 84, with ten proofs gained and 16 lost. The mean logit redistribution norms were 2.410 and 1.541. Both preserved every paid prefix. The dose difference and bidirectional proof flips prevent a witness-specific causal interpretation.

The fit used 7,421 recorded source decisions and a public planner with 197 reachable action states. The sealed fit SHA-256 is `8e4cb44e43d32f22b5df75f037c7411df17b00038017c5d1b82f2d4257f387fa`. The frozen planner exactly reproduced all 240 archived traces. Lean independently compiled 154 distinct successful scripts and replayed eight sampled traces, covering both successes and failures. Fitting and scoring ran on logical CPU 2, with peak process-tree committed memory of 0.55 GiB under the 2 GiB cap. All 113 native-enabled gym tests passed. The pilot missed the registered utility floor and remains `FSI-01 REDESIGN_OR_PILOT_ONLY`.

[Numerical receipt](results/ppt_hybrid_rl/ppt-guard-belief-dev-1123.json.gz) · [Archived bounded action control](results/ppt_hybrid_rl/ppt-guard-residual-audited-1109.json.gz)

```text
python -m configs.run_ppt_guard_belief fit BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS
python -m configs.run_ppt_guard_belief eval BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS --action-control ARCHIVED_ACTION_CONTROL
```
