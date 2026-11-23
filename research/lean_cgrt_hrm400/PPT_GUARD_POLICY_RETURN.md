# Tactic corrections trained on native proof return

The prior coherent guard-belief model lost to TRM and its directly supervised twin. Supervision gained eight seven-call proofs with a paid probe and lost four six-call proofs where probing is infeasible. Test whether distinct recurrent readouts can improve the values of committing to a witness and choosing a guard check while retaining the frozen planner's update after a paid observation.

Reuse the two-slot guard tokens and the native public transition graph. Emit four bounded outputs per guard: tight-budget commitment, commitment with room for a probe, probe choice, and in-branch verification order. An entry uses the mean of the two guard outputs for its witness and budget. A check uses the output for its own guard. Each action changes by at most ±0.075 utility. The frozen planner still supplies exact base values and conditions on all paid observations. Once just one possible false guard remains, all corrections vanish.

## Registration

Fit on the original 48 Lean-verified source root pairs using the same four-fold reader procedure for every arm. For the primary arm, optimize the exact expected proof return of the complete stochastic policy at each of the three budgets. Reuse the frozen transition graph, the original +1 for proof and −0.025 per charged operation, including the shared prefix charge. Use softmax temperature .045 during fitting and deterministic greedy choices during evaluation. The source false-guard identity selects the actual transition for reward accounting only. It never selects policy scores, features, or feasible actions.

Compare the same two-slot architecture and initialization trained to regress full-information source action Q values, TRM trained on the same exact policy return, and an affine return-trained calibrator with the same tactic readouts. Use the same 600 batches of 24 source worlds, AdamW at .002 and gradient clipping at 1. Include the unchanged frozen planner, source-only temperature calibration, and the two strongest archived policies, the bounded Bellman action correction and the supervised belief correction. As a development-selected comparator, route six-call cases to the archived action correction and the other budgets to the archived supervised belief correction. This routing rule was selected from prior development results. Verify the unchanged planner reproduces the archived traces before comparing them.

Score all 40 exposed development roots at 6, 7 and 8 calls for both paired worlds. The primary outcome is equal-root proof utility and the descriptive effect floor remains +.025 against every trained and calibrated control. Root, budget and world identities stay paired. The repeatedly examined cohort and absent prospective sham envelope limit the result to `FSI-01 REDESIGN_OR_PILOT_ONLY`.

For the BPD probe, swap the learned addends of the frozen and learned chosen tactics at their first disagreement where both select a tactic. Keep the supervised values and paid prefix, charge the action actually taken, and return to normal corrections at the next state. Report effective dose, action changes and bidirectional proofs. Reset the two commitment readouts or the two check readouts across an entire attempt as separate sensitivity controls. Compile every distinct successful Lean script and replay representative successful and failed traces.

## Development results

| Policy | Six calls / 80 | Seven calls / 80 | Eight calls / 80 | Proofs / 240 | Calls | Root-macro utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen supervised planner | 44 | 64 | 80 | 188 | 1,572 | 0.619583 |
| Temperature calibration | 42 | 64 | 80 | 186 | 1,568 | 0.611667 |
| Affine policy return | 50 | 58 | 80 | 188 | 1,586 | 0.618125 |
| Two-slot Q regression | 46 | 64 | 80 | 190 | 1,590 | 0.626042 |
| TRM policy return | 44 | 62 | 80 | 186 | 1,568 | 0.611667 |
| Two-slot policy return | 38 | 68 | 80 | 186 | 1,570 | 0.611458 |
| Two-slot commitment only | 38 | 64 | 80 | 182 | 1,560 | 0.595833 |
| Two-slot checks only | 44 | 68 | 80 | 192 | 1,580 | 0.635417 |
| Two-slot policy return, paired action swap | 44 | 64 | 80 | 188 | 1,570 | 0.619792 |
| Archived action correction | 49 | 64 | 80 | 193 | 1,581 | 0.639479 |
| Archived supervised belief correction | 40 | 72 | 80 | 192 | 1,570 | 0.636458 |
| Archived budget router | 49 | 72 | 80 | 201 | 1,591 | 0.671771 |

The full return-trained model missed the floor. It gained four seven-call proofs and lost six at six calls against the frozen planner. The commitment-only intervention reproduced the six-call loss. The checks-only intervention retained the seven-call gain, improving aggregate utility by 0.015833 over frozen. Relative to frozen, it won on eight roots and lost on nine. The archival router combines earlier development strengths and is the strongest observed policy on this cohort.

On source, the return objectives optimized seven-call stochastic utility to 0.75176 for two slots and 0.78084 for TRM. The frozen stochastic planner obtained 0.53172, and the matched Q objective obtained 0.54163. The large source rollout gain did not transfer at the same scale under greedy evaluation. Both the depth shift and the stochastic-to-greedy change are present in this comparison.

The predeclared full-policy BPD swap changed the next action in all 106 eligible attempts. It exchanged choices within a tactic family every time: 56 probe choices, 36 verification orders, and 14 witness entries. Twelve previously successful traces failed and 14 previously failed traces succeeded. The average two-action redistribution had L2 dose 0.1691. Utility increased under this intervention, so the full readout does not show a beneficial overall placement effect.

## Post-hoc checks-only comparison

After observing the reset result, apply the same zero-commitment reset to the already frozen TRM, Q-regression and affine controls. No checkpoint is retrained or chosen based on these diagnostics. The check-only policy is also probed by swapping two check addends at the first eligible disagreement between two probes or between two in-branch checks. Compare this swap only with the check-only policy whose prefix it preserves.

| Reset policy | Six calls / 80 | Seven calls / 80 | Eight calls / 80 | Proofs / 240 | Root-macro utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affine return | 44 | 58 | 80 | 182 | 0.594167 |
| TRM return | 44 | 62 | 80 | 186 | 0.611875 |
| Two-slot Q regression | 44 | 64 | 80 | 188 | 0.617917 |
| Two-slot policy return | 44 | 68 | 80 | 192 | 0.635417 |
| Two-slot policy return, paired check swap | 44 | 64 | 80 | 188 | 0.619583 |

The check-only gain over matched TRM is +0.023542 across all budgets and +0.071875 at seven calls. Across roots, the differences favor two slots on 11 roots, tie on 16, and favor TRM on 13. The +0.025 overall floor is still unmet. The check-only seven-call result also trails the earlier supervised readout's 72 proofs.

The check swap changed 98 of 98 eligible next actions. All 12 proof changes came at seven calls. Eight successful traces failed and four failed traces succeeded, reversing the net four-proof gain over frozen. The patched policy returned to the frozen planner's aggregate utility. The mean L2 change in the two action-value corrections was 0.1652 utility, with no change to prior values or paid prefixes. This is an exploratory placement effect on an exposed cohort.

## Provenance

The source fit SHA-256 is `12f4249b850e772c30ed2f67cc3ee16c868900e994bcdb894fc8b29a05ce66a7`. All registered trajectories reproduced exactly when the post-hoc controls were added. Lean independently compiled 154 distinct successful scripts and live-replayed 12 selected traces, including patched successes and failures for both swaps. Runs used logical CPU 2 and the 2 GiB process-tree cap, with peak committed memory of 0.56 GiB. All 115 native-enabled gym tests passed. The same frozen model hashes, source sampler hash, root contrasts and all paired trajectories are recorded in the receipt. [Numerical receipt](results/ppt_hybrid_rl/ppt-guard-return-final-1124.json.gz)

```text
python -m configs.run_ppt_guard_return fit BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS
python -m configs.run_ppt_guard_return eval BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS --action-control ARCHIVED_ACTION --belief-control ARCHIVED_BELIEF --check-controls
```
