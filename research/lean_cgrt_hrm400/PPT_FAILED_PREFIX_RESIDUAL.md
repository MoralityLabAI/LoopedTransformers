# Bounded reward corrections to the Lean recovery planner

Unconstrained reward training solved 119 of 240 failed-prefix attempts. The affine joint guard model with exact public planning solved 148, and a planner with actual guard truth solved 179. This pilot keeps the affine planner frozen and tests a small two-slot correction specifically at recovery decisions within its action envelope.

## Registered pilot

Use the existing failed-prefix bank with 48 source and 40 exposed development theorem pairs. The real Lean rejection, both saved snapshots, public continuation costs, and charge rules are unchanged. Train on four-fold out-of-fold affine guard posteriors, grouping both worlds of each theorem pair. At development, use the sealed original affine posterior and require that every frozen-planner attempt reproduce its archived trace.

The exact planner removes worlds contradicted by paid failures. Feed its conditional success probabilities and action values into slots that distinguish the retained and candidate witness. The recurrent actor adds at most ±.15 utility to each feasible action, with zero adjustment initially. Keep the value of stopping fixed at zero and retain the native affordability mask. Fit the actor by optimizing exact expected proof return using source native route observations. Compare the same architecture trained on full-information Q targets, TRM and affine actors trained with the same return objective, and the frozen affine planner. All new arms share 360 batches of 16 source worlds and AdamW at .002. The primary endpoint is root-macro proof utility across three, five and seven recovery calls after the common paid prefix. The descriptive floor is +.025 versus every non-oracle control. Report a separate actual-truth upper bound restricted to actions the ±.15 correction could make optimal.

At the first disagreement with the frozen planner that has three affordable routes and two distinct non-stop choices, exchange the learned additions for the frozen and learned routes. The matched sham moves the same L2 dose in the orthogonal, zero-sum direction that preserves the chosen routes' logit difference. Both keep the exact affine values, observed failures and paid prefix. Restore the natural policy after that decision. This reused-cohort pilot has no calibrated sham envelope and stays at `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

| Policy | Three recovery calls / 80 | Five / 80 | Seven / 80 | Proofs / 240 | Checks | Root utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived unconstrained two-slot return | 19 | 44 | 56 | 119 | 2,174 | 0.269375 |
| Frozen affine planner | 25 | 53 | 70 | 148 | 2,246 | 0.382708 |
| Two-slot return correction | 23 | 51 | 65 | 139 | 2,260 | 0.343750 |
| Two-slot Q correction | 25 | 53 | 68 | 146 | 2,259 | 0.373021 |
| TRM return correction | 23 | 54 | 69 | 146 | 2,248 | 0.374167 |
| Affine return correction | 24 | 55 | 74 | 153 | 2,264 | 0.401667 |
| Two-slot return, chosen-route swap | 23 | 52 | 69 | 144 | 2,237 | 0.366979 |
| Two-slot return, orthogonal sham | 23 | 51 | 68 | 142 | 2,267 | 0.355521 |
| Actual-truth upper bound within correction envelope | 26 | 68 | 79 | 173 | 2,200 | 0.491667 |

The frozen planner reproduced all 240 archived attempts. The two-slot reward correction recovered 20 proofs relative to the unconstrained policy and trailed the frozen anchor by nine. Its paired utility improved on nine roots, tied on ten and declined on 21. The affine reward correction added five proofs and .018958 utility over frozen, below the .025 floor. It improved 13 roots, tied on 13 and declined on 14. The restricted upper bound leaves room for 25 more proofs over the frozen anchor at the same correction scale.

The registered chosen-route swap changed the next action in 43 of 43 eligible attempts, gaining ten proofs and losing five. The orthogonal sham changed 16 actions, gaining four proofs and losing one. Their L2 doses matched at every eligible decision and averaged .3652 utility in score space. Both perturbations improved the natural recurrent policy's aggregate utility. This intervention does not identify a beneficial placement effect in the recurrent additions.

## Post-hoc intervention on the affine correction

After measuring the affine improvement, apply the same matched intervention to that frozen correction without refitting any model. All registered trajectories reproduced exactly in this second evaluation.

| Affine reward policy | Proofs / 240 | Checks | Root utility |
| --- | ---: | ---: | ---: |
| Natural | 153 | 2,264 | 0.401667 |
| Chosen-route swap | 149 | 2,251 | 0.386354 |
| Orthogonal sham | 152 | 2,265 | 0.397396 |

The affine swap changed 42 of 42 eligible decisions, losing seven proofs and gaining three. The sham changed nine of the same 42 decisions and lost one proof. The matched mean L2 dose was .0952 utility. This exploratory result ties most of the observed proof gain to the learned affine score placement on this cohort. It falls below the registered gain floor and remains a pilot result on exposed development roots.

The correction fit SHA-256 is `499c93aa19e1a1fb20dfe12e49c69773a05f7469deb67c589a99687d5c5833ef`. The final evaluation independently compiled 111 distinct successful Lean scripts and live-replayed 16 selected traces, including genuinely patched successes and failures for each intervention. All paid prefixes were preserved. Execution used logical CPU 2 and stayed below 0.57 GiB of committed memory in the 2 GiB process-tree cap. All 131 native-enabled gym tests passed. [Numerical receipt](results/ppt_hybrid_rl/ppt-failed-prefix-residual-audited-1129.json.gz)

```text
python -m configs.run_ppt_failed_prefix_residual fit BANK BPD MATHCAN WARM PRIOR ANCHOR FIT RESULTS
python -m configs.run_ppt_failed_prefix_residual eval BANK BPD MATHCAN WARM PRIOR ANCHOR FIT RESULTS --archive ARCHIVE --affine-intervention
```
