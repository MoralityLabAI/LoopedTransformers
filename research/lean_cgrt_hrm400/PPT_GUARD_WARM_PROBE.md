# Reward fine-tuning of the supervised Lean probe selector

The 1125 supervised two-slot selector solved 72 of 80 seven-call development proofs, compared with 62 for the same architecture trained on return from its initial weights. The return model won on shallow source and lost on development even under the same stochastic selection rule. This follow-up holds the supervised checkpoint fixed as an anchor, compares two source curricula for reward adaptation, and tests both objectives on the deeper curriculum.

## Registration

The original source holds 48 shallow Lean proof pairs. A separate namespace generates 48 new source pairs at the original development depths, with equal counts from the BPD recurrence and MathCan mask. Both worlds in a root are kept together. The source bank is checked by native Lean and its successful orders independently compiled. Pair, world and full statement identities must be disjoint from both original partitions. Individual closed computation expressions may overlap and are counted separately. The generation rule takes no development outcomes as input.

Reuse the archived two-slot and affine supervised weights. Train an analogous TRM supervised anchor using the same shallow source objective and 480 updates. Each new deep pair is assigned to one of the same four shallow reader folds by a public hash of its pair ID. No part of the deeper corpus refits the frozen base reader. Each adapter begins at its architecture's anchor. Its inference uses 65% of the anchor's log-weight correction and 35% of the adapter's proposal, limiting any change from the anchor to 0.525 log-weight units. The original correction range remains ±0.75.

The primary is the two-slot adapter fitted to exact native proof return on the deeper source pairs. Compare its identical shallow-source variant, the deep-source supervised two-slot and TRM adapters, the reward-trained TRM and affine adapters, all three frozen anchors, and the archived cold return and frozen planner. Match source counts and use one sampler for 240 updates of 16 worlds at LR .0004, with a .025 penalty on KL divergence of the probe distribution from the frozen anchor. Apply the same blend and penalty to the supervised adaptations. All fits use only source rewards or source guard truth. Compare greedy root-macro utility at seven and eight calls across both worlds of the 40 exposed development roots, with +.025 utility as the descriptive floor over the strongest control. Check that all six-call trajectories match the frozen planner.

At the first probe disagreement with the frozen two-slot anchor where all four probes are feasible, intervene on only the new log-weight increments. Exchange the increments associated with the two chosen guard hypotheses and keep the anchor weights untouched. A matched control transfers the identical signed changes to the two bystander guards. Require equal dose and the same paid prefix. Independently check winning scripts and replay sampled traces for both interventions. The reused cohort and source expression overlap limit evidence to `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

| Policy | Proofs / 240 | Proofs with probe budget / 160 | Root utility with probe budget |
| --- | ---: | ---: | ---: |
| Frozen planner | 188 | 144 | 0.720938 |
| Archived cold two-slot return | 186 | 142 | 0.710000 |
| Frozen affine supervised anchor | 182 | 138 | 0.685625 |
| Deep affine return | 184 | 140 | 0.697188 |
| Frozen TRM supervised anchor | 190 | 146 | 0.732813 |
| Deep TRM supervised | 190 | 146 | 0.733125 |
| Deep TRM return | 190 | 146 | 0.733125 |
| Frozen two-slot supervised anchor | 196 | 152 | 0.769688 |
| Shallow two-slot return | 196 | 152 | 0.769375 |
| Deep two-slot supervised | 196 | 152 | 0.770000 |
| Deep two-slot return | 196 | 152 | 0.769688 |
| Deep two-slot return, chosen-probe swap | 196 | 152 | 0.769688 |
| Deep two-slot return, bystander swap | 196 | 152 | 0.769688 |

Deep two-slot return retained the supervised advantage over matched TRM by six proofs and 0.036563 utility at the probe budgets. Its proof count and utility matched the frozen two-slot anchor. Deep supervised adaptation used two fewer paid calls than either at the same proof count. All models preserved the full six-call traces. At eight calls all arms closed every proof.

Seven-call stochastic utility improved on the deep source from 0.57380 to 0.64166 for two-slot return and to 0.64854 for two-slot supervision. The shallow reward update moved source utility from 0.80511 to 0.80518. The deeper curriculum produced a larger train-side update while greedy development remained at the anchor's 72 seven-call proofs. The generated bank contains 140 distinct closed expressions. Of those, 63 also occur in the exposed development bank and zero occur in the shallow source. Its full theorem pairs, world IDs and statements are disjoint from both original partitions.

The registered BPD probe was eligible in ten attempts. Swapping the learned increments changed six probe selections and traded two previously successful proofs for two previously failed ones. The bystander control changed one selection and no proofs. The average L2 dose of 0.4784 log-weight units was identical. Both retained the same aggregate proof utility as the natural adapter.

## Post-hoc strength diagnostic

After observing the registered plateau, score every saved proposal at full strength, using the same checkpoints with no extra updates or model selection. Evaluate stochastic development utility at the training temperature as a separate diagnostic.

| Full-strength proposal | Seven-call proofs / 80 | Total proofs / 240 | Root utility with probe budget |
| --- | ---: | ---: | ---: |
| Shallow two-slot return | 72 | 196 | 0.769375 |
| Deep two-slot return | 74 | 198 | 0.781250 |
| Deep two-slot supervised | 72 | 196 | 0.770000 |
| Deep TRM return | 64 | 188 | 0.721250 |
| Deep TRM supervised | 64 | 188 | 0.721250 |
| Deep affine return | 59 | 183 | 0.691250 |

The full-strength two-slot reward update gained two greedy proofs and 0.011563 utility over the frozen supervised anchor. Four roots improved, 32 tied, and four declined. This fell below the +.025 floor and was chosen for inspection after the registered readout. Its seven-call stochastic utility was 0.67466, compared with 0.68737 for the anchor and 0.69361 for the deep supervised adapter at the registered strength.

The matched interventions on the full-strength update shared 36 eligible attempts and mean L2 dose 1.5057. The chosen-probe swap changed 26 selections, lost two proofs, and gained two. The bystander control changed 14 selections, lost five proofs, and gained two. Both preserved all paid prefixes. The chosen-probe swap left the two-proof net gain intact, so this probe did not isolate a beneficial placement effect.

The sealed warm fit SHA-256 is `be855e9c7367f19e2df1865e17e7d33e371bc70958d1e98485d47c037deeb1b4`. The deep bank SHA-256 is `5c43ffa6d889edb1b0137c6fc5b43e0867b760b0f6d1c15ddbc00a0f67eef5fa`. The final evaluation reproduced all registered trajectories and independently compiled 134 development proof scripts. Lean replayed 12 selected successes and failures, including swapped traces for both strengths. The 192 source certificates are recorded separately. Execution used logical CPU 2 and the 2 GiB process-tree cap, with peak committed memory of 0.52 GiB in the final evaluation. All 122 native-enabled gym tests passed. [Deep source bank](results/ppt_hybrid_rl/ppt-guard-deep-source-1126.json.gz) · [Numerical receipt](results/ppt_hybrid_rl/ppt-guard-warm-probe-audited-1126.json.gz)

```text
python -m configs.freeze_ppt_guard_curriculum BANK BPD MATHCAN CURRICULUM
python -m configs.run_ppt_guard_warm_probe fit BANK BPD MATHCAN WARM PRIOR PLANNER CURRICULUM WARM_PROBE FIT RESULTS
python -m configs.run_ppt_guard_warm_probe eval BANK BPD MATHCAN WARM PRIOR PLANNER CURRICULUM WARM_PROBE FIT RESULTS --archive ARCHIVE --proposal-diagnostic
```
