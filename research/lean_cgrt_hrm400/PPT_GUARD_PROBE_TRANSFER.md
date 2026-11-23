# Unseen-expression transfer for the frozen Lean probe policies

The 1126 full-strength reward update gained two proofs over the supervised anchor on exposed development roots. The gain was selected after the registered blended readout and fell short of the utility floor. This is a separate, frozen transfer pilot for that candidate and its predeclared controls.

## Locked before native evaluation

Select 32 paired theorem roots for the BPD recurrence and 32 for the MathCan mask at the original development depths. Use the fixed `probe-transfer-unseen-expressions-1127` generator namespace and published order. Filter public statements by the left side of each closed guard equation. Exclude every exact expression from the original 48 shallow source pairs, 40 exposed development pairs, and 48 deeper source pairs. Exclude duplicate unordered guard menus, paired theorem roots, statements and world IDs. The accepted set has 63 distinct left-hand expressions, with up to nine roots sharing one inside the fresh cohort. Distinct expressions can still share the underlying computation family, inputs or depth. This audit covers the three registered guard banks.

Hash the public statements and freeze the fitted model, calibration and training-bank identities before running native Lean or scoring any model. Certify both successful native verification orders for each new theorem. Score frozen checkpoints once. The primary is the full-strength deep reward update. The frozen comparison family is the two-slot supervised anchor, the full-strength two-slot supervised update, the full-strength matched TRM reward update, and the chosen-probe versus bystander intervention. Also report the existing affine, shallow-return, blended, frozen-planner and temperature-calibrated controls. No parameter updates or choice of blend are made on the new cohort. [Locked inputs and comparisons](configs/ppt_guard_probe_transfer_lock.json)

The primary outcome is root-macro utility with seven or eight charged calls. Report both paired worlds at each budget, giving 256 primary rows across 64 roots. The 128 six-call attempts test whether learned probe corrections preserve the no-probe traces. One proof earns one unit and a charged native operation costs .025. The descriptive practical floor is +.025 utility over the strongest control. Report stratified paired-root bootstrap intervals as descriptive sensitivity to root resampling.

For BPD, intervene at the first disagreement with the supervised two-slot anchor where all four probes are available. Swap the new log-weight increments on the two chosen guards. The control distributes the exact same signed increments across the two bystanders. Both retain the anchor weights, paid prefix and true subsequent Lean observations. Log the L2 dose and recheck a selection of traced successes and failures.

The 40-root retrospective pilot had a mean effect of +.01156, a root standard deviation of .16971, and a worst observed sham contrast of .475. The 95th-percentile sham envelope could not be calibrated from that cohort. For the four frozen comparisons, the EvalMath preflight estimates a resolvable effect near .096 at 64 roots and about 514 roots for 80% planning power at .05 over a .025 floor. Recurring expressions make the assumptions for that estimate optimistic. This transfer study remains `FSI-01 REDESIGN_OR_PILOT_ONLY`. A larger sealed study would need a reliable sham envelope, a larger development signal and an audit of shared computations.

## Transfer results

| Frozen policy | Six calls / 128 | Seven calls / 128 | Eight calls / 128 | Total proofs / 384 | Root utility at seven and eight calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen planner | 82 | 108 | 128 | 318 | 0.742383 |
| Temperature calibration | 80 | 108 | 128 | 316 | 0.742383 |
| Affine supervised anchor | 82 | 104 | 128 | 314 | 0.726953 |
| Deep affine return, full | 82 | 106 | 128 | 316 | 0.733984 |
| TRM supervised anchor | 82 | 110 | 128 | 320 | 0.750195 |
| Deep TRM return, blended | 82 | 110 | 128 | 320 | 0.750586 |
| Deep TRM return, full | 82 | 104 | 128 | 314 | 0.727539 |
| Deep TRM supervised, full | 82 | 100 | 128 | 310 | 0.712500 |
| Two-slot supervised anchor | 82 | 116 | 128 | 326 | 0.773047 |
| Shallow two-slot return, full | 82 | 114 | 128 | 324 | 0.765625 |
| Deep two-slot return, blended | 82 | 114 | 128 | 324 | 0.765430 |
| Deep two-slot return, full | 82 | 116 | 128 | 326 | 0.772852 |
| Deep two-slot supervised, blended | 82 | 112 | 128 | 322 | 0.758398 |
| Deep two-slot supervised, full | 82 | 118 | 128 | 328 | 0.780664 |
| Full return, chosen-probe swap | 82 | 116 | 128 | 326 | 0.773047 |
| Full return, bystander swap | 82 | 119 | 128 | 329 | 0.783887 |

The exposed-cohort reward gain did not transfer. At the same proof count as its supervised anchor, the full-strength update used two more calls and changed utility by −.000195. It improved 10 roots, tied on 42, and declined on 12. At seven calls it exchanged two BPD proofs for two MathCan proofs relative to the anchor. The full-strength supervised update solved two more than either. Return adaptation gained .045313 utility over full-strength TRM return, .022266 over the strongest TRM control, and .030469 over both frozen planner and temperature calibration at the primary budgets.

| Frozen contrast | Paired root-macro difference | Descriptive 95% interval |
| --- | ---: | ---: |
| Full two-slot return minus its supervised anchor | −0.000195 | [−0.036914, 0.036719] |
| Full two-slot return minus full two-slot supervision | −0.007813 | [−0.023047, 0.000391] |
| Full two-slot return minus full TRM return | +0.045313 | [0.000000, 0.096680] |
| Chosen swap minus bystander swap | −0.010840 | [−0.032813, 0.005957] |

Both interventions had 76 eligible attempts and a mean L2 dose of 1.8934 log-weight units. The chosen-probe swap changed 62 actions and traded two successful proofs for two previously failed ones. The bystander swap changed 40 actions, gained four proofs and lost one. All proof changes occurred at seven calls. The chosen swap did not supply evidence that the learned reward increments were beneficial where placed.

All learned policies reproduced the frozen six-call traces exactly. The fixed lock was committed as `7244bc2` before native certification and model scoring. Lean certified 256 alternate-order scripts in the cohort, then independently compiled 230 distinct successful evaluated scripts and replayed ten complete traces, including patched successes and failures. Native certification and scoring used logical CPU 2, with peak committed memory of 0.68 GiB under the 2 GiB cap. All 124 native-enabled gym tests passed. [Sealed native cohort](results/ppt_hybrid_rl/ppt-guard-probe-transfer-cohort-1127.json.gz) · [Numerical receipt](results/ppt_hybrid_rl/ppt-guard-probe-transfer-results-1127.json.gz)

```text
python -m configs.run_ppt_guard_probe_transfer freeze BANK BPD MATHCAN WARM PRIOR PLANNER CURRICULUM WARM_PROBE FIT COHORT RESULTS
python -m configs.run_ppt_guard_probe_transfer eval BANK BPD MATHCAN WARM PRIOR PLANNER CURRICULUM WARM_PROBE FIT COHORT RESULTS
```
