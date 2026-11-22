# Learning to finish Lean proofs on time

The previous first-action experiment improved on readout calibration across seeds. Continued verified-path imitation came close to first-action RL on the same proof-success objective. Here both initial tactics lead to axiom-free proofs. Under the gym's public Horn tactic menu, their minimum costs are six and eight Lean tactic checks. The companion world swaps those costs and keeps the rule declarations and initial action menu fixed. Finding a valid proof alone leaves the more efficient tactic unresolved.

The cohort contains 32 training layout pairs and 24 development layout pairs. Twelve of the development pairs use three-premise rules, while the paired cost training uses two. The native reference evaluator forces each permitted first action, chooses the shortest public Horn continuation for the audit, and compiles both witness proofs. The training pipeline uses observations and rewards from its own executed episodes. The registry reserves 480 unused layouts, with older conjunction and ARG-03-N holdouts preserved.

The candidate keeps the warm-start Plan A/B hybrid's first-action adapter and unchanged continuation policy. An additional learned projection observes support accumulated across recurrent passes. Early support contributes over more passes before the decision. The projection begins at zero, so adding it preserves the warm-start action logits. The hybrid arms initialize from the same full imitation checkpoint. Graph TRM and its calibration arm start from the same previously trained graph TRM checkpoint, which saw the same prior training sources under the same rollout budget. This is a comparison of warm-start policies with different prior objectives and equal training exposure in the deadline experiment.

All policies can spend up to eight checks and retain both on-time and late standalone certificates. For RL on the paired cost layouts, a late valid proof earns 0.35 of the early proof reward, with the usual per-tactic cost applied to both. Deadline-filtered imitation learns from the on-time successes in its own rollouts, while all-proof imitation also uses its late successes. The other controls are the same warmed hybrid with the recurrent body frozen and its first-action readout trainable, the warmed hybrid with only the original terminal recurrent readout, full graph TRM, and frozen-backbone graph TRM. Every arm is exposed to the same source roots and tries both initial tactics in each training world. Later trajectories can differ as policies adapt.

The mixed batches include LoopedTransformers Horn pairs, BPD-LG-02 exact rows, the manifest-verified EvalMath training subset and the original audited MathCan replay. MathCan contributes archived certificate rewards or verified-path demonstrations. The causal comparison removes the accumulated-support contribution at the first decision and uses an equal-norm common shift of the same logits as a control. The primary outcome is a pair in which both worlds independently close within six checks. The development study uses seed 151 and a five-point meaningful-effect target for any prospective confirmation.

Advancement to independent confirmation requires a gain over the strongest deadline-aware imitation and readout-calibration controls, enough independent layouts to resolve that gain above the matched sham, and replication across training seeds. The pilot is a diagnostic comparison, with the reserved layouts held for prospective scoring.

```text
python -m configs.freeze_ppt_deadline results/ppt_hybrid_rl/root-adaptation-cohort.json DEADLINE-COHORT.json
python -m configs.run_ppt_deadline DEADLINE-COHORT.json MATHCAN-TRAIN.zip SEED-151-RUNS RUNS
python -m configs.probe_ppt_hybrid_bank DEADLINE-COHORT.json RUNS/arrival_rl.sqlite RUNS/arrival-probe.json --arrival
python -m configs.analyze_ppt_deadline DEADLINE-COHORT.json RUNS SEED-151-RUNS RUNS/arrival-probe.json summary.json evidence.json.gz
python -m configs.probe_ppt_deadline_continuation DEADLINE-COHORT.json RUNS continuation-probe.json.gz
python -m configs.probe_ppt_deadline_continuation DEADLINE-COHORT.json RUNS direct-facts.json.gz --prefer-direct-fact --arms arrival_rl readout_calibration deadline_imitation all_proofs_imitation terminal_readout_rl trm_rl trm_calibration
python -m configs.probe_ppt_deadline_continuation DEADLINE-COHORT.json RUNS direct-facts-cut.json.gz --prefer-direct-fact --arms arrival_rl --root-edit ablate-arrival
python -m configs.probe_ppt_deadline_continuation DEADLINE-COHORT.json RUNS direct-facts-sham.json.gz --prefer-direct-fact --arms arrival_rl --root-edit control-common-shift
```

## Development results

| Policy | Faster first tactic out of 48 | Certified by six checks | Certified by eight checks |
| --- | ---: | ---: | ---: |
| Accumulated-support hybrid, RL | 27 | 4 | 24 |
| Hybrid readout calibration | 24 | 4 | 24 |
| Deadline-filtered verified-path imitation | 26 | 4 | 25 |
| All-proof verified-path imitation | 27 | 4 | 26 |
| Terminal-readout hybrid, RL | 24 | 4 | 24 |
| Full graph TRM, RL | 24 | 12 | 33 |
| Frozen-backbone graph TRM | 24 | 1 | 6 |

No arm solved both worlds of a pair within six checks. In the hybrid RL arm, 24 of 32 paired training batches contained no on-time proof. The learner updated in all 24 of these batches under the graded reward. The deadline-filtered imitation arm updated in zero of those batches. The different update opportunities still yielded the same four on-time development certificates.

Removing accumulated support at the first decision changed nine proof paths and left the on-time score at four. The matched common shift changed no proof paths. Forcing the witnessed faster first tactic lifted the hybrid to 9 of 48 and graph TRM to 22 of 48 without changing their continuations. Among the hybrid's 23 original deadline misses after a feasible first action, 21 first diverged from the native witness by passing over an available direct fact. The other two first diverged at a rule choice.

The complete seven-arm CPU run averaged 6.50 logical processors' worth of CPU time. It peaked at 3.20 GiB committed under the 6 GiB process-tree cap. The observed one-seed result leaves the reserved confirmation layouts untouched.

## Shared direct-fact continuation

After diagnosing the misses, all seven frozen policies were evaluated on the same development layouts with a shared continuation rule. If a local hypothesis has exactly the active goal's type, the rule prefers a direct proof using that hypothesis. It operates after the first tactic, preserves each policy's learned first decision, and leaves proof acceptance to Lean.

| Policy | Certified by six checks with the shared rule | Both worlds certified by six checks |
| --- | ---: | ---: |
| Hybrid with recurrent support history, RL | 25 | 2 |
| Hybrid readout calibration | 22 | 0 |
| Deadline-filtered verified-path imitation | 24 | 2 |
| All-proof verified-path imitation | 25 | 2 |
| Terminal-readout hybrid, RL | 22 | 0 |
| Full graph TRM, RL | 12 | 0 |
| Frozen-backbone graph TRM | 13 | 0 |

Under the shared rule, removing the learned support history lowered the hybrid from 25 to 22 on-time certificates and from two to zero pairs solved in both worlds. The matched shift preserved 25 on-time proofs and two complete pairs. The hybrid ties the strongest verified-path imitation control on both endpoints. This one-seed, post-readout diagnostic gives a specific continuation failure to address and shows a conditional advantage over the tested TRM and readout controls.
