# Learning the continuation choice in Lean proof search

The previous deadline study identified two different problems. The hybrid often chose the faster first tactic and then passed over a direct local proof. Giving all frozen policies the same direct-fact rule recovered most of those misses. The resulting hybrid beat graph TRM and its own readout calibration, and tied the strongest imitation control.

This time the shared direct-fact rule runs during both training and evaluation. The main decision follows a compulsory first tactic. Both continuation tactics lead to independently compiled Lean proofs. Their minimum costs under the public Horn tactic menu are six and eight checks, including the compulsory prefix. Swapping supporting facts swaps which continuation meets the deadline. The native reference evaluator forces both continuations and compiles both proofs, and supplies a separate audit artifact. The learning arms receive their own executed trajectories.

The candidate routes the warm-start hybrid's evidence adapter to continuation decisions. It learns a projection of accumulated support across recurrent passes alongside the terminal evidence projection. Before training, the copied evidence branch produces the warm-start logits because the new projection starts at zero. The original policy weights stay fixed. The calibration arm trains the adapter's action readout and gain with the recurrent terms frozen.

The experiment compares continuation RL with that calibration arm, deadline-filtered imitation, return-weighted imitation, all-proof imitation, full graph TRM, and frozen-backbone graph TRM. Return-weighted imitation weights each retained verified action by the positive return from its proof. It shares the graded preference for faster proofs, while RL can also learn from unsuccessful trajectories. All arms get identical paired proposals at the interior branch and the same rollout budget. The hybrid arms share the earlier full imitation checkpoint. The TRM arms share the earlier full TRM checkpoint, trained under the same rollout budget and source mix with its own objective.

The study uses seed 151, 24 training pairs, and 24 development pairs. Twelve development pairs have three premises at a join, while the paired continuation training uses two. All arms also see LoopedTransformers, BPD-LG-02, and manifest-verified EvalMath native declarations through the mixed cohort and MathCan through separately audited replay. The primary outcome is independent Lean certification in both worlds within six checks. The causal comparison removes support history at the paired continuation decision and compares it with an equal-norm common shift of the same logits.

The registry reserves 480 disjoint layout pairs. Results here are for development. Any confirmation requires a prospective test with a five-point meaningful-effect floor, comparison to the strongest imitation and calibration controls, and replication across seeds.

```text
python -m configs.freeze_ppt_continuation_deadline results/ppt_hybrid_rl/root-adaptation-cohort.json COHORT.json
python -m configs.run_ppt_deadline COHORT.json MATHCAN-TRAIN.zip SEED-151-RUNS RUNS
python -m configs.probe_ppt_hybrid_bank COHORT.json RUNS/continuation_rl.sqlite RUNS/arrival-probe.json --arrival
python -m configs.analyze_ppt_deadline COHORT.json RUNS SEED-151-RUNS RUNS/arrival-probe.json summary.json evidence.json.gz
```

## Paired continuation results

| Policy | Certified by six checks out of 48 | Both worlds certified by six checks out of 24 | Certified by eight checks |
| --- | ---: | ---: | ---: |
| Hybrid continuation RL | 27 | 3 | 48 |
| Hybrid continuation readout calibration | 24 | 0 | 48 |
| Deadline-filtered imitation | 29 | 5 | 48 |
| Return-weighted imitation | 29 | 5 | 48 |
| All-proof imitation | 29 | 5 | 48 |
| Full graph TRM, RL | 8 | 0 | 31 |
| Frozen-backbone graph TRM | 11 | 0 | 33 |

The shared schedule and adapted hybrid continuations achieved full coverage by eight checks. The six-check scores therefore distinguish search effort on the hybrid arms. Removing support history at the second action changed three paths and reduced the RL hybrid to 24 on-time proofs and zero completed pairs. The matched shift preserved all 27 on-time proofs and three completed pairs. All three imitation controls finished more proofs on time than this RL update.

During training, the hybrid RL arm collected 29 on-time and 43 late certificates in 96 deadline rollouts. Each imitation arm collected 43 on-time and 37 late certificates in 96 rollouts. The seven-arm campaign averaged 5.17 logical processors' worth of CPU time and peaked at 3.30 GiB committed under its 6 GiB process-tree cap.

## Follow-up using sampled action values

The follow-up starts from the same warm checkpoint and repeats the same paired proposals. The two tried tactics at each public continuation produce two sampled returns. The model fits their difference using its logit margin. Twice the observed return difference is the regression target, fixed before running this arm. Failed and late searches can both inform this target. The later actions keep the original actor-critic objective. The independent confirmation layouts stay outside the follow-up.

```text
python -m configs.run_ppt_deadline COHORT.json MATHCAN-TRAIN.zip SEED-151-RUNS VALUE-RUNS --action-value-study results/ppt_hybrid_rl/continuation-action-value-study.json
python -m configs.analyze_ppt_continuation_value COHORT.json results/ppt_hybrid_rl/continuation-action-value-study.json evidence.json.gz VALUE-RUNS value-summary.json value-evidence.json.gz
```

The action-value arm fitted 48 paired action-value comparisons. It closed 26 of 48 worlds within six checks and completed both worlds in two pairs. The original continuation RL arm reached 27 and three. Each imitation control reached 29 and five. All of these hybrid policies certified every world by eight checks. The follow-up leaves the RL-specific advantage unresolved on this development family. The confirmation sets remain sealed.
