# Learning when to abandon a Lean proof branch

The continuation study gave every policy a comparable path to a verified proof. Imitation could learn the preference for shorter proofs from the successful searches. This study adds an action at the point where search can still recover from a poor choice. Every policy spends one check on the same speculative tactic. At the resulting Lean state it can continue, or restore the previous checkpoint and exclude the abandoned tactic. The restore spends a search decision and preserves the tally of executed Lean checks.

Each layout pair contains two worlds with the same initial tactic menu. Swapping the facts swaps which recovery choice can produce a certificate within eight checks. When the speculative branch is viable, keeping it permits a seven-check proof. When the opposite branch is viable, an early restore permits an eight-check proof. Deferring the restore costs at least one additional check because the current branch still has a legal continuation. A separate native witness tries both recovery decisions, independently replays each search and compiles the successful proof.

The hybrid controller compares the saved branch point with the current proof state. It uses the warm-start Plan A/B policy's parent and current action scores and a learned projection of the recurrent support history. Recovery RL and imitation update 629 of the hybrid's 1,663,309 parameters. Hybrid readout calibration updates 597 and freezes the recurrent support terms. The comparison also includes graph TRM RL, which updates all 1,663,365 parameters, and its 581-parameter frozen-backbone calibration control. The hybrid arms share the same earlier full imitation checkpoint. The TRM arms share the earlier graph TRM checkpoint, trained with the same earlier rollout budget and source mix under its own objective.

Training budgets one sampled recovery decision per world. The two worlds in a pair receive the same sampled decision. Imitation trains on the decisions from independently certified searches, including successful early restores. RL uses the difference in verifier returns between the paired worlds and trains a value estimate from both. Every arm observes the same source exposures. Other training batches use LoopedTransformers, BPD-LG-02 and manifest-verified EvalMath Lean declarations, with archived MathCan proof replay handled separately.

The development experiment uses 32 training pairs and 24 development pairs. Twelve development pairs use four premises at a join. The primary endpoint is independent certificate success in both worlds at the eight-check limit. The causal comparison removes the recurrent contribution at the single recovery decision and compares it with an equal-norm common shift of the same logits. The common shift preserves the immediate action ordering. This checks whether removal of the recurrent contribution changes that ordering. The cohort reserves 192 disjoint layouts. A confirmatory comparison requires the strongest imitation and calibration controls and a five-percentage-point scientific minimum, with resolution and matched-sham variation also included in the effect floor. The previous held-out sets remain sealed.

## Development results

The second and third runs used seed-specific warm checkpoints and the same development layouts as the primary run. Each cell shows fully solved pairs out of 24, followed by individually certified proofs out of 48.

| Policy | Seed 103 | Seed 151 | Seed 197 |
| --- | --- | --- | --- |
| Hybrid recovery RL | 8 / 31 | 13 / 36 | 13 / 37 |
| Hybrid verified-search imitation | 7 / 31 | 10 / 34 | 10 / 34 |
| Hybrid readout calibration | 5 / 28 | 5 / 28 | 9 / 33 |
| Graph TRM recovery RL | 0 / 12 | 0 / 16 | 0 / 11 |
| Graph TRM readout calibration | 0 / 11 | 0 / 16 | 0 / 8 |

Hybrid RL led verified-search imitation by one, three, and three fully solved pairs. This is a 9.7 percentage-point mean difference across the three seeded runs on the same 24 layouts. At seed 103, both trained hybrid policies certified 31 individual proofs, with RL solving one more complete pair. The matched hybrid calibration left more pairs incomplete at every seed. Both graph TRM arms certified individual proofs and solved zero complete pairs.

The hybrid RL training logs recorded 13, 11, and 9 updates on recovery batches where both worlds failed to yield a certificate. The corresponding imitation arms made zero updates on their own no-certificate batches. Each arm in each seed saw 64 LoopedTransformers recovery worlds, 64 BPD exact-proof episodes, 12 EvalMath exact-proof episodes, and 24 MathCan archived episodes. Every development score required a native Lean certificate. The ledger records MathCan archived closures separately from native certificates.

The two uninterrupted five-arm runs averaged 8.13 and 7.83 logical cores and peaked at 2.69 GiB of process-tree commit. The seed-151 job resumed from persisted checkpoints. Its resource receipt covers the resumed segment. The eight-check outcome counts use the complete ledgers for all seeds.

## Recovery-history intervention

The BPD-style intervention edits the recurrent history contribution once per world at the keep-or-abandon readout. After removal, Lean certified both worlds in 2, 3, and 3 pairs, down from 8, 13, and 13. The norm-matched common shift preserved all baseline recovery choices and certified pair counts. Native replay verified every intervention search. Across the three seeds, removal changed 27 recovery choices and lowered the mean fully solved pair rate by 36.1 percentage points on the development layouts.

The layout pair is the unit of analysis. Both worlds and all three seeded checkpoints contribute to that unit. The resolution calculation uses a five-percentage-point minimum, five comparisons, and the observed layout variance. Under an independence assumption and a zero sham floor, the 192 reserved layouts would resolve an RL-over-imitation gain of roughly 11.2 percentage points at 80% power. Resolving the observed 9.7-point gain under the same assumptions would take about 328 independent layouts. The observed causal contrast is large enough for that provisional power calculation, with 192 layouts resolving roughly 12.6 percentage points. Its 95th-percentile sham bound remains uncalibrated with 24 development layouts. Even a maximum-based 95% bound at that quantile requires at least 59 independent sham units. Shared branch templates also leave layout independence unverified. Both comparisons remain development findings, and the reserved layouts remain unscored.

The per-seed summaries and native witness and intervention traces are in `results/ppt_hybrid_rl/branch-recovery-*`. The combined summary records the five-comparison preflight, layout-level effects, and trace file hashes.

## Reproduction

```text
python -m configs.freeze_ppt_branch_recovery results/ppt_hybrid_rl/root-adaptation-cohort.json COHORT.json
python -m configs.run_ppt_branch_recovery COHORT.json MATHCAN-TRAIN.zip SEED-151-RUNS RUNS
python -m configs.probe_ppt_branch_recovery COHORT.json RUNS/hybrid_recovery_rl.sqlite recovery-probe.json.gz
python -m configs.analyze_ppt_branch_recovery COHORT.json RUNS SEED-151-RUNS recovery-probe.json.gz summary.json evidence.json.gz
```

For seeds 103 and 197, supply the corresponding initialization directory and append `--seed-study results/ppt_hybrid_rl/branch-recovery-seed-study.json --seed SEED` to the training and audit commands. Use distinct output directories for each seed.
