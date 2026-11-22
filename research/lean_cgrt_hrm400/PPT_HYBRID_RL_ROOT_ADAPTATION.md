# First-action routing and training stability

## What transferred across seeds

The full interaction model trained from random initialization showed a seed-specific advantage. This replication kept the original training sources, six matched arms and 24 development pairs. Its two new seeds reuse those layouts, so they measure checkpoint sensitivity.

| Training seed | Hybrid RL | Frozen-backbone hybrid | Same hybrid, verified-path imitation | Graph TRM | Frozen graph TRM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 103 | 34 | 24 | 23 | 7 | 8 |
| 151 | 7 | 23 | 27 | 10 | 8 |
| 197 | 18 | 24 | 28 | 6 | 6 |

Counts are independently certified Lean proofs out of 48 per seed. At seeds 151 and 197, full RL chose a feasible first tactic in 28 worlds each but failed after that choice in 21 and 10 worlds respectively. The same imitation architecture failed after a feasible first tactic in two and zero worlds. The mean-only hybrid reached 24, 7 and 13 proof closures. Its results show that removal of the interaction alone does little to address the continuation failures.

## What the learned readout can do

For seed 103 on the 96 reused asymmetric calibration pairs, the learned interaction ranked the feasible first action in all 192 worlds for both RL and imitation. For the RL model's 67 original first-action errors, the remaining logit components overrode the correct interaction ranking. With the verifier, fourfold multiplication of this signal and then use of the signal alone produced the following diagnostic results. Each change was applied at the first decision in both full checkpoints. Later decisions used the original policies.

| First-decision readout | RL proofs out of 192 | Imitation proofs out of 192 | RL pairs solved twice | Imitation pairs solved twice |
| --- | ---: | ---: | ---: | ---: |
| Original | 122 | 91 | 26 | 10 |
| Fourfold interaction | 176 | 123 | 80 | 37 |
| Interaction alone | 187 | 169 | 91 | 76 |

At the interaction-only setting, each policy started with the feasible tactic in all 192 worlds. The remaining difference in proof closures came from continuation search. This is a post-readout intervention on reused layouts with one seed. It measures the effect of mechanism-specific calibration on these checkpoints. It leaves the registered 480-pair confirmation set and the original branched and ARG-03-N holdouts unscored.

## Continue from imitation

The warm-start study loads the completed, full imitation checkpoint from each of the three seeds. The candidate clones its recurrent fact transport for the first decision and trains that copy with paired verifier rewards. The original proof workspace, gated memory and continuation transport stay fixed. A readout-calibration arm updates only the copied action projection and a bounded interaction multiplier. Further verified-path imitation trains the same first-action architecture. Whole-model RL starts from the identical imitation weights.

The common training mix includes the original LoopedTransformers Horn layouts, the BPD-LG-02 exact rows, six training declarations from EvalMath's manifest-verified BPD-LG-01-v2 fixture and the audited MathCan archive. The EvalMath importer passes only public propositions through to the learner, with binder names normalized to the existing tactic grammar. Lean accepts or rejects all executed native tactics, and successful native proofs compile as standalone results.

```text
python -m configs.freeze_ppt_root_adaptation results/ppt_hybrid_rl/conjunction-pilot-cohort.json EVALMATH/train.jsonl EVALMATH/manifest.json ROOT-COHORT.json
python -m configs.run_ppt_root_adaptation ROOT-COHORT.json MATHCAN-TRAIN.zip IMITATION-WORK RUNS
python -m configs.analyze_ppt_root_adaptation ROOT-COHORT.json RUNS IMITATION-WORK summary.json evidence.json.gz
```

| Seed | Starting imitation | First-action RL | First-action readout calibration | Further first-action imitation | Whole-model warm-start RL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 103 | 23 | 30 | 26 | 30 | 35 |
| 151 | 27 | 43 | 31 | 41 | 31 |
| 197 | 28 | 39 | 33 | 39 | 28 |

These are certified worlds out of 48 at each seed on the reused development layouts. First-action RL improves on the starting policy and on the readout calibration control for every seed. Verified-path imitation of the same first-action module is close, so the study leaves an RL-specific advantage over continued imitation unresolved. The adapted model adds 33 weights, all trainable in the first-action RL arm. The readout control trains nine. All arms try the same source theorems and identical paired root proposals within each seed.

On the primary endpoint, first-action RL certified both worlds in 10, 19 and 15 of the 24 pairs. The readout calibration control solved 6, 9 and 9 pairs twice. Continued imitation solved 10, 17 and 15 pairs twice. The audit confirmed bit-identical continuation weights and found that proof paths in the first-action arms changed only when the first tactic changed.

The twelve-arm CPU campaign averaged 9.16 logical processors' worth of CPU time on the 20-logical-processor host and peaked at 3.28 GiB committed under a 6 GiB whole-process-tree cap. Each worker used two PyTorch threads. All arms attempted twelve EvalMath native rollouts each and produced six standalone, axiom-free certificates each on those rollouts. MathCan closures remain reported separately as archived replay.
