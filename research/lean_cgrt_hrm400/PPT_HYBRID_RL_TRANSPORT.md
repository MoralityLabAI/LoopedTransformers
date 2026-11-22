# Residual fact transport development protocol

The shared-graph pilot exposed the distant fact to both architectures. The hybrid's readout changed by less than a millionth of a logit point at a typical margin of 0.351. The next candidate therefore adds eight fact channels with an additive recurrence over the public premise relation. It reinjects the same trainable local fact representation at each hop. A channelwise retention gate and mean incoming implications control the delivered message. The candidate readout starts at zero and is fitted from actor-critic proof returns. Mean pooling can transmit a partial premise and supplies neither a Boolean proof decision nor a verified label.

This development contrast reuses the fixed 12-pair graph cohort and seed 103. Compare the residual-transport hybrid with its frozen-backbone readout control, the graph hybrid with a GRU bank, full graph TRM, and graph TRM with frozen backbone. Keep the same archive, training-root order, 32 native-update batches, eight MathCan replay batches, eight-check budget and per-proof resource caps. The primary diagnostic is how many pairs are certified in both worlds. A second readout records whether switching the only distant fact reverses the first choice. Improvement over the original controls is insufficient on its own to show that training the recurrent body matters.

The intervention removes the transported candidate contribution and compares it with an equal-norm common shift to those same logits. A constant shift is a negative control for a ranking policy. Altered proof paths and separately compiled native Lean certificates provide the outcome. The development cohort is reused and the study has one model seed. A causal confirmation requires new independent graph pairs, a preregistered meaningful gain of at least five percentage points, and EvalMath FSI-01. For a strong mechanism claim, the planned sample must also resolve the matched-sham floor before opening the independent readout.

For this development run add `--residual-fact-transport` to the hybrid trial with `--graph-context --seed 103 --updates 32`. Add `--head-only` to freeze its recurrent body and fact-channel transport. Native Lean and archived MathCan verification keep the same boundary as the original graph pilot.

## Paired reinforcement credit

The frozen-backbone transport arm certified 19 worlds, including both worlds in seven pairs. Cutting its transport readout reduced it to 12, while the common-shift control stayed at 19. Full-model reinforcement learning stayed at 12. The original standard-training branched confirmation remains unscored under its activation gate.

The follow-up fits both fact-switch worlds in one batch. A uniformly sampled root action is shared and executed separately in each native Lean session. Ordinary paired RL importance-weights its root actor gradient. The contrastive actor instead weights the difference in the chosen action's probabilities by the difference in certified returns. This cancels a shared lexical logit preference when the two inputs reach identical logits. BPD-LG-02 and MathCan replay retain their original actor-critic update.

Compare `--alias-training contrast` on the residual hybrid, the same hybrid with `--head-only`, and full graph TRM. Also run the full residual hybrid with `--alias-training paired` to compare credit objectives under the same proposal law and matched batches. Use the same development pairs, seed, archive, proof budgets and number of updates. The reward combines verifier outcomes with the existing fixed tactic-check cost. This comparison uses the known development cohort.

## Development results

All arms started at 12 of 24 certified worlds, with one successful world per pair. Each counted proof was replayed and compiled as a standalone, axiom-free Lean 4.19 theorem.

| Training on the shared graph | Certified worlds of 24 | Both worlds certified, of 12 pairs |
| --- | ---: | ---: |
| Full residual hybrid, standard RL | 12 | 0 |
| Frozen residual hybrid, standard RL | 19 | 7 |
| Full graph TRM, standard RL | 12 | 0 |
| Frozen graph TRM, standard RL | 12 | 0 |
| Full residual hybrid, ordinary paired RL | 15 | 3 |
| Full residual hybrid, paired credit | 17 | 5 |
| Frozen residual hybrid, paired credit | 18 | 6 |
| Full graph TRM, paired credit | 12 | 0 |

The full hybrid has 1,663,258 parameters. Paired credit trained it to switch its first tactic in five of the twelve pairs. Removing the transport contribution from this checkpoint changed five proof paths and brought the score to 12. The equal-norm common shift preserved all 24 paths and the 17 successes. This establishes dependence on the transport readout for these development trajectories. The frozen-backbone hybrid reached 18 with paired credit and 19 with standard RL, so the experiments leave the value of training the recurrent body unresolved.

Each paired arm used the same roots and verified 32 shared root proposals within pairs. Across arms, the same uniform proposal law yielded different sampled actions as the rollout RNG advanced. The training schedule included LoopedTransformers Horn pairs, BPD-LG-02 exact rows with EvalMath generator provenance, and eight replay batches from the original MathCan train archive. Archived MathCan closures contributed replay reward. The native Lean proof counts in the table come from the Horn development pairs.

Both campaigns used PyTorch CPU with two CPU threads per arm, a 2 GiB per-arm cap and an enforced 6 GiB cap over the concurrent process tree. The standard and paired campaigns peaked at 2.23 and 2.31 GiB committed, respectively. Evidence covers one seed on twelve reused development pairs. The separate 80-pair branched registry was fixed before the standard pilot readout. Its original activation gate rejected the standard checkpoints, so the branched registry remains unscored.

## Reproduce the workflow

Run these commands from this directory with PyTorch CPU, the Lean 4.19 REPL and the audited MathCan train archive configured. Keep the proof and training budgets from the first command for each arm.

```text
python -m ppt_gym.hybrid_rl run results/ppt_hybrid_rl/graph-alias-pilot-cohort.json RUNS/103-contrast.sqlite --architecture hybrid --graph-context --residual-fact-transport --alias-training contrast --seed 103 --updates 32 --ram 2 --mathcan-archive ARCHIVE.zip
python -m configs.analyze_ppt_transport_pilot results/ppt_hybrid_rl/graph-alias-pilot-cohort.json RUNS paired-pilot.json --paired
python -m configs.probe_ppt_hybrid_bank results/ppt_hybrid_rl/graph-alias-pilot-cohort.json RUNS/103-contrast.sqlite contrast-intervention.json --transport
```

| Session suffix after `103-` | Architecture and training flags in addition to `--graph-context` |
| --- | --- |
| `transport` | `--architecture hybrid --residual-fact-transport` |
| `transport-head` | `--architecture hybrid --residual-fact-transport --head-only` |
| `trm` | `--architecture trm` |
| `head-only` | `--architecture trm --head-only` |
| `contrast` | `--architecture hybrid --residual-fact-transport --alias-training contrast` |
| `ordinary` | `--architecture hybrid --residual-fact-transport --alias-training paired` |
| `trm-contrast` | `--architecture trm --alias-training contrast` |
| `head-contrast` | `--architecture hybrid --residual-fact-transport --alias-training contrast --head-only` |

Omit `--paired` in the analyzer for a standard-RL campaign. The experiment records are in `results/ppt_hybrid_rl/transport-summary.json` and `results/ppt_hybrid_rl/paired-transport-pilots.json.gz`.
