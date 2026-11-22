# Paired RL for jointly supported Lean premises

The preceding pilot showed that full-model RL could use distant facts, while its best frozen-backbone control still solved more proofs. Linear fact transport also assigns identical support to two fully supplied premises on the same rule and two facts split across competing rules. This study tests whether learning a nonlinear recurrent update captures that difference.

Each pair has two root actions. Both branches offer the same number of rules and receive the same number of facts at every incidence depth. One branch has an entire rule supported by facts. In the other branch, those facts are split across two rules. The companion world swaps these roles. Atom names and hypothesis order are shuffled before exposing them to any model. The successful proof uses eight Lean tactic checks, and a wrong first tactic consumes the ability to finish it in budget. The development suite holds out shapes and includes eight pairs with four-premise rules. Training uses only two- and three-premise conjunction shapes, plus the existing BPD-LG-02 exact rows with EvalMath generator provenance and the audited MathCan archive.

The candidate keeps the Plan A workspace, the Plan B gated memory and the linear fact bank. In each graph pass it adds a residual for disagreement among the premises of an individual rule. The residual is initialized to zero, with one trainable signed coefficient per channel. It can learn to reward or penalize disagreement. Rule support uses the visible Lean state and both policies see that same state. Lean supplies the reward by running the proposed tactic and certifying the resulting proof. For paired batches both root actions are tried in both worlds, with identical proposals across arms. The root update compares paired returns. Continuation actions receive actor-critic credit conditioned on the proposed root.

The primary diagnostic is how often both worlds of a pair are independently certified. Each world counts toward certified accuracy as a secondary outcome. Compare against full and frozen graph TRM, the same hybrid with its recurrent body frozen, the full mean-only hybrid, and verified-path imitation using the same candidate architecture and source exposure. The imitation arm receives successful paths from native Lean and successful archived MathCan replays. The fallback for all RL arms on BPD and on MathCan is the existing actor-critic.

The study uses one seed and 24 newly generated development pairs. Its design and cohort are frozen before policy readout. Evidence is development-only, with the earlier branched cohort and the ARG-03-N holdout unscored. A positive development result would justify planning an independent study at a minimum meaningful gain of five percentage points and applying the EvalMath FSI-01 gate. The causal comparison is removal of the learned disagreement contribution against an equal-norm common shift of those same logits. A common shift preserves the current ranking and provides the matched intervention control.

Run the campaign from this directory with a CPU PyTorch environment and the pinned native Lean 4.19 build.

```text
python -m configs.run_ppt_conjunction_pilot results/ppt_hybrid_rl/conjunction-pilot-cohort.json MATHCAN-TRAIN.zip RUNS
```

## Development findings

| Model or training control | Certified Lean worlds out of 48 | Pairs solved in both worlds out of 24 |
| --- | ---: | ---: |
| Full interaction hybrid, RL | 34 | 10 |
| Frozen interaction hybrid, RL | 24 | 0 |
| Mean-only hybrid, RL | 24 | 0 |
| Full interaction hybrid, verified-path imitation | 23 | 3 |
| Full graph TRM, RL | 7 | 0 |
| Frozen graph TRM, RL | 8 | 0 |

Every arm started with 9 certified worlds and no pair solved in both worlds. The full model has 1,663,266 parameters, eight more than the mean-only hybrid. The frozen version trains 586 weights. All arms completed the same 40 available native batches under the registered 48-batch ceiling, with 24 conjunction batches and 16 BPD exact batches, plus ten archived MathCan batches. The ledger verified that both root actions were proposed in each training pair, in identical order across arms. Imitation updated only after a retained certified path, including archived successes, while RL could also learn from failures and tactic costs. The CPU-only process tree peaked at 3.35 GiB committed under the 6 GiB cap.

The native witness tried both root actions in each of the 48 development worlds. Exactly one could reach a standalone, axiom-free certificate in eight checks, and the successful first action switched within every pair. A nonzero test readout on the linear transport gave a maximum root action gap below one millionth of a logit. The learned model's first choice switched in ten pairs. Removing just the interaction contribution on the first decision changed ten proof paths and took the model from 34 to 24 certified worlds. The equal-norm common shift changed zero paths and preserved all 34 certificates.

On the eight development pairs with four-premise rules, the full RL model and verified-path imitation each solved both worlds in two pairs. The mean-only and frozen interaction hybrids solved no pair twice. The development evidence for the full model's advantage over imitation therefore comes from the two- and three-premise shapes in this study.

These findings come from 24 structurally distinct development pairs at seed 103. The intervention dose is removal of the entire learned first-action interaction contribution on each world. Across the five training controls and the matched intervention, differences are reported descriptively. Zero sham changes on 24 pairs establishes the observed development reference and leaves the population tail uncalibrated. Both the independent mechanism claim above a five-point effect floor and broader RL gains remain unconfirmed.

```text
python -m configs.check_ppt_conjunction_pilot results/ppt_hybrid_rl/conjunction-pilot-cohort.json RUNS/native-witness.json
python -m configs.probe_ppt_hybrid_bank results/ppt_hybrid_rl/conjunction-pilot-cohort.json RUNS/103-hybrid.sqlite RUNS/interaction-intervention.json --interaction
python -m configs.analyze_ppt_conjunction_pilot results/ppt_hybrid_rl/conjunction-pilot-cohort.json RUNS RUNS/native-witness.json RUNS/interaction-intervention.json summary.json evidence.json.gz
```
