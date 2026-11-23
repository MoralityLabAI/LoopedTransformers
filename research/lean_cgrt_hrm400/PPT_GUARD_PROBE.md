# Learning which Lean guard to probe

In the previous return-trained pilot, removing learned commitment corrections retained a four-proof gain at seven calls. Swapping the learned check scores removed that gain. This comparison restricts learning to the identity of a guard probe. The frozen planner chooses when to probe, which witness to enter, and which tactic to use after entering. Six-call attempts cannot probe. In eight-call attempts there can be a second decision after a true probe; all arms receive the actual paid observation.

## Registration

Use the 48 original Lean-verified source root pairs and four-fold out-of-fold base reader forecasts. Give the existing two-witness recurrent representation the remaining budget and the paid guard bits in each slot. The readout adjusts the log weights of the four possible false guards by at most ±0.75. The planning layer eliminates outcomes incompatible with observations and translates the adjusted weights to per-probe utilities using the exact frozen continuation. Each score is anchored at its original value when the correction is zero.

For the reward-trained arm, maximize exact stochastic policy return across the remaining probe decisions at budgets seven and eight. All other actions follow the fixed greedy planner, so the policy cannot save calls by altering nonprobe choices at the same state. Training uses the native-verified source outcome only for transition and reward accounting, with temperature .035 for probe selection. The matched supervised twin predicts the native-checked false guard among the hypotheses consistent with observed bits. Compare TRM with the return loss and affine readouts fitted with each loss. All models share the input features, source sampler, scoring layer, and 480 batches of 16 source worlds with AdamW at .002. Matching recurrent and affine twins start from identical weights. Greedy evaluation uses the frozen planner for all nonprobe actions.

The primary development endpoint is equal-root utility at the two budgets that allow probing, with both paired worlds in each of the 40 exposed roots. Report the six-call invariant and all three budgets. Require +.025 utility versus the matched controls, the unchanged frozen planner, the earlier checks-only actor, and the strongest archived supervised belief correction at the same budgets. Retain `FSI-01 REDESIGN_OR_PILOT_ONLY` because the same roots informed the readout and the sham has no independent prospective envelope.

At the first differing probe choice with four feasible probes, swap the two log-weight corrections assigned to the frozen and learned picks. A matched intervention applies the identical signed shifts to the other two guards. Both preserve the prior, the total learned correction, the L2 dose, the paid prefix, and the underlying Lean outcomes. Restore the learned correction after that decision. Compare proof changes at the same eligible decisions and independently compile every distinct successful proof.

## Development results

| Policy | Six calls / 80 | Seven calls / 80 | Eight calls / 80 | Proofs / 240 | Calls | Seven- and eight-call root utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen supervised planner | 44 | 64 | 80 | 188 | 1,572 | 0.720938 |
| Affine supervised | 44 | 58 | 80 | 182 | 1,558 | 0.685625 |
| Affine policy return | 44 | 60 | 80 | 184 | 1,560 | 0.697813 |
| TRM policy return | 44 | 62 | 80 | 186 | 1,574 | 0.708125 |
| Two-slot policy return | 44 | 62 | 80 | 186 | 1,562 | 0.710000 |
| Two-slot supervised | 44 | 72 | 80 | 196 | 1,580 | 0.769688 |
| Two-slot return, chosen-probe swap | 44 | 68 | 80 | 192 | 1,580 | 0.744688 |
| Two-slot return, bystander swap | 44 | 60 | 80 | 184 | 1,552 | 0.699063 |
| Archived checks-only return | 44 | 68 | 80 | 192 | 1,580 | 0.744063 |
| Archived supervised belief | 40 | 72 | 80 | 192 | 1,570 | 0.769688 |

All newly fitted arms preserved the same six-call traces. The return-trained two-slot model saved 12 calls at the same proof count as TRM, gaining 0.001875 utility across the two probe budgets. It trailed the frozen planner by 0.010938 and its supervised twin by 0.059688. The supervised probe restriction retained the archived supervised model's seven- and eight-call counts and recovered four six-call proofs by reverting to the frozen commitment rule. Its seven-call gains over frozen were six BPD recurrence proofs and two MathCan mask proofs. The return fit matched the frozen BPD proof count and lost two MathCan proofs.

The 74 eligible BPD interventions shared their paid prefixes and mean L2 dose of 1.462 log-weight units. The chosen-probe swap changed 52 of those actions, gaining six proofs and losing zero. The bystander swap changed 34 actions, gaining five and losing seven. The proof changes all occurred at seven calls. The result places a source-trained correction at a consequential decision, with an observed effect in the opposite direction from the intended policy gain. This was a reused development cohort.

A follow-up readout held the stochastic selection temperature fixed across source and development. At seven calls, the two-slot reward fit fell from 0.80200 source utility to 0.59181 development utility. TRM fell from 0.73640 to 0.60254, and the supervised twin fell from 0.79422 to 0.68737. The reversal therefore also occurs under the training-time selection rule. The source and development guard depths differ, so a deeper source curriculum or a reward update anchored to the supervised fit would test separate explanations for the transfer loss.

The fit sealed to SHA-256 `0fda12df6ab2e46414c40b990cd992c9b859bd497537f6903825a3b00ea5486c` and used six reachable states with a probe choice. Adding the stochastic diagnostic reproduced all greedy traces. Lean independently compiled 138 distinct successful scripts and replayed ten selected successes and failures, including both BPD interventions. Logical CPU 2 handled training and scoring under the 2 GiB process-tree cap. Final evaluation peaked at 0.53 GiB committed. All 119 native-enabled gym tests passed. [Numerical receipt](results/ppt_hybrid_rl/ppt-guard-probe-final-1125.json.gz)

```text
python -m configs.run_ppt_guard_probe fit BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS
python -m configs.run_ppt_guard_probe eval BANK BPD MATHCAN WARM PRIOR PLANNER FIT RESULTS --belief-control ARCHIVED_BELIEF --return-control ARCHIVED_RETURN
```
