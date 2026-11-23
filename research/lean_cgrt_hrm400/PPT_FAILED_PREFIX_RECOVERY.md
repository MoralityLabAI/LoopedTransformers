# Lean recovery with a retained conjunct

The previous guard acquisition study held the continuation planner fixed. Reward fine-tuning tied its supervised anchor on fresh expressions. This pilot puts the decision after a real Lean rejection, where continuing with a partially proved witness competes with restarting a different witness.

## Registered pilot

Generate 48 source and 40 distinct development theorem pairs from the original BPD and MathCan Lean definitions. Each member of a pair shares the same forced prefix and displayed continuation lengths. Rotating closed guards changes which continuations succeed. One or two of the three routes succeed for each world. Use the existing shallow source and deeper development depth ranges. The shared six paid tactics open an existential witness, discharge its first conjunct, and attempt a derivation of its second conjunct that Lean rejects. Save both the initial witness fork and the one-goal snapshot. Administrative snapshot restoration has no charge. A recovery action must execute a registered tactic fragment: retain the first conjunct and repair its second, or abandon the witness and restart one of two alternatives. Each rejected fragment spends all tactics through its failed check. A subsequent action can return to either original snapshot. Fragment costs to fail and to succeed can differ, and remaining affordability is public.

Give all models the same public typed guard descriptions, costs, budget and paid rejection history. The two-slot return model binds the retained witness and proposed continuation in separate slots. TRM receives the same tokens. Both optimize exact expected return of the stochastic finite recovery policy under native-checked source outcomes. Use matched two-slot controls trained on full-information action Q regression and direct native guard labels. For the latter, place the three frozen probabilities into a joint prior conditioned on the known support of one or two viable routes. An affine joint calibration control predicts that support directly. Both pass their prior to exact public planning. Include uniform-prior planning and an actual-truth oracle as structural references. The warm features are inherited from the four-source MathCan, EvalMath, LoopedTransformers and BPD prior.

Use 360 batches of 16 source worlds, shared order and matching initialization for the recurrent variants, AdamW at .002, and temperature .12 for stochastic return fitting. Freeze before greedy development scoring. Score both paired worlds with three, five and seven recovery calls after the common paid prefix. The endpoint is root-macro proof reward minus .025 per charged call, with a +.025 descriptive floor over the strongest non-oracle control. Independently compile completed native proofs and replay successful and failed attempts including histories with more than one route where available.

At the first recovery decision with at least two options, exchange the two input slot tokens and restore their normal order at the next decision. Preserve the actual saved prefix, affordable menu, and physical native outcomes. Treat this as slot-order sensitivity. The observational control is the unchanged model and there is no calibrated equal-dose sham envelope. The data and scope support `FSI-01 REDESIGN_OR_PILOT_ONLY`.

## Development results

| Policy | Proofs / 240 | Charged checks | Root-macro utility |
| --- | ---: | ---: | ---: |
| Two-slot policy return | 119 | 2,174 | 0.269375 |
| Two-slot Q regression | 130 | 2,200 | 0.312500 |
| TRM policy return | 128 | 2,207 | 0.303438 |
| Two-slot supervised guard planner | 134 | 2,154 | 0.333958 |
| Affine joint calibration and planning | 148 | 2,246 | 0.382708 |
| Uniform-prior planning | 145 | 2,217 | 0.373229 |
| Two-slot return with the role slots swapped | 121 | 2,177 | 0.277396 |
| Actual-truth planner | 179 | 2,119 | 0.525104 |

The reward-trained two-slot model missed the registered floor and trailed every fitted control. The strongest non-oracle policy estimated joint route truth with an affine head and used the actual saved-prefix cost structure for planning. Uniform-prior planning also outperformed both reward fits, so the ability to look ahead through affordability and rejection already explains much of the reachable gain. The actual-truth planner measures the remaining structural opportunity within the registered routes and budgets.

| Policy | Three recovery calls / 80 | Five / 80 | Seven / 80 | Proofs after an additional failed attempt |
| --- | ---: | ---: | ---: | ---: |
| Two-slot policy return | 19 | 44 | 56 | 15 |
| Two-slot Q regression | 15 | 48 | 67 | 25 |
| TRM policy return | 22 | 46 | 60 | 19 |
| Two-slot supervised guard planner | 18 | 53 | 63 | 18 |
| Affine joint calibration and planning | 25 | 53 | 70 | 28 |
| Uniform-prior planning | 23 | 52 | 70 | 37 |
| Two-slot return with the role slots swapped | 19 | 45 | 57 | 18 |
| Actual-truth planner | 26 | 73 | 80 | 0 |

The two-slot return policy stopped 17 times while a route remained affordable. Turning every such stop into a proof would raise it to 136 proofs, below the affine and uniform planners. The slot swap was eligible in 168 attempts and changed 49 first recovery choices. It gained seven proofs and lost five. The paid prefix and subsequent physical outcomes matched native replay. This sensitivity provides no evidence that the learned allocation of saved and candidate slot roles improved proof outcomes.

The native bank holds 286 independently compiled recovery proofs across source and development. The evaluation independently compiled 120 successful scripts and replayed eight selected successes and failures. The fit SHA-256 is `9e7a31e9036a163662b0e680bdc944aeda2524915e79204125ce75f012d0b4dc`. All experiments used logical CPU 2 with the 2 GiB process-tree cap. Evaluation peaked at 0.53 GiB committed. All 127 native-enabled gym tests passed. [Native recovery bank](results/ppt_hybrid_rl/ppt-failed-prefix-bank-1128.json.gz) · [Numerical receipt](results/ppt_hybrid_rl/ppt-failed-prefix-dev-1128.json.gz)

```text
python -m configs.freeze_ppt_failed_prefix BPD MATHCAN BANK
python -m configs.run_ppt_failed_prefix fit BANK BPD MATHCAN WARM PRIOR FIT RESULTS
python -m configs.run_ppt_failed_prefix eval BANK BPD MATHCAN WARM PRIOR FIT RESULTS
```
