# LOGRANK-BPD-13: costed acquisition with basis induction and end-to-end decoding

Development only. Protocol frozen before scoring. 12 training worlds; 24 disjoint small evaluation worlds (r=6/8, d=9/11), eight additional scale worlds (r=21/45, d=24/48). Uniform hidden physical sign bits, random invertible binary frame, known independent noise .15/.35. Initial independent observations followed by sparse quotient checks. Public induced frame uses observed rows then candidates ordered by cost and index; no planted map or target answers used in selection. Initial generation order exposes the old independent directions; this is a useful structured prior. Planted BP is an oracle-basis diagnostic. All policies receive a cost cap of 24 units beyond 4r initial observations, with identical candidate costs (1..3) and shared call-ordinal noise draws. Candidate pool may overlap initial observations; no candidate acquired twice or direct target observed. The 13-feature ridge policy learned exact gain/cost from 28,215 correlated candidate examples across training worlds, not independent examples. Its learned acquisition is a linear regressor, not a native TRM/BPD model.

## Reproduce

```sh
cd experiments/logrank_bpd_13
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py train
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py eval
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py scale
OPENBLAS_NUM_THREADS=1 python audit.py
OPENBLAS_NUM_THREADS=1 python gauge.py
```

NumPy only. BP uses the same damped 40-iteration implementation as 12. Factorized query moments are products of approximate marginal means, set to zero outside the exact observation span. They are not certified posterior probabilities. Exact scoring reference computed separately on every small-state trajectory. Cost-aware policies maximize one-step Brier gain per unit cost. All stop arms halt once >=80% weighted target mass has nominal confidence >=90%, or when no probe is affordable; fixed arms exhaust affordable probes. All target predictions, including abstained ones, contribute to Brier and forced-answer accuracy. Confidence coverage and weighted accuracy among covered targets are separate. Total cost excludes initial observations and local computation; timing is descriptive.

## Small evaluation: 24 worlds per noise rate

| Policy | .15 Brier | .35 Brier | .15 mean cost | .35 mean cost |
|---|---:|---:|---:|---:|
| exact_stop | .02988 | .21439 | 6.792 | 23.958 |
| exact_fixed | .00046 | .21439 | 24 | 23.958 |
| raw_bp | .23954 | .24990 | 24 | 24 |
| induced_bp | .01885 | .24115 | 7.250 | 24 |
| induced_fixed | .01749 | .24115 | 24 | 24 |
| planted_bp | .01858 | .23529 | 7.583 | 24 |
| random_induced | .07486 | .23937 | 17.792 | 24 |
| learned_induced | .02152 | .23663 | 7.500 | 23.750 |

At .15 induced BP achieves weighted accuracy .9861 with mean cost 7.25; learned acquisition .9792 with cost 7.5. All 24 stop at confidence in both arms. Exact-fixed has 100% forced accuracy and mean Brier .00046; induced-fixed .9674 and .01749, despite its same-trajectory exact posterior Brier .00038. Thus more calls alone do not repair approximate decoding. Realized early-stop loss below exact-stop loss is a descriptive sample comparison, not a contradiction of Bayesian optimality conditional on the same observations. At .35 no confidence stops; exact acquisition/decoder leads. Approximate confidence is miscalibrated: induced BP's covered-weight accuracy .8030 at .35. Neither learned policy nor one-step approximate acquisition establishes superiority.

## Same exact-fixed trace decoder comparison

| Basis | .15 Brier | .35 Brier |
|---|---:|---:|
| raw physical | .22929 | .25008 |
| public induced | .01428 | .23709 |
| planted oracle | .00368 | .23720 |

Same observations, same BP message rule, different frames. Exact posterior is basis invariant under invertible GF(2) transforms with a uniform state prior and consistently mapped masks. BP is not invariant. If every parity factor has degree >=2, all-zero sum-product initialization is a fixed point: each factor-to-variable message contains at least one zero incoming tanh factor, and the variable-to-factor sum stays zero. Combining repeated observations does not remove that obstruction. All 48 raw trajectories without singleton factors had uniform predictions; 58/64 raw trajectories were uniform overall. Public induction transforms initial independent observation masks to singleton coordinates without changing the true inference problem. Greedy ordered independence is GL-equivariant: a linear bijection preserves span membership, selected row order and the unique coefficient vector of each mask. gauge.py passed 131,540 coefficient equalities over 128 invertible transformations of 32 worlds. This is a finite check of frame equivariance, not a claim about arbitrary neural BPD gauges.

## Scale diagnostic: four worlds per width

| d | noise | induced BP Brier | forced accuracy | coverage | covered-weight accuracy |
|---:|---:|---:|---:|---:|---:|
|24|.15|.06301|.9375|.9583|.9348|
|48|.15|.31686|.5859|.6875|.6515|
|24|.35|.24999|.4375|0|n/a|
|48|.35|.25000|.5182|0|n/a|

At d48, p=.15, 3/4 induced runs stop at nominal confidence yet aggregate covered-weight accuracy is only 65.15%. Planted BP covered accuracy 83.98%, random induced 74.53%. Do not deploy the confidence threshold as a calibrated certificate. No exact posterior at large widths. The initial observation budget scales with rank but extra budget remains fixed; sparse quotient checks also have increasing combinatorial support, especially targets. This is not a matched constant-cost-per-rank scaling law. Four roots per width, no statistical generalization or native causal claim.

## Audit

PASS, 448 trajectories, 5,669 budget/noise/rank steps, 702 random choices; 12,288 exact same-trace reference probabilities and 3,072 duplicate exact-final probabilities, 11,264 final BP and 4,608 same-trace BP probabilities independently replayed. Max exact delta 4.44e-16 and BP delta 9.44e-16. Independent audit uses integer likelihood weights in physical coordinates and a separate edge-list BP implementation. No ridge refit, nonrandom acquisition-choice replay, or independent intermediate stop-probability replay (recorded stop coverage checked); no CI or confirmation. See summary_receipt.json for aggregate outcomes and hashes; the full root-level summary is in the experiment bundle. Full raw receipts and all scripts in downloadable experiment bundle.

## Next

Freeze a new development cohort; compare BP confidence against repeated frame ensemble, residual likelihood repair, and exact small conditional subproblems, including cost of additional verifier calls. Use appropriate proper scoring, coverage, held-out root calibration and cost; at high noise abstention may dominate forced answers. Match initial redundancy and extra cost/rank across scale. No protected cohort opened, no native BPD training or causal intervention. Apply EvalMath preflight with real independent pilot effects and matched shams before confirmation.
