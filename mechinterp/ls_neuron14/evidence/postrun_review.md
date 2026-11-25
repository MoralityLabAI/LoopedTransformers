# LS-NEURON-14 independent statistical review

## Result

The reported counts, exact paired tests, uncertainty output, and study labels agree with the raw certificate rows. Both prespecified operational gates failed. The FSI verdict is `inconclusive` for each comparison, and the overall evidence class remains `PILOT_ONLY`.

| Comparison | Bound repairs | Point repairs | Bound-only wins | Point-only wins | Exact two-sided p | Bonferroni p | Paired difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single moves, primary | 4/24 | 7/24 | 0 | 3 | 1/4 | 1/2 | -1/8 |
| Paired moves, secondary | 9/24 | 9/24 | 2 | 2 | 1 | 1 | 0 |

The primary point-only successes are seeds 447, 452, and 459. For paired moves, bound-only successes are seeds 459 and 463; point-only successes are seeds 452 and 462. The paired arms therefore certify different sets of seeds despite equal totals.

All 24 unique initial states have exact point-violation witnesses. The 96 final arm/seed endpoints comprise 29 certificates and 67 point violations. No initial or final check is unresolved, so the stricter repair count and the broader certificate-acquisition count coincide in this cohort. The effective replication count remains 24 initialization draws. The 29 certified endpoints contain 20 distinct decoded-weight hashes. Eleven seeds receive a certificate in at least one arm; that pooled count is descriptive and is outside the registered single-method comparisons.

## Independent checks

I recomputed paired outcomes directly from `certificates.json`, using integer binomial sums and rational fractions for the exact tests. I checked both outcome JSONL files against those outcomes and verified their recorded SHA-256 values. The certificate digest agrees with the audit and statistical report:

`767519e2c70424e4d20b72cb8aa62fea279305894ce5b4bc39ba6ab8a815a60c`

The marginal Clopper-Pearson endpoints were checked against an independent beta-quantile implementation. Every stored lower bound lies below its corresponding beta quantile and every stored upper bound lies above it, with each outward discrepancy at most `2^-18`.

| Arm | Repair rate | Stored 97.5 percent marginal Clopper-Pearson interval |
| --- | ---: | ---: |
| Point, single | 7/24 | [0.11016846, 0.53956604] |
| Bound, single | 4/24 | [0.03841400, 0.40295792] |
| Point, paired | 9/24 | [0.16825485, 0.62137985] |
| Bound, paired | 9/24 | [0.16825485, 0.62137985] |

The report's `alpha` is 0.05. Its field named `familywise_alpha` contains 0.025, which is the allocated alpha after dividing by the two planned comparisons. The marginal intervals thus have 97.5 percent nominal coverage under the declared iid seed-generator assumptions. These four marginal intervals do not jointly establish 95 percent simultaneous coverage across all four rates.

The stored paired bootstrap intervals replayed exactly with 4,096 resamples and seed 20261125. Their endpoints are [-0.29166667, 0] for single moves and [-0.20052083, 0.16666667] for paired moves. These are approximate percentile intervals at the allocated 97.5 percent level. Their asymmetry can arise from resampling and quantile interpolation. Their small-sample behavior does not override the exact paired tests or the prespecified effect-floor gate.

The Hoeffding radius equals `sqrt(2 log(40)/24) = 0.5544426220774891`. The reported bounds therefore match the observed effects: [-0.67944262, 0.42944262] for single moves and [-0.55444262, 0.55444262] for paired moves. Each displayed side comes from a one-sided bound at the allocated level. For the positive-floor claim, simultaneous control over the two lower bounds is the relevant familywise guarantee.

## Interpretation and reporting limits

The primary cohort shows three fewer certified repairs under bound-based selection, a measured difference of -12.5 percentage points. The paired test supplies insufficient evidence to infer population harm or improvement from this cohort. Equal secondary totals leave the population difference uncertain, with different seeds succeeding in the two arms. An equivalence claim would require a prespecified equivalence margin and a sufficiently narrow interval.

The required effect floor is +25 percentage points, and each operational gate requires at least 12 candidate repairs together with an advantage of six. The primary candidate produced four repairs; the secondary produced nine. Both failed their absolute and relative count requirements.

The evidence concerns this fixed perturbation generator, mod4 template, and search budget. Floating numerical ranking scores remain proxies; the successful endpoints receive their exact-real invariant claim from the rational verifier. The audit checks recorded candidate states and complete profile ranking while preserving the stated limitation that SciPy optimization trajectories were not independently retrained.

No correction to the numeric results or current statistical labels is needed. If confidence intervals appear in the report, retain their 97.5 percent marginal label and the assumptions stated here. Count the 29 successes as certified final arm/seed endpoints. A later comparison obtained by selecting or pooling successful seeds across methods would have a different endpoint and search budget.

No locked file was changed during this review.
