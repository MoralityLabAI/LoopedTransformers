# LS16 evidence plan

## Question and independent unit

The registered comparison asks whether retaining three distinct codebooks raises the yield of exact-real certificates when both methods receive the same search opportunities. One independently perturbed initialization is one unit. The cohort contains 24 fresh seeds, 493 through 516, at the inherited 15-percent perturbation dose. The 32 transition cases and all rollout positions remain measurements within each seed.

The candidate arm is `beam`; the control arm is `greedy`. There is one planned hypothesis comparison. Numerical bounds, rollout accuracy, ancestry, unique codebooks, and time are descriptive diagnostics. They will not create additional inferential claims. Earlier studies informed this design; their seeds are development data.

## Outcome and gates

The primary binary outcome is an initial `POINT_VIOLATION` followed by a final `CERTIFIED` result. Every assigned seed remains in the denominator. An initially `UNRESOLVED` seed contributes zero primary repairs, with any later certificate acquisition reported separately. Existing certificates and lost certificates are recorded separately. Search uses numerical scores; exact certificate checks follow completion of every search.

The operational gate requires at least 12 new certificates among the 24 beam runs and at least six more than greedy. The scientific effect floor is a 0.25 increase in the probability of a new certificate under the fixed perturbation generator. The exact paired McNemar test, marginal Clopper-Pearson intervals, descriptive cluster bootstrap, and conservative Hoeffding effect bounds come from the archived official EvalMath FSI module. The family has one comparison at alpha 0.05.

At 24 independent seeds the Hoeffding radius is approximately 0.4996442296. A lower bound above 0.25 needs at least 18 net repairs, together with the other FSI gates. The operational count gate and the confirmation gates serve different purposes. A failed gate does not establish equivalence.

## Search control

Both methods share one initial fit and one bootstrap expansion. Each subsequent round allocates three parent expansions. Greedy updates its incumbent between those expansions. Beam freezes its three parents at the start of the round, expands each, then keeps the best three distinct codebooks. Every expansion uses both inherited scale anchors, 177 profile slots per anchor, and three fitted proposals per anchor, including the no-op. Each shortlisted proposal receives the inherited least-squares fit and one conditional bound LP.

The budget charged to each arm is 37 expansions, 223 least-squares calls including the common initial fit, 222 LP calls, and 13,098 profile slots. Shared computations are counted once in actual execution totals. Equal opportunities do not imply equal function evaluations or elapsed time. Codebook deduplication suppresses alternate scale basins for the same code tuple. Beam retention and breadth allocation form the treatment being tested.

## Pre-data resolution

The archived LS15 `bound_lp` and `max_lp` arms each repaired 11 of 24 seeds, with identical seed outcomes. These 24 paired effects are zero. They provide historical context for the search redesign. There is no measured development effect for LS16 beam search and no matched beam sham distribution.

The official preflight schema requires a sham field. Each supplied zero is a computed identity difference between a historical `max_lp` endpoint and itself. This identity check carries no independent null information. The external floor of one is the universal support bound on an absolute paired binary difference. It prevents those zeros from becoming a fabricated null calibration. The reported `upper_sham_quantile=1` therefore denotes this supplied bound, not an empirical quantile. Zero historical variance cannot establish power for a hypothetical positive beam effect.

The strict preflight exits with status 2 and `REDESIGN_OR_PILOT_ONLY`. Its reasons include a target below the supplied variation bound, zero historical signal, uninformative zero variance, and unavailable confirmation power. LS16 remains a bounded pilot for algorithm development regardless of its operational count result. A future confirmation needs an independent beam development contrast, a suitable matched control calibration, and prospective resolution planning on fresh units.

## Selection exposure and scope

The source, registration, audit code, and protocol are frozen locally and committed remotely before seeds 493 through 516 are generated. All 32 transition targets are known to the search. Freshness applies to the perturbation seeds. This experiment does not test unseen task families. The certificate concerns the ideal real-valued GELU recurrence with stored decoded weights, state-region radius 1/8, transition error at most 3/32, and disturbance budget 1/32. Floating-point rollout checks have their own numerical scope.

The paired test assumes independent draws from the declared perturbation generator and the equal-marginal paired-label null. Those assumptions depend on experimental provenance; the statistics code cannot prove them. Positive template weights supply a structural certificate control. They do not supply an independent sham sample. The resulting evidence class is `PILOT_ONLY` for comparative search yield, with exact certificates for the specific passing weights.

## Replay

Run from the LS16 directory:

```bash
python evidence/preflight.py --pilot evidence/preflight_input.json --out evidence/preflight_results.json --strict
python -m unittest discover -s evidence -p test_stats16.py
python evidence/stats16.py --certificates certificates.json --outdir evidence/results
```

The expected strict preflight status is 2. The receipt tests use synthetic verdict labels and never generate the fresh perturbation cohort. The final reporting step validates full seed/arm coverage and writes all paired outcomes with source and registration hashes. Official source provenance and the historical certificate hash are recorded in `source_provenance.json`.
