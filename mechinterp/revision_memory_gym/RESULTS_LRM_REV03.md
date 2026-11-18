# LRM-REV-03: exact continuation memory and Bayes reminder frontier

**Evidence:** exact finite mathematics under the LRM-REV-02 12-old/6-replacement law. All twelve independent unit tests passed locally, including exhaustive event likelihood checks. This is a public analytic baseline, with no newly trained model, no heldout confirmation and no neural mechanistic claim.

## Online memory lower bound

There are 4,096 possible named old streams and 924 future six-slot correction schedules. We computed the entire 4,096 × 924 table of resulting accepted-match counts using fixed all-zero replacements. **All 4,096 old strings have distinct continuation signatures.** Because accepted count uniquely determines the three-regime posterior, exact processing of every future correction schedule needs 12 unconstrained state bits. If the initial count remains a separately retained label, the largest fiber contains `C(12,6)=924` distinct states, requiring **10 additional fixed-width bits** within that fiber.

This is a finite online-repair constraint with a known stochastic law and a finite legal action family. A learned 32-dimensional real GRU has sufficient raw numerical dimension in principle; whether it learns stable per-position retrieval and overwrite is a separate empirical issue.

## Exact count-only posterior recovery

A count-only observer initially stores `sum(X)` and later receives correction positions S and accepted values Y. It can also receive truthful reminders of original X at named positions. Unlike the trained GRU, this observer never received the named original stream, so a reminder can convey previously discarded information.

The source enumerates the entire population with rational arithmetic, using the declared priors and Bernoulli laws and integrating the overlap between S and a uniformly sampled six-position reminder set. For every reminder stage, the endpoint is population-mean squared error relative to the **full named-history Bayes posterior** `p_H`.

| Number of old-value reminders | Corrected-position policy | Uniform random-position policy | Random minus corrected |
|---:|---:|---:|---:|
| 0 | 0.064462690845 | 0.064462690845 | 0.000000000000 |
| 1 | 0.058482963087 | 0.056911479999 | -0.001571483088 |
| 2 | 0.051316389909 | 0.049735970154 | -0.001580419755 |
| 3 | 0.042580145020 | 0.042916081997 | +0.000335936976 |
| 4 | 0.031688246641 | 0.036431238724 | +0.004742992083 |
| 5 | 0.017863266733 | 0.030263964967 | +0.012400698234 |
| **6** | **0.000000000000** | **0.024401568463** | **+0.024401568463** |

The final corrected-position oracle recovers the full-history posterior exactly because all original values at corrected positions are disclosed. A random six-position set generally leaves some corrected originals unknown. The random policy performs better after only one or two reminders, since observed unchanged bits are directly informative about the regime. This early crossover comes from the information law, rather than a learned mechanism.

The value **0.02440156846261376** is an exact-population expectation expressed as a decimal. It is a theoretical information gap for the count observer; its exact rational digest and every stage's digest are stored in [the deterministic receipt](results/lrm_rev03_exact_v1.json). The existing LRM-REV-02 four-GRU average, **+0.008553539884**, is a development observation from a different information interface. The GRUs initially received every named old bit, making truthful repetitions redundant in the ideal online representation. That difference in observability matters when interpreting the effect.

## A precise teacher for the next learned-state experiment

The helper `partial_named_posterior` returns the correct target after each of the six replacement events. It integrates all still-hidden future correction positions under the finite uniform schedule distribution. With no revealed replacements it gives the initial count-marginalized posterior; after all six it gives `p_H`. Independent brute-force enumeration over remaining schedules matches it at every intermediate stage.

The proposed LRM-REV-04 should fit the same GRU using this **stepwise correction supervision**, with the original three-checkpoint objective as a control trained from new source-disjoint model roots. Additional supervision changes the training objective and may carry more gradient computation; record and match updates, source exposure and measured CPU/GPU cost. The expected improvement is in post-correction competence against full-history Bayes. Preserve the reminder-policy contrast as a secondary diagnostic and score native hidden-state interventions only after matched-sham preflight. Use an explicit addressed twelve-slot overwrite memory as a declared algorithmic positive control rather than evidence of emergent recurrence.

## Verification

Run from the repository root:

```bash
python -m unittest discover -s mechinterp/revision_memory_gym -p 'test_lrm_rev03_exact.py' -v
python mechinterp/revision_memory_gym/benchmark_lrm_rev03.py --check
```

The suite verifies every six-slot continuation signature, the maximal count fiber, 32 arbitrary mixed-disclosure exact likelihoods against a separate exhaustive old-vector enumerator, 42 partial-correction teacher comparisons against direct future-schedule enumeration, terminal Bayes equality and exact population-risk stages. An isolated CPU workflow reruns the receipt bytewise. This experiment adds no training and consumes no GPU memory.
