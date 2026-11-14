# FC-04: adaptive LR-01 recurrence, first frozen CI result

**First successful test:** [GitHub Actions #43361328736](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43361328736),
13 November 2026. Both dependent jobs passed, after one earlier
training-only unit-test failure (Python verifier method-binding) was
corrected before the new test became accessible.

The [pre-registration](https://github.com/MoralityLabAI/Compactification-Theory/blob/ce7190129617d8387a7acf4ceed3e8d0570cfce4/research/refutation_bench/fc04/PREREGISTRATION.md)
was committed before any FC-04 test generation or inference. The
[source-certified FC-04 corpus](https://github.com/MoralityLabAI/Compactification-Theory/pull/107)
contains 384 new, independent finite Horn arguments: 192 positive Lean
theorems and 192 exact negative countermodels.

Training source SHA:
`7aca2bc3fc5586c5b7997fba868bd179db761121`

Held-out generator SHA:
`1b85412aa237d7522341c40d029654052e1d3267`

Canonical held-out JSONL SHA-256:
`08e57752f02c886e0947e36687c0572d84cfc815a467e1ea0c2554a1a044c04c`

Each 11,521-parameter LR-01 student trained on FC-02 train only.
The post-training validation grid selected θ=0.70 (seed17)
and θ=0.55 (seed29), without access to any FC-04 example.

## Verified refutation certificates and compute

| Model | Split | Adaptive certified /96 invalid | Fixed12 certified /96 | Adaptive mean passes | Uncertified positive errors | Abstained |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Seed17 | same-depth | 91 | 0 | 5.09 | 5 | 0 |
| Seed17 | deep | 95 | 0 | 7.67 | 1 | 3 |
| Seed29 | same-depth | 96 | 96 | 5.01 | 0 | 0 |
| Seed29 | deep | 96 | 96 | 8.45 | 0 | 37 |

The exact checker accepts only a full model-generated assignment for
which every premise is true and the asserted conclusion is false.
**All negative certificates passed that check**, with zero false
refutations of valid finite formulas. Positive calls are explicitly
`established_provisional` and remain unverified.

The [public long-form results receipt](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc04-adaptive/research/refutation_bench/results/fc04_adaptive_20261113.md)
also reports fixed4/fixed8 arms, calibration scores, selective error,
abstentions, throughput per 1,000 cell calls and measured CPU overhead.

## Decision boundary

All four registered adaptive-vs-fixed12 comparisons met the
predefined conditions: same or higher certificate recall, lower mean
recurrent calls, no false verified countermodels and no increased
positive-provisional error rate. This result is **conditional on the
fixed12 comparator**. Fixed4 on seed17 same-depth is more efficient
in certificates per call (117.19 vs 93.05 per 1,000 calls), though
it finds one fewer negative certificate. Fixed12 seed17 collapses
to unverified positives, an extreme baseline pathology; the
comparison should be understood in that light.

The policy implements true early stopping: each sample executes only
its recorded number of recurrent cell operations. The CI job records
model-forward and checker CPU milliseconds separately, and checks
checkpoint SHA-256 before test inference.

## Reproduce

With the pinned FC-02 source in `../Compactification-Theory-fc02`
and the FC-04 generator commit in `../Compactification-Theory-fc04`:

~~~sh
export FC02_ROOT="$(realpath ../Compactification-Theory-fc02)"
python -m unittest discover -s experiments/refutation_transfer \
  -p 'test_fc04_adaptive.py' -v
python experiments/refutation_transfer/fc04_adaptive.py train \
  --fc02-root "$FC02_ROOT" --out /tmp/fc04_weights

python experiments/refutation_transfer/fc04_adaptive.py eval \
  --fc04-root "$(realpath ../Compactification-Theory-fc04)" \
  --weights /tmp/fc04_weights --out /tmp/fc04_results
~~~

The two steps belong in isolated environments for the registered
access boundary: the training process must not see the FC-04
generator. The CI workflow enforces this with separate dependent
jobs and frozen artifact files. Its output artifact named
`fc04-selective-43361328736` includes `FC04_report.json` and 20
complete per-case JSONL prediction files (30-day retention).

These results are specific to bounded Horn reasoning. The exact
symbolic solver achieves 100% on these tasks, and a semantic
translation from real mathematical manuscripts is not yet included.
