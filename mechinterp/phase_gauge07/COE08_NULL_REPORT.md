# COE-08: truth-conditioned phase-history null audit

**Executed 2026-11-15.** Follows native COE-07/COE-07R, using the **same existing pretrained LoopedTransformers/LoopScope-02 output telemetry**. The underlying checkpoint was *not re-inferred* and the base model was *not retrained*. This experiment is a post-hoc falsification check of the `20/1795` phase-invariant-order-over-count observation.

## Why the new control matters

The earlier eight-step phase-invariant correctness history distinguishes 1,389 of the 1,795 different-target-digit pairs in `S4_stalled` while a count-only trace distinguishes 1,369. But each correctness bit is `1{prediction == target_digit}`, and thus requires knowledge of the answer. A label-blind predictor can produce rich, variable correctness traces. Any pair-separation statistic must first exceed the distinction rate expected from the **same trace histogram with its labels shuffled**.

## The registered source and the exact null

- Repository source: `mechinterp/loopscope02/results/benchmark_bundle.json`, Git blob `01ec63b4b2a4ae5399f783c6036a7fc018825a87`.
- Output trace: `mechinterp/loopscope02/results/trm_results.json`, Git blob `3c35daeb71da32f5c9ebbf03da539b8ee718b0ee`.
- Pinned pretrained model: `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att` `step_21700`, SHA-256 `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`.
- `N=64` stalled Sudoku boards, target-label multiplicities `[10,8,2,5,8,7,6,10,8]`, evaluator pairs across different digits `M=1795`.
- The null permutes **entire eight- or sixteen-step correctness histories**, keeping all temporal correlations and all label frequencies intact. Every permutation is independent of the original answer-to-trace association.

| History | Observed dihedral minus count | Exact null mean | Exact null variance | Null 95% empirical interval | Upper-tail p (10k permutations) |
|---|---:|---:|---:|---|---:|
| T=4 | 0 | 0 | 0 | [0,0] | 1.000 |
| T=8 | 20 | `1795/96 = 18.69792` | `11285015/5809152 = 1.94263` | [15,21] | 0.31477 |
| T=16 | 6 | `1795/252 = 7.12302` | `46069057/60043032 = 0.76727` | [5,8] | 0.95800 |

For `T=8` the exact standard deviation is 1.39378, putting the observed `20` about 0.934 SD above the permutation mean. Exact means/variances use rational arithmetic and **account for overlapping evaluator pairs**. Monte Carlo upper-tail probabilities use plus-one correction and fixed seed `20261115+T`, 10,000 independent shuffles of the fixed label multiset.

The eight-step increment is compatible with a label-independent reallocation of the actual trace codes. This removes its interpretation as positive evidence for extra **task-specific** phase coding. The measurement remains a valid descriptive fact about those 64 output histories.

## A mathematically stronger negative control

If a model independently guesses one of nine digits uniformly at random at every visit, then each truth-conditioned correctness bit is Bernoulli(1/9), independently of the true digit. For eight visits, **about 82.8%** of pairs of independent ordered correctness histories still differ. Thus history distinction is an unsafe surrogate for capability or label information, even with a high score.

The precise null identity and a dependent-pair second-moment theorem are proved in `patrickdugan/MathCan/research/compactification/OBSERVER_PERMUTATION_NULL.md`. Their finite arithmetic cells are included as `MathCan/ObserverPermutationNull.lean`.

## Reproduce

From the LoopedTransformers repository root:

```sh
python -m unittest discover -s mechinterp/phase_gauge07 -p 'test_label_shuffle_null.py' -v
python -m mechinterp.phase_gauge07.label_shuffle_null --repetitions 10000 --seed 20261115 --out /tmp/coe08_s4_null.json
```

The CLI verifies both original Git blob hashes before reading, and the existing TRM checkpoint/dataset pin before scoring. Its exact rational mean and variance have independent exhaustive finite permutation tests. All tests run on Python standard library, without external inference or GPU.

## Next experimental gate

With the full pretrained checkpoint available on local RTX 5080, capture native `z_H`, `z_L`, head-visible and orthogonal states for the existing locked `S1+S3` probe-fit and `S2+S4` evaluation split. Select phase-sensitive, phase-invariant and count-only one-bit readouts using **only fit-set activations and labels**. Before heldout scoring, freeze hyperplanes and calibration, the number of candidate heads, the selection objective, and comparison with final-bit and 8-bit budget controls. Replay held-out predictions and apply a label shuffle at group-disjoint board level. Follow with causal patching of the chosen readout direction to evaluate impact on task accuracy.

**Limitations:** this audit is post-hoc on one checkpoint's truth-conditioned outputs. It supports an epistemic correction to earlier numerical interpretation; it does not establish a new inference result for pretrained activation probes.
