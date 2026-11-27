# OLR-06F: metered anytime abstraction results

**Status:** native engineering PASS; registered total-work allocator headroom PASS.
**Executed commit:** 20a45eb6da5e292c5b5a98c440a778945473a748.
**Successful workflow:** [45322490776](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45322490776), native job 138204125531. The earlier run 45322445596 was cancelled by concurrency when the README changed. Use the successful replacement, not the cancelled run.
**Registration:** [06F](OLR_06F_REGISTRATION.md).
**Artifact:** [14887878324](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45322490776/artifacts/14887878324).

24 alpha-disjoint roots, 96 target obligations, 192 bounded policy/root trials and 24 unrestricted references. All five protocol tests passed. 270 distinct native leaf paths replayed from reset and compiled independently; 91 complete conjunction declarations compiled. Total 361 axiom-free declarations. All 24 unrestricted references close. No learned weights or scientific promotion.

## Complete conjunctions under total work caps

| Policy | DEV 128 | DEV 256 | DEV 512 | Transfer 128 | Transfer 256 | Transfer 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cached independent BFS | 2 | 3 | 11 | 0 | 2 | 6 |
| Demand shared ledger | 5 | 6 | 10 | 3 | 4 | 8 |
| Hamming weighted | 0 | 2 | 3 | 0 | 0 | 1 |
| Eager width <=2 | 0 | 7 | 8 | 0 | 1 | 5 |
| Eager width <=3 | 0 | 0 | 4 | 0 | 0 | 0 |
| Cyclic quarter | 0 | 1 | 8 | 0 | 0 | 5 |
| Demand quarter | 0 | 2 | 8 | 0 | 0 | 6 |
| Demand half | 0 | 3 | 8 | 0 | 0 | 7 |

Denominator 12 independent roots per panel. Abstract settled-state and fresh concrete neighbor enumerations both consume one illustrative work unit. This is not equal measured FLOPs. Cache hits consume logical schedule slots, no fresh enumeration units. Queue rekeys, scan overhead and timing are measured separately.

| Policy | DEV mean total work | Transfer mean total work | DEV mean ms | Transfer mean ms |
| --- | ---: | ---: | ---: | ---: |
| Cached independent | 268.92 | 431.33 | 7.66 | 7.53 |
| Demand ledger | 245.75 | 333.50 | 12.16 | 22.29 |
| Hamming | 443.25 | 511.83 | 9.70 | 12.85 |
| Eager2 | 320.08 | 422.25 | 10.12 | 17.09 |
| Eager3 | 497.33 | 512.00 | 6.44 | 6.10 |
| Cyclic quarter | 383.58 | 480.42 | 21.66 | 35.01 |
| Demand quarter | 352.33 | 437.08 | 24.55 | 45.43 |
| Demand half | 347.33 | 441.08 | 22.99 | 41.69 |
| Unrestricted reference | 327.92 | 642.58 | 4.83 | 11.03 |

Means include bounded misses, so they are not time-to-completion comparisons. Native leaf replay cost and standalone compilation are outside these planner timings. Full campaign 42.63 s, native leaf execution/replay 33.48 s.

## Registered headroom

Best fixed per-panel choices at cap512 are cached independent (11/12 DEV) and demand (8/12 transfer): 19/24, 79.17%. Hindsight choice of one of the eight bounded schedules for each root solves 23/24, 95.83%. Gap +16.67 percentage points. The two registered thresholds (at least 20% unsolved, portfolio gap >=10 pp) both hold. This is a small exploratory sample. The hindsight portfolio is privileged, not an algorithm, and its per-panel baseline is selected on these evaluation roots. The gate is a reason to test a learnable selector on fresh roots, not proof of a useful neural mechanism.

## Custody

Downloaded artifact and checked all 3 registry and 8 evaluation manifest entries by SHA-256.
- Registry manifest: 6e81000eef3dab7d3d26ef1b284dcdbb8bd4dd25072c0a49132ce84e7626b89c
- Evaluation manifest: e7ea38b54819270c6f6f8610232859856e9ec78364588d100a6b9246d6d17c29
- Archive SHA256: 0694208e4ca13377d3f2127420a05dd807a1bf34f68af5fcab213e5a627a0988

## Next gate

[06G](OLR_06G_REGISTRATION.md) tests one-shot selection among fixed budgeted schedules, including source-calibrated two-stage restart portfolios. It is a deliberately simpler problem than online abstraction/frontier scheduling. Freeze all selectors before new outcomes. Include simple feature-based and full-input controls, plus all fixed schedules as separate reference trials. All actual bounded work is charged. If a cheap deterministic selector or restart strategy captures the available gap, prefer that engineering result. A recurrent mechanism requires a prospective advantage over these controls.
