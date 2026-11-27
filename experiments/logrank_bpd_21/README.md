# LOGRANK-BPD-21: marginal separator-memory allocation

Development protocol registered before execution. 27 fresh worlds (three per width 12/48/96 and family chain/two-hub chain/global random triples), 54 paired noise conditions. All 36 structured conditions admitted; 18 dense conditions refused scope >6 before table allocation. Two joint regimes, stable sources/local updates and shifting sources/distributed updates, each four epochs of 16 queries with eight carried pairs. One fresh noisy repeated unary factor between epochs. Eleven arms per admitted regime, 792 trajectories. Regimes change two variables jointly, so their effects are not separately identified. No native model training, protected cohort or confirmation claim.

## Mathematics

For directed message e, let K(e) be dependencies, w_e local message arithmetic, and d_e fixed direct unsigned request count. For warm pinned cache C, R_e=0 on C and w_e+sum_{f in K(e)}R_f otherwise. D_e=0 on C and d_e+sum_{parents p}D_p otherwise. The dependency DAG is ordered by source-side component size. Unsigned work W(C)=sum_e d_e R_e=sum_e D_e w_e. For e outside C the marginal reduction W(C)-W(C+e)=D_e R_e: count expanded occurrences of e, each with residual subtree cost R_e. Equivalently, W(empty)-W(C) is weighted coverage of local-work occurrences in the expanded request forest, each covered when its path contains a cached edge. Hence fixed-demand warm savings are monotone submodular. Lazy priority keys remain upper bounds. Unique nonbacktracking paths in the elimination tree permit subtracting D_e on uncached descendants and R_e on uncached ancestors when adding e. The exact ratio greedy is not globally optimal for knapsack. The proof excludes fill cost, cache invalidation, future demand, and changing query plans.

Cache policies: LRU, online observed request count times cold cost per byte, planned direct public batch demand times cold cost per byte, incremental marginal greedy using current public batch, and lagged marginal with previous batch demand (first current). Plans retarget then fill lazily. All arms use the same fixed conservative byte reservation per edge: side-specific factor count +3 and bag count bound integer weight size; recursively counted dictionary/key/value object sizes plus 256 B. Actual charges asserted below reservation. Retained logical cache cap 16/64 KiB, excludes temporary contractions, local factors, allocator overhead, policy objects and receipts. Current batch is public; no future query/evidence/answer used. Lagged computes current demand for audit only. Actual runtime and arithmetic include fill/invalidation costs, unlike warm objective; arithmetic excludes policy calculations, measured runtime includes them.

## 96D hub results

Six paired noise conditions per regime (three independent worlds). Work is counted products plus addition contributions. Original width sweeps concurrent with rotated per-condition arm order. Separate isolated 96D rerun on exposed worlds reproduces all 18 non-timing records. Times descriptive.

| Regime | KiB | Policy | Work | Original ms | Isolated ms |
|---|---:|---|---:|---:|---:|
| stable | 64 | LRU | 55586 | 31.55 | 31.36 |
| stable | 64 | online cold | 43221 | 29.81 | 30.13 |
| stable | 64 | planned cold | 31655 | 20.95 | 20.52 |
| stable | 64 | marginal | 31193 | 28.36 | 29.49 |
| stable | 64 | lagged | 33603 | 33.33 | 30.69 |
| shift | 64 | LRU | 83933 | 43.94 | 45.21 |
| shift | 64 | online cold | 64380 | 39.90 | 41.33 |
| shift | 64 | planned cold | 50068 | 29.04 | 28.91 |
| shift | 64 | marginal | 50068 | 37.66 | 40.12 |
| shift | 64 | lagged | 55490 | 43.96 | 40.33 |
| stable | 16 | LRU | 82547 | 44.17 | 44.74 |
| stable | 16 | online cold | 59610 | 33.14 | 34.82 |
| stable | 16 | planned cold | 62984 | 34.84 | 34.89 |
| stable | 16 | marginal | 38908 | 31.39 | 31.66 |
| stable | 16 | lagged | 41586 | 32.28 | 32.73 |
| shift | 16 | LRU | 114969 | 101.49 | 60.84 |
| shift | 16 | online cold | 86516 | 47.88 | 46.87 |
| shift | 16 | planned cold | 88525 | 45.92 | 48.70 |
| shift | 16 | marginal | 60129 | 42.69 | 44.13 |
| shift | 16 | lagged | 65918 | 44.14 | 45.29 |

Direct 202051 work in either regime, original 67.28/66.26 ms and isolated 64.46/65.17 ms. At 64 KiB both current plans achieve zero warm forecast unsigned work; marginal reduces stable arithmetic by 1.46% vs planned cold and ties shifting arithmetic, but runtime rises. At 16 KiB it improves arithmetic vs planned cold in all six conditions per regime (38.2% stable and 32.1% shifting), original runtime wins 4/6 and 6/6. Warm work averages 6301.5/8859.5 vs planned cold 28975.5/35420.3, actual costs include fills and signed contractions. The lagged forecast uses more arithmetic than current. Strong simpler baseline at 64 KiB; marginal selection useful candidate for tighter budgets. No statistical superiority claim.

## Verification

PASS: 54 graph replays, 288 exact partitions, 4608 independent reference queries, 393216 exhaustive states, 50688 arm-query comparisons, 3168 arithmetic ledgers, 2880 direct demand replays, 326950 cache transitions with order/eviction/admission replay, 2160 invalidation replays, 1728 plan replays and 18779 independently recomputed greedy steps. Fixed reservations independently reproduced from source-side components. Fixtures pass 6601 marginal/diminishing-return comparisons and 192 greedy-plan checks. Exact independent references are enumeration at 12D and chronological frontier DP at larger structured widths. No independent timing or allocator measurement, Lean proof or CI claim.

## Reproduce

Python 3.13 standard library, pinned copies of 19/20 code included.

```
python audit21.py fixture
for n in 12 48 96; do python pilot21.py "$n"; python audit21.py "$n"; done
```

For isolated repeat preserve receipt96.json, rerun pilot21.py 96, rename new receipt to timing_repeat.json and restore original, then python summarize21.py. The full downloadable bundle contains 132 full-precision aggregates, all 792 root-level records, raw receipts, timing repeat and audits. Compact summary in Git includes primary rows and raw/source digests. Next: resident-state-aware planning with explicit cache fill/eviction cost and expected invalidation survival, cheap warm-zero routing, independent factorial workload variation. Native MMS/BPD integration remains outstanding.
