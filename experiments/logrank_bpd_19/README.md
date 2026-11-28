# LOGRANK-BPD-19: reusable separator-message memory

Development only. 27 fresh worlds (3 seeds per 12/48/96 variables and relabeled chain, two-hub chain, global triples), 54 paired noise conditions (.15/.35), 64 distinct unobserved pair queries per world. Uniform hidden state and known BSC likelihood. A public minimum-degree elimination plan determines bags and parent separators. The scope cap is six; refuse before allocating likelihood tables if a bag exceeds the cap. All 36 structured noise conditions fit (chain scope 3, hub scope 4). All 18 dense conditions are refused (scopes 9–61). Same compiled tree, observations and exact integer factors in every admitted arm. Three independent worlds per stratum; noise conditions and query prefixes are repeated measurements. No native model training, causal claim, protected cohort or confirmation.

## Method

Assign each observation factor to the bag of its earliest eliminated coordinate. A bag product is phi_i(x_i), where x_i denotes its assignment. An elimination-tree edge has separator S_ij = bag_i intersect bag_j. The running-intersection property ensures that removing an edge separates the two collections of factors conditional on that separator. Calibrate both directed messages using

    m_i->j(s) = sum_{x_i: x_i|S_ij=s} phi_i(x_i) prod_{k in nbr(i) except j} m_k->i(x_i|S_ki).

Add an empty super-root for disconnected components. All weights are unnormalized exact integers. For a pair (a,b), Z_ab is the signed partition sum obtained by inserting (-1)^h_a and (-1)^h_b into their owning bags. Then P(h_a xor h_b=1 | obs)=(Z-Z_ab)/(2Z). The direct baseline repeats a full upward contraction per query (plus one unsigned root contraction); all local tables and the compiled tree are shared. For a cached pair, propagate the signed contribution along the unique path from bag a to bag b; use calibrated incoming messages from off-path neighbors, and contract at b with its sign. For a batch of pairs sharing a source, propagate once along each edge in the union of target paths. The orientation is chosen by a greedy public maximum-incidence rule on the remaining query pairs. This computes exactly the same signed partition sums; negative intermediate messages are valid because they are linear signed contractions, not probability distributions. Complexity depends on bag size, degrees, path lengths and the number of shared source-path edges, not solely on the latent state dimension.

## Results

Mean ms including graph plan, local table construction and unsigned root contraction (direct) or conservative cache preflight and both-direction calibration (cached). Six paired noise conditions per size/family. Baselines share local table construction; timing is sequential in a common process, without randomized arm order or controlled CPU affinity. Counted arithmetic = incoming message multiplications + sum contributions, excludes graph planning, local factor products (same setup), bit complexity and Python overhead.

| n | family | queries | direct | path | batch |
|---:|---|---:|---:|---:|---:|
| 12 | chain | 64 | 2.933 | 1.851 | 1.445 |
| 12 | hub | 64 | 4.640 | 2.891 | 1.998 |
| 48 | chain | 64 | 15.043 | 7.402 | 6.060 |
| 48 | hub | 64 | 28.393 | 13.286 | 10.809 |
| 96 | chain | 64 | 35.373 | 18.210 | 15.486 |
| 96 | hub | 4 | 10.322 | 10.052 | 9.539 |
| 96 | hub | 16 | 21.512 | 15.227 | 10.872 |
| 96 | hub | 64 | 65.771 | 32.640 | 22.489 |

At 96D hub and 64 queries, batch is 2.92x faster than direct; mean counted arithmetic 193137 direct, 78800 path, 41244 batch (78.6% reduction vs direct). At 64 queries batch is faster in all 36 structured noise conditions; at 4 queries the advantage is inconsistent. All outputs match exactly, so this is a computational efficiency result, not an accuracy benefit. Query grouping uses public masks with deliberate common-source clusters; do not generalize the speedup to arbitrary workloads.

## Cache costs

Conservative preflight counts the retained directed-message dictionaries, tuple keys, assignment keys and value objects using a per-entry upper integer size based on 2^n max(u,v)^m. Repeated referenced objects are counted pessimistically. This excludes local factors, temporary signed contractions, graph planning, allocator overhead and receipts. The benchmark builds full caches only after the 1 MiB preflight passes. The smaller-budget flags report conservative admission/fallback decisions; separate runs under those lower limits were not executed.

| n/family | mean counted cache bytes | maximum preflight bytes | smallest admitted cap (all 6 conditions) |
|---|---:|---:|---:|
|12 chain|15097|16800|64 KiB|
|12 hub|20700|23520|64 KiB|
|48 chain|79235|109296|256 KiB|
|48 hub|128452|188072|256 KiB|
|96 chain|205185|325432|1024 KiB|
|96 hub|356418|596056|1024 KiB|

All 36 actual cache object estimates were below their conservative preflight bounds. The 96D chain cache object estimate is below 256 KiB, but the conservative admission rejects it. Partial cache admission and message lifetime planning are next. Process high-water RSS across the sequential benchmark was 97424–101904 KiB; this is not per-arm incremental memory. No hard total-RAM cap or observed process-RAM reduction.

## Reproduce and audit

Python 3.13, standard library only, Linux resource RSS.

```
python pilot19.py
python audit19.py
```

PASS: 54 independent graph/running-intersection replays, 36 exact partition checks, 2304 exact pair probability checks against direct enumeration (12D) and independently ordered chronological frontier DP (large structured graphs), 73728 enumerated 12D states including refused dense references, 324 exact work-ledger comparisons, 144 conservative cache admission comparisons. All 18 dense refusals checked. The pilot asserts equality across all three methods; the independent reference checks their common outputs. Audit does not independently replay each cached message table or Python object-size estimate, and does not validate timings. First timing run omitted unsigned root time for direct and cache-preflight time for cached arms. Preserved initial source/receipt/log; reran with both included in aggregate timing, added 1 MiB cache assertion; all 54 non-timing comparisons matched. Protocol written before execution, Git commit after execution. Full receipts in downloadable bundle; repository contains source and compact aggregate summary.

Next: budget-aware partial separator cache with sound recomputation, message-use value per byte, trace-dependent admission, cache invalidation on verifier updates and amortized cost under fresh query workload distributions. Keep exact reference and dense refusals. Native MMS/DataGoblin integration, Lean formalization and CI remain outstanding.
