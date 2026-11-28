# LOGRANK-BPD-20: partial separator caching under evidence updates

Development only. 27 fresh worlds, three seeds per size (12, 48, 96) and family (relabeled chain, two-hub chain, global triples), 54 paired BSC conditions (.15/.35). 36 structured conditions admitted at max bag scope six; all 18 dense conditions refused before factor-table allocation. Four epochs of 16 queries, eight pairs carried from the previous batch, one fresh independent noisy observation on an existing public factor mask before each later epoch. This preserves the graph and its owner assignment. Uniform hidden bits and known noise law. All arms see the same observation stream and public batches. No native neural training or protected data. Unit of independence is world; paired noise and query rows are repeated measures. Protocol was committed before execution.

## Algorithm and invariant

Use the exact signed source-batched contraction from 19 with on-demand unsigned separator messages. A cache miss recursively recomputes the corresponding factor subtree from current evidence; a hit returns the retained table. An insertion may evict entries. The LRU policy evicts oldest accesses. The value policy orders residents and incoming candidates by cumulative observed request count times cold subtree arithmetic contributions divided by the logical charged bytes. Requests include recursive misses' dependencies; the score is an online heuristic, not actual marginal benefit or an optimal cache policy. It uses neither future query answers nor future query batches. Its sorting cost is charged to time. The 1024 KiB LRU arm is a full-admission reference (no evictions in any admitted condition), not proactive calibration. Two flush controls discard all messages on each evidence update at 64/1024 KiB. Direct is independent upward contraction for each query and root.

For a tree edge (i,j), delete the edge and let T_i be the component containing i. The unsigned message i->j is an exact contraction of factors owned by bags in T_i. When an observation changes a factor at owning bag o, the direction remains valid whenever o is outside T_i, since its summands are unchanged. If o is inside T_i, invalidate it conservatively. Traversing outward from o marks one direction on each tree edge; the reverse direction survives. For several changed owners use the union of their marked directions. With fixed topology and correct factor ownership, induction over the recursive message computation proves that all recomputed and retained unsigned messages equal fresh exact contractions for the current evidence. Signed path and batch contractions therefore remain exact. This invalidation rule is structurally sound but may discard numerically unchanged messages. Graph changes or changes to the probability model require a new graph/model version and appropriate invalidation. The experiment only appends repeated masks under the original model.

The byte cap counts retained message dictionaries, key/value Python object sizes, edge tuple and integers plus a fixed 256 B per message. Logical charge is checked on every insertion; eviction precedes admission, with incoming-message rejection permitted. This is not an allocator cap or a guarantee of process RSS. It excludes local factors, signed/unsigned temporary tables, policy counters, dictionary allocator slack outside the accounting rule and receipt storage. The cold work score includes products/addition contributions over a whole side with no cached descendants, so it can overestimate marginal savings when descendant caches overlap. Recursive requests are policy-dependent. Timing is sequential in a fixed arm order, single measurement per condition and includes update, policy overhead and event logging. Shared graph planning excluded from arm times (same plan across arms). No significance or causal claims.

## Results

96D hub-chain, means across six paired noise conditions (three worlds, each at .15/.35); arithmetic is counted message products plus addition contributions across all four epochs. Each admitted arm returns exactly the same answers as the direct baseline.

| Policy | Budget KiB | Arithmetic | Time ms | Mean peak logical bytes |
|---|---:|---:|---:|---:|
| Direct | 0 | 202051 | 64.09 | 0 |
| LRU selective | 16 | 140742 | 72.95 | 16384 |
| Value selective | 16 | 113513 | 74.00 | 16382 |
| LRU selective | 64 | 105970 | 53.32 | 65536 |
| Value selective | 64 | 82635 | 55.41 | 65535 |
| LRU flush | 64 | 108117 | 54.79 | 65536 |
| LRU selective | 256 | 60260 | 49.48 | 262127 |
| Value selective | 256 | 60177 | 30.25 | 262119 |
| LRU selective | 1024 | 59007 | 27.65 | 319072 |
| LRU flush | 1024 | 67263 | 31.71 | 317401 |

Primary comparison at 96D hub/64KiB: value cuts mean arithmetic by 22.0208%, wins all six conditions on arithmetic, but increases mean elapsed time 3.9224% (wins two of six timings). Value ranking cost and fixed-order timing variability are confounds for runtime claims. At 96D chain/16KiB, value reduces arithmetic by 34.3455% and runtime by 29.3736%, winning six of six paired conditions on both descriptive measures. No conclusion of general superiority. The tight-budget policy can be slower than direct contraction, including 96D hub/16KiB. At 1MiB the selective LRU cache reduces arithmetic by 12.2746% and time by 12.8122% versus flush on 96D hubs, winning all six conditions on both. At 64KiB the corresponding gains are small (1.9855% arithmetic and 2.6976% time); flush can free space that a partial cache needs. All full-admission arms have zero evictions. See compact_summary.json for every admitted family/cap aggregate; the full-precision aggregate and paired root rows are in the experiment bundle's summary_receipt.json.

## Audit

Independent exact references: direct exhaustive summation at admitted 12D, chronological frontier DP in offline structural order at larger chain/hub graphs. Recompute likelihood and all signed query values after every evidence update. PASS: 54 graph replays, 18 scope refusals, 144 exact partitions, 2,304 independent reference query values, 196,608 enumerated 12D states, 23,040 arm query checks, 360 trajectories. Independently verify changed factor owner, source-side connectivity, exact invalidation set, logical cache ledger on 286,054 transitions, 1,296 epoch ledgers and 972 invalidation replays. No timing, request order, charged object sizes or individual eviction ranking independently reimplemented. The audit checks logged transitions and exact outputs, not full allocator instrumentation. No model training, Lean proof or repository CI claimed.

## Reproduce

Python 3.13, standard library only. Source-locked copies of 19's solver and independent reference included.

```
for n in 12 48 96; do python pilot20.py "$n"; python audit20.py "$n"; done
```

Complete raw receipts and logs are in the downloadable experiment bundle, along with full-precision aggregates and root-level rows in summary_receipt.json. Git contains source, protocol, report and compact summary. Next: replace full sorting with an incremental indexed priority structure; use marginal saved work conditional on already cached descendants, and independent direct-use frequency to limit self-induced request inflation. Test changed evidence schedules, graph versions, recency and size-matched controls on fresh worlds, and measure true allocator and wall-time costs. No confirmation cohort.
