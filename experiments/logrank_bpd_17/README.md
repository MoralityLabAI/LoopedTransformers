# LOGRANK-BPD-17: cutset-conditioned certified inference

Development-only comparison registered in PROTOCOL.md before execution. 18 new worlds, three per dimension (12/24/48) and family (randomly relabeled two-hub chain/global triples); 36 noise conditions and six arms = 216 runs. Four distinct pair targets per world, two unary observations per coordinate, 3n triple observations. Paired .15/.35 noise and shared observations. The selectors use public masks, and scope2 additionally uses public target masks. Offline hub identities are used only for generating worlds, checking hub recovery and exact-reference ordering, never to choose the cut. No native BPD training, protected cohort or confirmation.

## Method

For a cutset C of k coordinates, enumerate its 2^k assignments. The assignments partition the uniform state space. In a branch h_C=x, restrict every factor row a to a on the complement of C and XOR its observed bit with parity(a intersect x). Retain integer likelihood scaling. Condition residual target parity using the pivot substitution from 16. If the target is determined by the cut assignment, route the root cell bounds entirely to the corresponding target sign. Otherwise compute both conditioned weights with the scope-3 mini-bucket kernel. Sum lower and upper weights separately across disjoint branches. If Z_b lies in [L_b,U_b], the posterior event interval is [L_1/(L_1+U_0), U_1/(U_1+L_0)]. It is exact if all contributing weight bounds are exact. The certificate is conditional on the specified uniform prior and known BSC likelihood, not a guarantee of realized correctness. Each branch may reuse the same bounded-size factor kernel; total space includes all other live tables, bookkeeping and receipts. Runtime may grow exponentially in cut size. Scope cap counts the eliminated variable itself; a table of scope c has at most 2^c entries per bound. This is a static cutset, not the adaptive disjoint prefix search of 15.

## Results

All nine hub-chain worlds have their two planted hubs recovered by both degree2 and scope2 (checked offline). Prefix2 and random2 recover neither complete planted cutset on any of these nine worlds. Both graph-selected arms give exact posteriors for all 72 structured target/noise cases, as does no-cut scope 5. At .15 each exact arm certifies 28/36 structured queries at 90% confidence, all correct in this sample; at .35 none certifies any. The original no-cut scope 3 certifies none at either rate. Across all dense controls, none of six arms issues a certificate; even at n12, scope-5 and two-coordinate cuts produce near-vacuous intervals. This is a negative result for these algorithms, not general dense-inference hardness.

48D hub-chain at .15 (three roots, four queries each):

| Arm | Mean interval width | Exact /12 | Certified /12 | Mean ms | Peak counted live table slots |
|---|---:|---:|---:|---:|---:|
| none3 | ~1 | 0 | 0 | 72.5 | 1488 |
| none5 | 0 | 12 | 10 | 83.6 | 1488 |
| prefix2 | ~1 | 0 | 0 | 324.0 | 1418 |
| random2 | ~1 | 0 | 0 | 339.5 | 1426 |
| degree2 | 0 | 12 | 10 | 167.7 | 550 |
| scope2 | 0 | 12 | 10 | 382.7 | 550 |

Degree2 reduces the maximum counted live slots by 63.0% versus none5 (550 vs 1488), and reduces mean factor-entry work (23240 vs 29033), but takes about 2x wall time. Its 36 kernel calls vs 9 for none5 account for overhead. Scope2 chooses identical cuts to degree2 on the structured worlds; its selection alone costs about 215 ms at 48D/.15. No evidence for preferring this expensive lookahead here. Process RSS is around 80 MiB for all arms; counted table-slot savings did not yield a measured peak RSS reduction. RSS includes interpreter/forkserver pages and accumulated result receipts, not exclusive solver workspace. Timings use four concurrent fresh workers, descriptive only. The six arms do not have equal wall time or table caps; the four cut arms share cut size and scope cap. At 48D dense/.15, scope2 costs approximately 899 ms and still has interval widths rounding to one. Use the cheap degree heuristic as a candidate when the public graph contains clear hubs, with a preflight width diagnostic; no general claim about cutset optimality.

## Audit

Independent replay PASS: 216 runs, 648 disjoint partition cells, 5832 integer kernel weight replays, 864 posterior intervals, 144 exact root-weight enclosures, 576 exact interval enclosures on 24 conditions. Absolute-mask mini-bucket arithmetic and independent conditioning from 16; exact reference uses exhaustive state summation for all n12 and chronological frontier DP with offline hub-first ordering for larger structured worlds. The audit verifies recorded grouping and weight arithmetic, query conditioning, cut partitions and prefix/random cuts. Degree/scope selection and cost counters are not independently reimplemented; no timing replay. Raw receipts contain all branch bounds, grouping traces and root outcomes. Three independent roots per stratum, no significance test or causal claim.

## Reproduce

Python 3.11+, standard library only. Linux process RSS and forkserver used for timing runs.

```
python pilot17.py
python audit17.py
```

The kernel and auditor from 16 are included as pinned copies. The compact summary receipt includes all 72 aggregate groups, with floating-point values rounded to nine decimal places, and source/raw hashes. Complete receipt, full-precision summary, source and logs are in the experimental bundle; raw receipt is not committed to git. Next: adaptive cutset growth with budget-gated exact/approximate routing, compare residual width prediction and certified interval reduction per cost on fresh graph families. Prioritize reusable factor messages and streaming rather than retaining all branch receipts in RAM. A true RAM cap needs allocator/workspace instrumentation. Native BPD/MMS integration, Lean formalization and CI remain outstanding.
