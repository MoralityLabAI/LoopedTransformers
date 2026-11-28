# LOGRANK-BPD-15 / 15B: certified likelihood-tree repair

Development only. 12 fresh worlds (four each d=11,24,48), paired noise .15/.35. 72 primary searches and 24 adaptive 15B searches on the same exposed worlds. Shared 13 induced-fixed acquisition trace, 24 extra cost units, observed-first quotient, common BP modal warm start, static weighted-degree bit order. No protected data or native BPD training. Protocol was written locally before the primary run; no pre-run Git registration. Node budgets count distinct tree nodes, not equal FLOPs, peak RAM or all function calls (root and singleton likelihood additionally checked). Timings include interval generation and exclude shared acquisition and BP seed. No inferential superiority claim.

## Reproduce
NumPy >=2.0, Python 3.11+.
```sh
cd experiments/logrank_bpd_15
for r in 8 21 45; do OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot15.py "$r"; done
python audit15.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python weighted15.py
python audit15.py weighted
```
Source, full receipts, logs and independent audits are retained in the experimental bundle outside Git. base13.py and bp.py are pinned copies. See summary_receipt.json for compact aggregates and raw receipt digests.

## Mathematics
Uniform prior, rational BSC noise u/(u+v), 0<u<=v. Exact integer weight is u^e v^(m-e). A prefix fixes d of R observable coordinates. Group unresolved factors by identical nonzero residual mask a. Their combined likelihood values are g_a(0),g_a(1). Fixed factors contribute F. For an independent residual-mask set I, each parity vector occurs exactly 2^(R-d-|I|) times. Bound cell likelihood weight by
```
L = F * 2^(R-d-|I|) * product_{a in I}(g_a(0)+g_a(1))
    * product_{a notin I} min(g_a(0),g_a(1))
U = F * 2^(R-d-|I|) * product_{a in I}(g_a(0)+g_a(1))
    * product_{a notin I} max(g_a(0),g_a(1))
```
Grouped uses I empty, naive bounds individual observations separately. Independent selects by group multiplicity. Weighted 15B selects by absolute adjusted vote imbalance |n0-n1|. On the homogeneous BSC, replacing max by average multiplies the upper bound by (1+(u/v)^|n0-n1|)/2; replacing min by average multiplies the lower bound by (1+(v/u)^|n0-n1|)/2. The improvement magnitudes have identical ordering. Linear matroid greedy selection optimizes both bounds within this product-bound family. This is not general optimal inference. Pointwise dominance does not imply trajectory dominance at equal node count.

Exact leaf weight W and disjoint frontier bounds L,U certify retained mass in [W/(W+U), W/(W+L)]. For event A, sum lower/upper weights for A and its complement. A cell fixing the target contributes to its corresponding side; an unresolved cell adds zero lower and its full upper to both. Intersect [L_A/(L_A+U_notA), U_A/(U_A+L_notA)] with [W_A/(W+U),(W_A+U)/(W+U)]. Outside-span queries exactly uniform. Answer only when confidence lower >=.9. Certificate is conditional on the specified probabilistic model; it does not guarantee individual outcomes. Visited-only probabilities are not certified.

## Results
Mean retained-mass lower bound, .15 noise, 4096-node cap:

| d | naive | grouped | independent | weighted (15B) |
|---|---:|---:|---:|---:|
|11|1|1|1|1|
|24|.000002188|.496260|.831677|.900950|
|48|6.15e-21|4.03e-8|.207754|.233368|

At d11 and .15, independent certifies all targets at 256 nodes, naive 50%. At d24 and .15, independent and weighted coverage .4140625; all covered answers correct in sample. Weighted mean interval width .097985 vs .167349. At d48 and .15, original arms issue no predictions. Weighted certifies .916667 of target weight in one of four worlds, all correct there; mean coverage .229167, mean width .766477. Mean d48 decode time .546s weighted, .282s independent, .053s naive. At .35 all arms abstain, even exhaustive d11. Weighted retained lower at d24 .35 is 1.22e-6, at d48 1.71e-15. Visited-only Brier at d24/35% .42833 and d48/35% .35200. Most large-state runs score just two initial leaves. Severe negative scalability finding; no replacement for approximate repair in 14.

## Independent audit
Primary: 3,888 exhaustive fixture cells, 282,552 independently recomputed node bounds, 141,240 splits, 43,342 leaf weights, 216 snapshots, 6,912 intervals (2,304 checked against full small-state posterior), 72 exact mass bounds. 15B: 94,184 independently recomputed bounds, each checked against static independent bound on the same cell; 47,080 splits, 14,590 leaf weights, 72 snapshots, 2,304 intervals (768 exhaustive checks), 24 exact mass bounds. Independent LSB elimination, integer arithmetic, prefix partitions, priority replay, trace cost/noise/rank checks. No independent BP seed or nonrandom acquisition-policy replay. No observed decline in retained lower bound across three snapshots; no general monotonicity proof. No Lean compilation, CI verification, or hard RAM cap.

## Next
Exploit residual-factor components by exact local elimination where rank fits a cap; mini-bucket or cutset bounds otherwise. Test fresh low-treewidth and dense negative-control families with measured peak RSS and wall time. A separate MMS controller allocates compute or verifier calls according to certified interval width and cost; retain abstention and exact small-rank reference. No confirmation release.
