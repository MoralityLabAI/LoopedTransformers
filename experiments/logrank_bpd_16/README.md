# LOGRANK-BPD-16: bounded-width posterior elimination

Development only. 36 fresh parity graph worlds (12/24/48 variables, four roots per size and family), paired BSC noise .15/.35, 360 solver runs. See PROTOCOL.md, written before the first local execution. Each world has two unary observations per coordinate, 3n triple observations, and eight unobserved pair queries. Identical observations per arm; structural controls differ in graph connectivity, not observation count or arity. Four adjacent and four randomly selected pairs. Uniform hidden state and known noise law; no native training, intervention, protected cohort or confirmation. The kernel comparison evaluates root and conditioned weights; it does not yet replace the search loop in 15.

## Method

For a nonnegative factor graph, split the bucket containing variable x into groups with joint factors G_j(x,z), each on at most c variables including x. Fix the remaining assignment z. Choose one anchor j*. Then

```
(sum_x G_anchor) * product_other min_x G_j
  <= sum_x product_j G_j
  <= (sum_x G_anchor) * product_other max_x G_j.
```

Every factor is nonnegative, so iterating the inequalities preserves the global lower and upper bounds even when prior steps produced distinct lower and upper factors. Exactly one group sums over x, which avoids overcounting x and preserves exactness whenever a bucket fits without splitting. All integer weights and tables use arbitrary-precision arithmetic. Deterministic minimum-neighbor ordering and greedy scope packing; c counts x itself. Separate exact component arm integrates a connected component in its GF(2) quotient if its rank <=7 (including free-coordinate multiplicity); otherwise it falls back to the likelihood-weighted independent-factor bound of 15. The mini-bucket arms intersect their raw weight intervals with the independent bound. Bound tightness need not be monotone across caps because grouping and induced scopes may differ.

To obtain an event probability for nonzero parity mask q, condition on q dot h=b by eliminating its lowest-index variable through substitution. For an observed row containing that pivot, replace a by a XOR q and y by y XOR b. The remaining-coordinate map is bijective to the event hyperplane. If L_b <= Z_b <= U_b, then

`L_1/(L_1+U_0) <= P(q dot h=1 | observations) <= U_1/(U_1+L_0)`.

A .9 threshold certificate is issued only when the lower bound for the predicted sign is >=.9. It is a posterior guarantee under the likelihood model, not a guarantee of the realized answer. These observations have positive weights. A factor table has at most 2^c entries (per bound); total storage also depends on the number of live factors, integer bit lengths and bookkeeping. If every elimination bucket fits, inference is exact, including large connected low-treewidth graphs. Rank and elimination width measure different constraints.

## Results

48D, .15 noise, mean interval width, certified coverage and mean elapsed solve time (17 weight-bound calls: root and both conditioned sides of eight queries):

| Graph | Method | Width | Coverage | ms |
|---|---|---:|---:|---:|
| block | independent | ~1 | 0 | 7.8 |
| block | component7 | 0 | .96875 | 14.3 |
| block | mb3 | ~1 | 0 | 99.6 |
| block | mb5 | 0 | .96875 | 90.0 |
| chain | independent | ~1 | 0 | 7.6 |
| chain | component7 | ~1 | 0 | 14.6 |
| chain | mb3 | .237002 | .65625 | 94.4 |
| chain | mb5 | 0 | .875 | 107.2 |
| dense | independent | ~1 | 0 | 11.4 |
| dense | mb7 | ~1 | 0 | 762.2 |

All 384 query intervals from width-5 elimination on block/chain graphs (all dimensions and both noise levels) are exact. At .15 noise the combined structured-family coverage is 170/192 (88.54%), all covered decisions correct in the sample. At .35 it is 3/192, all correct. Dense 12D width-7 mean width .789726 at .15 with four of 32 query predictions certified (all correct); dense 48D every width-7 interval rounds to width 1 even though exact integer endpoints may be slightly tighter than [0,1]. Zero coverage in all dense 24/48D strata. More computation without useful separator structure is ineffective here. No claim of general dense-inference impossibility or statistical superiority.

Memory: width-5 chain at 48D reaches at most 4 variables per factor join (16 assignments), with a mean per-world maximum of about 900 counted live table slots across its 17 calls. Dense width-7 uses at most 128 assignments per join, ~4039 counted live slots on average, while producing vacuous intervals. Peak process RSS at 48D: independent 81,676–81,932 KiB; component 81,676 KiB; mb3 82,444–83,852; mb5 82,444–83,980; mb7 82,444–84,236. RSS includes interpreter/forkserver inherited pages; not exclusive allocation. Measured times use four concurrent workers; descriptive, not controlled FLOP or latency comparisons.

## Measurement correction and audit

First run used fork workers spawned from a parent that accumulated results. RSS was confounded by growing inherited process memory. Preserved original source, log and receipt, reran with clean forkserver workers (one task per worker). All 360 non-timing result records matched exactly. Corrected RSS/timing used in report. No pre-run Git registration; no changes to algorithms or root roster in rerun.

Independent audit: 6,120 partition-weight bounds reproduced using LSB elimination, absolute-mask table arithmetic and independent group/substitution logic; 2,880 event intervals reproduced; 4,760 exact weight enclosures and 2,240 exact event enclosures against chronological frontier DP (56 conditions). Further direct exhaustive crosscheck of 98,304 12D states and 216 partition/event weights. 1,080 small fixture enclosures (original solver). The audit replays certificate arithmetic and grouping, not independent performance. Large dense graphs have no exact reference. Complete root-level receipt and logs in bundle; Git contains source and aggregate summary/digests. No Lean or repository CI claim.

## Reproduction

```
python pilot16.py fixture
python pilot16.py run
python audit16.py
python exhaustive16.py
```

Python 3.13, standard library only. Linux process RSS via resource and forkserver. The `summary_receipt.json` in the bundle includes every root; repository compact summary records aggregates and hashes. Future work: separator-aware cutset conditioning, exact sums over linear factor bases with bounded residual dependency rank, per-target re-use of elimination messages and costed branch-and-bound. Keep the dense negative control, match total runtime and memory, and measure certified interval reduction per cost. For DataGoblin/MMS use graph diagnostics to choose local component, low-width elimination, or approximate search and explicit abstention. No runtime integration yet.
