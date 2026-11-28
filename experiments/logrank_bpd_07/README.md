# LOGRANK-BPD-07: compositional memory retrieval

Development-only CPU pilot. Six fresh program-generator roots (three at each k=16,32), each with randomized distinct signed literals, parity of two/three coordinates and signed conjunction of two/three coordinates. Fixed random bijective query address map; Bob sends log2(k) bits and Alice replies with one bit. The local dictionary learner receives the full matrix of training labels but no generating programs. It searches a specified family of signed supports/operators of arity <=1,2,3. This is discrete program fitting with an explicit strong prior; it is not neural TRM training or native BPD decomposition. The rank-half and rank-full static affine baselines are gradient-trained for 350 Adam updates (unequal learning compute and hypothesis classes). No sealed data or confirmatory causal claim.

## Reproduce
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/logrank_bpd_07/pilot.py --out receipt.json
OPENBLAS_NUM_THREADS=1 python experiments/logrank_bpd_07/audit.py receipt.json
```
256 training and 1024 disjoint heldout memory vectors per root; all query coordinates scored. Candidate dictionaries have 34/754/6354 entries at k16, 66/3042/52642 at k32. Exact integer-valued training correlations fit every candidate. The independent bitset audit recomputes all candidate-label distances, tie counts, optimal training scores, predictions, canonical certificate equality and saved static affine head outputs. It passed 5,358,624 candidate-label comparisons and 737,280 scored decisions. It does not replay optimizer trajectories.

| Arm | correct / 147456 | exact generating-program certificates |
|---|---:|---:|
| depth cap 1 | 108245 | 33/144 |
| depth cap 2 | 130306 | 90/144 |
| depth cap 3 | 147456 | 144/144 |
| static rank-half | 117308 | n/a |
| static rank-full | 118190 | n/a |

All depth-three best-fitting programs are unique and exactly equal to the generating canonical programs. This certifies their Boolean functions over the whole memory hypercube. The maximum depth is three local reads, mean 309/144=2.146; each read selects a single coordinate of persistent k-bit memory, applies the sign and accumulates multiplication (parity) or minimum (AND) on a one-bit accumulator. The full coordinate bank retains rank k. This is a per-pass rank-one active write, not a rank-one total-state claim. Fixed query codes are a deliberate isolation of retrieval, not learned addressing. No native BPD run in this receipt.

## Theory
For uniform bipolar x and f=s prod_{i in S} x_i, |S|=d, any polynomial p of degree <d has E[fp]=0 by Walsh orthogonality. Exact threshold sign(p) with ties positive would force fp>=0 everywhere and fp>0 on all negative inputs. Contradiction. Degree d is attained by p=f. For d>=2, no static affine threshold head of any rank can represent parity exactly. In the one-coordinate-per-read model, flipping an unread relevant coordinate proves exact parity requires d reads on every path; signed AND needs d on its all-satisfying input. This bound is specific to coordinate reads and excludes arbitrary dense projections. See companion MathCan and Compactification-Theory notes.

Native BPD integration is in BPD PR #166: identity V,U factors, pass-specific one-hot masks, exact residual-zero, full-mask parity and zero-mask tests, scoped CPU CI. This is an engineering fixture with planted factors. Its native execution is separate from this local experiment; no native causal effect asserted.

All root counts, source and raw receipt digests in summary_receipt.json. Raw receipt and audit are retained in local bundle, not committed. No PR merge or Lean proof claimed.
