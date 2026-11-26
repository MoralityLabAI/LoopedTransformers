# Parity-query identifiability under a fixed coordinate map

For bipolar x and unknown signs s_i=(-1)^{h_i}, suppose a known coordinate map associates token i with x_i. Each clean parity training query has incidence row a in F_2^k. Its output divided by the known unsigned product of the selected x_i is (-1)^{a dot h}; thus a clean training set supplies linear equations Ah=b. For any consistent b, the solutions are h_0+ker A.

An unseen query q has identifiable sign if and only if q lies in the row span of A. Indeed q dot h is constant on h_0+ker A iff q annihilates ker A. The orthogonal complement of ker A is row(A) by rank-nullity. Conversely, outside row(A), some g in ker A has q dot g=1; h and h+g give identical training outputs but opposite query outputs on every memory. Under a uniform prior on this affine solution space, no estimator can exceed 1/2 expected accuracy on such a query. This holds even with arbitrary compute, recurrence or perfect memory. It is a sign-identifiability result conditional on known coordinates, known parity operators, clean observations and the full solution prior; it does not cover unknown coordinate maps, mixed AND queries, noisy observations or side information.

Example: all even-weight parity queries are insensitive to a simultaneous global sign flip, whereas an odd-weight query is sensitive. A full-rank training incidence matrix removes this sign ambiguity without requiring singleton examples.

## Singleton baseline

If the target for token t is a signed coordinate, a clean singleton label vector on m sampled memories identifies its signed coordinate whenever the k column signatures are distinct up to sign. With iid uniform bipolar memories, a given pair of columns agrees or is opposite with probability 2^(1-m), so by a union bound the failure probability is at most k(k-1)2^(-m). This is a sufficient identification bound under the specified distribution, not a necessary sample count. Exact finite signature equality can be checked without a statistical estimate. LOGRANK-BPD-09B used the clean singleton labels already present in training; all six worlds had unique exact matches.

## Exact finite check

The companion identifiability.py enumerates all 512 binary 3x3 matrices, their consistent right-hand sides, and every query incidence vector. All 20,952 checks passed. Mathematical proof above; Lean formalization remains outstanding.

## Next gym

Build training sets of controlled GF(2) rank, without singleton task labels. Register in-span and out-of-span unseen queries separately. Compare a GF(2) solver, noisy parity decoding and a learned hard component router; use ambiguity-aware abstention and charge verification queries. Do not interpret a failure to recover an unidentifiable target as a model deficiency. Unknown coordinate permutations and mixed operators require distinct controls.
