# SPC-09: causal phase and deferred-observability subspaces

## Registration (before any SPC-09 evaluation)

Base: `6205e0c4f4cc84e0eb34cca31abb853160403464` on `main`.
The frozen JSON in `configs/spc09_frozen.json` fixes all dimensions, radii,
seeds, losses, and thresholds. Its SHA-256 accompanies every receipt.
No test result selects a radius, rank, loss weight, or checkpoint.

Priority: A exact SPC-08 replay; B compression; C steering; D bidirectional
transfer; E continuation only after A and frozen interventions validate.
Missing archives produce an explicit incomplete receipt, never a replacement
checkpoint represented as an original. Two original checkpoints are available
in principle; independent seed 41524 is unreplicated unless its own provenance
is supplied. No large models or cloud compute are used.

SPC-08 fits answer-token phase increments and prompt-local future derivatives
in width 48. Its pooled deferred basis is not a common pointwise nullspace.
SPC-09 tests this geometry causally; phase registration alone is not capability.

## Hypotheses and estimands

H1: original rank-eight overlap is <=0.1 and original phase/deferred causal
rates reproduce within 1e-5 absolute tolerance, with discrete counts identical.
Report every numerical divergence, independently of whether H1 passes.

H2: for each preregistered rank/depth, a compression preserves capability only
if the lower paired cluster-bootstrap 95% confidence bound on its accuracy
difference from baseline is >=-0.02. Phase registration is a separate outcome,
with the same 0.02 criterion. Also report immediate/future logit distortion,
intervention norm, removed state energy and removed natural-update energy.
All rank/depth results are reported; unadjusted intervals are descriptive,
not simultaneous guarantees or evidence for a test-selected optimum.

H3: within each fixed radius, local steering prediction is acceptable if
median relative error <=0.2 and nonzero-margin sign agreement >=0.8. Zero
predicted effects have undefined relative/sign error and are counted separately.
Numerical rank uses relative 1e-3 and absolute 1e-10 cutoffs; exact rank is
never inferred from floating-point SVD.

H4: paired Procrustes transfer is compared with raw, shuffled-pair, random
orthogonal and target-fitted bases in both directions. Intervention rank and
norm are equal. Confidence intervals and failures accompany each direction.

H5: continuation results are provisional. Fixed losses are mean squared error
between student and frozen teacher projected answer-token natural increments
(P), and projected answer-token states (D), divided by teacher target mean
square plus 1e-8. Coefficients are both 0.1. CE supervises hops 1..3.
Every arm receives identical graph batches, AdamW settings and initialization,
at most 300 updates, snapshots every 50. Evaluate all snapshots, never select
one using test data. Baseline and every arm use the same test graphs/depths.

## Splits, interventions and resources

A reuses the exact original generated prompt cohorts, auditing both prompt
and canonical graph overlap. It is a reproduction, not a new split guarantee.
B--E partition canonical successor maps by SHA-256: buckets 0..5 training,
6..7 validation, 8..9 test. Start node and edge token order do not affect the
partition. Exact canonical graph IDs and prompt hashes are saved. Training
phase/PCA/derivative/alignment fits all use training graphs. Test graphs are
never used for fitting. This disjointness concerns SPC-09 basis/continuation
fitting: SPC-04 used procedural training and may have encountered the same
finite graphs; absence from the original model's training is not claimed.

All new interventions affect the answer token once. Context tokens continue
with the native recurrence; this is not global compression of all nine tokens.
Hidden state is centered by a training-only mean for state compression.
The P-retain/D-suppress and converse arms use oblique maps that fix the retained
basis and annihilate the suppressed one, only when their union has full rank.
Ill-conditioned cases are recorded as invalid. Union rank r uses r/2 P and
r/2 D directions, then orthonormalizes: its total budget is r. PCA and random
controls have rank r. Suppress-immediate-invisible uses the local readout row
projector, with its actual rank reported (usually seven), not relabeled r.
Full-state baseline retains all native channels. For each compression there is
a fixed-seed random direction with the exact same per-item displacement norm.
Energy, operator norm, actual rank and any oblique amplification are reported.

Inference is microbatched <=16 (default eight); derivatives are sequential
CPU float64 JVPs of the 48-dimensional answer-token injection. The remaining
token states evolve in full: no frozen-context approximation. Small output
Jacobians are at most 91x48, never dense full-state 432x432 Jacobians.
Rank-exact small SVD is used here; randomized SVD is unnecessary at this size.
GPU allocation is capped at 3.2e9 bytes with automatic CPU fallback. RSS is
sampled and checked between bounded operations against 8e9 additional bytes.
Receipts include runtime, peak RSS, allocation and device/fallback reasons.

## Mathematical bounds (paper proofs, not Lean certificates)

1. Let J0:V->Y, B:V->Z and K=ker J0. The deferred quotient is
K/(K intersect ker B). Rank-nullity for B restricted to K gives dimension
rank(B|K)=rank([J0;B])-rank(J0), at most min(dim V-rank J0, dim Z).
Projection from im([J0;B]) onto im J0 has kernel {0}xB(K), proving the identity.
This is local linear algebra at one state, not nonlinear identifiability.

2. For x[t+1]=F[t](x[t]) and compressed z[t+1]=Q[t+1]F[t](z[t]), let
e[t]=||x[t]-z[t]||, q[t]=||Q[t]|| and assume F[t] is L[t]-Lipschitz on
a set containing both trajectories. Then
e[t+1] <= ||(I-Q[t+1])x[t+1]|| + q[t+1] L[t] e[t].
Add/subtract Q[t+1]F[t](x[t]) and use the triangle inequality. Iteration
gives e[T] <= e[0] product(q L) + sum of each omitted energy square root
times the subsequent product(q L). Orthogonal projections have q<=1;
oblique retention requires its measured operator norm. Affine centering
uses ||(I-Q)(x-mean)|| in the omitted term.

3. For observation Jacobians O and O+E, G=O*O obeys
||G'-G|| <= 2||O||||E||+||E||^2 by expansion O*E+E*O+E*E.
For deferred O N and perturbed O' N',
||O'N'-ON|| <= ||O'-O||+||O|| ||N'-N|| for orthogonal N'.
An eigenspace stability claim additionally requires an eigenvalue gap;
near-zero singular values can make numerical nullspaces unstable.

4. If a reference phase has distance gap g>0 to every competing phase,
a state perturbation of norm <g/2 preserves its nearest-reference label:
the true distance is <g/2 and all competing distances >g/2. After T updates,
a sufficient condition is product(L[t])||delta||<g/2. This proves phase
registration only, not correct algorithmic output. At equality ties can occur.

5. If C is M-Lipschitz on the trajectory tube, output error <=M e[T].
The bound in (2) therefore controls future output error using square roots
of omitted energies. For one-shot differentiable perturbations with
Hessian norm <=H on the perturbation segment,
||O(h+delta)-O(h)-DO(h)delta|| <= H||delta||^2/2, by integrating the
Jacobian along that segment. Local Jacobian norms at sample points alone
do not certify tube-wide L, M or H. Empirical measured norms are diagnostics,
and no global bound is asserted for the trained transformer.

New finite Lean witnesses and exact arithmetic tests live in MathCan. They
are explicitly separate from these analytical proofs and neural measurements.

## Status

Implementation and validation receipts are recorded under `receipts/spc09/`.
Scientific results require original hash-verified checkpoints. Incomplete
stages and unavailable independent replications remain explicit.

