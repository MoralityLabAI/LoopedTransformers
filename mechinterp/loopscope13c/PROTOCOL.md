# LOOPSCOPE-13C / LGA-10C: admission after physical FP32 addition

18 November 2026. Prospective CPU engineering protocol, before initializing
model seed 61066 or generating `fixed_data(8,61066)`. This protocol precedes
the implementation lock and the first excluded-source run.

## Motivation and scope

LGA-10B's immutable first pilot at implementation commit
`57cb53c686d2f9e3932614d3ca842dd2091e6e3e` produced eight valid FP64 shams in
64/64 cases. Physical FP32 addition caused 48 cases to exceed its fixed
curvature tolerance, leaving 16/64 admitted against 61 required. The first
archive, report and failed verdict remain unchanged. Its raw vectors account
for the observed curvature drift through `2 e^T H z + e^T H e`.

The primary question is whether moving that physical check inside the
bounded candidate search supplies all eight controls for at least 61/64
fresh cases. A separate gate asks whether those selected controls also meet
the constraints and curvature resolution when the recurrent continuation
and derivative oracle run in native FP32.

The experiment has one untrained 44,001-parameter model initialization and
eight independent input bases, each with eight correlated goal variants.
It scores no edited task correctness. Its evidence class is
`SOURCE_ONLY_UNTRAINED_ENGINEERING`. Neither the case count nor the number
of generated shams supplies independent trained model replication.

## Source, model and common proposal geometry

Initialize one FP32 `TinyRecursiveModel` with `torch.manual_seed(61066)`;
its exact promoted copy supplies the FP64 model. Generate eight inputs with
`fixed_data(8,61066)`. Keep goals `(0,10,20,30,40,50,60,70)` and the existing
legal predecessor mutation `d -> d%9+1`. Preserve the original and donor
source labels before derivative calculations. The observer remains
`E(h)=0.5*sum(h*h)` with mu zero and std one.

The primary geometry uses the FP64 reached state h64 after two native tied
blocks and the existing legal donor direction at its natural pair radius.
Run the unmodified LGA-10B matcher once per case. Its original 66 directional
HVPs, orthogonal complement basis and seven extrema pairs define both arms'
proposal geometry. The complete attained and outer intervals are retained.
The frozen cosine grid is `(-.85,-.65,-.35,0,.35,.65,.85)` in that order.

Both arms use the original LGA-10B SHA seed formula, including teacher seed,
base index, goal, sham index, slice index and attempt. Each sham has at most
32 attempts per slice. Spherical interpolation and its 64 bisections remain
unchanged. The new arm replays this proposal arithmetic from the captured
geometry; its different accepted set can subsequently change diversity
rejections. A baseline diversity failure leaves the complete geometry
available to the new search. Geometry, solver and empty-grid failures are
shared. All failed cases remain in the denominator.

Compute a separate full 96D FP64 Hessian from exactly 96 coordinate HVPs at
h64. Preserve its raw columns and their skew before symmetrizing. Require
skew Frobenius norm <= `1e-8*max(1,||Hraw||F)`. This dense matrix evaluates
the rounded vectors, whose rounding components may leave the 66D span
measured by the proposal oracle. It never replaces the original HVPs used
to construct proposals. Fresh directional HVPs validate each final applied
vector independently of the dense matrix.

## Registered arms and candidate decisions

**Baseline:** the immutable LGA-10B matcher accepts eight intended FP64
controls, then evaluates their actual FP64 and FP32 additions with the
existing physical gates. It receives the same fresh source/model geometry
as the new arm.

**Roundtrip admission:** form every proposed intended edit with the original
FP64 construction. Apply

```python
actual32 = (h64_pair.float() + intended.float()).double() - h64_pair.float().double()
```

Check its curvature against the applied selected donor and all other
physical constraints, including diversity against previously accepted
applied shams. A failed check consumes the registered attempt; the search
then follows the next pre-seeded proposal. Check intended geometry and
intended diversity as well. Preserve every proposed vector, normalized
random start, interpolation parameter, seed, gate value and rejection reason.
Evaluation of physical diagnostics on a proposed vector does not change
the precedence of intended-geometry or diversity rejections.

The donor's intended curvature `Qtarget` fixes
`tau_Q = 1e-8*max(1,abs(Qtarget))`. Both arms retain exactly the existing
LGA-10B tolerances:

| Constraint | Applied FP64 | Applied FP32 |
|---|---|---|
| Curvature parity with applied donor | tau_Q | tau_Q |
| All ten future linear responses | `1e-8*max(1,||J donor_actual||)` | coefficient 1e-6 |
| K residual and error from intended radius rho | `1e-7*max(1,rho)` | coefficient 1e-6 |
| Head residual | `1e-7*max(1,||W||2*rho)` | coefficient 1e-6 |
| E differential residual | `1e-7*max(1,||g||*rho)` | coefficient 1e-6 |
| Absolute donor/dark cosine | `.85+1e-12` | `.85+2e-6` |
| Pairwise signed dark cosine | `.98+1e-12` | `.98+2e-6` |

Actual dark vectors use `P_N=P_K-RR^T` and their own measured norms. A zero
dark norm fails. Final FP64 curvature also agrees with the intended target
within tau_Q. All final physical curvatures use fresh native FP64 HVPs at
h64. A failure of the final validation remains a case failure; it does not
restart selection. Captured dense-matrix curvature and fresh HVP curvature
must agree within `1e-10*max(1,abs(Qtarget))`.

Exhaustion after rounded-candidate rejection is
`ROUNDED_CANDIDATE_BUDGET_EXHAUSTED`. It reports search coverage under this
budget. The smooth extrema describe the intended FP64 slices. FP32 rounding
is discontinuous, and exhaustion supplies no global absence claim on the
rounded lattice. Negative smooth-grid claims retain LGA-10B's outer-bound
and unresolved-interval distinction.

The physical coordinate system is fixed by the original two native 48D token
channels. Orthogonal gauge tests continue to apply to the FP64 geometry.
Coordinatewise FP32 rounding has no general orthogonal gauge invariance.

## Independent native FP32 continuation gate

After each arm's selection, compute the actual FP32 reached state h32 by
running the same input through the FP32 model's first two blocks. The
native check receives those selected intended edits and never feeds a
rejection back into the primary search. Evaluate both arms whenever all
eight shams exist, including cases that failed primary physical validation.
Preserve partial cases with `native_ready=false`.

Apply each intended edit at h32. Denote its exact promoted physical
difference by delta and its FP32-representable HVP direction by
`v32=delta.float()`. Record the direction-casting error. At the identical
clean state `h32.double()`, the promoted FP64 model provides both the HVP
on `v32.double()` and the HVP on exact delta. All norm and dot-product
diagnostics accumulate in FP64.

Define the native physical curvature estimate as
`Q32phys(delta)=delta^T HVP32(v32)`. Define the promoted physical reference
as `Q64phys(delta)=delta^T HVP64_same_h32(delta)`. The native per-case
readiness gate is the conjunction of four named checks:

1. `native_physical`: all applied native-state edits satisfy the FP32 table
   above, including native Q32phys sham/donor parity <= tau_Q. Use the
   native clean J32, its K-restricted rowspace, current-head projector and
   quadratic-energy gradient at h32. Dose references the primary intended
   natural rho; source angles reference the applied native donor.
2. `promoted_curvature`: every exact applied delta has Q64phys sham/donor
   parity <= tau_Q at that same native reached state.
3. `curvature_resolution`: both the native-versus-promoted physical parity
   discrepancy and each same-probe arithmetic error
   `abs(v32^T HVP32(v32)-v32^T HVP64_same_h32(v32))` are <= tau_Q.
4. `derivative_agreement`: the clean full-J Frobenius error, each same-probe
   HVP Euclidean error and the clean forward-logit infinity error are
   <= `1e-6*max(1, corresponding reference norm)`. Both models receive the
   identical native reached state and identical representable probes for
   these arithmetic comparisons.

Record every component separately. The clean state drift h32-h64 and
J64(h32)-J64(h64) are distinct diagnostics. Scalar native-Q error relative
to exact physical Q64 is retained for every vector. The broad vector/J
sanity tolerance cannot substitute for the strict scalar-resolution gate.
These checks concern the numerical instrument and its matched controls.

## Outcomes and evidence rules

For each arm, preserve per-case and per-base counts for primary admission,
native readiness, and their conjunction. The new arm passes primary
engineering if >=61/64 cases have eight controls passing the primary
fresh-HVP validation. Native readiness and overall readiness each also
require >=61/64; overall is computed per case as primary AND native.

The registered primary comparison is new-arm minus baseline primary case
admission on the same inputs. Report the paired outcome table and equal-base
macro difference descriptively. The comparison concerns one initialization
and eight bases. The pilot makes no population-confirmation claim. Native
results remain a separately registered numerical endpoint. Report all
gates, even when the primary gate passes.

EvalMath's independent units are the eight input bases here. Physical dose
is each donor's natural pair norm. A scientific causal effect floor and a
matched-sham population floor are uncalibrated in this source-only study;
causal power and binary correctness inference are inapplicable. The old
61065 result was exposed during method design, while 61066 stays unopened
until this protocol and the complete implementation are locked. Trained
model replication, native task controls and a separately sealed readout
remain prerequisites for a later mechanism study.

## Execution, validation and preservation

Use CPU only, one Torch thread, Python 3.11, Torch 2.7.1+cpu and NumPy 2.2.6.
Source-free and exposed-61064 tests precede the implementation freeze. The
tests cover physical rejection/retry, deterministic paired proposals,
bounded exhaustion, actual addition, native oracle comparisons at a common
state, and a synthetic raw-receipt tamper check. Reuse the frozen LGA-10B
geometry tests.

The first CI push on the new branch supplies its immutable commit SHA to
the runner. Before initialization, require a clean checkout of all imported
source/protocol/test/workflow files and record their SHA256. Generate the
excluded model and inputs only after that guard. Preserve all 64 paired
outcomes, raw arrays, rejected candidates, HVP counts by purpose, elapsed
and CPU times, peak RSS, versions, source/model hashes and code provenance.
Use compressed NPZ without pickle and an independent NumPy auditor that
imports neither model nor matcher. Archive the first raw result and its
hashes even when either engineering gate fails.

Science seeds **31059, 31064 and 31065** remain ungenerated. Trained
teacher/potential tensors, prior native cohorts, ARG-03-N, AMM and CQ remain
closed. Passing this source-only stage still requires a separately frozen
first-native protocol and source lock before any trained-native release.
