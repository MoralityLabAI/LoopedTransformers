# LOOPSCOPE-13D / LGA-10D: reached-state geometry and continuation precision

19 November 2026, Asia/Singapore. Prospective source-only engineering
protocol, before initializing model seed 61067 or generating its inputs.

## Question and prior evidence

LGA-10C's frozen first run admitted 64/64 cases after physical FP32 addition
and admitted 0/64 under its complete native FP32 gate. Seventeen selected
donor probes exceeded its scalar resolution limit. Captured-array analysis
also found that changed additions and changed reached-state Hessians broke
the old controls. That result and its 1e-8 bound remain unchanged.

The primary question here is whether evaluating the continuation derivatives
at the actual reached state supplies eight usable controls for a declared
promoted continuation. Both arms share the same native donor, dose and
physical application. Native FP32 readiness remains a separate registered
endpoint and receives its original precision checks.

The evidence class is `SOURCE_ONLY_UNTRAINED_PRECISION_ENGINEERING`.
There are eight independent input bases with eight correlated goals each,
under one untrained model initialization. This study measures numerical
constraints and oracle consistency. It scores no edited task correctness.

## Sources and common physical fiber

Initialize one FP32 `TinyRecursiveModel` with `torch.manual_seed(61067)`.
Use its exact promoted FP64 copy. Generate `fixed_data(8,61067)` once and
retain goals `(0,10,20,30,40,50,60,70)`. The legal predecessor mutation is
the existing `d -> d%9+1`. Its source labels determine the signed donor
margin before derivative calculations.

For each recipient and legal donor, measure the actual FP32 reached state
after two tied blocks, denoted h32 and hdon32. Also retain the all-FP64
recipient trajectory h64. Promote h32 and hdon32 before subtraction:

```
hN = h32.double()
hD = hdon32.double()
dN = (hD-hN)[pair]
rho = norm(dN)
```

The observer is `E(h)=0.5*sum(h*h)`, with mu zero and std one. Construct the
joint current-head/energy tangent projector K_N at hN and the selected
donor `zN=rho*K_N*dN/norm(K_N*dN)`. Both arms use exactly this natural donor,
radius, K_N and selected vector. Physical additions occur at h32.

Reuse `plan_pair(model64, observer, hN, hD, ...)` unchanged for the common
parent. Its existing auxiliary random and first-margin-matched constructions
remain part of source eligibility. They do not enter the new comparison.
All source failures remain in the denominator and affect both arms.

One separately counted fresh HVP at hN measures the common intended donor
target `Q_N=zN^T H64(hN) zN` before either arm's construction. The common
final tolerance is `tau_Q=1e-8*max(1,abs(Q_N))`.

## Paired derivative-location arms

| Property | `reference_derivatives` | `native_derivatives` |
|---|---|---|
| Full J and margin H evaluation state | All-FP64 trajectory h64 | Actual reached state hN |
| Selected donor, natural dose and K | Common native parent | Same common parent |
| Physical addition base | h32 | h32 |
| Proposal algorithm | Frozen LGA-10B spectral search | Same algorithm |
| Rounded candidate filter | Frozen LGA-10C filter | Same filter |
| Final promoted validation | Common J64(hN), H64(hN), K_N | Same common validation |

The reference arm includes native donor and application repairs. It is a
new comparator with a declared derivative-location difference. The original
LGA-10C procedure remains immutable.

Each arm builds its own K-restricted full-J rowspace and the original
66-HVP spectral geometry. It also measures a separate dense 96-coordinate
Hessian at its derivative state, retaining raw columns and skew. Require
skew Frobenius norm <= `1e-8*max(1,||Hraw||F)` before using the symmetric
matrix for that arm's filter.

Preserve the original cosine grid `(-.85,-.65,-.35,0,.35,.65,.85)`, its
order, 32 attempts per sham per slice and 64 interpolation bisections.
Use the original `LGA10B` SHA namespace with seed 61067 and the same base,
goal, sham, slice and attempt keys in both arms. These keys generate common
96D ambient draws. Different derivative geometry produces different
projected starts and proposal vectors. Exact vector pairing is checked
within each geometry's own unfiltered-to-roundtrip replay.

Both filters form the actual native addition:

```
delta = (h32_pair + intended.float()).double() - h32_pair.double()
```

Each filter retains its own construction target and the original
`1e-8*max(1,abs(construction_target))` curvature tolerance. Record it
separately from the common final tolerance. The target normalization is
part of the unchanged algorithm evaluated at each derivative location.
All other candidate constraints and rejection precedence remain LGA-10C's.
No endpoint result feeds back into selection.

Archive every attempt, normalized start, physical diagnostic and rejection.
After selection, measure nine fresh HVPs on actual FP64 additions and nine
on actual FP32 additions at each arm's own derivative anchor. These
selection-space checks retain the original physical tolerances and the
`1e-10*max(1,abs(construction_target))` dense/fresh quadratic agreement
budget. Report `selection_validated` separately from the common endpoints.

## Execution contracts and final endpoints

The promoted contract executes the prefix and physical addition in FP32,
then lifts the resulting state exactly and executes the remaining two tied
blocks and head with the promoted FP64 model. The native contract executes
the continuation in FP32. Both use the same original FP32 parameter values;
the promoted copy contains their exact lifts.

### Promoted continuation readiness

For either selected set, use its exact physical deltas at h32. Common final
physical validation uses K_N, the common selected donor/radius, the energy
gradient at hN, J64(hN) and its K-restricted rowspace. Fresh HVPs at hN
come from the independent precision helper described below.

Require all of the following for `promoted_ready`:

- An eligible complete set of eight shams, with preserved actual additions.
- All final native-application physical constraints at hN: full-J, K, dose,
  current head, energy, dark angles and curvature parity under the common
  target. Use the existing FP32-application tolerances from LGA-10C.
- A valid native-state dense Hessian and dense/fresh quadratic disagreement
  <= `1e-10*max(1,abs(Q_N))` for every applied vector.
- The first-gradient precision witness and its independent fine-scale
  sham/donor curvature parity check, both defined below.

The final physical conditions are common to both arms. A stale-anchor
selection-space measurement does not silently become an additional final
condition. Every stated component remains separately visible.

### Native FP32 readiness

Run the unchanged LGA-10C `native_precision_check` on the same selected
sets, using the common Q_N tolerance. The native physical gate, promoted
curvature gate, both scalar-resolution checks and broad derivative-agreement
checks retain their original definitions and constants. Keep the exact
physical delta separate from its representable FP32 HVP probe.

Use the actual h64 and J64(h64) as the drift references for both arms. Thus
state drift diagnostics retain their original trajectory meaning even when
selection uses hN. The shared selected donor, native state and representable
probe imply a shared donor-only precision check across arms; remeasure it
on 61067 and report the resulting ceiling descriptively, with the measured
donor coverage. A donor left unmeasured by an incomplete set cannot be
counted as a measured pass or failure.

If reference geometry fails before J64(h64) is available while the native
arm has a complete set, attempt a diagnostic-only J64(h64) calculation
after both selections. Record this fallback and its ten first-derivative
calls. It supplies the same drift reference and cannot change selection.

Both selections finish before either precision endpoint is evaluated.
Evaluate both contracts for every complete eight-sham set, including a set
that failed its other endpoint or selection-space validation. Exceptions
are isolated by arm and endpoint. A complete set never loses its second
measurement because the first measurement raises. No partial set passes.

## Independent first-gradient witness

For each exact applied delta, set `u=delta/norm(delta)`. At hN, measure the
pair gradient of the legal signed margin using first derivatives at the
six displacements `+/- h*u`, with fixed physical step sizes
`h=(2^-8,2^-9,2^-10)`. The displacement norm uses these fixed steps and
does not scale with the donor dose.

Form the centered gradient estimates and adjacent Richardson estimates:

```
C_h = (g(+h*u)-g(-h*u))/(2*h)
R_coarse = (4*C_(2^-9)-C_(2^-8))/3
R_fine   = (4*C_(2^-10)-C_(2^-9))/3
Q_coarse = norm(delta)^2 * dot(u,R_coarse)
Q_fine   = norm(delta)^2 * dot(u,R_fine)
```

The helper measures its own nine fresh HVPs on the exact physical deltas;
its API accepts no cached HVP or dense Hessian. For every donor and sham,
require `abs(Q_coarse-Q_fresh)`, `abs(Q_fine-Q_fresh)` and
`abs(Q_coarse-Q_fine)` each <= `tau_Q/10`. Also require every fine-scale
sham-minus-donor parity <= tau_Q. A failed witness remains a final failure
with no retry signal.

Preserve all displaced first gradients, intended and realized probe
displacements, fresh HVPs and scalar estimates. The witness is numerical
consistency evidence. It provides no rigorous interval enclosure of the
mathematical Hessian. Its fixed steps and budget are tested with analytic
nonlinear functions and the exposed 61064 fixture before the implementation
lock; a biased HVP that agrees with its own dense matrix must be rejected.

## Counts, comparisons and evidence scope

Each execution contract has its own >=61/64 engineering threshold. The
primary paired contrast is `native_derivatives` minus `reference_derivatives`
in promoted continuation readiness. The corresponding native FP32 contrast
is secondary. Report every component, case, base and paired transition.
Keep the two execution-contract verdicts distinct.

The physical dose is the common native donor pair norm for each case. The
independent units are eight input bases; the 64 goals and 512 shams are
correlated observations. There is one initialization and zero newly trained
model roots. A causal mechanism floor and population matched-sham floor
remain uncalibrated. Causal power and population confirmation are outside
this engineering study; the paired differences are descriptive.

The exposed 61066 result informed this design. Seed 61067 remains unopened
until this protocol and the complete implementation are committed. A later
confirmation claim requires independent model/source replication and a
separately sealed task readout.

## Execution and preservation

Use CPU, one Torch thread, Python 3.11, Torch 2.7.1+cpu and NumPy 2.2.6.
Repeat the existing 61 LGA-10B/C tests and the new tests before source
generation. Before initializing seed 61067, require an exact clean checkout
of the implementation commit, verify commit membership and byte equality
of every imported source/protocol/test/workflow file, and record SHA256.

For 64 complete cases, budget 27,712 HVPs: 8,448 spectral, 12,288 dense,
2,304 selection-space validation, 64 common-target, 1,152 promoted witness
and 3,456 native endpoint calls. The witness also uses 6,912 first-gradient
probes. Preserve available purpose counts if failures interrupt a measurement.
If an endpoint helper raises before returning its counters, record
`call_counts_complete=false` and identify the failed endpoint; reported
totals are then lower bounds. An unavailable count is never a known zero.
Complete captures must match the full expected counts. Also retain
wall/CPU time, peak RSS, model/source hashes and environment versions.

The new raw format retains every logical array name while sharing exactly
equal NPY payloads through a content-addressed alias map. Equality includes
dtype, shape and value bytes. Repeated measurements retain separate
receipts and call counts. Verify the alias map, stored payload hashes and
semantic relationships through an independent NumPy auditor that imports
no model, matcher, precision oracle or Torch.

The first CI push captures the experiment once. PRs run tests only, and
report/result-only commits do not rerun source generation. Archive the
first result and all failed endpoints. Integrity failure fails CI; an
engineering threshold failure remains a preserved experimental outcome.

Science seeds 31059, 31064 and 31065 remain ungenerated. Trained weights,
earlier native cohorts, ARG-03-N, AMM and CQ stay closed. A promoted-contract
pass permits planning under that explicit execution contract. A subsequent
trained study still requires its own frozen protocol and source lock.
