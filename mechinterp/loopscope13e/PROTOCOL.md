# LOOPSCOPE-13E / LGA-10E: native FP32 error attribution and prospective repair

**Registered 19 November 2026 (Asia/Manila).** This is a preparation protocol and
source-lock plan. It authorizes retrospective replay of the already exposed 61067
archive. It does **not** authorize initialization of a fresh source, generation of
seed 61068, release of trained tensors, or task-correctness claims.

## Background and scope

The frozen LGA-10D experiment used one untrained 44,001-parameter model, eight
input bases and eight correlated goals per base. It established 64/64 promoted
FP64 continuation readiness using derivatives at the actual FP32 reached state;
the old derivative location passed 9/64. Under unchanged native FP32 contracts
both passed just 1/64. Its original 61067 source generated 27,712 HVPs and 6,912
independent first-gradient evaluations.

Twenty-one of 64 **donor-only** representable-probe resolution checks exceed
their 1e-8 tolerance. For the fixed donor, source, and oracle this gives a
hard descriptive ceiling of 43/64 before sham constraints. Editing only shams
cannot overcome a failing donor-only requirement. A 61/64 native-FP32 claim
requires either a different donor satisfying the original constraints or a
**separately named** revised contract, whose impact must be shown explicitly.
No retrospective edit may change the LGA-10D verdicts.

## Phase R: frozen original 61067 retrospective, permitted now

Read the immutable first-run archive from GitHub Actions run **44363670991**,
artifact `lga10d-excluded-61067-native-state-precision-feasibility`.
Pin `raw.npz` SHA-256:

`20f264a7fe0044ca2b142a5d1b70ffb27b6915a914bb91f96c30ccb5de65246d`

and `feasibility.json` SHA-256:

`223d7a247e1a4e581839017e6f44b1aad11a4ae20004e5de9757616a17a9a44f`.

Run the original independent LGA-10D NumPy auditor before computing any new
diagnosis. Preserve all 64 cases including source failures and both arms.
Use the aliased NPY arrays, retaining float64 dot products for diagnosis.
For each actual edit `a`, its representable FP32 probe `p`, FP32 HVP
`H32(p)`, and promoted HVPs `H64(p)` and `H64(a)`, define:

```
q_native = a · H32(p)
q32_probe = p · H32(p)
q64_probe = p · H64(p)
q64_actual = a · H64(a)

C_contraction = q_native - q32_probe
C_arithmetic  = q32_probe - q64_probe
C_probe       = q64_probe - q64_actual
D_total       = q_native - q64_actual
D_total = C_contraction + C_arithmetic + C_probe
```

Call the components respectively **left-contraction displacement**,
**same-representable-probe HVP arithmetic**, and **representable-probe
substitution**. They are a signed, exact scalar telescoping identity, not
causal effects. Decompose sham-minus-donor discrepancies by subtracting each
component's donor value from its sham value. Preserve both signed contributions
and their magnitudes. Rounding of the physical edit is recorded independently
as `a - intended`; the probe-displacement contribution may contain both
direction rounding and the HVP response to it.

Replay and verify all four archived scalars, actual FP32 addition, probe lift,
and stored resolution errors from raw arrays. Reject altered or missing data,
incorrect shapes, nonfinite values, inconsistent capture identities, unexpected
seed/contract metadata, and checksum mismatches. Record all 64 cases, nine edits
per arm, both native and reference derivative selections, and per-base results.

Report descriptive zeroing-one-component-at-a-time diagnostic contrasts only
as **arithmetic counterfactuals**, never as corrected model evaluations or
prospective pass counts. Phase R runs entirely on CPU with NumPy and no source
initialization or Torch model import. CI may retrieve the existing artifact
through `actions/download-artifact@v4` with pinned run-id and explicit
`actions: read` permission. The artifact expires; permanently preserve its
original ZIP outside the 90-day Actions retention window. The model/source
digest, all LGA-10D files and all registered verdicts remain unchanged.

## Phase P: fresh quantization-aware source study, implementation pending

Use a **new, independently frozen implementation and test commit** before
any fresh source generation. Reserve the excluded engineering seed **61068**;
do not open it in this preparation PR. Science seeds 31059, 31064, 31065,
trained checkpoints, ARC/ARG sealed sets, AMM and CQ remain closed.

A future implementation must explicitly construct and compare:

1. **Frozen native-state LGA-10D baseline:** same donor and dose, reached-state
   derivatives and original native FP32 check.
2. **Sham-only quantization-aware construction:** maintain the original donor;
   use native-FP32-representable proposals and an FP32-only construction
   objective. Retain the donor-only upper bound, and count failures against
   the whole source set.
3. **Source-constrained donor-plus-sham repair:** allow a new *physically
   realizable* donor within prospectively declared dose/angle/source limits.
   Preserve the original natural-donor direction as an audited reference and
   account for perturbation size. Original-donor and repaired-donor targets and
   verdicts must have different field names; the repaired target must meet the
   exact original source-margin sign and physical constraints. All source
   denominators stay fixed, and donor-only resolution is evaluated on the
   repaired donor.

Any FP64 computation used as a *selection* oracle disqualifies an arm from a
native-FP32-only construction claim. Independently assess each selected set
using fresh FP32 HVPs, promoted FP64 checks and the existing first-gradient
witness. The future source protocol must precommit precise source-preserving
repair limits, candidate budgets, stopping criteria, runtime/precision limits,
and the criteria under which a changed donor counts as the same source
intervention. No optimization that is evaluated on the original 61067 archive
can count as an independent confirmation.

The prospective engineering threshold remains **61/64 complete cases**
per registered arm, with all eight source bases retained and case-level
paired contrasts reported. That threshold is not attainable for the 61067
fixed donor under the old donor-only check (43/64 ceiling). Report any new
endpoint as its own contract, independently of the prior promoted FP64 result.
A later trained-model phase requires a separate protocol, independent seeds,
sealed task readout and causal controls.

## CI guardrails and deliverables

Preparation CI validates synthetic analytic edge cases, then fetches and
hash-checks the **existing** 61067 capture, runs the pre-existing independent
auditor, and writes an explicitly retrospective error-attribution JSON and
per-edit CSV. It must never initialize seed 61068 or train/load protected
weights. The workflow runs on pull request and explicit dispatch only;
there is no automatic fresh-source command.

The phase-R output is an audit aid for prospective hypothesis selection.
Do not use it to retune LGA-10D selection, rewrite the original capture, or
claim a trained-model mechanism result.
