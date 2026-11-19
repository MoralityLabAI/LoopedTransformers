# LGA-10D implementation freeze

19 November 2026, Asia/Singapore, before the first excluded 61067 model
initialization and input generation. Prospective protocol-only commit:
`aa0e397c89778ba8c023f7126e5454e71a7d4844`.

The immutable commit containing this file completes that protocol. The CI
push supplies its full `GITHUB_SHA`; the runner checks HEAD, tracked-file
membership and exact committed bytes before touching the source RNG. Its
receipt records SHA256 for the reused source closure, earlier frozen
protocols and tests, and every new source, test, protocol and workflow file.

## Geometry and execution contracts

One FP32 initialization and its exact FP64 copy define both arms. The
common donor comes from promoting the reached recipient and donor states
before subtraction. The common native parent supplies the selected vector,
natural dose and current-head/energy tangent K. Each arm applies at the
same FP32 reached state. Only the derivative location changes: the reference
arm uses the all-FP64 trajectory, and the native arm uses the lifted actual
FP32 trajectory.

Each arm retains the unchanged spectral construction and roundtrip filter,
including its own construction target and tolerance. The final promoted
contract uses the independently measured common native target and native
J/H geometry. It requires the final physical constraints, dense/fresh
agreement and an independent witness built from first derivatives. Each
selected set also receives the original LGA-10C native FP32 precision
checks. The separate endpoint names are `promoted_ready` and
`native_fp32_ready`; each has its own 61/64 engineering threshold.

Both selections finish before precision measurements. An exception in one
endpoint leaves the other measurement available. Complete sets retain both
measurements even when selection-space validation fails. The native drift
reference remains the actual all-FP64 trajectory and its Jacobian. A
diagnostic fallback may recover that Jacobian after selection if the
reference construction failed earlier.

The promoted witness measures nine fresh HVPs internally and 54 displaced
first gradients at fixed physical steps `(2^-8,2^-9,2^-10)`. Its two
Richardson curvature estimates, their mutual discrepancy and independent
fine-scale sham/donor parity retain every registered budget. The witness
provides numerical consistency evidence. It supplies no interval enclosure.

## Independent reconstruction and preservation

The NumPy auditor reuses the frozen LGA-10C geometry, proposal, selection
and native validators through explicit per-arm views. New checks rebuild
the common native fiber and both Richardson scales from captured arrays.
The auditor imports no model, matcher, precision oracle or Torch.

The archive maps logical names to content-addressed NPY members. Each
payload hash covers its stored bytes, including dtype and shape. The
reader checks the alias-map digest, exact member set and byte counts.
Identical payload storage leaves separate measurement receipts and call
counts intact. Source exceptions and failed constraints remain visible.

A complete run uses 27,712 HVPs and 6,912 displaced first-gradient probes.
Geometry receipts distinguish completed and attempted calls. If an
endpoint helper raises before returning counters, the record identifies
that endpoint, marks `call_counts_complete=false` and reports available
totals as lower bounds. Missing internal counters cannot become measured
zeros.

## Validation and first run

All 90 tests pass locally: the unchanged 27 LGA-10B tests and 34 LGA-10C
tests, plus 29 LGA-10D tests (nine geometry, nine precision, three archive
and eight receipt tests). The combined new suite passed in 31.037 seconds.
The new tests cover the native source geometry, derivative location, analytic
Richardson order and dose scaling, biased HVP detection, archive integrity,
full receipt reconstruction, endpoint isolation and the pre-generation
commit guard. Model-based development tests use only exposed seed 61064.

The local environment is Python 3.12.14, Torch 2.5.1+cpu and NumPy 2.3.5.
The first CI run repeats the full test suite with Python 3.11, Torch
2.7.1+cpu, NumPy 2.2.6 and one Torch thread before opening seed 61067.

The production command fixes seed 61067 and offers no source override. It
rejects existing capture paths. The first push runs the source experiment
once and preserves its raw archive, receipt, independent audit and hashes.
The push guard requires run attempt one and this implementation commit's
direct parent to equal the protocol-only commit above. Later code pushes
and workflow reruns cannot reopen the source through this workflow.
PRs run tests only. Result and report updates cannot trigger another source
capture. Integrity failure fails CI; an engineering threshold failure
remains a preserved experimental outcome.

The eight input bases are the independent source units. Their eight goals
are correlated measures under one untrained initialization. Protected
science seeds 31059, 31064 and 31065, trained weights, earlier native
cohorts, ARG-03-N, AMM and CQ stay closed. A subsequent trained study needs
its own prospective protocol and source lock.
