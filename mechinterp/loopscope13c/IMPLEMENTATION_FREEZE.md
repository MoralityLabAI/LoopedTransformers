# LGA-10C implementation freeze

18 November 2026, before the first excluded 61066 initialization and input
generation. Protocol-only commit:
`bcb8def743ab405267ebc5080671eab90b8bfbc2`.

The immutable implementation commit containing this file completes that
protocol. The CI push passes its full `GITHUB_SHA` to the runner, which
checks HEAD, tracked source paths and clean file contents before seeding
the model or generating inputs. The receipt records SHA256 for the complete
imported source closure, both protocols, reused and new tests, and workflow.

## Implementation decisions

The baseline imports the unchanged LGA-10B matcher. Both arms consume its
original 66 HVPs, basis, slice extrema and exact proposal seed namespace.
The new matcher reads that captured geometry without mutating it. It calls
the same interpolation routine and follows the same intended rejection
precedence, evaluating physical diagnostics for every numerical proposal.

The dense primary Hessian uses 96 further coordinate HVPs. Its raw skew
gate precedes admission. The new physical filter requires the entire set
of already accepted shams plus the proposed sham to pass the unchanged
FP32 physical checks. Accepted controls receive fresh directional HVP
validation in both application dtypes. Dense and fresh quadratic values
must agree within the registered `1e-10*max(1,abs(Qtarget))` bound.

Both selections finish before either native precision receipt is examined.
The native oracle checks all available complete eight-sham sets, including
sets that fail primary admission. It preserves exact FP32 additions,
promoted physical deltas, representable probes, both derivative dtypes,
state drift and all four native gates. Each case's overall readiness is
the conjunction of primary admission and native readiness.

The independent auditor imports NumPy and the unchanged LGA-10B numeric
receipt checks. It imports no Torch, model, matcher or precision oracle.
It reconstructs the physical constraints, candidate paths, proposal keys,
native comparisons and paired aggregates from raw arrays. Deliberate
synthetic receipt corruptions must fail before any excluded data are read.

## Validation and execution

The test stage includes the original LGA-10B sphere, spectral and HVP suite,
plus source-free matcher tests, common-state precision tests and a complete
synthetic paired receipt. Any model-based fixture uses exposed seed 61064.
The first CI run repeats these tests with Python 3.11, Torch 2.7.1+cpu,
NumPy 2.2.6 and one Torch thread before opening seed 61066.

Local pre-freeze validation passed all 61 tests: the unchanged 27 LGA-10B
tests and 34 LGA-10C tests (12 precision, 15 matcher, 7 receipt and fault
tests). That environment used Python 3.12.14, Torch 2.5.1+cpu and NumPy
2.3.5. The source-free paired receipt covers all 64 base/goal identities,
and rehashed raw-vector and native-HVP corruption is detected. Independent
reviews of the geometry, precision oracle and runner/auditor integration
closed the identified defects before this implementation commit.

The production command fixes seed 61066 and offers no source override.
It rejects existing output paths. Every source case stays in the receipt,
including search, geometry, physical or native failures. A failed
engineering threshold is an experimental outcome. Failed integrity checks
fail CI. The workflow uploads the original capture and hashes after either
outcome and runs only tests on pull requests. Report and result-only
commits do not trigger another source capture.

Protected science seeds 31059, 31064 and 31065, trained weights and prior
native cohorts stay closed. Source-only success does not release them.
