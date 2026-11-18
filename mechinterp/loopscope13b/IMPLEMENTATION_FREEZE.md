# LGA-10B implementation freeze

18 November 2026. This addendum fixes implementation details before the first
excluded-source 61065 run. The original `PROTOCOL.md` remains unchanged. The
first source-only workflow commit containing this file, matcher, runner, tests,
auditor and workflow is the implementation lock. Its complete commit SHA and
the SHA256 of each imported source file are captured in the pilot receipt.

## Scope and units

The pilot initializes one untrained 44,001-parameter recurrent model with seed
61065 and generates `fixed_data(8,61065)`. Eight registered goals share each
base input. The experiment has eight independent input bases, 64 dependent
source-goal cases and one untrained model initialization. The observer is the
fixed artificial quadratic energy from LGA-10. All model computations and
derivatives use FP64. The separate edit-addition check below uses FP32 values.

Source-only engineering PASS requires at least 61 of 64 cases to supply all
eight admissible shams. A case must pass both FP64 geometry and FP32 addition
accuracy. This is a fixed engineering threshold; it supplies no binomial
confidence guarantee or causal effect estimate. There are no trained model
roots or task-correctness outcomes in this run. Per-base admission counts and
all failed cases are retained.

Science seeds 31065, 31064 and 31059 remain ungenerated. Trained bundles,
ARG-03-N, AMM, CQ and earlier native cohorts remain closed. A passing pilot
still requires a separately frozen first-native protocol and source lock.

## Mathematical scope corrections

1. The seven registered cosine slices remain exactly
   `(-0.85,-0.65,-0.35,0,0.35,0.65,0.85)`. A valid global minimum and maximum
   exhaust each individual slice. Failure on all seven is recorded as
   `GRID_NO_ATTAINABLE_TARGET`; it leaves intervening angles unresolved.
   For example, with `z0=e0`, `q0=e1`, `H01=H10=-0.6`, `H11=1` and all other
   Hessian entries zero, `Q_c=c*c-1.2*c`. The donor target is -0.2. The allowed
   angle c=0.2 attains it, while every registered slice misses it.
   A negative grid result requires all conservative outer intervals to
   exclude the target at the frozen tolerance. A target that falls between
   an attained endpoint and its outer bound yields
   `SPHERE_INTERVAL_UNRESOLVED`.
2. Ordinary sign bisection on a spherical path finds a bracketed crossing.
   Several crossings can occur along one path. Their order has no role in
   this pilot's curvature-equality test.
3. Sphere minimization uses an equality constraint. For
   `f(v)=v^T A v+2 b^T v`, a global minimum obeys
   `(A-lambda I)v=-b`, `||v||=1` and `A-lambda I` positive semidefinite.
   Lambda may have either sign. The hard case includes a nonzero b orthogonal
   to the entire lowest eigenspace; its complementary solution is completed
   to unit norm inside that eigenspace. Maximum uses the minimum of `(-A,-b)`.
4. All spectral bounds and certificates use ordinary FP64 arithmetic.
   Stored residuals and dual bounds quantify numerical consistency. Formal
   interval arithmetic is outside this implementation.

The sphere characterization follows W. W. Hager, *Minimizing a Quadratic over
a Sphere*, SIAM Journal on Optimization 12(1), 188–208 (2001), Lemmas 2.1–2.2:
<https://people.clas.ufl.edu/hager/files/sphere.pdf>.

## Frozen construction

The physical K projector is formed by applying the existing projection to
the 96 coordinate vectors. The complement projector is
`P_K-R R^T-u u^T`, symmetrized before `torch.linalg.eigh`. Eigenvalues greater
than 0.5 select the complement. The projector spectrum must be within 1e-8
of {0,1}; K idempotence, K symmetry, row-basis orthogonality and complement
residuals have absolute Frobenius or Euclidean tolerance 1e-8. Each basis
column's largest-magnitude coordinate is made positive, breaking equal
coordinate magnitudes by first index. Native cases require dimension 64.

There are exactly 66 construction HVPs for each complete native geometry:
64 complement basis vectors, z0 and q0. The projected Hessian is symmetrized.
Its unsymmetrized residual and cross-HVP symmetry are checked against
`1e-8*max(1,||M||F,||H z0||,||H q0||,||H V||F)`.

All seven extrema pairs are computed and recorded before drawing shams.
The sphere solver scales the problem and uses 64 secular bisections in the
logarithm of the positive pole offset, including its documented hard-case
tolerances and residual checks.
The complete solver source and its tests belong to this lock.

For each sham index, visit cosine slices in registered order. At each
attainable slice, try attempts 0 through 31. Each ambient 96D standard-normal
draw has a generator seeded by the first eight big-endian bytes of SHA256
of `LGA10B|ambient|teacher_seed|base_index|goal|sham_index|slice_index|attempt`,
masked to 63 bits. Project into the complement and normalize. Interpolate
along the shortest spherical path toward the opposite extremizer. An
antipodal tie uses the first least-magnitude coordinate of the starting
vector to form a perpendicular tangent. Bisection has 64 iterations, with
endpoint acceptance at the frozen curvature tolerance. A diversity rejection
consumes one attempt. Search continues to the next registered slice after
32 misses. Accepted shams may therefore occupy different registered slices.

The eight shams are separately seeded and jointly filtered for diversity.
Their statistical dependence follows from the common recipient geometry and
the rejection rule. Same-environment replay must reproduce the vectors.
Orthogonal gauge tests compare transported physical witnesses, extrema and
constraint residuals. Equal RNG seeds in independently diagonalized gauges
need not produce identical physical vectors within repeated eigenspaces.

## Numerical gates fixed before the pilot

Let rho be the intended selected norm, r its dark-component norm, W the
current head matrix, g the energy gradient, J the clean full ten-logit
Jacobian, and `Qtarget` the intended selected curvature. Curvature, J and
source-angle parity use each dtype's physically applied selected edit as
their reference. Dose is checked against the intended natural rho. All
curvatures use the same FP64 clean native Hessian.

| Check | FP64 | FP32 addition |
|---|---|---|
| Curvature difference | `1e-8*max(1,abs(Qtarget))` | same |
| Full-J response difference | `1e-8*max(1,||J selected_actual||)` | coefficient `1e-6` |
| K residual and dose error | `1e-7*max(1,rho)` | coefficient `1e-6` |
| Head residual | `1e-7*max(1,||W||2*rho)` | coefficient `1e-6` |
| Energy differential residual | `1e-7*max(1,||g||*rho)` | coefficient `1e-6` |
| Source absolute cosine upper bound | `0.85 + 1e-12` | `0.85 + 2e-6` |
| Pairwise signed cosine upper bound | `0.98 + 1e-12` | `0.98 + 2e-6` |

Planned FP64 vectors also retain the direct rowspace, K and dose tests in
the matcher. Actual edits are evaluated after adding them to the reached
two-token state and subtracting that state. The FP32 expression is exactly
`(h.float()+edit.float()).double()-h.float().double()`. Dark angles use
normalized actual dark components obtained with `P_N=P_K-R R^T`.
A zero actual dark norm fails. Raw mismatches, dark norms and
the ratio r/rho remain visible. The angle allowance is fixed, including when
r/rho is small.

Fresh native HVPs independently evaluate every physically applied selected
and sham vector in both dtypes. A complete case uses 18 validation HVPs in
addition to its 66 construction HVPs. Intended-to-applied selected curvature
drift is reported. This check evaluates FP32 addition under the FP64 Hessian;
future native FP32 model derivatives require their own accuracy gate.

## Tests, capture and independent audit

Before first generation, run synthetic feasible and infeasible quadratic
tests, general sphere hard cases, scaling, gauge transport, deterministic
replay, the seven-grid counterexample and central finite differences of the
native HVP. Native derivative tests use the already excluded seed 61064.

The first pilot runs in GitHub Actions with Python 3.11, CPU Torch 2.7.1 and
NumPy 2.2.6. Record exact package versions, platform, wall time, process CPU
time, peak resident memory, code hashes, source hash and commit SHA. The
receipt preserves each of the 64 cases, seven attained intervals where
available, attempt history, failure category, FP64 result and FP32 result.
Raw numeric arrays use a compressed NPZ with pickle disabled on read.

An independently implemented NumPy auditor reads the JSON and NPZ without
importing the matcher or sphere solver. It reconstructs the constraints,
checks stored extremizers and verifies native HVP parity. SHA256 receipts
cover all output files. CI archives outcomes even when the engineering gate
fails; an admission failure is an experimental result. A failed integrity
audit fails CI. The result report must identify the first run and preserve
the frozen source-only outcome before proposing any further revision.
