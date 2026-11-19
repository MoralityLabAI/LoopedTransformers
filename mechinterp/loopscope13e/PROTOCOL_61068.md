# LGA-10E/P1 prospective source-only native FP32 curvature matching

Registered 19 November 2026 (Asia/Manila). The first model and source seed
**61068** is excluded engineering data, and remains unopened until a separate
implementation freeze is committed and source-guarded CI tests pass.

## Fixed source and comparison

A single untrained 44,001-parameter FP32 TinyRecursiveModel with
torch.manual_seed(61068), an exactly lifted FP64 copy, and
fixed_data(8, 61068). Each independent input base has goals
(0,10,20,30,40,50,60,70). Donors use the legal predecessor mutation
d -> d%9+1; each source must pass the frozen LGA-10D parent source eligibility.
All 64 base/goal cases remain in the denominator.

Compare three explicit arms:
- \`frozen_baseline\`: LGA-10D reached-state spectral construction and frozen
  LGA-10C native_precision_check on exactly the original selected donor.
- \`fp32_sham_filter\`: original donor and frozen FP64 construction geometry,
  followed by native-FP32-HVP-based candidate rejection, without FP64
  precision-oracle feedback during the selection filter. The FP64 spectral
  geometry makes this a **mixed-precision construction with FP32 filtering**,
  not FP32-only construction.
- \`mixed_donor_repair\`: at most 17 source-preserving donor candidates including
  the original donor. An oracle-assisted precision filter may use FP64 HVP
  checks to select the donor and shams. This arm is clearly labelled
  **mixed-precision construction, native-FP32 execution**. The FP32 endpoint
  remains independently evaluated after selection.

All arms apply the same FP32 state addition and continue their tasks with the
unmodified FP32 model. The promoted model participates in diagnosis,
construction and final reference checking; no mixed-precision inference claim
is credited to native FP32.

## Original physical edit and donor repair

Let h32 be the actual two-block recipient state. Let hN=h32.double(), z the
native source's frozen LGA-10D selected first-order K-tangent donor, and rho
its fixed natural dose. Let R be the frozen J64(hN)-restricted row basis,
u=(z-RR^T z)/||z-RR^T z||, and V the orthonormal complement inside K and R^⊥
to u. Candidate 0 is z. For each of four deterministic ambient seeds
\`LGA10E|61068|base|goal|direction_index\`, project onto V and normalize to
unit w. For θ in (-0.12,-0.06,+0.06,+0.12) radians, in that fixed order,
construct donor

z'(θ)=RR^T z + r cos(θ) u + r sin(θ) w,

where r=||z-RR^T z||. This holds the K-projector, future J rowspace, first
margin derivative, and L2 dose at the *intended* level. Check:

* abs(||z'||-rho) <= 1e-7 max(1,rho);
* ||z'-z|| <= 0.2 rho, including original natural-source displacement;
* ||Kz'-z'|| <= 1e-7 max(1,rho);
* ||J64(hN)(z'-z)|| <= 1e-8 max(1,||J64(hN)z||);
* abs(Q64(z')-Q64(z)) <= τ=1e-8 max(1,|Q64(z)|);
* same signed future-margin derivative as z within the J64 bound;
* nonzero physically realized dark radius.

Reject candidates that fail any check. Only the \`mixed_donor_repair\` arm
may use the promoted HVP oracle to select a donor, testing the exact
FP32-applied realized delta and its representable probe. Retain all attempted
donors, selected index, geometry deviations, signed-margin checks and
rejection reasons. First accepted donor wins. Source eligibility cannot be
reclassified after selection.

## Sham generation and precision filtering

Use the frozen LGA-10B seven cosine slices (-.85,-.65,-.35,0,.35,.65,.85),
32 ambient attempts per sham per slice, the 64-step crossing solver and
eight shams. Use unchanged source and angle constraints. For the baseline,
run the frozen LGA-10D plan without any new filter.

For the sham-filter and mixed-donor-repair arms, replay the same spectral
source family using the **frozen validated dense FP64 Hessian at hN**.
The FP64 matrix is part of the construction geometry. A candidate must first
pass LGA-10C's native physical-addition gate. Next:
- \`fp32_sham_filter\` may use only fresh native FP32 HVP values on rounded
  candidate and donor for its additional acceptance rule; require native
  curvature sham-minus-donor <= τ. This is an FP32 arithmetic filter layered
  over mixed-precision geometry.
- \`mixed_donor_repair\` additionally requires fresh promoted FP64 HVPs on
  both exact physical delta and representable probe. Require promoted
  sham/donor curvature parity <= τ, native parity <= τ, paired-resolution
  mismatch <= τ and absolute probe-resolution <= τ. Precompute the selected
  donor's reference HVP before sham search; never reuse a final endpoint HVP.
  Donor selection itself must meet probe-resolution <= τ.

All selection oracles consume their own counted HVPs. A candidate that raises
or is nonfinite is rejected, with the attempted call recorded. Each loop is
bounded; incomplete sham sets fail the case. No endpoint verdict feeds back
into selection or increases a budget.

## Registered primary endpoint

\`repaired_donor_native_fp32_ready[b,g]\` is exactly the conjunction of:
1. Source and donor eligibility as above;
2. complete eight-sham physical selection within the registered budget;
3. \`native_precision_check(...).receipt["native_ready"]\` with the **unchanged
   frozen LGA-10C limits** and the original natural donor's Q64 target
   (not a retuned donor target).

Native Precision checks each of all nine applied vectors for native physical
constraints, promoted curvature parity, both curvature-resolution scalars,
HVP/J/forward derivative agreement and the dark-angle conditions. The
original LGA-10C q tolerance is τ=1e-8 max(1,|Q0|). All NaNs, incomplete
sets, caught errors, source failures, and endpoint exceptions score zero.

The 64-case endpoint **passes iff at least 61/64 succeed** in the repaired
arm. Treat eight goals from each input as correlated, report per-base totals
and paired results, and preserve all 64 in the denominator. The 61/64
threshold is an engineering pilot criterion, not a statistical population
claim. Report the frozen-baseline and fixed-donor sham-filter endpoints
separately, without sharing success counts. Also report strict source-direction
deviation, donor-only oracle resolution, native physical pass, and the
promoted FP64-physical check as separate diagnostics.

## CI freeze, analysis and preservation

Use Python 3.11, Torch 2.7.1+cpu, NumPy 2.2.6, one Torch thread, deterministic
algorithms. Before initializing seed 61068: require the exact implementation
commit to be directly after this protocol commit, require first push/attempt,
verify imported source/test/workflow byte identity against Git and all file
SHA256 hashes. CI first runs synthetic analytic/unit and exposed 61064
model fixtures and inherited LGA-10C/D tests, then generates 61068 **once**.
Preflight PRs and staging pushes run tests only. Later result commits and
reruns cannot reopen this first run.

Preserve an aliased-NPY original capture, complete endpoint and selection
receipts, rejected candidates, source/model provenance, CPU/RSS and HVP/gradient
counts, original SHA256SUMS, and an independent NumPy audit. The auditor
recomputes LGA-10C gate outcomes from the captured arrays, without importing
Torch, model, matcher or selection oracle. Scientific engineering failures
remain preserved CI successes if receipts and audit are internally correct.

Protected science seeds 31059, 31064 and 31065 remain unopened. Trained
models, task scores, ARG-03-N, AMM and CQ are excluded.
