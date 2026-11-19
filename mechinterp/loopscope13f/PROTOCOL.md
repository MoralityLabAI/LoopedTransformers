# LOOPSCOPE-13F / LGA-10F: source-constrained curvature-level-set donor repair

**Prospective preregistration, 19 November 2026 (Asia/Manila).**

## Motivation and source independence

The LGA-10E first 61068 run passed independent audit and failed its registered
native-FP32 61/64 endpoint (35/64). All 38 admitted donor candidates were the
unchanged natural donor; 416 fixed-angle alternatives failed the original
curvature preservation check. Its 26 donor search and 3 sham budget failures
are retained as the fixed negative result. This exposed cohort informs this
design and is not an independent confirmation.

Initialize exactly one untrained 44,001-parameter FP32 TinyRecursiveModel at
new excluded engineering seed **61069**. Use its exact FP64 parameter lift and
fixed_data(8,61069), retaining the legal predecessor mutation and goals
(0,10,20,30,40,50,60,70). The denominator is the full 64 case-by-goal
source roster, with eight independent bases and correlated goals. All source
failures, incomplete sets and numerical exceptions count zero.

All experiments run on GitHub-hosted CPU Actions. Trained tensors, new
science seeds (31059, 31064, 31065), task correctness, ARG-03-N, AMM and CQ
remain closed.

## Frozen geometry and physical source semantics

At actual FP32 reached state h32 after two tied blocks, let hN=h32.double().
The exact lifted donor difference is dN, radius rho=||dN||, selected donor
z = rho K dN/||K dN||, using frozen LGA-10D source code. Let R be the full
native-state J64(hN)-restricted row basis. Define:

    z0 = R R^T z
    q0 = z - z0
    r  = ||q0||
    u  = q0/r
    V  = 96×64 orthonormal complement inside K∩ker(J64) orthogonal to u.

H is the validated symmetric dense 96×96 native-state signed-margin Hessian
from the frozen LGA-10D geometry; its skew gate and SHA file provenance
remain unchanged. Define Q(x) = x^T H x.

The natural donor is candidate zero. Compute curvature tangential gradient

    g = V V^T (H z)
    e = g/||g||

and for each of four deterministic ambient vectors (stable
\`LGA10F|61069|base|goal|j\`), construct unit w in span(V) after removing its
projection onto e. Skip a degenerate g or w with explicit failure receipts.
For each j=0..3 and s in
(-.12,-.06,-.03,-.015,+.015,+.03,+.06,+.12), in that order, seek t in
[-.12,+.12] such that

    z'(s,t) = z0 + r*(u+s*w+t*e)/sqrt(1+s²+t²)
    Q(z'(s,t)) = Q(z).

Use a fixed 48-bisection bracket after checking the two endpoints and their
signs. Reject an unbracketed root; do not change s or t bounds during the
search. Reject any candidate with |Q(z')-Q(z)| above 1e-11 times max(1,|Q(z)|).
The root is numerical, not an interval certificate. The admitted curvature
level-set candidate must satisfy:

* distance ||z'-z|| <= 0.20 rho;
* original dose ||z'|| = rho within 1e-7 max(1,rho);
* original tangent ||Kz'-z'|| <= 1e-7 max(1,rho);
* signed-margin derivative preserved:
  ||J64(hN)(z'-z)|| <= 1e-8 max(1,||J64(hN)z||);
* original Q target preserved under **physical FP32 edit**:
  |Q64(delta_z')-Q_original| <= tau, where
  Q_original = z^T HVP64(hN,z) and
  tau = 1e-8 max(1,abs(Q_original));
* original donor probe scalar resolution:
  |p^T HVP32(h32,p) - p^T HVP64(hN,p)| <= tau,
  with delta=(h32+float32(z')).double()-h32.double() and
  p=float32(delta).double();
* fresh HVP64 vector agreement with HVP32 <= 1e-6 max(1,||HVP64(p)||);
* nonzero physically realized donor and dark radius.

Selection may use the validated dense H64 to precheck physical curvature;
fresh FP64 and FP32 HVPs must validate the chosen candidate separately.
Do not retune the original Q target. Preserve the exact natural donor as
candidate zero, all 32 level-set proposals, root failures, raw donor vectors,
physical rounding, HVP receipts and rejection order. First accepted donor
wins; the candidate search is bounded and case-independent.

## Three prospective paired arms

1. \`frozen_baseline\`: unchanged native-state LGA-10D spectral construction
   and independent native_precision_check with the original natural donor.
2. \`fixed_donor_oracle_filter\`: natural donor, source-constrained spectral
   shams with FP32+FP64 precision-aware acceptance. This uses FP64 geometry
   and FP64 construction oracles. It measures an unmodified donor with the
   same sham filter used in the repaired arm.
3. \`levelset_repair\`: candidate-zero natural donor or the first admissible
   curvature-level-set donor, followed by the same source-constrained sham
   filter as arm 2.

Both new arms are **mixed-precision construction, native-FP32 execution**.
The frozen LGA-10C gate is a separately computed endpoint oracle after
selection and is not reused to choose candidates.

Sham search: original seven fixed cosine slices
(-.85,-.65,-.35,0,.35,.65,.85), 32 attempts per sham per slice, 64
interpolation bisections, eight shams in registered order. Use the source
and physical admissibility limits from LGA-10C/B and the identical
\`CurvatureSelectionFilter(mode="mixed")\` as used in LGA-10E. The selected
donor and each sham are physically applied in FP32. A failed donor or
incomplete sham set causes a case failure. All attempted HVPs and proposal
rejections stay in the capture.

## Registered 61/64 endpoint

Primary endpoint is \`levelset_repair.native_fp32_ready\`. A case is 1 iff
(a) frozen source parent eligible; (b) donor passes source/tangent/physical
constraints and the original curvature target; (c) exactly eight shams pass
the registered selection budget; (d) the unchanged LGA-10C
\`native_precision_check(...).receipt["native_ready"]\` is true for all nine
physically applied vectors with original \`target_curvature=Q_original\`.

This means all native physical gates, promoted sham-donor curvature parity,
both native/promoted scalar-resolution tests, and HVP/J/forward agreement
must pass. Any numerical exception, rejected donor, incomplete plan, or
failed component is 0 in the fixed denominator. The experiment passes iff
sum_{b=0..7,g∈G} I_bg >= 61. Preserve per-base counts and paired transitions
against arm 2, with exact component failures. The 61/64 condition is an
engineering pilot and does not establish population success.

Declare separately:
- count of **repaired donors** (selected candidate index > 0);
- number of cases rescued by such donors relative to fixed-donor oracle arm;
- donor-only floor and numerical rejection reasons;
- promoted-physical-only success and any other changed execution contract.

A 61/64 overall pass with zero repaired donors cannot be attributed to
donor repair.

## Freeze, GitHub Actions and audit

First commit this prospective protocol alone. Build preflight
implementation/tests on a staging branch, using exposed seed 61064 only.
Preflight test: exact level-set geometry, deliberate unbracketed roots, zero
tangent gradient, source violations, analytic symmetric Hessian, corrupted
raw receipts, and one exposed-model case reconstructed by the independently
implemented NumPy physical/native gate.

When preflight CI passes, create an immutable implementation commit whose
**direct parent is this protocol commit** and whose tree exactly equals
the preflight-tested source tree. The source-branch first push and run
attempt one are the sole authorization for the fresh 61069 capture.
A failed first source must be preserved; the source is never regenerated.
PRs and analysis-only later pushes run tests only. Use Python 3.11, Torch
2.7.1+cpu, NumPy 2.2.6, one thread, CPU, pinned source hashes and clean
worktree, with a 35-minute job limit.

Archive full 64-case JSON/NPY receipts with SHA-256, every mathematical
candidate (including failed roots where a vector exists), raw probe/HVP
measurements, per-purpose counts, resource accounting and failure
precedence. An independent NumPy auditor must rebuild each completed
nine-edit LGA-10C native gate, the 64-case totals and the donor level-set
source constraints, without importing Torch, model or proposal algorithms.

CI integrity success is separate from the registered engineering threshold
failure. Results are analyzed only after frozen source generation and
independent audit pass. An independent trained-model confirmation, if
undertaken, requires its own prospective lock and sealed task-correctness
readout.
