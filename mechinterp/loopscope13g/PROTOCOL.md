# LOOPSCOPE-13G / LGA-10G: lattice-aware sham rescue on the native curvature fiber

**Prospective protocol, 19 November 2026 (Asia/Manila).** Register before generating any model or input from engineering source seed **61070**. Reuse the fixed LGA-10F frozen donor geometry and LGA-10C native FP32 endpoint. This paper tests a bounded correction of candidates that exhausted frozen sham search.

## Prior evidence, independence and scope

LGA-10F seed 61069 reached 50/64 with 19 genuinely changed donors. Under the same donor solver 3 cases exhausted all donor candidates and 11 cases admitted a donor but failed sham selection. Among those 11, 8 failed the first sham, 1 failed the second, and 2 failed the fourth. Retrospective inspection showed many physically admissible candidates whose native HVP scalar parity or same-probe resolution exceeded the registered 1e-8 threshold. This cohort is exposed and may inform the registered design, not count as a new confirmation.

Initialize exactly one FP32 untrained 44,001-parameter TinyRecursiveModel with torch.manual_seed(61070), its exact promoted FP64 copy, fixed_data(8,61070), and the frozen predecessor mutation. The eight independent bases have eight correlated goals (0,10,20,30,40,50,60,70). Retain the denominator of 64 cases. Science seeds 31059, 31064, 31065, trained weights, sealed tasks, ARG, AMM, CQ and causal correctness outcomes remain unopened.

## Paired arms and unchanged donor-selection rules

1. \`frozen_baseline\`: original LGA-10D reached-state spectral shams and unchanged LGA-10C native precision check.
2. \`fixed_donor_oracle_filter\`: original natural donor plus the LGA-10F FP32/FP64 filter on the frozen seven-slice spectral shams.
3. \`levelset_repair\`: the first admitted donor from LGA-10F's source-constrained curvature-level-set search (including natural candidate index 0), then the same LGA-10F sham filter.
4. **\`levelset_sham_repair\` (primary)**: the **identical selected donor** as arm 3, followed by the same original sham search, with a deterministic local curvature-level-set correction **only when that sham's frozen 7×32 proposal budget is exhausted**.

The donor selection is performed exactly once per case. Both donor arms use the same selected donor, original natural curvature target and current-head/energy tangent fiber. The new selection step may consult the FP64 dense Hessian plus fresh FP32 and FP64 HVPs. Its construction is **mixed precision**, while FP32 reached state, physical addition and subsequent continuation remain native FP32. Corrected candidate evaluations use separate selection-stage HVP calls, never final endpoint cached values.

## Frozen sham repair geometry, allowed candidate family

For an incomplete sham k, collect every finite **rejected original trial** in that sham's already-exhausted original fixed grid. The original trial is

    x = z0 + c q0 + r sqrt(1-c²) V y,

where z0=RRᵀ z is the donor's full-J row component, q0=z-z0, r=||q0||, c is its registered cosine slice and V is the frozen orthonormal 96×64 complement to q0 inside the tangent/full-J nullspace. y is its unit coordinate in V. Its original dense quadratic target is the selected donor's own frozen construction curvature.

Among original **physically admissible** candidates rejected by the native scalar filter, rank by the maximum of the four absolute scalar-gate residuals divided by the original tau=1e-8 max(1,|Q_original|): native sham/donor parity, promoted sham/donor parity, paired mismatch, and same-probe FP32/FP64 resolution. Ties are broken by original candidate index. Retain **at most three** parent trials per exhausted sham. If none qualifies, record the sham as failed.

For each ranked parent in increasing rank, construct two local correction directions j=0,1 with deterministic SHA256 seed namespace \`LGA10G|61070|base|goal|sham_index|parent_candidate_index|j\`. Let M be the validated native-state dense Hessian and let H(x)=xᵀMx. Express the source sham as normalized V-coordinate y; define tangential curvature gradient \`e=normalize((I-yyᵀ)∇_y H(x))\`. Draw a fresh normalized w orthogonal to both y and e, inside V-coordinate space. If e or w is degenerate, record explicit failure.

At each j test the fixed signed offsets

    s = (-0.010,-0.005,-0.0025,+0.0025,+0.005,+0.010).

Solve for t in [-0.08,+0.08] by a **32-step** sign-bracketed bisection around t=0, using

    y'(s,t)=(y + s w + t e)/sqrt(1+s²+t²),
    x'(s,t)=z0+c q0+r sqrt(1-c²) V y'(s,t),

subject to \`|H(x')-Q_construction|<=1e-10 max(1,|Q_construction|)\`. The first finite, bracketed root with valid scalar and physically applied geometry is eligible for filtering. Check physical LGA-10C constraints, original donor-sham angles, and the unchanged FP32/FP64 selection HVP limits. A candidate rejected in any stage remains rejected; the first passing corrected candidate is selected for sham k and original search resumes with sham k+1.

The fixed fallback budget is **3 original parents × 2 directions × 6 offsets = 36 additional corrected-candidate checks per exhausted sham**. The original frozen 7×32×8 proposals, 64-step original interpolation bisections and eight-sham limit remain unchanged. Extra HVPs are counted by purpose. Each failed-root and each evaluated candidate has a frozen index, source identity, root receipt, rounding result, signed residuals, HVP measurements and rejection reason. Preserve both intended and physically realized vectors; correction must be the one passed into the subsequent original native endpoint.

## Registered endpoint and scientific interpretation

For each of the 64 source cases, \`levelset_sham_repair.native_fp32_ready\` is one only when: original legal source passes; the frozen donor is admitted; all **eight** shams are complete inside the registered original+fallback budgets; and the unchanged \`native_precision_check(...).receipt["native_ready"]\` is true for all **nine** physically applied vectors against the **original natural donor's Q64 target**, including the full native physical K/J/dose/energy/head/dark-angle gate, promoted curvature parity, paired and probe scalar-resolution gates and HVP/J/forward derivative agreement. Incomplete, exception and source-ineligible cases score zero. The experiment's prospective **engineering pass threshold is >=61/64**.

Record separately:
- changed donor count and genuinely repaired-donor successes;
- number of cases where a local corrected sham was selected;
- paired \`levelset_sham_repair\` vs \`levelset_repair\`: both-pass, new-only, old-only, both-fail;
- failure stage, attempted HVPs, CPU time, peak RAM and source-case physical limits;
- case-level eight-base clustering. One untrained initialization cannot establish population or trained-model generalization.

Any local correction must preserve the original donor and source semantics. Do not change the native gate, loosen tolerance or reclassify postselection as an FP32-only proposal method.

## CI source freeze and evidence preservation

Commit this protocol alone. Implement and test on a staging branch using exposed seed **61064** exclusively: analytic quadratic controls, successful local curvature roots, unbracketed/degenerate roots, finite and casting failures, all original candidate budgets, exact corrected-sham recording, independent NumPy native-gate reconstruction and deliberately rehashed corruption. Preflight and PR checks execute tests only.

Create a single implementation commit that directly follows this protocol commit and whose tree is byte-identical to the passing preflight tree. The first push on the dedicated \`research/loopscope13g-61070-sham-source-20261119\` branch with run attempt one is the only authorized source generation, on GitHub Actions Ubuntu CPU, one Torch thread, Python 3.11, Torch 2.7.1+cpu, NumPy 2.2.6 and a 35-minute job timeout. Before RNG, verify exact clean checkout, commit ancestry and every imported source/protocol/test/workflow SHA-256 byte-for-byte. A failed first run remains the registered result and is **not rerun**. Later commits/PRs run tests only.

Archive all 64 cases, original donor and sham attempts, correction trials, failed source/endpoint exceptions, input and model hashes, exact resource and HVP call counts, original NPY arrays, independent NumPy reconstruction and SHA256SUMS. The independent auditor imports no Torch, source model, construction oracle or matchers, and validates every completed native LGA-10C endpoint and exact correction-parent geometry against stored arrays. Engineering success/failure must be reported separately from CI integrity success.
