# LOOPSCOPE-13B / LGA-10B: spectral feasibility of curvature-matched dark-fiber controls

**Prospective source-only registration, 18 November 2026, before implementing this replacement matcher or generating source seed 31065.** The independent prior LGA-09 full-ten-Jacobian first-science result passed 3/3 exposed frozen 44K TRM teachers. Its original model, source and audit receipts remain immutable. Distinct original LGA-10 source-only excluded-seed feasibility **FAILED**: 28/64 (43.75%) cases admitted all eight angularly separated curvature-matched shams versus 95% required. That failed random-plane algorithm and its protected unused seed 31064 remain archived unchanged.

## Question

Does the signed future donor-vs-original margin Hessian level set contain **geometrically separated** same-dose, same full-ten-Jacobian-response controls inside the 65D first-order future-dark physical pair subspace, even when a seeded random-plane search fails?

This stage distinguishes a stochastic sampler miss from a **curvature-level-set nonexistence at fixed angular separation**. Source-only feasibility does not establish a trained recurrent causal mechanism.

## Frozen original physical model and future source

Freeze exact LGA-02 44,001-parameter teacher bundle SHA256 `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`, manifest `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`, LGA-05 learned potential SHA256 `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`, potential manifest `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`. Teacher seeds 103/211/307 are historically exposed, for future development evidence only. No teacher or scalar observer refitting.

If and only if the Stage 0 source-only feasibility passes, reserve a separately first-generated future native cohort `fixed_data(64,31065)`, disjoint from all previously GENERATED LGA IID sources including 31060,31061,31062,31063 and historical 31041..58. Earlier seed 31064 remains protected and UNUSED, as does separately reserved 31059. Eight fixed goal positions (0,10,20,30,40,50,60,70), each with one legal predecessor digit donor mutation `d→d%9+1` at input `p=(j−1)%81`; donor truth comes from its right-cyclic-shift source law, rather than edited model output. 64 independent sources/model root with correlated eight goal variants. Source-specific effects remain descriptive within the three exposed old model roots.

## Physical first- and second-order constraints

At the reached state h2, physical edit support is two 48D token channels (predecessor p and goal j), 96D real state. The current head-row null plus global frozen E differential gives K (generic 75D). The entire ten-class future-logit Jacobian `B:D(h2[S])L_4|K→ℝ¹⁰` has historical numerical rank ten and yields a 65D **first-order future-silent** space N=K∩ker B.

Let source-selected legal donor edit `z=z0+q0`, where `z0` lies in the B-rowspace and `q0∈N` with `r=||q0||`. The natural pair physical L2 radius is `rho=||z||`. Every admissible strict sham keeps z0, norm r of its N component, and both K constraints; thus all ten linearized future class logits and the full physical dose match exactly.

Fix source-derived donor/original classes `d,o`, and `m(h2)=logit_4(d)−logit_4(o)` under the original two remaining tied native blocks. Its clean-state Hessian `H_m` supplies the signed second-order scalar `Q(z)=zᵀH_m z` (half of Q is the Taylor quadratic coefficient). This oracle-matched curvature control uses only the source law and frozen model derivatives, never any edited donor-correctness outcome.

## Global constrained spectral matcher

For a prescribed cosine `c∈[-0.85,0.85]`, parameterize every separated future-dark component by `q=r(cu+s v)`, where `u=q0/r`, `s=sqrt(1−c²)`, `v∈N∩u⊥`, `||v||=1`. The complement has dimension 64 when N has dimension 65. Define `a_c=z0+r c u`. Then

```
Q_c(v) = a_cᵀH_m a_c
       + 2 r s (H_m a_c)ᵀ v
       + r² s² vᵀ H_m v .
```

Build the exact FP64 orthonormal 96×64 basis V of N∩u⊥ by rank-revealing projector eigen-decomposition; fix eigenvalue cutoff and deterministic basis sign convention. Compute `VᵀH_m V` by exactly 64 Hessian-vector products, source-only before any outcome scoring. Symmetrize its numerical roundoff and record raw spectral eigenvalues. Two more HVPs for z0 and q0 supply the affine linear terms. The original runtime, RAM and HVP counts must be recorded.

Freeze cosine grid **c = (-0.85,-0.65,-0.35,0,0.35,0.65,0.85)** in that exact order. For each c, solve the sphere-constrained scalar quadratic minimization and maximization deterministically using the eigensystem of the 64×64 projected Hessian and a frozen **64-iteration secular-equation bisection** for the extremal Lagrange multipliers; handle hard-case zero linear terms by selecting the extremal eigenspace with deterministic sign. Report the proven numerical interval [Q_min(c),Q_max(c)] and an independent residual check for each extremizer.

If target `Q(z)` lies within the attained numerical min/max interval at any c, construct eight independent shams using a SHA-seeded random v on the same 64D sphere and a fixed interpolation from that v to the **opposite curvature extremizer** (min when v exceeds Qtarget, max when it falls below). Use exact spherical interpolation and 64 bisection iterations on the scalar Q to find the first curvature-equal level-set crossing. The independent seed includes teacher/root, source base, goal, sham index and attempt. Try at most **32** distinct random-v initializations/sham/c value. Require physically distinct q orientations among the eight shams (pairwise cosine ≤0.98), source donor angle `|c|≤0.85`, complete K and full-ten-J response parity, exact natural rho and curvature mismatch `≤1e−8 max(1,|Qtarget|)`. Any curvature extremal bound or mismatch failure produces an explicit admission failure, not an adapted threshold.

Record whether failures came from (a) no accessible Qtarget interval at any c, (b) solver residual/hard case, (c) insufficient eight distinct shams, or (d) dtype/FP32 native edit accuracy. The distinction between **attainable curvature range** and random sampler coverage is the primary source-only diagnostic.

## Source-only pilot and release rules

Before any trained weights or 31065 science, run source-only CPU algebra with an artificial symmetric quadratic form (known feasible and infeasible targets), then the untrained 44K recurrent architecture on excluded **61065**, eight independent base inputs × eight legal donor goals. Use its real two-tied-block future signed margin HVP and an artificial fixed quadratic E. Check source legality and full native HVP parity via central finite-difference gradients, spectra, trust-region extrema, exact curvature/noise, full ten-J/dose constraints, orthogonal gauge transport and deterministic seeds. Verify the captured feasibility JSON hash in CI; preserve all 64 outcomes including failures.

**Source-only engineering PASS** requires ≥**61/64** (95.3125%) of the 64 excluded untrained source-goal cases construct all eight separated curvature-level-set controls without numerical constraint violations. Otherwise source-only **FAIL**, hold all pretrained-native source access and publish original failed pilot. This fixed 95% source-only gate precedes any release; if spectral-level-set nonexistence dominates, record the bound rather than silently relaxing angular diversity.

If the source-only gate passes, separately freeze a complete first-native protocol and model/implementation/test/workflow SHA bundle **before** training-data or first `31065` generation, verify original teacher/potential tensors, and upload the pretest source lock with `new_test_generated=false`. Only an explicitly unique first merge-commit release may generate the 64 new prospective sources. First native causal arms must include selected donor z, eight spectral curvature-matched shams, eight original full-J-matched random shams, head-only/natural pair/full-state positive controls, and complete 81 class predictions, ten goal logits, actual raw physical vectors and original 2,000 whole-source bootstrap decisions. The registered strict development effect requirement remains **+10 percentage points selected over median curvature-matched shams**, plus +10 over full-J controls and competence gates, in ≥2/3 old teachers, with positive within-source bootstrap lower bounds. No result from old 31063 can count as new data.

All source-level conclusions remain **DEVELOPMENT_ONLY**. A scientific causal claim beyond this tiny source-specific cyclic-copy model requires newly trained independent teacher roots, EvalMath root variation/power preflight, ≥59 independent sham-calibration roots for a 95/95 maximum tolerance guarantee when claimed, and separately protected confirmation. ARG-03-N, AMM, CQ and all earlier LGA cohorts remain closed.
