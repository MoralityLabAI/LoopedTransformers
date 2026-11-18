# LOOPSCOPE-12 / LGA-09: full-ten-logit first-order-matched dark pair fibers

**Prospective registration, 18 November 2026, before LGA-09 implementation, source generation or model scoring. Commit this protocol FIRST.** LGA-08 original first-science [RESULTS](../loopscope11/RESULTS_FIRST_NATIVE_20261118.md) and its independent stdlib audit passed; its 31062 sources and all earlier 31060/31061 science are exposed and frozen. Never reinterpret an old donor case as independent confirmation.

## Scientific question

At native h2 of a fixed 44,001-parameter four-pass TRM, a legal donor difference on the predecessor/goal token pair lies partly in the **75-dimensional current-head/energy-tangent physical fiber** K. LGA-08 demonstrated 3/3 old teacher **development** gains +21.97/+27.05/+26.95 percentage points in correct donor targets over eight equal-dose same-K shams matched on **one** signed future donor-vs-original-class margin derivative. The task's ten-class argmax depends on other competing logits. This next study tests whether donor-specific transfer persists after matching the **ENTIRE ten-logit future first-order Jacobian response**.

Only the original three historically examined frozen teacher seeds 103,211,307 and original LGA-05 externally fitted shared energy are reused, for development evidence only. Any new scientific model-root confirmation requires its own independently trained root population and EvalMath root-level mechanism power/variation check.

## Frozen source, weights and labels

Original first-science teacher/SAE bundle SHA-256 `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`; train source manifest `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`. Original trained scalar E bundle `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`; fit manifest `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`. Use the old tensor-fingerprint-verifying loaders. Freeze teacher, E, model input processing, current linear ten-class head, source normalization, and original two future tied blocks. No retraining, fine tuning or reweighted observer.

First prospective source: `fixed_data(64,31063)`, exactly **64 independent IID 81-digit strings** x∈{1,...,9}^81 and y=one-right-cyclic-shift(x), generated only after the Stage A pretest source lock. Exclude exact strings from previously generated LGA seed rosters 31041/42/43/44/48/49/50/52/53/54/55/56/57/58, 31060 (64), 31061 (64), and 31062 (64). The distinct reserved seed 31059 stays unopened and excluded from any generation test. Each base has **eight** fixed goal positions `(0,10,20,30,40,50,60,70)`. For goal j set p=(j−1) mod 81 and construct one legal donor by changing x_p=d to d%9+1 only, shifting donor input for its independently defined correct target. Retain all donor structures/labels/source SHA and explicit base-ID clustering. **512 goal cases per historical teacher, 1,536 in all** are ITT rows even if geometry/sham matching degenerates.

## Pair K and full future readout Jacobian

Use the exact LGA-08 source-only 2×48 physical pair K projector: per-token rank-ten current head W, per-token 38D head null P_H; pair-head-null has dimension 76; intersect the single restricted full frozen E derivative row to obtain **dim K=75** when nondegenerate. Apply FP64 numeric tolerance rtol=1e−9 and atol=1e−12, and preserve physical Euclidean metric, full head and finite E violations.

At h2, before any donor-correctness or edited task scoring, define full native future goal output `L(h2)=head(F(F(h2)))_goal∈ℝ^10`. Obtain all **ten** class Jacobian rows `J_{c,:}=∂L_c/∂h2[p,j]` using genuine frozen native two-block autodiff. Project each row into the physical pair K: `B=J P_K∈ℝ^{10×96}`. Use a frozen full FP64 SVD and threshold above to obtain `rank(B)≤10` and its orthonormal right singular-row basis R; report the complete spectrum, rank/cutoff, output derivatives and physical closure. With dim K=75,

`dim(K∩ker B)≥75−rank(B)≥65`.

The donor source-only selected physical direction z is `rho U_K/||U_K||` at **natural pair rho=||h2_donor−h2_recipient||_pair** (unchanged from LGA-08). The entire ten-class first-order response is `t=Bz`. Let `z0=R Rᵀz` be its minimal-norm rowspace component, and `q∈K∩ker B` a random unit vector orthogonal to z0. Define

`z_fullJ_sham=z0+sqrt(rho²−||z0||²)q`.

Generate exactly **eight** SHA-seeded Gaussian q vectors independently, project them into K and off B's rowspace, normalize, and produce eight fixed full-J-matched shams. Each must have the SAME current-head/energy tangency, exact physical L2 rho, and **every one of ten signed first-order future digit-logit changes** `Bz_sham=Bz` (FP64 ≤1e−8 relative-to-scale, actual applied f32 ≤2e−4 absolute or 5e−5 relative). Do not match only donor-vs-original margin. The full future Jacobian is model-only and uses ALL ten classes, so its construction does not select the correct target class.

A source-goal with zero donor-K dose, K rank drift, rank/near-singularity failure, nonfinite gradients or `rho²−||z0||²` indistinguishable from zero is **INELIGIBLE_FULL_J_MATCH**. The candidate/full-J/random/signed-match trial contributes identity edit to ITT rather than reselection. Log every case and fail parent geometry/admission engineering gates if fewer than 95% sources per teacher qualify. No training/adaptation after outcomes are observed.

## Frozen arms and oracle controls

At each legal source case:
- `pair_donor_k`, source-only selected 75D projected donor at natural pair rho.
- `full_j_sham_0..7`, same-K same-L2 and exact all-ten-future-logit-Jacobian matched shams, **STRICT PRIMARY null**.
- `margin_sham_0..7`, original LGA-08 target-aware signed donor-minus-original future margin-gradient-matched null family, same K and rho, to reproduce the weaker conditional effect on this *new source*.
- `random_k_sham_0..7`, label-blind same K and rho.
- `pair_head_only` at rho, `natural_pair` at its natural pair dose, `goal_only_natural` and `predecessor_only_natural` at their own natural doses, `pair_quarter` at 0.25rho, and `full_donor_state` exact entire reached h2 donor positive control, distinct natural whole-state dose.

All selected, full-J and random K directions derive only model and legal source states; the margin-matched comparator is explicitly target-aware and uses source-determined legal class indices. Match no edited outcomes or final task labels when fitting/selecting directions. Every source planning decision, basis rank, matching residual, physical pair dose and edit FP32 SHA is frozen before true target correctness is scored.

## Native endpoints, independent units and gates

Run all frozen arms with identical **two actual tied-block continuations** from h2 to the original trained k4 readout, preserving all original 81 final predicted classes, ten goal logits, true donor target CE/margin and correctness, actual physical edit vectors, source-donor/full gradient fingerprints and finite external-energy/current-head changes. Full state donor replacement must exactly reproduce clean independently reached donor all-81×10 logits to max abs ≤1e−4.

**Primary registered ITT difference per case:** `I(selected legal donor correct)−median_8 I(full-J-matched sham correct)`. Aggregate the eight fixed goals within each independent base string, then compute teacher-mean and 2,000 SHA-keyed whole-64-base 95% percentile bootstrap CIs. Secondary registered comparisons are the same selected outcome vs median 8 signed-margin shams and vs median 8 random-K shams. Report all fixed comparisons regardless of result; each teacher is one previously exposed model. Tokens, shams and legal donor variants within a source are repeated measurements.

A teacher passes the **STRONG_FULL_J_INCREMENTAL** development gate only if all hold: ≥95% pair geometry/full-J sham source admissions, ≥80% jointly clean competent legal donor/recipient goals, whole-state full-donor exact-output parity and ≥95% correct among jointly competent, natural pair control ≥90% correct among jointly competent, and (a) selected minus full-J-matched correct-donor rate **≥+10 pp** with within-teacher whole-source bootstrap 95% lower >0, plus (b) selected minus signed-margin-matched ≥+10 pp and vs random-K ≥+15 pp with each bootstrap lower >0. **DEVELOPMENT_STRONG_FULL_J_PASS** only if at least 2/3 old teacher roots pass all gates; otherwise original registered verdict **DEVELOPMENT_STRONG_FULL_J_FAIL**. Never substitute weaker signed-only success as full-J success. A full-J null with signed-margin success is informative evidence that competitor-digit *first-order* sensitivity can account for the prior mechanism contrast.

## Stage chronology, independent audit and evidence scope

Stage A **before source 31063**: commit/run excluded-seed `61063` synthetic/algebraic tests with freshly initialized untrained 44K TinyRecursiveModel plus analytic linear and nonlinear heads. Verify K dimension 75, full-J rank/null dimension, all ten output derivative equality and exact physical dose under random shams, orthogonal channel gauge transport, degenerate rank/radius handling, real native 2-loop trajectory and protected seal source exclusions. Verify original trained artifact bytes, source tensor fingerprints, protocol/test/workflow code SHAs in a first uploaded immutable `source_lock.json` recording `new_test_generated=false`.

Stage B only on a unique **first source-release main-branch merge commit** with exact code lock (no PR/test source opening): check Stage A artifact, generate prospective 31063 once, run full native frozen study and archive every physical edited pair FP32 vector, source delta, E gradient and full-ten-logit Jacobian row, full true targets and original raw complete output and preregistered engineering verdict. Keep both PASS and FAIL raw receipts.

Stage C independent Python-standard-library-only audit: download FIRST frozen source lock and result, verify every source x/y/legal donor SHA, original 10×48 head and K, all actual edited pair vectors and full ten first-order matched derivatives, physical doses, complete logits/CE and full-donor positive-control parity, all original per-base bootstrap samples, fixed root gates and original verdict. The source-root selection and audit executable hashes are pinned before Stage B; no model/Torch imports in the auditor.

**Scientific inference: DEVELOPMENT_ONLY.** The original three old teachers are already exposed; even a 3/3 PASS over new source strings cannot establish generalization across new model initializations. For proper confirmation, first design a separately pretrained independent model-root training population and a new EvalMath mechanistic-resolution preflight. A nonparametric 95th-percentile/95%-confidence sham maximum bound requires ≥59 independent calibrated roots under iid assumptions and a separate sealed confirmation population. Preserve ARG-03-N, AMM-03/04/05, CQ-08 and all earlier LGA scientific first receipts; no protected inference or future work is authorized by this registration.
