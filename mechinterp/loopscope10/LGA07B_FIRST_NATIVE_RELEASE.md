# LGA-07B / LOOPSCOPE-10B: first native execution lock

**Prospective execution contract, committed BEFORE native evaluation code or opening 31060.**
This document refines the already frozen [DONOR_PROTOCOL.md](DONOR_PROTOCOL.md) and preserves all its original negative/positive verdicts and the distinct LGA-07 seed 31059. The three teachers 103, 211, 307 are historical, exposed model roots; this release is **development-only** and cannot be called scientific confirmation.

## Original model and source lock

Exact trained artifacts (from original CI):
- LGA-02 teacher/SAE `bundle.pt` SHA256 `d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266`
- LGA-02 `train.json` SHA256 `7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d`
- LGA-05 `potentials.pt` SHA256 `24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53`
- LGA-05 `fit.json` SHA256 `69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d`.

**Stage A:** run source-independent tests on 61060 and the already registered 61059 geometry; fetch and SHA-check original artifacts and run original LGA-02/05 canonical tensor manifest loaders. Emit an immutable `source_lock.json` containing the bytes SHA of all four, first registration and this execution protocol SHA, code SHA, teacher list, seed 31060 reserved, `new_test_generated=false`, `test_labels_read=false`. Upload this FIRST. No call to `fixed_data(64,31060)` until Stage B receives and checks this source-lock artifact. The PR source-only workflow must never execute Stage A/B against the protected sources.

**Stage B:** exactly 64 independent IID 81-token strings generated with `fixed_data(64,31060)`, reject source collision with the earlier LGA-01..06 IID roster and every duplicate within the new IID base. Eight fixed goals `(0,10,20,30,40,50,60,70)`. One deterministic legal donor per base×goal, changing only the predecessor from d to d%9+1. These are 64 independent base-source clusters and 512 correlated donor-goal interventions *per teacher*, never 512 independent source roots. Each recipient/donor state is obtained by actual native tied-block passes from its own legal input.

### Frozen primary metric

At k=2, select the signed edit `P_K(h_donor[k,j]-h_recipient[k,j])`, where `K=ker(W_head)∩ker(dE_shared/dh_j)`, and normalize to `rho=0.25*||F(h_recipient)[j]-h_recipient[j]||`. The head/gradient nullspace is computed with the existing two-stage scale-stable FP64 SVD, relative tolerance 1e-9 and absolute tolerance 1e-12. Generate eight deterministic same-K random shams with equal physical L2 radius, seeded by `LGA07B|sham|teacher|base_index|goal|k`. All candidate and sham directions are frozen before any ground-truth donor label enters task scoring.

Native final readout is exactly two additional tied blocks (k2→4). Primary per-case effect is `I(projected_donor argmax = donor target) - median_{8 shams} I(sham argmax = donor target)`. Equal-donor-target ITT rate is averaged across all original 512 cases for each teacher, including degenerate projected or dose cases. Ineligible intervention is **identity/no edit** for selected and all shams, so both score the clean recipient prediction and contribute zero paired difference. Separately report eligibility-stratified effects. Clean recipient and donor competence at four passes uses their respective correct legal targets, separately computed from actual clean model readouts; require >=80% jointly competent cases/teacher for interpretability.

**Development engineering gate:** >=2/3 original exposed teachers satisfy ITT mean effect >=+0.15, a positive lower point of a two-sided 95% **2,000-resample base-source-cluster percentile bootstrap**, AND full naturally reached donor-token replacement obtains donor target on >=95% of *clean recipient+donor competent* cases. The last control is a natural larger-dose **single-token** donor state replacement, not a full-state source swap; record its success without automatically interpreting failure as incorrect model/source. If fewer than two teachers meet every threshold, register `DEVELOPMENT_NULL_OR_INELIGIBLE`. Retain all root failures and every edited case.

### Frozen secondary arms and observations

- Selected energy-tangent projected donor, its negative, eight same-K dose-matched shams, same-dose unprojected donor difference, full natural donor **token** replacement. A clean unedited baseline is retained for every source.
- At k=6→12 apply precisely the same selection and sham scheme, *separately*. This long-horizon stress is descriptive only, excluded from the registered development gate.
- Each case records input, donor, target hashes, original base ID, eight fixed goal indices, clean recipient/donor competence, full ten goal logits for every arm, all 81 final predicted classes, true donor target correctness and goal CE/margin; original source clean predictions; finite instant current-head change, finite energy change, linear energy tangent residual, actual physical L2 dose, and edit vector SHA256 and numerical vector for independent verification. Batch execution is allowed if exact native sequential continuation and clean closure are verified.
- The score must use all 64 base IDs even if a source has zero admitted goals. No goal replacement, label-informed gate, new random-angle sweeps, dose changes or post-observation endpoint substitutions.

### Independent audit

**Stage C:** separate Python-standard-library-only auditor downloads the *FIRST* Stage B artifact. Rebuild source donor inputs and rotated labels, check exact little-endian int64 x/y SHA and all four original artifact bytes/protocol+source-lock SHA. Recompute every arm goal prediction from ten logits, goal CE and clean-matched margin, physical edit FP32 hash, edit norms, linear head/energy constraint residual and donor output counters. Recompute the within-base median of eight shams, all 2,000 paired source-cluster bootstrap replicates, and registered gate from original raw case records. Verify all 3×64×8 source-goal ITT cases. Do not use Torch in auditor, re-fit weights, or run new inference.

## Limits

Native whole-state energy descent is not equivalent to donor-target task improvement. A direction lying in the current goal's head/energy tangent is only **first-order** silent; capture finite deviations. The legal donor is source-reachable, while the single-token hybrid is an off-trajectory intervention. Sample size is three exposed teacher checkpoints, so statistical confidence across new *models* and generalization requires a new preregistration with independent-model roots, EvalMath sham-variation floor, appropriate multiple testing and a sealed confirmation cohort. Protected ARG-03-N, AMM-03/04/05 and CQ-08 rosters remain untouched.
