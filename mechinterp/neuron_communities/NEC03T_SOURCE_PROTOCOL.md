# NEC-03T: full-affine SAE source-only necessity ablation

**Prospective registration before scoring, 21 November 2026.** Follow-up to the independently audited NEC03S source-only offset factorial. The experiment asks whether the source-ranked active SAE features are *necessary* when the original full-affine SAE restoration is already present, using an equal-physical-dose disjoint active-code sham as control.

## Cohort and protected splits

Exactly eight previously trained five-input development Boolean worlds, dev5-000 through dev5-007, with paired composed/additive frozen native reset1800 models and original 128-feature TopK16 sparse autoencoders. Frozen original first40-fit/last40-rank source selection. Score **only original training-source groups 24–31**, eight full seven-bit context groups ×32 focal masks = **256 already-seen source histories per model arm**. Native and SAE optimizer updates = zero. Every arm and root remains represented. Roots 3/5 have preregistered zero effects in the primary eight-root ITT because their earlier paired source instruments failed.

No development-evaluation, reserved, 88 recruited-world non-source, 32 five-input embargoed or 64 original four-input confirmation contexts may be read. The 69 independently source-eligible recruited worlds are preserved for a later separately registered causal campaign.

## Full-affine restored background

Use the original native source-selected recurrent visits and their graph hubs. On each source input take the original clean 64D GELU activation, obtain frozen TopK16 active codes, and reconstruct **all 16 active decoder contributions plus the source normalization mean and decoder bias**, masked to the selected original native nonhub coordinates. Keep hub channels live in the modified 12-visit recurrent trajectory. The original unmodified native clean prediction fixes the logit-margin orientation.

Run a *full-affine restored* recurrent trace with gradients. The comparator uses the physical **32D post-MLP-down gradient of this restored trajectory**, rather than the earlier hubs-only lesion gradient. Save the original clean, hubs-only, native-full-nonhub, full-affine and sixteen-code contribution-only logits as descriptive ceilings.

## Source necessity and physically matched disjoint sham

At every selected recurrent site and input, the original feature budget n equals the number of source-selected native nodes at that visit. Rank the 16 positive active codes using the frozen source-only ranking, select its top n and calculate the native **64D GELU decoder contribution** removed from the **same full-affine background**. Enumerate **every n-code subset of the remaining 16−n positive active codes**, with the same physical mask. Select the disjoint candidate minimizing the absolute difference of source/sham output-32D cosines against the restored-background margin gradient. The maximum allowable absolute cosine gap is 0.25; gradient norm must exceed 1e−7.

Calculate the physical 32D L2 norm of both unattenuated removal vectors and use their **minimum** as one achievable dose. Apply attenuation only, with gains in [0,1] and direct physical L2 equality within 1%. Require the shared physical dose to exceed 1e−6. Both source and sham begin from the **identical full-affine donor**, preserving the same mean/bias offset. Source removal is subtracted from this full restoration, sham removal is similarly subtracted. Run complete nonlinear recurrent trajectories for both.

If any selected visit lacks exactly 16 positive codes, a suitable gradient, an angular-matched control or an equal positive dose, both source/sham removals become **zero for the entire input at every selected visit**. The trajectories return identical full-affine restored behavior and contribute zero paired difference. Do not drop a root, arm, case or visit after reading outcomes. Teacher truth labels enter only after all routes, donor vectors and doses have been fixed.

Secondary ablations use widened min(2n,8) active-code budget and a deterministic hash-selected original-n disjoint active sham, salt NEC03T-source-full-affine-necessity-random-v1. Each secondary source/sham removal pair is independently equalized to its common attainable dose; the same invalid-case policy applies. A disjoint 16-code sham is combinatorially impossible and never claimed.

## Fixed single primary root-level effect

For each original paired-source-eligible root, S is the source-ranked code-removal correct-count and H is matched disjoint sham-removal correct-count among its 256 original source inputs per teacher arm. The positive necessity contrast is **H − S**, measuring extra correctness lost when removing the source-ranked codes. Subtract the same quantity in the additive teacher from the composed teacher:

\[
D_r=\frac{(H_{r,\mathrm{composed}}-S_{r,\mathrm{composed}})
-(H_{r,\mathrm{additive}}-S_{r,\mathrm{additive}})}{256},\qquad
\bar D=\frac{1}{8}\sum_{r=0}^7 D_r.
\]

Roots 3 and 5 contribute fixed zero to the eight-world macro mean. Original meaningful positive effect floor **+0.02 correctness fraction**, one primary family. Per-arm clean/full-affine reconstruction, widened-budget necessity and hash shams are secondary diagnostics. Exact two-sided root sign-flip p is exploratory and assumes symmetric root signs under its null.

## Hosted audit and inference stop

Freeze the scorer, independent auditor, stdlib aggregate, tests and workflow by Git blob SHA1 before any new source scoring. Pin the original 5-input registry SHA256 **6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217** and 4-input confirmation spec SHA256 **670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39**. Hosted preflight checks original NEC03O A/B independent model+SAE ancestry, source group membership, missing-site behavior and physical dose matching.

Run exactly eight source root workers using frozen checkpoint bytes and already-trained source groups 24..31. Each independently audited root must rederive every true task label, affine/background restoration, source-ranked code-removal direction, **every disjoint active-code sham candidate**, equal physical dose and full recurrent logits, with raw JSONL SHA256. All 16 model arms and original zero roots stay in the aggregate. Publication needs **8/8 independent root audits**.

The first 59-root sham-tail project stays deferred until a matched mechanism clears the prespecified source-only +2pp effect floor and realistic independent-root power gate. Prior NEC03Q, NEC03R and NEC03S negative observations remain intact.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
