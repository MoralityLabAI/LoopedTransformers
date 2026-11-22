# NEC-03W/F01: negative source-only factorial interaction, root 0

**Study:** Original dev5-000 source-trained teacher, groups16..23.
**Status:** developmental calibration, REDESIGN_OR_PILOT_ONLY.
**Native + independent 12-pass replay:** [GitHub Actions run 45003603923](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45003603923), [artifact 14745929793](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45003603923/artifacts/14745929793).

## Prior physical and source custody

Original root0 source-trained native and TopK16 SAE checkpoint identities were already authenticated against NEC03T/NEC03V and the independent source-fit positives. New source-behavior inputs were restricted to original SAE-training groups16..23, precisely 256 cases per teacher arm. Source quartet and 2+2 partition selection used previous outcome-blind feature support and projected decoder geometry. The full 512-case physical roster was frozen and independently audited [in run 45002476714](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45002476714) *before* any source behavioral inference.

Original physical roster SHA-256: b3f13ab106b0d60f5c7d97e8bd521fa5890195fe8f025cf843126169e0f58513. Composed 256/256 physically admissible source cases, additive 152/256; the remaining 104 additive cases were paired zero-intervention no-ops. All 78,336 admitted 32D projected coordinates were independently audited.

Composed source quartet [8,27,31,64] at original recurrent visit0, split A=[8,27], B=[31,64]. Additive source quartet [72,102,115,124] at visit10, split A=[72,102], B=[115,124]. Original-source disjoint shams, shared nonamplifying 32D doses and physical pair matching were frozen for every individual case.

## Ten-mode behavior

| Fixed original source mode | Composed correct | Composed errors | Additive correct |
|---|---:|---:|---:|
| Full-affine source baseline | 237/256 | 19 | 256/256 |
| Remove pair A | 238/256 | 18 | 256/256 |
| Remove pair B | 236/256 | 20 | 256/256 |
| Remove both pairs A+B | 240/256 | 16 | 256/256 |
| Matched sham remove C | 235/256 | 21 | 256/256 |
| Matched sham remove D | 236/256 | 20 | 256/256 |
| Matched sham remove C+D | 234/256 | 22 | 256/256 |
| AB lesion plus target-direction late injection | 231/256 | 25 | 256/256 |
| AB lesion plus sham-direction late injection | 228/256 | 28 | 256/256 |
| Same-visit A+B exact addback identity | 237/256 | 19 | 256/256 |

Preregistered 2×2 excess-error interaction:

    J_group = error(AB)-error(A)-error(B)+error(baseline)
            = 16-18-20+19 = -3 errors.
    J_sham  = error(CD)-error(C)-error(D)+error(baseline)
            = 22-21-20+19 = 0 errors.
    J_group - J_sham = -3/256 = -1.171875 percentage points.

The additive teacher retained 256/256 correctness in every mode, giving zero additive interaction. Therefore the source root0 composed-minus-additive factorial effect is **-1.171875 percentage points**. It has the wrong sign for the proposed redundancy-type complementarity mechanism and misses the positive +2pp research-promotion floor.

The late matched-control contrast is (28-25)/256 = **+1.171875 percentage points** favoring the original target-direction injection over the sham direction. The original AB lesion alone had 16 errors; target injection increased this to 25, and sham increased it to 28. Thus both delayed injections **worsened** task accuracy relative to the unrescued lesion. The target direction was comparatively less harmful, which does not establish restorative memory computation. Same-visit exact addback reproduced baseline by construction and is only an engineering identity control.

## Verification

The native source-only scorer completed 160 original-source CPU batches, each running all twelve tied recurrent passes. A separately implemented memory-token prehook, recurrent block posthook and donor reconstruction independently replayed another 160 original-source batches, verifying every 32D physical component, original case/mask, original teacher label, saved native binary logit, correct-count ledger and root factorial/delayed contrasts. **Independent audit PASS, maximum absolute within-run logit replay difference = 0.0.** Original model and SAE weights were frozen.

The first two F01 behavioral runs [45003130136](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45003130136) and [45003388866](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45003388866) stopped during the zero-forward input gate, before task labels or model inference. The second isolated a clerical transposition in a manually copied original calibration coefficient SHA, corrected in the independently audited behavior v3 registration, while the original 512-case physical roster and full model scoring method stayed unchanged. The correct original coefficient SHA is 688f7c92beb7015b575d1addff606a357dbf2af360c81b048cc5bd271af919d1. This correction and failed-attempt evidence are recorded on [PR #309](https://github.com/MoralityLabAI/LoopedTransformers/pull/309).

## Scientific decision

**Source-calibration root0 F01 FAIL** for positive joint necessity and direct effective delayed rescue under these selected original feature pairs. The positive target-minus-sham late contrast was 3/256, below the provisional 2pp meaningful research gate, and both late injections made AB lesion performance worse.

This is ONE development world, with 256 correlated source-training cases and a source SAE already fitted on original groups0..39. The root0 test is developmental, without a fresh population, uncertainty guarantee, causal generalization or independent confirmation.

The other seven original roots have independently audited label-blind candidate selections and positive code coefficients. Their full source-only physical feasibility can be screened with zero new neural inference. Any extension of behavioral scoring beyond root0 requires a **separate preregistration** that acknowledges this root0 result, fixes root-level analysis and maintains original ITT zeros for roots3/5. Query groups32..39, all 88 recruited NEC03P roots, reserved and embargoed contexts, and old four-input confirmation cohorts remain sealed. FSI-01 REDESIGN_OR_PILOT_ONLY.
