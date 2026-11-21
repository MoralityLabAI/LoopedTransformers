# NEC-03K-B: causal replay with direction-matched active controls

**Registered after the successful NEC-03K source-only feasibility stage and before any new direction-matched evaluation outputs.** The original source-only feasibility workflow [44892904616](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44892904616) passed its independent model/checkpoint/SAE/source and exhaustive 32D direction checks. All eight previously opened Boolean worlds showed adequate source angle-matching and physical signal.

## Preserved inputs

Reuse exactly the original eight NEC-03H-v2 dev-008..015 native/SAE 900-step frozen checkpoints from GitHub Actions run 44878411694, the previously frozen source-top and four disjoint hash active control profiles from NEC-03I-B run 44885003105, and the source-only 16-context NEC-03K geometry calibration artifacts from run 44892904616.

Each world has one previously trained and source-ranked native 64D memory-token GELU circuit, one TopK16 SAE with 128 decoder features, and one source-selected recurrent native graph/fallback site roster. The initial model and SAE bytes and existing prior correctness/ITT receipts are never modified. All source indices and full old 64-world confirmation-spec commitments are unchanged.

The control algorithm is exactly the previously source-frozen algorithm in NEC03K_SOURCE_LOCK.json: choose the nearest-angle disjoint active n-code donor among all C(16−n,n) subsets, using the clean-chosen logit margin gradient from the **hubs-only** native trajectory in the physical 32D post-MLP-down output coordinates. The original source donor is the n highest-ranked currently active SAE codes. True Boolean teacher labels are used only to score final correctness, after both donors and their equalized physical dose have been fixed.

The source-only pilot required an absolute cosine gap no greater than 0.25 on at least 80% site/input pairs, and minimum direct physical coverage of 80% plus 25% median retained original source dose. Every source model met these bounds, with mean source-only angle and dose coverage 99.327%.

## New outcome endpoint and failure accounting

Evaluate exactly the **48 previously opened development-evaluation context quartets** for each of the eight original worlds, each containing all 16 related focal-bit histories. The 48 reserved context groups, unused old development worlds dev-016..063 and all 64 original confirmation worlds stay unopened.

For each context, verify the clean and hubs-only frozen model logits against the archived NEC-03I-B hash-control raw record. Compute a single clean-chosen final binary margin orientation and hubs-only gradient per site/input, then choose the angle-matched control with the frozen source-only algorithm. Equate direct MLP-down donor physical L2 norm by **attenuation only**, gains at most one. Source and control then undergo full **twelve-visit nonlinear forward recurrence**, with all sixteen binary output logits retained.

Record the signed exact correctness gain of source-ranked versus angle-matched control over 768 case histories within each world, plus the exact nonlinear clean-oriented margin difference, its first-order hubs-only Jacobian approximation, the nonlinear remainder and the source/control 12-visit memory-state drift. Signed class correctness remains the behavioral outcome; Jacobian sign and nonlinear drift remain mechanistic diagnostics. This is a *new development-only outcome family*, so it cannot revise any prior PEF/NEC protocol or original NEC-03H-v2 ITT score.

The new directional-control world is *admissible* only if its original composed head was at least 90% correct on the original development contexts and at least 80% of evaluated site/input pairs have positive shared direct output dose and meet the locked cosine-angle gap threshold, with numerical physical norm matching on at least 99% and median retained original source dose at least 25%. A failed root contributes **zero** to the new, explicitly posthoc ITT descriptive mean; its observed raw outcomes and failure reasons are retained. The eight teacher worlds are the unit of analysis, while contexts, focal masks, active-code subset candidates and recurrent visits are correlated repeated observations.

## Independent external audit

The pure-standard-library audit reconstructs every source and disjoint-control feature subset from its sixteen logged active 32D decoder projection vectors, independently enumerates all combinations and verifies the selected direction minimizes the source cosine gap. It then recomputes each exact true Boolean label, original clean/hubs frozen output, source/control binary correctness, direct physical donor L2 norm, attenuation gains, first-order gradient displacement, nonlinear margin difference, twelve-element memory drift, context signs, and original model/SAE/source lock digests. It fails closed on a missing source calibration, altered control policy or any reserved/confirmation context.

The GitHub Actions runner first verifies exact code Git blobs against NEC03K_DEV_LOCK.json. It must restore each source-only calibration from **run 44892904616** before starting a development causal replay and enforce source_null_feasible=true. The new code, causal thresholds and source-population contract are committed **before scoring the new comparison**.

## Scientific limits

These eight model and Boolean-function worlds have already been inspected repeatedly. The source matching algorithm was calibrated on training-source contexts, but the development-evaluation outcomes of these worlds are already known from NEC-03H/NEC-03I. Even a large positive or negative effect against this new null therefore constitutes **retrospective development evidence only**, useful for deciding which independent-world mechanistic study to preregister.

The original NEC-03H-v2 world-mean ITT +0.423 percentage points remains below its two-percentage-point meaningful-effect floor. The EvalMath FSI-01 matched-null 95/95 upper tolerance requirement for at least 59 independent sham worlds remains unresolved, and the original 64 confirmation world SHA256 commitments remain untouched. **REDESIGN_OR_PILOT_ONLY** remains the required scientific classification.
