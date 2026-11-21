# NEC-03P: root-level power gate and source-only recruitment

**Registration: 21 November 2026.** Research issue [#263](https://github.com/MoralityLabAI/LoopedTransformers/issues/263). All 88 new source-training roots and the analysis script are frozen by Git blob SHA before the first new native optimization. The previous NEC-03O [source refit phase A](NEC03O_RESULTS.md) and [prospective phase B](NEC03OB_RESULTS.md) both passed independent hosted auditors and were merged. The strict continuous-AdamW NEC-03N study remains failed.

## Definition of the next 88 source roots

- The root unit is one original NEC-03L independently seeded five-essential-input Boolean world, with a composed native teacher and independently seeded class-matched additive teacher **paired within that root**. Each model's 32 focal masks, contexts, recurrent visits and 16 active SAE features are correlated observations.
- Recruit **all and only** the predeclared 88 DEVELOPMENT roots \`dev5-008..095\`, indices 8 through 95 inclusive, from original registry SHA256 \`6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217\`. Already trained roots \`dev5-000..007\` remain in the historical source-pilot cohort. The 32 \`emb5-000..031\` worlds remain entirely untrained. Original four-input 64 confirmation spec stays \`670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39\`.
- Every attempted root stays in the intent-to-treat denominator, including missing checkpoint artifacts, teacher heads under 90% source accuracy, absent source graph candidates, failed AE reconstruction and failed control geometry. **Continue through root 95 regardless of how many eligible roots have already appeared.** Any failed workflow job is recorded as a failed attempt. Never adapt root membership, training updates or graph fallback based on source outcomes.
- Each new root receives 80 complete seven-bit original source-training context groups × 32 focal masks, 2560 source histories per model arm. **The 24 development-evaluation and 24 reserved groups remain unscored**, as do embargoed and old confirmation world outcomes. Generating registry partitions for source ownership is administrative, not permission to score their labels or run the models on them.

## Native recipe and frozen source geometry

1. First train 900 AdamW source updates per arm, using exact NEC03M model architecture PEF07 Looper hidden32, twelve tied recurrent visits; lr=.003, weight decay=0, batch128, norm clip1 and seed \`world.native_model_seed\` (plus 100000 for additive). Save exact stage900 checkpoint bytes, tensor SHA256 and all 2560 source predictions.
2. Reload the saved stage900 bytes and initialize a **fresh AdamW optimizer with empty moments**. Generate minibatches with \`torch.Generator(seed*7919+59)\` after exactly 900 historical draw advances. Run exactly 900 further updates on identical source rows, then freeze stage1800 checkpoint and full source predictions. This is a **two-stage reset optimizer condition**, not a recreation of an uninterrupted original AdamW optimizer.
3. Select a native candidate from the first 40 complete source groups with the registered graph parameters hub_rate .9, min_pair_count 3, min_phi .12 and original candidate size/rate criteria. Retain the same deterministic two-per-visit frequent-nonhub fallback at third and fourth chronological focal positions. Hub-saturated missing candidates remain source failures.
4. For an available candidate, fit a **new 128-feature TopK16 sparse autoencoder** to the frozen native stage1800 64D GELU MLP-down inputs, 480 source-only updates on the first 40 original groups, with new source-only normalization. Rank each of the 12 visits on the disjoint last 40 original source groups. Freeze model, SAE, normalizer and feature-rank SHA256.
5. On the first eight original source groups ×32 focal masks, compare the original feature direction with **every disjoint n-active-code sham subset** of the remaining 16-n active codes. Choose the minimum absolute cosine gap against the 32D post-MLP-down hubs-only clean-chosen margin gradient. Equalize the physical output L2 dose by attenuation alone. Retain full source active-code contributions for independent \`C(16-n,n)\` enumeration, including low-dose or unavailable cases.
6. Fixed **source-only eligibility** for an arm: ≥90% native source training accuracy, a source-selected native candidate, normalized SAE MSE ≤.03, angular gap ≤.25 on ≥80% of site-inputs, physically matched positive-dose coverage ≥80%, at least 99% equalized L2 match, and median retained source dose ≥25%. A whole root is source-eligible only when **both** paired arms pass. All 88 registered roots remain in the denominator.

## Statistical preflight and causal separation

The nonparametric **95th-percentile / 95%-confidence** sample-maximum sham tolerance bound needs at least

\[
n_{\mathrm{sham}}=\left\lceil\frac{\log(1-0.95)}{\log(0.95)}\right\rceil=59
\]

**independent eligible worlds with causal sham outcome measurements** under a fixed intervention and an exchangeable conditional-root population. Distinct registered roots and a passed source-only control feasibility flag establish neither iid sampling nor the behavioral sham tail; those assumptions and outcomes require a separate audit. Source selection defines a **conditionally eligible population** and cannot silently stand for all Boolean worlds. Missing-source roots remain failures in the overall ITT estimand.

The eight already observed source worlds produced **six of eight paired-source-eligible roots**. The Clopper-Pearson exact 95% interval on this small proportion is broad, about 35%–97%. For 88 further independent Bernoulli eligibility draws, the chances of ≥59 passing are approximately 96.45% if p=.75 and 38.98% if p=.65. These are sensitivity cases assuming an iid Bernoulli model, **not causal evidence or assured future recruitment**. The source-only 88-root recruitment run has a diagnostic success threshold of 59 *source-eligible* worlds. Completing that gate permits **planning** an independently preregistered causal sham calibration. It does not produce a 95/95 behavioral tail bound.

Under [EvalMath FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md), a g-root equal-cluster effect with m registered comparisons has the conservative one-sided Hoeffding margin \(\sqrt{2\log(m/\alpha)/g}\). At α=.05 and m=1, g=32 gives a 43.27-percentage-point margin; g=59 gives 31.87 points; g=88 gives 26.09 points. The prior NEC03H meaningful causal effect floor of two percentage points would require a 28.09-point observed mean for a positive g=88 confirmation **under this conservative gate**. A future causal pilot must measure effect and matched-sham variation before calculating a scientifically meaningful power plan.

## Next causal work, separately locked

The source-only model/SAE receipts of \`dev5-000..007\` can support a **development-only causal pilot** if an explicit new evaluation protocol is frozen before reading their 24 untouched development-evaluation groups. Those eight worlds were already selected and heavily inspected on source outcomes, so they are development units rather than prospective population confirmation. Do not open their development-evaluation labels in this recruitment task.

Before any development-evaluation score, preregister a physically matched intervention-versus-sham paired correctness contrast, exact donor/routing policy and intervention dose, all model arms, comparison family, scientific effect floor, root-level signed inference and ITT failure handling. Use [EvalMath mechinterp power preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) and FSI-01; keep the 32 embargoed five-input roots and old 64 four-input confirmations unopened. At least 59 eligible *independently measured causal sham root outcomes* would then be required for the registered sham tail; any new-root generalization claim needs separate root-level effect evidence and a realistic resolution gate.

## Audit outputs and stops

The hosted run emits: all-attempted source-root ledger, each root's 900/1800 native file and tensor SHA256, original and new complete source confusion/prediction records, source roster/graph hash, 40/40 checkpoint-specific SAE hash and rank, 8-group 32D exhaustive null receipts, independently reproduced source labels/dose math, missing-root reason, original five-input/confirmation registry SHA256 and CI run IDs. Its final standard-library aggregator counts **all 88 attempts**, conservatively treats any missing auditor as ineligible and reports the source-only ≥59 readiness flag separately from any causal/sham effect.

**Scientific evidence state remains \`REDESIGN_OR_PILOT_ONLY\`.** A native source learner or well-fitted SAE is an engineering prerequisite, never evidence of a positive causal treatment effect.
