# NEC-03P: sealed development causal-resolution pilot design

**Status: registered design only.** This protocol reuses the eight **previously source-trained** five-input development worlds \`dev5-000..007\`, with completed source-only model and SAE provenance in NEC03O A/B. It creates a machine-checkable manifest and runs tests on *registry metadata only*. All 24 development-evaluation and 24 reserved context groups per world, all 32 new five-input embargoed worlds and all 64 old four-input confirmation roots remain **unscored**.

## What the causal pilot would test

The question is whether a source-attributed TopK16 SAE direction causally improves five-bit composed-rule correctness more than its locally **direction-matched active-code sham**, after accounting for the equal-prior additive teacher. The hypothesized mechanism is specific to recurrent composed reasoning rather than general restoration of a network's confidence.

The unit is one five-input Boolean world. Two native teacher arms are paired inside each root. Each arm uses the exact frozen **900+900 reset-AdamW source-only checkpoint** and a source-only **checkpoint-specific** 128-feature TopK16 SAE with ranks learned on the original 40/40 source split. No model or dictionary is trained on the pilot evaluation contexts. The measured population consists only of **development** roots already inspected on source-training outcomes, never prospective population confirmation.

### Primary contrast

For root r, arm t in composed/additive, and the 768 distinct development histories x across 24 source-disjoint seven-bit contexts ×32 focal counterfactuals, let

\[
A_{r,t} = \frac{1}{768}\sum_x
\left[\mathbb{1}\{\operatorname{SAE}_{r,t}(x)=y_{r,t}(x)\}
-\mathbb{1}\{\operatorname{sham}_{r,t}(x)=y_{r,t}(x)\}\right].
\]

The preregistered **one primary** root effect is \(\Delta_r=A_{r,\mathrm{composed}}-A_{r,\mathrm{additive}}\). Its primary estimate is the macro mean \(\bar\Delta=\frac1{8}\sum_{r=0}^{7}\Delta_r\). We predict a positive mean and retain the prior NEC03H scientific minimum **0.02 absolute correctness fraction**.

The source-only screening already identified **six paired-source-eligible roots**: indices 0, 1, 2, 4, 6 and 7. Root 3 lacks an additive candidate after visit-8 source hub saturation; root 5 has composed native score 2237/2560 under the fixed 90% gate. **Both stay in the eight-root ITT denominator with registered root effect zero.** Missing per-input active nulls on an admitted root are also carried as zero by using the same hubs-only donor for source and control at that input. Input failures, weak gradients and poor angle/dose matches are never dropped after scoring task correctness.

### Mechanism and matched-sham policy

Within each already-trained root, **freeze the original source graph** and either its original discovered sparse recurrent community or the original deterministic fallback at the third and fourth focal visits. The TopK SAE rank comes solely from the second 40 original training-source groups.

For each development input and selected recurrent visit, trace the model's clean-chosen margin and hubs-only baseline. Choose \`n\` source-ranked active TopK16 features using the fixed source ranking; enumerate every disjoint \`n\`-subset among the remaining \`16-n\` active features. The sham minimizes the absolute 32D cosine gap against the hubs-only **post-MLP-down clean-chosen margin gradient**. The maximal permitted source/sham absolute cosine gap is .25, with the exact original source NEC03K feasibility rule. Attenuate source and sham 64D GELU donor vectors to equal 32D output norm, using \(\min(\|W_{\rm down}s\|_2,\|W_{\rm down}h\|_2)\), with no amplification. Record every input's physically equalized dose, clean logits, original source/active-code IDs, exhaustive matched control and gradients. The native recurrence has twelve tied visits. Compute the **full nonlinear outcome**, because tangent cosine matching alone is an observational geometry constraint.

Any input/site without a valid nonzero, angle-matched, physically comparable null is assigned the same hubs-only donor under both interventions for that input, giving a signed correctness difference of zero. Frozen orientation and control routing never consult true task labels. Labels enter only the final scoring phase. Store full raw 768-case paired correctness, counts and per-root effect, with complete failure reasons.

### Inference and stop gate

Use one primary difference-in-differences comparison, with root-level signed effects \(\Delta_r\) and exact root sign-flip inference **only if** root-sign exchangeability under its null is defensible. Report native/SAE/source eligibility as exposure history and finite-sample selection conditions. The historical source pilot roots were examined extensively before this design; this is **development-only hypothesis resolution**.

Even if all eight observed root effects are positive, a small p-value would have limited inferential scope. The FSI-01 single-comparison conservative effect lower margin at g=8 is **86.54 percentage points**. This pilot therefore serves to estimate an actual contrast and matched-sham variation for later root-power design; it cannot establish a two-percentage-point population causal mechanism under the FSI-01 confirmation gate. Any additive-only effect, memory-trajectory decomposition or model-confidence changes are **secondary descriptive** measurements without separate confirmatory claims.

Before actually opening **any** of the 24 development-evaluation groups, the causal execution code must be independently reviewed, source-checkpoint provenance and all root eligibility failures verified, every planned comparison and dose frozen in the same audit manifest, and an explicit executable protocol **separately committed before outcome generation**. The current module writes **only the sealed manifest**, with \`development_evaluation_contexts_scored=0\`. A later user request may authorize implementation of that executable protocol and its development-only pilot. Old 32 five-input embargoed and original 64 four-input confirmation worlds stay closed.

**FSI-01: \`REDESIGN_OR_PILOT_ONLY\`.**
