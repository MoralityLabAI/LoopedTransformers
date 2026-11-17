# LOOPSCOPE-04 / LGA-01: Do supervised TRM loop updates descend a task loss?

**Prospectively frozen 18 November 2026 before any LGA-01 training or heldout evaluation.** New independent study at mechinterp/loopscope04, motivated by SMC-26/27 and BPD-OUT06's distinction between trainable model weights, inferred causal direction and cost-aware intervention selection. Existing SMC, LOOPSCOPE-02/03, LS-SAE and ROC outcomes are unchanged.

## Architecture and task

Use a source-attributed MIT mirror of the original 44,001-parameter TinyRecursiveModel(48,4) in MoralityLabAI/Compactification-Theory/research/levin_sieve_v1/pretrained_trm/upstream_arch.py: one shared residual token-mixing/channel-mixing block F across four passes, 81 tokens, 48 hidden channels, and shared 10-logit readout H. This is an independently initialized TRM-inspired 44K model, not a 7M attention TRM nor a 3B looped LM.

Choose an independently generated deterministic token-routing task: 81 IID digits in {1,...,9} per sequence, with target at token j equal to input digit at (j-1) modulo 81, i.e. a one-position cyclic shift. This requires cross-token information rather than input-token identity. Freeze train (512 sequences, generator seed 31041), validation (128, seed 31042), and heldout test (256, seed 31043). No example crosses splits; preserve SHA256 of ordered x/y tensor bytes. Entire 81-token sequences, not individual tokens, are statistical clusters.

Three initialization seeds: 103,211,307. Exactly 200 AdamW steps, lr .001, weight decay .0001, batch size 16 and fixed seed-specific randomized batch order. No early stop or post-test checkpoint selection. Two paired-seed arms using identical input minibatches, architecture, optimizer and four F calls:
- final_only: train loss CE(H(h4),y).
- deep_020: train loss CE(H(h4),y) + (0.20/3)*(CE(H(h1),y)+CE(H(h2),y)+CE(H(h3),y)).
Both use full BPTT of the tied block. Pin torch==2.7.1 CPU on GitHub, two CPU threads. Preserve model checkpoint and all loop-level diagnostics at steps 0,50,100,200; validation accuracy is descriptive. Chance class accuracy is 1/9; final heldout accuracy >0.20 is the fixed task-learning admission gate.

## Distinct parameter and activation ledgers

At training steps 50,100,200, on the current train minibatch before AdamW, capture the FULL shared parameter gradient G and the after-before actual optimizer update Dtheta. Report cos(Dtheta,-G), -G dot Dtheta and norms. This is the parameter-optimizer direction, which includes AdamW preconditioning and weight decay.

At each heldout model checkpoint and loop k=0..3, capture h[k] and h[k+1]=F(h[k]) on all 256 test sequences. Define delta[k,j]=h[k+1,j]-h[k,j] and local diagnostic gradient g[k,j] of the current shared head cross-entropy at h[k], holding F fixed:
g = (softmax(H(h[k])) - onehot(y)) W_head.
Compute per-token cosine a=cos(delta,-g), local first-order predicted CE decrease -g dot delta, and actual head CE decrease CE(H(h[k]),y)-CE(H(h[k+1]),y). Record positive fractions, physical update/gradient norms, and Pearson association of first-order predicted to actual decrease. Aggregate 81 tokens WITHIN each sequence before computing any confidence interval. The gradient is **label-conditioned and diagnostic only**; it does not control model or policy selection on new data. Recurrent training BPTT gradients and instantaneous head gradients are fundamentally different objects. Nonlinear CE can change against the local first-order prediction.

Measure head rowspace projection of each delta using FP64 SVD of learned 10x48 output weights W, with singular values admitted above 1e-10 relative to the largest. Record the fraction ||P_head delta||^2/||delta||^2. Head-null movement can affect future output after later F even though it is currently invisible to H.

## Native causal controls

On the first 64 heldout sequences only, for k=1,2 and eight input-only fixed positions floor(81*j/8), j=0..7, compute the genuine proposed next F update delta[k]. From exactly the same h[k] source state, inject at one selected token: no edit, +delta, -delta, and isotropic seeded equal-L2 random edit. Continue all arms for exactly 4-k original tied-block passes and read out their final full ten logits. Freeze random arm generator seed by SHA256 of task/model seed, training arm, heldout sequence ID, k and position, before looking at targets. Check clean versus zero injected baseline; report mean correct-digit margin (true minus strongest clean rival) and CE differences from clean, including matched physical norm and 64-sequence clustered descriptive 95% intervals. The +delta edit is a DOUBLE proposed update before remaining passes. The proposed delta was already computed with F; these are mechanism diagnostics, not claims of inference compute savings. Do not select goal or dose using test labels.

A companion head-only classwise decision diagnostic uses the clean head logits at h[k] and the candidate margin effects for +delta versus -delta. Every hypothetical class y=1..9 has strongest CLEAN rival from full 0..9 logits. Combine candidate local effects under q=softmax(clean digit logits, temperature 1 fixed). Compute mass q_plus,q_minus,q_neutral with frozen true effect neutral band +/-0.15 logits and BPD-OUT06 utilities E U(+)=2q_plus-1.1, E U(-)=2q_minus-1.1 and E U(0)=q_neutral. Choose the highest expectation with abstention on ties. Compare action rate and retrospective actual decision utility to uniform-class Bayes and always-abstain, scoring only after target label is revealed. The diagnostic has no registered scientific success gate. It is HEAD-ONLY and POST-proposed-update, and neither calibrated to actual true label uncertainty nor a claimed native remaining-loop effect.

## Registered inference and reporting

For all three paired seeds and both arms, at final step 200 report per-loop mean cosine of the tokenwise local gradient versus the actual activation update and 95% bootstrap percentile intervals from 256 WHOLE sequences (2,000 draws, root seed 41041). A seed has a positive gradient-descent classification only if heldout final-readout accuracy >0.20 and at least TWO of the four depth-specific bootstrap lower bounds are >0. A study-level positive result requires this seed-level classification for >=2/3 final_only seeds, plus mean cosine at those depths greater than corresponding step-zero checkpoint cosines. Report all other seeds and any negative/inconclusive outcomes. Deep_020 versus final_only is a paired-seed secondary contrast and cannot change final_only classification.

Preserve heldout CE/accuracy at steps 0,50,100,200 across loops 0..4, optimizer update angles at steps 50,100,200, update norm fractions, local prediction-versus-actual CE gains, and native final-output intervention margins. Head-visible projection tests do not establish native causal mediation. If head-null counterfactuals are not actually executed, report HEAD_NULL_CAUSAL_NOT_RUN rather than claiming a positive fiber effect.

All results are about a 44K supervised cyclic-copy routing model. Source SMC-27 original FAIL, larger 7M Sudan TRM, BPD feature necessity and deeper looped LLM mechanisms require independent tests.
