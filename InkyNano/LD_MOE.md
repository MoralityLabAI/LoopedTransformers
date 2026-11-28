# LD-MoE: per-token lattice-controlled expert routing for InkyNano

Date: 2026-11-25. Status: theoretical design and development registration. No new model-quality result or trained router in this note.

## Motivation and evidence

The existing PROMPT_MATRIX.md path assembles a fixed subset and copies the corresponding original router rows. At capacity eight, every available expert is selected on every visit; mixture weights can still vary. That is a very strong intervention on a token-wise MoE. Random or prompt-selected collations should remain negative controls. Scoped 12-expert teachers also restrict the original support; they cannot establish unrestricted teacher parity.

The earlier 16-root cached native pilot (SilicoExperiments PR 3, commit e43c01156e82585402f24183d30516c346754444, PILOT_RECEIPT.md) measured generated-continuation H0 hindsight mass of 0.4505/0.6408/0.8381 at k=8/16/32 and H32 mass of 0.5046/0.7073/0.8976. Consecutive-pass overlap was 0.5891 and adjacent-token overlap 0.4220, measured as intersection / 8, not Jaccard. These are observational measurements from a small development sample with capped continuations, not proof of quality loss or cache hit rate. Stage 1's synthetic matched 17,440-parameter pass mesh showed no consistent gain over its layer-mesh control. The full GGUF bank now exists; see GGUF_CONVERSION.md. The native offload path is the next gate. Do not rebuild an entire compact GGUF per token.

## Literature and specific predictions

- [Thinking Machines Lab, Introducing Inkling 2 Mini (2026-11-20)](https://thinkingmachines.ai/news/inkling-2-mini/): 384-to-128 expert pruning during midtraining followed by 1T teacher-distillation tokens. Motivates exact offload before permanent pruning; this is not an estimate of the tokens necessary for InkyNano healing.
- [Goodfire, Interpreting Inkling 2 Mini (2026-11-20)](https://www.goodfire.com/research/interpreting-inkling-2-mini): six-layer block boundaries, prior-loop messages, and coordinate 2027 (in their indexing) implicated in cross-loop communication. Predicts that causal earlier-pass/block state can improve expert demand forecasts. This is a hypothesis, not evidence of routing mediation; verify index conventions and use matched random-coordinate/feature controls.
- [Noam Shalev, Does Inkling 2 Mini use the same experts on every loop? (2026-11-24)](https://noamshh.github.io/blog/inkling-loop-routing/): independent prefill-only analysis over 14.5M tokens. The same router weights act on different loop states. Prefill overlap need not equal decode overlap; measure both.
- [Zhu et al., ReMoE, ICML 2026](https://proceedings.mlr.press/v306/zhu26ab.html): learned reuse-biased native router fine-tuning. Strong baseline for a cache-aware residual.
- [Wu et al., Cache-Aware Joint Router Adaptation, arXiv:2609.04895](https://arxiv.org/abs/2609.04895): temporal same-layer retention plus causal predecessor-state prefetch, with joint model adaptation while retaining native top-k selection. Match budget, prediction timing and native semantics in the corresponding controls.
- [Lasy, Cai and Ayonrinde, RouterInterp, ICML 2026](https://proceedings.mlr.press/v306/lasy26a.html): evidence for fine-grained superposed expert specializations. Broad prompt domain labels are a weak replacement for actual token state. Learn conditional output compatibility, not a universal expert equivalence class.

## Two coupled control problems

Physical layer l=0..23, pass p=0 outside the core and p=0..2 inside [6,18), token t. A visit v=(t,l,p) has hidden state h_v and native router ranking

  s_e = sigmoid(w_e^T h_v)
  a_e = s_e + b_e
  S_v = Top8(a)
  q_e = 8 s_e / (sum_{j in S_v} s_j + 1e-20), e in S_v
  y_v = sum_{e in S_v} q_e E_{l,e}(h_v).

Verify actual quantized runtime router parity with the pinned reference (including bias, scaling, ties and FP32 accumulation). Bias is used for ranking only. Physical expert key (l,e) is shared across passes.

Problem A (exact): all 128 candidates remain eligible. Fetch missing native experts, using a bounded slot/byte cache and an explicit slow CPU fallback where necessary. Prefetch or evict based on demand forecasts. At fixed precision and identical native arithmetic, changing scheduling alone preserves the selected IDs and mixture. Numerical parity must be tested in the actual backend, since kernels can change rounding. This is the quality anchor; its speed is an open question.

Problem B (approximate): change the selected set only under a calibrated substitution policy. Given causal features z_v, memory state C_v and measured transfer price c_e, define

  a'_e = a_e + Delta_theta(z_v,e) - lambda*c_e.

Select top eight by a' over all candidates; exact fetch remains possible. For a controlled initial arm compute native mixture weights over the newly selected set, gathering s_e rather than a'_e. Later jointly learned mixture weights are a distinct arm. Zero correction and cost recover the native rule (subject to native tie handling). The provided NumPy reference implements ranking and mixture, not a CUDA router.

Cache residency is independent of selection: the same expert may be absent when selected, and an admitted expert need not execute. Never use a GPU slot index in place of original expert ID. A predicted miss is not a prohibition. Charge actual packed bytes by expert, demand loads and prefetches, and wasted speculative transfers; keep cold starts and prompt prefill.

## Mesh and lattice

Four logical six-layer blocks. Each block has a small shared predictor over native scores, current visit, causal earlier hidden sketches, prior-token demand, prior-pass messages, cache mask and budget. Use per-layer/pass positional tags and bounded communication from already executed visits; no same-step all-layer pooling of future states. A block-end message from pass p can inform pass p+1 only after it exists. Use previous-token summary for future-layer prefetch. At a target layer, the exact native score vector is available and cheaper to evaluate than moving expert weights; do not discard it.

Per expert e maintain a set P_v(e) of possible replacements within the same physical layer. The product of powersets ordered by inclusion is an explicit lattice; meet is intersection and join union. Only runtime facts (valid bank identity, completed copy, compatible shape/type, byte budget) are hard constraints. Learned compatibility and cache predictions are soft evidence. Empirical tolerances cannot certify arbitrary inputs. Within an epoch retain hard feasibility; at a new visit reset relevant candidates. Do not make permanent deductions from router affinity or a missing cache slot. The existing ExpertLattice implements residency admission for uniform costs; a global unequal-byte pool needs separate verified allocator.

Compare:
- exact native route plus LRU (demand only);
- exact native route plus next-use hindsight eviction (diagnostic, with explicitly matched prefetch/admission assumptions);
- temporal forecast, prior-pass reuse and causal predecessor prefetch;
- ReMoE-style scalar recency bias;
- parameter-matched independent, TRM, layer-mesh and pass-mesh predictors;
- random and fixed prompt collations as negative controls.

## Rank-margin certificate and output compatibility

Let gamma = a_(8) - a_(9). If gamma>0 and ||a'-a||_infinity < gamma/2, every original selected score remains above every originally excluded score. Proof: pairwise score gap is at least gamma, and the pairwise perturbation is at most 2||a'-a||_infinity. With zero gap, no positive uniform strict radius follows. This certifies top-k membership only (native mixture is preserved if the original unmodified s and scale are used). It says nothing about replacing an expert or global model correctness. For floating-point implementations require a measured numerical safety margin. Stable tie-break in the reference is deterministic, not a guarantee of native tie parity.

For a fixed h and native q define D_h(e,f)=||E_e(h)-E_f(h)||_2/(||E_e(h)||_2+epsilon). An output-distance model can propose substitutes; validate it on held-out roots and on-policy student activations.

If each original selected expert e is mapped to f(e) and q is held fixed, the triangle inequality gives

  ||y-y_sub|| <= sum_e |q_e| ||E_e(h)-E_{f(e)}(h)||.

This is a local bound for the specified substitution and weights. If weights change, add a weight-change term: on the union of supports, ||sum_e (q_e-q'_e) E_e(h)|| <= sum_e |q_e-q'_e| ||E_e(h)|| (or use an appropriate decomposition around the substituted outputs). The exact measured output error can be smaller because of cancellation. Repeated use through attention, residuals, loops and autoregression can amplify errors. No end-to-end accuracy certificate follows. If uncertainty exceeds the calibrated budget, fetch native experts.

## Learning objective

For teacher/student on common prefixes:
  L = KL(p_teacher || p_student) + alpha*normalized mixture output error
      + beta*route demand loss + lambda*expected transfer cost.

Use soft normalized native selected demand over 128 experts as a target for forecast training; for true router training include native full score vector, selected IDs and mixture coefficients. Costs use the physical bank bytes and measured stall time. Optimize route/cost through a differentiable surrogate or explicit policy gradient; do not pretend discrete top-k or cache misses are differentiable. Validate with actual discrete replay. Stage A trains scheduler only; Stage B learns residuals with frozen experts; Stage C optionally heals experts/backbone using distillation. Use the grant only after quality and resource gates. The reference module provides deterministic primitives; training and runtime integration are future work.

## Mechanistic test

Train a predictor of next-pass demand with causal hidden/logit sketches. Compare same-size pass-mesh with no-pass and random/permute message controls, same feature width and training budget. Evaluate coordinate-2027 and block-end features versus variance-matched coordinates, shams and full sketch. First test predictive contribution, then interventions in the router at matched routing/transfer dose, and task scores under identical decoding. Do not label predictive gain a causal mechanism. Consult EvalMath mechinterp-power and FSI-01 per AGENTS.md.

See LD_MOE_EXPERIMENTS.md for staged gates and LD_MOE_CODEX_GOAL.md for the implementation handoff.
