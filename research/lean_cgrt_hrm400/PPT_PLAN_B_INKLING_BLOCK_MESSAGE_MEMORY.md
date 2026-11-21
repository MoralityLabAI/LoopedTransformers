# PPT Plan B: Inkling-derived block-message proof transformer with DRAM plasticity

**Research review / revision:** 2026-11-21  
**Status:** proposed architecture and experiments; untrained.  
**Location:** Lean-CGRT local gym.  
**Companion:** [PPT Plan A](PPT_PLAN_A_DRAM_PLASTIC_PROOF_TRANSFORMER.md).

## Why revise Plan A

Thinking Machines Lab released Inkling 2 Mini and its variable-loop companion on **2026-11-20**. The release, Kimi Team's Attention Residuals, Goodfire's interpretability analysis, NNsight/NDIF instrumentation, Ouro/LoopLM and Mixture-of-Recursions suggest a better *testable* structure than a single recurrent residual vector. Keep Plan A as the compact, falsifiable baseline.

**Main revision:** use a nonrecurrent proof **prelude**, two *weight-tied middle reasoning blocks* communicating through an explicitly gated previous-pass **message bank**, and a nonrecurrent typed action **coda**. Carry DRAM-plastic specialization primarily at the message/routing interface and a separate online value head; instrument bank read/write causality rather than assuming that output logits reveal intermediate proof reasoning.

This is an architectural hypothesis. Neither Inkling's reported improvements nor its interpretability demonstrations establish that small Lean specialists learn online, that CPU-updated gates improve proofs, or that more loops improve arbitrary Mathlib goals.

## Source review with design consequence

| Source | Grounded result / mechanism | Consequence for PPT |
|---|---|---|
| [Inkling 2 Mini release, 2026-11-20](https://thinkingmachines.ai/news/inkling-2-mini/) | 24 stored layers split into four 6-layer blocks. Prelude/final run once; middle 12 run three passes, for 48 layer visits. Middle vs final looping won 27/28 evaluated settings in the authors' comparison. Variable-loop model trained mostly at 3 passes with 1-5 exposure; 10-pass N-Hop performance improves beyond training depths, while no-CoT time horizon levels off after ~3. | Prefer middle-only looping and train a range of pass counts. Test beyond trained depth separately on serial proof dependencies. |
| [Kimi Team, Attention Residuals, arXiv:2603.15031](https://arxiv.org/abs/2603.15031) | Softmax mixtures of layer or block representations replace the single cumulative residual; block version reduces stored inter-layer states from O(Ld) to O(Nd), with input-dependent depth selection. | Use block-level mixture for proof-state context, and keep separate interpretable block summaries. |
| [Goodfire, Interpreting Inkling 2 Mini, 2026-11-20](https://www.goodfire.com/research/interpreting-inkling-2-mini/) | Cross-loop routing favors recent middle blocks and can depend strongly on a few high-magnitude coordinates; coordinate 2027 accounts for a reported 46% of squared norm in their measured setting. Direct intermediate logit lens performs poorly; tuned/coda lenses recover more of the final output distribution. | Instrument per-pass messages, block reads, outlier features and a trained **proof coda lens**; prefer causal patching to interpretations of ordinary logits. |
| [NNsight/NDIF guide, 2026-11-20](https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/) | The active 6-layer block, completed blocks, and cross-pass message bank jointly define Inkling's state. Per-pass KV caches are separate. The full forward state cannot be reconstructed from a single layer output. Remote probes can measure each pass. | Test the architecture/measurement contract on NDIF as external observational work; implement equivalent hooks on the compact PPT. |
| [Zhu et al., Scaling Latent Reasoning via Looped Language Models, arXiv:2510.25741](https://arxiv.org/abs/2510.25741) | A learned per-step exit distribution with entropy-regularized training and subsequent improvement-aware gate fitting allocates computation by task. Their specialized gating improves matched average-exit performance on reported MMLU settings. | Fit halting to observed *verified benefit* of another proof pass; include depth diversity and entropy control to prevent terminal-only collapse. |
| [Bae et al., Mixture-of-Recursions, arXiv:2507.10524](https://arxiv.org/abs/2507.10524) | Token-specific recursive depth routing and recursion-wise vs shared KV caching offer distinct efficiency-quality tradeoffs. | First implement goal/slot-aware skipping, then optionally per-token routing; account for both KV allocation and compute, which increase with passes. |
| [Yang et al., Looped Transformers Are Better at Learning Learning Algorithms, arXiv:2311.12424](https://arxiv.org/abs/2311.12424) | Shared-depth recurrence performs iterative algorithmic work in stylized in-context learning settings. | Use algorithmic task families for curriculum, but retain real Lean verification for research claims. |

**Limits of transfer:** Inkling is a 26.4B-total, 3.4B-active MoE trained at substantial scale; the novel PPT remains a proposed 20-55M dense specialist. Inkling's native NVFP4 training and multi-trillion-token post-training do not imply we can reproduce its gains with a small local model. The source findings motivate ablations, not promoted outcomes.

## Proposed PPT-B computation graph

~~~text
Lean goals / retrieved lemmas / failure history / check budget
                         |
                 [PRELUDE, once]
         typed embeddings + proof-graph masks
                         |
                 current block bank
                         v
       +----------------------------------+
       | MIDDLE A: goal and premise work  |
       | Block-AttnRes over public prelude|
       | and current partial block        |
       +----------------------------------+
                         |
       +----------------------------------+
       | MIDDLE B: candidate and repair   |
       | evidence, error and value work   |
       +----------------------------------+
                         |
             previous-pass bank:
             [message_A, message_B]
                  ^          |
                  |          v
       gated cross-pass Q/K/V/MLP inputs
                  |          |
          next pass over same A and B
             K = 1,2,3,5 (8,10 eval)
                         |
             [CODA, once on exit]
       typed tactic / next-state / value / HALT
                         |
           Lean executes and certifies
                         |
          DRAM experience and fast update
~~~

Prelude and coda weights are distinct from the two middle blocks. Middle A and B keep their own distinct weights, each *tied across passes*. This preserves the serial specialization in the original release while limiting the physical parameter count. For PPT-20M, sweep 1+2+1 distinct blocks or sub-blocks; for PPT-55M, sweep 2+2+2 distinct blocks with the middle group reused. These are architecture presets whose exact parameter counts require code measurement.

### Block and message equations

Let b_0 be the prelude embedding/proof-context output, b_{A,k}, b_{B,k} the current middle-block outputs at pass k, and p_{s,k-1} the saved previous-pass bank for each typed slot s.

At layer l, construct the current sublayer input via block attention:

x_(l,k,s) = sum_{j in available_blocks} alpha_(l,j,k,s) RMSNorm(b_(j,k,s)),
alpha = softmax_j(q_l^T RMSNorm(b_(j,k,s))/sqrt(d)).

The prior-pass message is a learned block mix:

m_(l,k,s) = sum_{j in {A,B}} beta_(l,j,k,s) RMSNorm(b_(j,k-1,s)).

Gate it as:

x' = (1 - g_forget) * x + g_message * m,

where g_forget and g_message are low-rank functions of [RMSNorm(x); RMSNorm(m)] and slot type. Initialize g_message = 0 and g_forget = 0, so the first looped model is equivalent to its unmessaged path at initialization. Let gates grow under supervised training, with a separate norm/causal intervention audit.

Use distinct Q/K/V/FF input gates as the ambitious variant; start with *one shared cross-pass gate per block* as the cheap pilot. Register the split-gate expansion as an ablation. Distinguish **algorithm-clock previous-pass messages** from **token-clock KV caches** and from **episode-clock DRAM learning**.

At every pass boundary, preserve only designated previous-pass summaries, not every residual vector. Retain the token prelude across all passes. Do not discard previous-pass verifier information that is part of the public proof history.

## Typed proof slots and decoding

Keep the semantic roles from Plan A, but distribute them over block summaries rather than treating a single z_k as the whole state. Proposed roles:
GOAL, CONTEXT, LEMMA, PLAN, PROOF_FRONTIER, ERROR, EVIDENCE, VALUE, MEM, HALT.

Middle A should attend heavily to GOAL/CONTEXT/LEMMA, while Middle B can access FRONTIER/ERROR/VALUE. Masks are configurable and may be relaxed during training. Do not claim learned role specialization until probing/patching confirms it.

The **proof coda** maps the full current block state to a typed action:
- valid candidate-ID ordering or grammar-bounded new Lean tactic;
- predicted successor goal(s) and distinction between accepted-nonterminal/closed/rejected;
- estimated gain from another loop / retrieve / delegate / stop.

Train a cheap affine/tuned lens on each pass for observational diagnostics, and compare it with the full coda applied offline to the saved pass bank. Cost the full-coda diagnostic separately: it is an additional forward computation, and running it every pass may erase adaptive-depth savings. Teacher prose explanations are attributed guidance, while kernel receipts remain authoritative.

## DRAM specialization: revise the update surface

Plan A allowed plasticity in a general specialist adapter. Plan B **focuses rapid local learning** on two well-identified read/write interfaces:

1. A **CPU resident fast value/routing head** over encoded (goal, candidate, typed-slot summary, budget, previous result). RLS or a bounded online SGD update uses only definitive support-set Lean labels. This is the high-frequency DRAM update.
2. A **low-rank message/gate adapter** modifying cross-pass communication, block-selection queries, or final coda projections. Train it periodically on checked replay batches, with weight masters/optimizer in DRAM. Use checkpointed, validated snapshots for GPU inference; do not claim this can train through a large GPU-frozen backbone without backward activation cost.

For d=768 and rank r=16, a single Inkling-style projection+two-output low-rank gate has roughly r*(2d)+2*(d*r)=49,152 trainable parameters, excluding biases. Eight gate sites (~two middle blocks x Q/K/V/FF) are ~393k parameters, or ~6.3 MB for FP32 weights+gradients+two Adam moments. One-gate-per-block is far cheaper. The actual prototype can choose lower rank and a shared query. This is a *storage estimate*, excluding GPU activations, recurrent KV caches, hardware transport and optimizer scratch.

The episodic store retains immutable (goal, evidence, tactic, post-state, certificate, model/weight-version) records, indexed by hash. It is a distinct retrieval-only comparator. A new adaptation state can be rolled back by version hash; the evaluator must label which parameter version influenced each episode.

**Avoid per-token CPU synchronization as the default runtime.** Execute the complete K-pass burst on accelerator or CPU, then update CPU fast weights between proofs/decisions. Batch DRAM gate/LoRA refreshes across verified support sets. Measure host->GPU transfer separately.

## Variable depth and curriculum revision

Inkling motivates *depth-diverse* training instead of fitting at a single K and hoping for extrapolation. Set training K in {1,2,3,5}, with matched presentations and per-pass supervision; evaluate 8 and 10 as extrapolation only. Add 16 later if measured benefits justify it.

A learned depth distribution p_phi(k | goal, slots, budget) is trained by

L = E[sum_k p_phi(k) * L_k] - beta H(p_phi) + lambda E[compute_k],

where L_k combines *predecision* action imitation, next-state prediction, and verifier-grounded outcome targets on training episodes. A second stage fits the depth gate to **observed held-out-on-train benefits** across all K with the model frozen; use a calibrated exit quantile at inference.

Use **goal/slot-level** routing before optional token-level routing. Per-token variable passes complicate cache alignment and may win fewer CPU/GPU-seconds at the small sequence lengths of Lean proof packets. Report live/allocated KV bytes for both recursion-wise and first-pass-shared caches. Fix identical public input and candidate menu across depth conditions.

Prefer direct feedback inside a block as the starting ablation, with RLT and matched damping retaining full paired controls. The Lean RLT negatives make a prescribed RLT schedule an unearned default.

## Concrete training and experiments

### B0: baseline equivalence and provenance

Build a direct tied-loop Plan A control and a matching prelude/middle/coda variant with zero-gated messages. Check exact initial logits/gradients where architectures permit, parameter counts, first-pass cache hashes, and deterministic replay. Freeze memory limits and split roster.

### B1: block-residual ablation

On synthetic Horn programs and fresh Lean train goals, compare ordinary residual, Block-AttnRes without cross-pass messages, gated previous-pass messages, and a compute-matched feedforward middle. Evaluate structured successor prediction and certified close@8. Block gains on synthetic tasks alone are engineering evidence.

### B2: depth conditioning and training

Train K=1/2/3/5 with all-pass action/next-state targets. Evaluate fixed K=1/2/3/5/8/10, frozen queries, and evidence of improvement at larger *dependency distances*. Track genuine proof coverage, ranker success, marginal extra-iteration gain, and wall/GPU time. Compare learned gate, entropy-only gate, random halting, and oracle compute allocator.

### B3: causal message and lens experiments

For same-theorem clean/altered-premise pairs, patch prior-pass messages or masked typed slots at recorded boundaries. Measure downstream action distributions, successor-state predictions and actual Lean-verified effects. Include patch-sham, norm-matched random, block-swapped, and within-goal time-shuffled controls. Report high-magnitude/outlier channel concentration; avoid inferring an explicit routing semantics from magnitude alone. Compare tuned lens and coda lens on held-out proof-state prediction.

### B4: support/query online-learning episodes

Each proof family supplies support goals with definitive verifier outcomes followed by family-related but unseen query goals. Reset DRAM parameters between independent tasks. Train the encoder/update rule through meta-train query loss, with query labels withheld from inner updates. Compare: frozen head, retrieval-only, fast weights, message-adapter consolidation, both, shuffled verifier labels, and same-cost longer inference. Measure learning curves after {0,1,2,4,8,16} support outcomes.

### B5: deployment, pool expansion and handoff

Use the existing four-automation / four-specialist check budget as one fixed allocation; compare with learned allocation only under total matched Lean calls, tokens and latency. Include synthesizing off-pool candidates and a Bonsai handoff; do not count historical c4 replay as a trained gain. Maintain per-checkpoint origin and role signatures. Calculate incremental certified rescues and memory cost per solved goal.

## Decision table: how Plan B revises Plan A

| Design axis | Plan A | Plan B |
|---|---|---|
| Core state | One typed latent workspace with residual feedback | Prelude + *current/completed block bank* + gated previous-pass messages |
| Depth | K={1,2,4,8,16} candidate sweep | Train 1/2/3/5; extrapolate 8/10; fit gain-aware early exit |
| Training feedback | All-pass proof-state targets and online meta-learning | Same, with explicit block-level probes and pass-specific decoding |
| Rapid DRAM weights | General value head plus optional LoRA | Prioritize online routing/value head; periodic gate/message adapter |
| Mechanistic evidence | Slot probe and state patch | Complete-state, message and block patch with matched shams |
| Resource boundary | Host memory and separate GPU copies | Also explicit per-pass KV caches, block-state transport and sync charges |
| Primary experimental risk | Plasticity helps only retrieved familiarity | Apparent loop gains arise from extra compute or gate/memory leakage |

## Projection and falsification

Expect the clearest possible benefit on *serial premise dependencies* and error-conditioned proof repairs. The Inkling sources do **not** predict a gain from more passes on factual recall. The existing Lean-CGRT and DataGoblin archive results suggest that closed-goal coverage may be dominated by automation and the candidate menu unless proposal generation increases witness coverage.

**What would change the research program:** A reproducible gain on fresh, lineage-disjoint proof queries *after DRAM updates*, exceeding retrieval-only and same-compute controls, with weight-reset ablation removing the gain and actual Lean kernel certification. A second independent finding would be a positive causal role for prior-pass messages in hard proofs at matched model calls. Both are empirical gates.

**Fail Plan B as a useful architecture** if it has no paired proof gain over the simplest tied loop at matched GPU-seconds, if a fixed no-message model matches it, if the DRAM updates only memorize support-query duplicates, if best-fit K varies by validation noise, or if per-pass cache/host sync cost overwhelms additional closures. Record partial and inconclusive results; retain Plan A.

### Resource-limited first run

1. Reuse the existing Lean-CGRT v4.19.0 native verifier and DataGoblin fresh-rollout/export contracts; do not train on c4 historical test states.
2. Implement a 20M-class *dense* four-stage prelude/A/B/coda prototype with two or four typed message roles. Time CPU-only and existing capped GPU leases independently.
3. Fit and score one K={1,2,3,5} depth sweep and direct-vs-message control on a *fresh* pre-registered cohort; delay expensive online adapter consolidation until a message-specific benefit appears.
4. Add the DRAM fast-head support/query pilot and exact memory-version receipts. Preserve the option to integrate in ALEPH-1B later.

**No model download, GPU training, production runtime, or scientific positive result is claimed in this note.**

## Source register and linked code

- Thinking Machines, *Introducing Inkling 2 Mini* (2026-11-20): https://thinkingmachines.ai/news/inkling-2-mini/
- Thinking Machines, model card: https://thinkingmachines.ai/model-card/inkling-2-mini/
- Kimi Team, *Attention Residuals* (2026): https://arxiv.org/abs/2603.15031
- Goodfire, *Interpreting Inkling 2 Mini* (2026-11-20): https://www.goodfire.com/research/interpreting-inkling-2-mini/
- NNsight/NDIF, *Opening the box of Inkling 2 Mini* (2026-11-20): https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/
- Zhu et al., *Scaling Latent Reasoning via Looped Language Models*: https://arxiv.org/abs/2510.25741
- Bae et al., *Mixture-of-Recursions*: https://arxiv.org/abs/2507.10524
- Yang et al., *Looped Transformers Are Better at Learning Learning Algorithms*: https://arxiv.org/abs/2311.12424
- [Earlier Lean-CGRT negative and positive findings](LOCAL_SCALING_RESULTS.md)
- [DataGoblin Inkling Variable-Loop Gym PR #5](https://github.com/MoralityLabAI/DataGoblin/pull/5)
- [DataGoblin Lean-CGRT harness PR #4](https://github.com/MoralityLabAI/DataGoblin/pull/4)
