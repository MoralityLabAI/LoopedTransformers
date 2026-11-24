# INK-NANO: spectral residency for recurrent sparse MoE

Status: design/preflight. All numeric results here are derived budgets, not measured quality.

## Model and indexing

The published teacher has 24 physical decoder layers, with layers 6–17 visited three times (48 total visits), hidden size d=2304, expert width m=1152, 128 experts per physical layer and top-8 routing. Shared weights are keyed by (layer, expert), never by pass. Each visit retains its own router state and message-passing state. Expert weights ship in mixed NVFP4/MXFP8; obtain the effective dequantized W using the pinned upstream implementation, without treating packed codes as float weights.

Let q(t,l,p,e) be the teacher's normalized selected routing weights, with zero elsewhere. Record the original weights and normalization denominator as well; preserve the native sigmoid routing and scaling when evaluating intervention. Residency set R(t,l) has capacity k; it can change only at permitted boundaries. The retained mass is sum of q over R. It is an observational proxy, not a task-accuracy guarantee.

For expert gate, up and down matrices W, use W_r = U[:, :r] diag(s[:r]) Vh[:r, :]. Store nested factors from a single decomposition; do not separately randomize lower ranks. Implement the nonlinear expert as down_r(silu(gate_r(x)) * up_r(x)), preserving all upstream scaling, biases if present, and accumulation dtype. Inference must execute factors without materializing dense matrices. SVD is optimal for Frobenius residual at fixed rank; it need not preserve activations or routing.

Ideal factor elements per expert = 3r(d+m)=10368r (with singular values absorbed into one factor). Full bank = 3072 experts. At rank 64/128/256/512, ideal packed INT4 bank sizes are 1.019/2.038/4.077/8.154 GB decimal. At rank 128, k=8 resident experts per physical layer cost 127.4 MB INT4, k=16 cost 254.8 MB, and k=32 cost 509.6 MB. These figures exclude all scale, padding, staging, activation, KV and non-expert tensors. Rank 512 BF16 factors cost 32.61 GB before metadata; avoid publishing this as the first atlas. Measure full GPU allocation including the non-expert backbone and context-dependent caches. Budget in GiB when comparing with the card.

## Residency oracle and causal policies

Given a segment I (a whole turn or H tokens) define A(l,e,I)=sum of normalized teacher weights over token and pass visits in I. With uniform byte cost and independent layer capacities, selecting top-k A maximizes retained mass for that segment. This follows by exchanging a selected item of smaller score for an unselected item of larger score. This hindsight oracle ignores replacement penalties and uses future target tokens, so never use its labels as input features at evaluation. Under variable ranks/bytes the capacity constraint is knapsack-like. Joint temporal optimization adds a cost for additions to the preceding set.

Compare at k=8,16,32 and horizons whole-turn,128,64,32,1:
- Full teacher, uncompressed.
- Full compressed bank with native routing, to isolate spectral loss.
- Fixed global per-layer frequency set from training only.
- Prefix-only per-layer set; next window uses only past visits.
- Last-window and LRU/cache controls.
- Hindsight mass oracle (upper bound only for fixed-set teacher mass).
- LDT and matched-size MLP/GRU with identical causal inputs, capacity, timing and bytes.

At exactly eight resident experts, a restricted top-8 router cannot change membership, although it can change weights. Two separate interventions: (A) retain only original selected experts within R, report any renormalization or zero-output policy; (B) select native top-8 logits restricted to R and apply native combination rule. Evaluate both with on-policy model states, not solely offline teacher traces. For k>8 the original router can choose within the resident bank. Avoid treating a teacher's saved routing weights as available at inference on future tokens.

Compare 6 persistent + 2 rolling slots and 8 fixed slots under the same byte cap and account for every prefetch. Compare cache persistence between turns and cold starts. Store independent physical layer experts; across loop passes keep the same cache key. Allow a pass-conditioned prediction only if its transfer budget is charged. Trace teacher-forced and free-generation separately; divergent free generations require task-level scores rather than token-position KL.

## LDT mesh hypothesis

One controller per six-layer block, with optional communication through a compact shared state. Inputs available causally: prompt-only summaries, layer ID, pass, prior router traces, current cache mask, and byte budget. Output predicts the next interval's expert demand, selecting k residents per physical layer. Train with teacher demand distribution, cost for entering experts and downstream distillation:

L = token distillation + lambda * (1 - retained teacher mass) + mu * transfer bytes / budget.

Normalize per root; freeze controller architecture and lambda/mu on development data. A simple global/last-window predictor or parameter-matched GRU may explain any benefit. Keep matched controls.

## Shared spectral bases (separate arm)

For each layer and projection, fit shared orthonormal bases U_l,V_l across development experts, and C_e = U_l^T W_e V_l. Expert W_e ≈ U_l C_e V_l^T. At r=128, one core has 16,384 coefficients, compared with 442,368 independent factor coefficients per matrix. Store shared basis once; charge all core and basis bytes, and all dense core multiply cost. Compare at matched measured byte budget and task fidelity; shared bases may discard expert-specific directions. Use covariance/eigen or randomized sketches without constructing a huge concatenated expert tensor. A shared basis is a separate experimental arm, not a prerequisite to the per-expert atlas.

## Physical target

Text-only, no vision or MTP path for the first deployment (verify backbone dependency), context initially 512/1024/2048, batch 1. Measure prefill and decode separately, cold/warm cache, p50/p95 token latency, peak allocated/reserved VRAM, RSS, actual H2D/D2H bytes, and task score. RTX 3050 lacks native NVFP4 tensor-core inference; use tested supported kernels or dequantize small tiles into supported dtype. Never count packed 4-bit storage as compute throughput. Compare an existing quantized runtime/offload baseline. Goal: a measured 4 GiB fit with positive generation throughput and calibrated quality, not simply a parameter-count estimate.
