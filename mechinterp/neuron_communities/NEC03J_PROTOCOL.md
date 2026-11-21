# NEC-03J | Native GELU Jacobians, margin direction and recurrent-memory transport

**Protocol frozen on November 21, 2026 before any new gradient or transport outputs.** This is a development-only, read-only mechanistic experiment on the eight already examined NEC-03H-v2 Boolean teacher worlds. The original eight native model checkpoints, TopK16 SAE dictionaries, frozen 12-visit source-attribution ranks, original 48-context heldout input assignments and all four NEC-03I-B active-feature control permutations remain unchanged.

## Scientific question

The source-ranked active TopK donor beat every independently hash-ranked, disjoint active feature donor in dev-009 (+30, +41, +15, +8 correct /768), yet lost to all four in fallback-selected dev-013 (−6, −16, −38, −15). All these previous comparisons had equal direct physical MLP-down output L2 dose and passed retained-signal feasibility. Can the **direction of the donor in the local decision-margin gradient**, together with nonlinear twelve-visit memory transport, explain the opposed signs?

A physically matched scalar output norm leaves the direction and the subsequent recurrent state unconstrained. A source attribution rank fitted on training contexts can point along a harmful downstream direction on a different teacher world. This experiment measures these directional quantities without changing the source model or the eligibility gates from the prior studies.

## Source, model and unit

Input native models are the exact 900-update PEF-07-style 32-D hidden, tied 12-visit checkpoints on the original development worlds **dev-008 to dev-015**, from [NEC-03H run 44878411694](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44878411694). The 128-feature TopK16 decoder and the native GELU memory-token 64-D coordinate basis were fitted only on original parameter-training contexts. The controls and archived complete 16-bit logits are from [NEC-03I-B run 44885003105](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44885003105).

No model or SAE training occurs. No input from the original 64 confirmation worlds, the unused dev-016..063 roots or any 48 reserved context quartets is used. This experiment revisits only the original 48 development-evaluation context quartets per world, with all sixteen related focal counterfactuals.

## Gradient and physical coordinate contract

For each historical input, first compute the model's **clean** binary logits and freeze its clean chosen class, \(c(x)=\arg\max_j\ell_j^{\mathrm{clean}}(x)\). The label-blind class margin is \(m(x)=\ell_{c(x)}-\ell_{1-c(x)}\); the same clean-chosen orientation is used in every modified trajectory, even when the modified prediction flips. The true Boolean teacher label is used only to score correctness after interventions, not to choose the derivative.

Trace all twelve recurrent visits, intercepting the GELU vector entering the shared MLP-down module at the carried-forward memory token and the corresponding 32-D MLP-down output. The direct output direction is the **frozen 32 × 64 linear weight** applied to the 64-D donor difference. On the intact clean trajectory and on the **hubs-only lesioned trajectory**, compute both the native GELU gradient and the post-MLP-down gradient of the complete final clean-oriented logit margin. This derivative includes all later recurrent transport.

For a matched donor pair A and B and a selected visit \(t\), define the GELU donor displacement \(\delta_t=a_t^A-a_t^B\), with direct 32-D output displacement \(d_t=W_{down}\delta_t\). The local first-order predicted margin difference is \(\nabla_{a_t}m\cdot\delta_t\), equivalently \(\nabla_{y_t}m\cdot d_t\). Both quantities are recorded and must agree in float32 within a maximum 0.001. The directional cosine is between \(d_t\) and the gradient in **32-D output coordinates**. This prevents a false comparison between incompatible native and residual geometries.

The first-order predictor sums these local terms at all selected visits, on both the clean and hubs-only reference trajectories. The **actual nonlinear margin difference** is taken from full fresh donor grafts into the original hubs-only lesion at all selected visits. Its remainder relative to the first-order hubs-only prediction is retained for each focal input, and its sign is compared with the local gradient prediction. A large remainder means linear local attribution is a poor approximation to the finite, multi-site intervention.

## Complete recurrent memory transport

A separate forward hook on the tied attention-plus-MLP block records the **32-D carried-forward memory state after every one of the twelve chronological visits**. For each matched control and source-ranked donor, calculate the 12-length per-case L2 norm sequences: source versus control, source versus hubs-only, and control versus hubs-only.

These memory trajectories carry full nonlinear interactions. Report the final 12th-visit drift and peak visit drift. They are repeated observations inside one root, so a case × visit count cannot be treated as additional independent teacher-world samples.

## Physical dose and original outcome integrity

For every source context and each of the four historical hash-control variants, regenerate the original input-conditioned **active** TopK source and hash-control donors using the same previously frozen rank policy and source SAE weights. Exactly reproduce the historical **clean, hubs-only, source-ranked, hash-control** logits before interpreting any derivative. Both donors are attenuated to their smaller attainable MLP-down output norm, with no gain above one. The original direct local dose arrays and actual classification outcomes are cross-checked by an independent pure-stdlib auditor.

Before opening the first new development-evaluation Jacobian, run a **source-training-only finite-difference check** on sixteen parameter-training histories: perturb one 64-D GELU coordinate at a registered source-selected visit by ±0.003, measure the final clean-chosen class margin, and compare the centered numerical derivative with the clean autograd gradient. Maximum absolute error must be ≤0.02 on this FP32 CPU model. The direct native GELU/post-MLP-down gradient inner-product identity must hold for every scored example/visit within 0.001.

The new raw evidence includes 8 roots × 4 original sham variants × 48 independent context groups = **1,536** complete context records, each with sixteen correlated focal histories, original teacher labels and every original paired logit, per-visit gradient displacement/cosine, nonlinear margin remainder, and twelve-visit memory drifts. This is an instrumented replay with unchanged native source states and unchanged original signed ITT effects.

## Inference and eventual next test

All eight worlds and their Boolean labels have been examined repeatedly, so even a strong correlation between local derivative direction and sign reversal remains **posthoc mechanism diagnosis**. Gradient alignment is a local sensitivity observable on a nonlinearly lesioned system and may have poor predictive quality at finite perturbation size. This study does not establish a unique sparse mechanism or behavior transferable to pretrained language models, autonomous self-models or recursive self-improvement.

The independent unit remains a **trained teacher world**, rather than source context, focal mask, variant or recurrent visit. The original NEC-03H-v2 world-level ITT mean of +0.423 percentage points and its 2pp scientific floor remain unchanged. Only two original worlds have complete valid matched sham families, so the EvalMath 95/95 upper-tail bound remains uncalibrated. The original 64 confirmation roots and reserved cohorts stay sealed, and **FSI-01 remains REDESIGN_OR_PILOT_ONLY**.

A future experiment could preregister the direction-alignment predictor and stronger null controls on a freshly generated, independently powered teacher-world population. It must establish its own sham floor and meaningful behavior threshold before any confirmatory readout.
