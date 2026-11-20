# NEC-02B: six-model, source-frozen neuron community cross-seed results

**Date:** November 21, 2026. **Evidence class:** posthoc-development-only. This is a retrospective extension of the CPU-trained PEF-07 pilot in [NEC-02](NEC02_RESULTS.md).

The [hosted six-model read-only run 44790504320](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44790504320) completed successfully with **20 passed tests**. It reused the six exact frozen checkpoint files and the two seed-103 raw-logit archives from [NEC-02 run 44788623483](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44788623483), without retraining. Four new seed/task arms completed train-only source selection and 48 heldout context quartets, each with the original clean logit pair and five interventions.

The resulting GitHub Actions artifact is **nec02b-six-model-raw-causal-and-crossseed-receipts**, artifact ID **14655419843**, ZIP digest **849b021efaad6196c4f6cb43441a52ef4bf9e9d6941c011d0e003190be7f2246**. It contains the six original hosted model checkpoints, all six discovery rosters and frozen SHA256 manifests, full 16-state causal logits, and the independent audit file **nec02b_six_seed_audit.json**.

## Cross-seed observations

| Seed | Task | Communities | Graph edges | Targeted ablation: added class errors across candidate screens | Best conditional rescue | Worst conditional rescue |
|---|---|---:|---:|---:|---:|---:|
| 103 | Coalition | 4 | 24 | 0 | +9 | 0 |
| 103 | Threshold | 5 | 36 | 0 | +25 | -7 |
| 131 | Coalition | 4 | 61 | 0 | 0 | 0 |
| 131 | Threshold | 2 | 30 | 0 | +7 | +1 |
| 167 | Coalition | 3 | 58 | 0 | 0 | -68 |
| 167 | Threshold | 2 | 22 | 0 | +14 | 0 |
| **Total** | | **20** | | **0** | | |

**Definitions.** Targeted ablation zeros just a selected community's visit:neuron occurrences from the intact MLP, retaining all other computation. Conditional rescue counts are errors from hubs-only minus errors from hubs plus the selected community at only its chosen MLP visits. Each displayed best or worst candidate was picked from that model's heldout results, so extremes have post-selection bias and serve only as descriptive diagnostics.

All 20 communities had zero changed class predictions from selective ablation of the intact model on the same 768 heldout examples per candidate. The frequency-matched sham ablations also generated zero class errors. This null behavioral result persisted across all six frozen model/task arms and sharply constrains a simple necessity interpretation.

The conditional rescue pattern was model dependent. The threshold task produced at least one positive rescue candidate in each of the three seeds, with maxima of 25, 7 and 14 recovered predictions. The coalition task showed a positive best candidate only in seed 103; seed 167 additionally had a community whose retention *worsened* the hubs-only system by 68 errors. The various community sizes, positions and graph densities differ across seeds. This provides development evidence for context- and network-dependent computational roles, including destructive interference when the supporting circuit has been thinned.

## Raw evidence and independent units

Across the six model/task arms, NEC-02B discovered **20 communities** and archived **960 candidate-by-context records**, with **16 focal variants each**, their labels, exact clean logits, and complete ablate/sham/retain/hubs/blank logits. The independent context unit for a fixed model is its eight-bit context group: 48 heldout groups, each containing 16 correlated counterfactual histories. Those same 48 source contexts and teacher task families are reused across the six models. Pooling the 960 records as independent statistical cases would be invalid.

The audit verifies model and roster SHA256, original PEF-07 source assignments, all six frozen model weights, complete heldout context coverage, unaltered source-only selection, finite logits and correctness deltas independently recomputed from raw predictions. All 20 community outcomes, including zero and negative rescue, are retained in the artifact. No local GPU, retraining or paid hosted compute was required.

## Statistical and mechanistic interpretation

**EvalMath FSI-01 verdict: REDESIGN_OR_PILOT_ONLY.** The community candidates were discovered from source-only train contexts, but the subsequent broad mechanistic hypothesis family and its smallest meaningful effect were not prospectively registered. All six seed/task arms share a source family, a large fraction of the observed effects are zero, and the strong conditional extremes were chosen using heldout results. A source-frequency-matched sham has yet to establish matched activation, downstream perturbation dose or mechanistic variation floor.

The strongest supported mechanistic pattern is **weak individual necessity plus context-dependent conditional recovery**. The data motivate tests for alternate reader circuits, compensating redundant features and task-specific masking effects. They do not establish a unique Boolean circuit, an interpretable self-model or generalization to pretrained language models.

[NEC-03, preregistered matched-dose native versus SAE comparison](https://github.com/MoralityLabAI/LoopedTransformers/issues/239) is the next research design task. Its confirmation split must use fresh task/model roots, with a precommitted causal effect floor and familywise error control. The prior November 15 local source-manifest SHA discrepancy remains documented on parent PEF-07 PR #67 and NEC-02's result report.
