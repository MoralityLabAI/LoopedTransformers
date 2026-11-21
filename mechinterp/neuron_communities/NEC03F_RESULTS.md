# NEC-03F: TopK16 source SAE, active routing, matched-dose causal outcomes

**Date:** 2026-11-21. **Research evidence class:** `posthoc-development-only`, on PEF-07's previously opened 48-context heldout source family. This is an instrument validation and retrospective comparison; it is not independent confirmation of native-neuron or SAE circuit specificity.

## Public CPU provenance

- [Successful GitHub Actions run 44861209471](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44861209471). Five TopK/causal unit tests passed, followed by source-only SAE training and two complete trained-model causal replays. The independent routing, model-source, output-classification and local dose audit succeeded.
- Source frozen native models, context partitions and nine seed-103 candidate communities: [NEC-02B artifact 14655419843](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44790504320). No native-model training or modification was performed.
- New evidence artifact `nec03f-topk16-active-source-matched-causal` **ID 14689528300**, GitHub ZIP SHA256 **91a676771ff26ab05863bf80e698687e983a05d99ee6bde480258d71e34a44c4**. The artifact contains both SAE checkpoint files and their source-only fit/rank manifests, 1296 candidate × pair × context raw records containing all clean/hubs/donor class logits, physical dose/attenuation arrays and full 16-state TopK feature routing selections, and the independent `nec03f_audit.json`.

## Trained SAE quality

Each original 32-hidden-unit recurrent model has a 64-D memory-token GELU input to the tied `mlp_down`. The TopK SAE uses 128 learned features, a positive ReLU encoder and exact TopK=16 activation masking, with a linear decoder trained for 480 CPU steps. All feature fitting uses the first 80 PEF parameter-training context quartets; all 12-visit feature attribution rankings come from the other 80. The 48 downstream-probe and 48 previously opened heldout contexts were excluded from TopK training and ranking.

| Property | Coalition seed 103 | Threshold seed 103 |
|---|---:|---:|
| Latent dictionary width | 128 | 128 |
| Number of active features enforced | 16 | 16 |
| Observed mean L0 | 16.0 | 16.0 |
| Fraction train inputs with exactly 16 positive codes | 100% | 100% |
| Normalized source-training reconstruction MSE | 0.0070764 | 0.0068343 |
| Trunk checkpoint altered | No | No |
| Heldout contexts used for SAE optimization/feature ranking | 0 | 0 |

The TopK SAE reconstruction remains close to the earlier dense SAE's MSE (~0.00712 coalition, ~0.00659 threshold), while reducing average active latent count from approximately 63 to **16**.

## Physical-dose matching and useful retained signal

At each candidate recurrent site, the policy selects exactly as many active TopK features as there are native neurons in the community at that visit. Its ranking is fixed using source-training contexts; for each test input, the policy selects the most source-attributed **active** features. The alternative control uses the lowest source-attributed active features, with identical count. This adds an input-conditioned selection policy to the causal intervention; it cannot be interpreted as a fixed neuron-community identity.

For each selected donor pair, both outputs are attenuated to \(\min(\|W_{\mathrm{down}}d_L\|_2,\|W_{\mathrm{down}}d_R\|_2)\). Gains stay in [0,1], and source-/checkpoint-locked raw logs verify the local physical doses.

All **27/27** candidate comparisons matched direct output dose within 1% for all site/example samples. **19/27** also passed the precommitted physical-signal floor of 80% positive shared dose and at least 25% median original left-donor dose retained.

| Task | Pair | Passing matched-dose + signal gate |
|---|---|---:|
| Coalition | Native vs native sham | 3/4 |
| Coalition | Native vs active TopK SAE | 4/4 |
| Coalition | TopK SAE vs lower-ranked active SAE | 0/4 |
| Threshold | Native vs native sham | 5/5 |
| Threshold | Native vs active TopK SAE | 5/5 |
| Threshold | TopK SAE vs lower-ranked active SAE | 2/5 |
| **Total** | | **19/27** |

Previous NEC-03E fixed-feature SAE policy had **8/27** physically feasible comparison pairs, with **0/9** native versus SAE pairs passing the signal gate. With the same native checkpoints and source contexts, a genuine TopK dictionary and active code routing enabled **9/9** native-versus-SAE comparisons to pass. This is a substantial **instrumentation improvement** in availability, not a prospective superiority result.

## Selected behavioral outcomes, exact correctness

The strongest threshold candidate remains `nec-b30c4edbbb42`, on which the hubs-only baseline made **59 of 768 errors**.

| Matched pair | Left donor errors | Right donor errors | Left minus right correct counts | Positive shared-dose fraction | Median left original dose retained |
|---|---:|---:|---:|---:|---:|
| Native vs native sham | 58 | 53 | -5 | 100% | 79.3% |
| Native vs active TopK | 38 | 34 | -4 | 100% | 82.9% |
| TopK versus low-ranked active TopK | 41 | 59 | +18 | 100% | 26.9% |

All three pairs passed the physical matching and source-signal gates. The active TopK policy recovered 25 of the 59 hubs-only errors in the native-versus-TopK comparison, and its lower-ranked active alternative recovered 0 in the TopK-versus-alternative comparison. The treatment and donor pairs differ across rows due to *pair-specific common attainable dose*, so error counts across rows are descriptive rather than a single fixed dose ranking.

The threshold candidate `nec-0cf33bd0b573` had hubs-only **5** errors, native recovery **5** and TopK recovery **0** when native and active TopK were dose matched, while the active TopK versus low-ranked-active control gave **0** versus **2** errors. These both passed the physical signal floor.

For the strongest coalition community `nec-68cd70ab69a8`, hubs-only made **10** errors. Native versus active TopK at exactly equal local dose produced native **0** errors and TopK **1**, a one-case advantage for native, with full 100% positive-dose availability. The rank-selected TopK versus alternative comparison gave 6 versus 15 errors (nine-case difference), but its median retained left-donor dose was **24.4%**, below the required 25%, so that particular comparison was marked low-signal.

All remaining communities, including zero-effect, native-sham superiority and threshold weak-coalition cases, are preserved in the full raw artifact and can be audited without the original SAE training code.

## Interpretation and next experiment

The **availability failure of fixed sparse donors was fixed by input-conditioned selection of active codes**, while retaining genuine TopK16 model sparsity and physical dose equality. This is a verified engineering result on a small synthetic recurrent model. The strongest observed threshold example favors the learned TopK feature router over an alternative active feature router, as expected from using downstream-gradient attribution to rank features on source training data. Since source-ranked feature selection itself is a supervised interpretability instrument, this finding alone does not identify a unique causal Boolean operation or prove that an SAE basis is intrinsically better than native neurons.

Every model, task teacher and 48-group PEF heldout source family has appeared in earlier experiments. The primary quantitative comparisons were not prospectively frozen on independent new roots, and 27 candidate-level matched comparisons share both task/world structure and repeated contexts. A confirmatory study must fix the rank/routing algorithm, intervention sites, dose and signal floor, meaningful correctness contrast, independent task/model roots, and complete multiplicity-corrected statistical evidence **before** new heldout results are exposed. Ideally, the next source generator should vary Boolean gate composition and interacting input positions across independent families, not only distractor contexts.

**EvalMath FSI-01 classification:** `REDESIGN_OR_PILOT_ONLY`. The result establishes a feasible source-only sparse feature intervention interface. It is not an inference about general intelligence, persistent self-models or natural-language model circuitry.
