# NEC-04F: intermediate XOR supervision succeeds locally in three roots, final original parity gate fails

**Completed 22 November 2026. Evidence class:** prospective three-arm native-model calibration-only experiment on untouched original NEC04A calibration WORLD roots **12–17**, followed by independent checkpoint/optimizer/model-logit replay. **Machine decision: STOP_AUX_XOR_STATE_CALIBRATION_PILOT.**

[Issue #305](https://github.com/MoralityLabAI/LoopedTransformers/issues/305) and the [NEC04F original training protocol](../mechinterp/neuron_communities/NEC04F_PROTOCOL.md) register the original six roots, immutable source programs and contexts, both model control arms, three native objectives, 900 continuous optimizer steps, 240/900 checkpoints, full root-level decision criteria and protections. Frozen training source commit **89c49e55b76725ac96a5af2ba8ef71cf06689db7** was committed **before any native training**.

## Verified hosted training and provenance

The original [**GitHub Actions experiment #45015714628**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45015714628) completed all **eight jobs successfully**, including **11/11** prospective source/native tests, independent original root/teacher/oracle preflight, six fully trained and individually audited world workers, and the final pure-stdlib six-world root macro aggregate.

The source-stage audit independently verified **15,360** composed original source labels, **4,608** owned calibration-world validation labels, **61,440** exact source XOR prefix-stage labels, **61,440** independent 16/16-balanced hashed source auxiliary pseudo labels and **18,432** owned validation XOR-prefix labels. Source corpus SHA256 **82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0**; world-root registry SHA256 **88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e**. All original 24-world source records came from previously accepted NEC04A [run #44983509977](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44983509977).

All **six original roots × three arms × two checkpoint budgets = 36/36 original native model-and-optimizer checkpoints** passed independent byte-SHA256, model-state/weight hash, AdamW step counter, identical source minibatch SHA across arms, and independently recomputed complete 12-visit final native **and four auxiliary-stage logits**, source/owned validation labels and metrics. This is **checkpoint/prediction replay**, not a second independent 900-step optimizer retraining.

Artifacts:

- Original six-world source/teacher preflight **14751816377**, ZIP SHA256 **d45adc43e78766188ba56ce60202fd6fb2750e8f0590cbb1af7fa23b5319ea07**
- Native root12 **14751701971**, root13 **14752095879**, root14 **14751577430**, root15 **14751617531**, root16 **14752185980**, root17 **14751747110**
- Final independently audited aggregate **14751582788**, archive name **nec04f-six-world-original-native-XOR-aux-independent-aggregate**, ZIP SHA256 **13aacc64772807154f866c4a468dd394acf6176716537cca96aeb6876fec9ec4**

Every original calibration root used the same **80 source contexts ×32 focal masks = 2,560 source cases** and **24 owned within-calibration-world validation contexts ×32 masks =768 validation cases**. These owned evaluation-context groups are distinct from the separate 24 NEC04A developmental WORLD roots, 16 confirmation WORLD roots and each original calibration world's untouched 24 reserved-context groups.

## Frozen original network and auxiliary controls

Each world trained three same-seed, equal-architecture native PEF07 32D/12-visit Looper models with one additional shared 32D→2 auxiliary classifier at exact arrivals of the 2nd, 3rd, 4th and 5th focal bits. Every arm started from exactly identical model/auxiliary-head weights and consumed the same original source minibatch sequence; single continuous AdamW at learning rate 0.003, batch64, gradient clip1, zero decay, 900 steps/arm, model and full optimizer snapshots at 240 and 900. The auxiliary prefix arm used a fixed 0.5-weighted mean four-stage exact XOR CE, the matched negative control received the same-sized loss against independent SHA-balanced 16/16 source pseudo labels, and vanilla retained the auxiliary head with zero loss weight.

The sole preregistered 900-step primary is **original composed full-task correctness on the MUX-selected parity validation branch**, comparing auxiliary XOR against shuffled auxiliary labels, paired by world root. The original exact five-focal parity branch is class-balanced 50/50 in each complete 32-mask context. Registered success required **at least +10 percentage points six-root macro**, **at least 4/6 positive roots**, **at least 60% macro absolute native parity accuracy in the XOR arm**, and **alternate-branch macro retention at least −5pp**. These conditions were fixed before new root scoring. All six roots remain in the denominator.

## Primary: 900-step original native correctness

| Original calibration world | Shuffled native parity accuracy | XOR-supervised native parity accuracy | Vanilla native parity accuracy | XOR − shuffled (pp) | Alternate branch XOR − shuffled (pp) |
|---|---:|---:|---:|---:|---:|
| cal-012 (easy) | 50.00% | **42.92%** | 50.21% | **−7.08** | −1.04 |
| cal-013 (medium) | 50.00% | **50.78%** | 50.00% | **+0.78** | +4.17 |
| cal-014 (hard) | 49.55% | **50.22%** | 50.22% | **+0.67** | −8.75 |
| cal-015 (easy) | 50.00% | **50.00%** | 50.00% | **0.00** | −11.46 |
| cal-016 (medium) | 50.00% | **39.32%** | 50.00% | **−10.68** | −14.58 |
| cal-017 (hard) | 50.31% | **44.38%** | 50.00% | **−5.94** | +18.97 |
| **Six-root macro** | **49.98%** | **46.27%** | **50.07%** | **−3.71** | **−2.12** |

The frozen primary **failed**: mean paired native parity accuracy delta **−3.7078pp** (root SD **4.8579pp**), with **2/6** positive roots and XOR-arm macro **46.2698%**. Alternate-branch retention was **−2.1156pp**, meeting its −5pp aggregate floor. The original whole-task clean validation macro difference was **−3.0382pp** (XOR − shuffled), with source accuracy 66.52% XOR versus 63.95% shuffled and owned-validation whole-task accuracy **60.70% XOR versus 63.74% shuffled**. The source/owned generalization difference is retained rather than selected away.

The prespecified 240-update intermediate diagnostic gave mean original native parity accuracy **49.98% XOR**, **49.68% shuffled**, **50.14% vanilla**, a **+0.3038pp** paired primary difference with 3/6 positive roots. That checkpoint had no budget-selection authority. The machine outcome at 900 is **STOP_AUX_XOR_STATE_CALIBRATION_PILOT**, with no native headroom checkpoint promoted.

## Secondary mechanism-relevant observation: some models learn XOR, but the original final task deteriorates

The auxiliary head's *own true prefix XOR* accuracy after 900 source-only updates was strong in exactly three original roots, with source and owned-context validation tracking closely. This is a secondary diagnostic, not the primary task endpoint:

| Original root | Auxiliary 4-bit XOR validation | Auxiliary 5-bit XOR validation | Final original task on parity branch | Fifth-order share of final nonconstant focal logit power |
|---|---:|---:|---:|---:|
| cal-012 | **98.44%** | **95.05%** | 42.92% | **54.34%** |
| cal-013 | 48.96% | 50.91% | 50.78% | 0.053% |
| cal-014 | 49.48% | 50.91% | 50.22% | 0.008% |
| cal-015 | 49.74% | 50.52% | 50.00% | 0.027% |
| cal-016 | **97.66%** | **90.23%** | 39.32% | **48.30%** |
| cal-017 | **97.92%** | **97.01%** | 44.38% | **30.63%** |
| **Six-root macro** | **73.70%** | **72.44%** | **46.27%** | **22.23%** |

The matched shuffled-auxiliary arm attained only **49.52%** mean true four-bit XOR accuracy and **49.85%** mean true five-bit auxiliary XOR accuracy. Its final-logit degree-five Walsh share averaged **0.0085%**, compared with **22.23%** for XOR auxiliary training. The original vanilla arm's degree-five mean was **0.0885%**. Each spectral number is an *average of per-context nonconstant Fourier-power fractions, then root-macro averaged*, not a pooled power ratio.

In roots 12, 16 and 17, the exact supervised intermediate head could decode five-focal raw XOR at **90–97%** validation accuracy. Their final original task prediction nevertheless fell below chance, while the final head's fifth-order Walsh component became substantial. This combination is consistent with a **context-dependent parity phase, final-head readout, or MUX selection/integration problem**, with no causal localization yet. The original NEC04A composed parity teacher includes context-dependent XOR offsets, so recovering the raw five-focal XOR alone does not guarantee correctness of the final original task. In particular, **Walsh power is unsigned** and cannot verify that the final model's parity sign is aligned with the original teacher in each context.

The three other roots failed even the direct auxiliary XOR learning objective, so any hypothesis must preserve all six original world failures. The full 240/900 native source/validation logits, true stage labels, all three controls and optimizer receipts are preserved by the artifacts.

## Scientific next question and protection boundary

Before more model training, analyze the **existing archived** 900-step final native logit margins and original source/owned parity contexts of all six roots for the signed five-focal XOR Fourier coefficient **conditioned on original context XOR phase**. Quantify teacher-aligned signed parity signal, contextual sign inversion rates and accuracy under strictly labeled **posthoc diagnostic** sign/selector decompositions, with exact original checkpoint/label SHA replay. A high Fourier-power but systematically misaligned sign in the three learned-auxiliary roots would support a routing/readout hypothesis. Retain root unit n=6, prior model results and zero new optimizer updates. A later prospective new model would need fresh untouched calibration roots, a matched context-phase supervision/control and separate preregistration before any validation readout.

The original 24 developmental WORLD roots, 16 confirmation WORLD roots and all original calibration reserved contexts remain uninstantiated/unscored, as do old NEC03Z source groups56–63, NEC03P recruited-world sham outcomes and old embargo/confirmation sets. No source SAE training, functional/physical sham or neuron-community causal mechanism was evaluated. The original NEC03P 69/88 source-eligible pool still has **zero independently scored causal-sham root outcomes**; ≥59 independent eligible sham worlds and a valid physically and functionally inert null are prerequisites for a later directional 95/95 tolerance campaign.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
