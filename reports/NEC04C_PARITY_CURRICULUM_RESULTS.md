# NEC-04C: matched source-parity curriculum failed to teach native five-bit parity

**Completed:** 22 November 2026. **Evidence class:** preregistered six-root native-model training and within-calibration-world owned-context validation. **Decision:** `STOP_PARITY_CURRICULUM_CALIBRATION_PILOT`.

[Issue #296](https://github.com/MoralityLabAI/LoopedTransformers/issues/296) registered the original NEC04A *calibration* roots 6–11, paired vanilla/curriculum native training arms and fixed stop conditions. Source-lock, optimizer recipe, root roster and code were committed **before training** at [`b6e7a9a2`](https://github.com/MoralityLabAI/LoopedTransformers/commit/b6e7a9a2e2b530e738967aa15d37c265643e98b4). Historical diagnosis from the original calibration roots 0–5 was independently audited in merged [PR #297](https://github.com/MoralityLabAI/LoopedTransformers/pull/297). The present experiment uses newly untouched calibration-world roots and is a distinct prospective training comparison.

## Execution and independent receipts

[**GitHub Actions run 44993859839**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44993859839) completed successfully: the pretraining code/seed/sampler test suite passed **6/6**, independently audited **15,360 original source** and **4,608 owned-calibration-world validation** Boolean labels, and ran all **6/6 CPU root workers**. Each worker trained two native 32D, twelve-recurrent-step Looper models with identical root-specific initial weights, 900 continuous AdamW updates (LR 0.003, batch 64, clip 1.0, no weight decay), and checkpoints at 240 and 900. All **24/24 checkpoint bytes, optimizer-state update counts and full source/owned-validation native logits** passed separate model-checkpoint replay. The independent auditor re-evaluates saved checkpoints; it does not retrain each optimizer trajectory from scratch.

**Source:** original NEC04A 24-calibration-world corpus run [44983509977](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44983509977), source SHA256 `82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0`, root registration SHA256 `88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e`. Exactly 80 source context blocks ×32 focal masks (2,560 cases) trained each root; exactly 24 original **owned validation-context blocks in that same calibration world** (768 cases) were scored only after both model arms' checkpoint manifests were sealed. All original *calibration* reserved contexts and the separate 24 NEC04A development WORLD roots and 16 confirmation WORLD roots remain sealed.

**Pretraining artifact:** `nec04c-frozen-original-six-root-oracles-before-model-training` (ID 14742886436). **Six individual native model + audit artifacts:** root6 ID 14742124547; root7 ID 14743106877; root8 ID 14743221555; root9 ID 14742897690; root10 ID 14743391255; root11 ID 14742099591. **Final independent six-world aggregate:** artifact **14742039872**, `nec04c-six-root-parity-curriculum-independent-native-paired-audit`, ZIP SHA256 `34b8e4795897ccedd9ce8597daa769fc51fcb01d8b741aa874542cfcbc9ecfac`.

## Preregistered comparison

The **vanilla** arm sampled all source examples uniformly for 900 updates. The **curriculum** arm sampled only original context-MUX-selected parity cases for exactly 240 updates, then used a fixed 32 parity / 32 alternate source minibatch for 660 more updates, always with the original composed Boolean labels, identical architecture and the same root-specific starting weights. Both used continuous AdamW moments. The second arm's curriculum was fixed before the newly untouched six roots were trained.

The **one primary endpoint** was the six-root macro mean of (curriculum − vanilla) owned-validation correctness on the MUX-selected five-focal XOR parity branch **at 900 optimizer updates**. Prospective meaningful-gain conditions were at least **+10 percentage points** and **at least 4/6 strictly positive paired root differences**. The separate alternate-branch retention constraint required macro curriculum-minus-vanilla accuracy at least **−5 percentage points**. A complete 32-mask parity branch has exactly 50% class balance.

### Per-root original owned-world validation at 900 updates

Every row is one independent calibration-world root. All percentages are measured on that root's 768 owned validation cases, with variable original parity/alternate-case allocations.

| Calibration root | Tier | Vanilla parity | Curriculum parity | Parity Δ (pp) | Vanilla alternate | Curriculum alternate | Alternate Δ (pp) | Whole-task Δ (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cal-006 | Easy | 50.00% | 49.55% | **−0.45** | 72.50% | 86.25% | +13.75 | +5.47 |
| cal-007 | Medium | 50.00% | 49.72% | **−0.28** | 76.44% | 79.33% | +2.88 | +1.43 |
| cal-008 | Hard | 50.57% | 51.70% | **+1.14** | 97.12% | 91.35% | −5.77 | −2.60 |
| cal-009 | Easy | 49.48% | 50.00% | **+0.52** | 90.36% | 94.79% | +4.43 | +2.47 |
| cal-010 | Medium | 50.00% | 48.61% | **−1.39** | 78.33% | 84.58% | +6.25 | +3.39 |
| cal-011 | Hard | 50.00% | 50.00% | **0.00** | 84.82% | 82.37% | −2.46 | −1.43 |

At the preregistered 900-update endpoint, the **six-root macro parity accuracy** was **50.0079% vanilla** versus **49.9309% curriculum**, a paired difference of **−0.0770 percentage points**. Exactly **2/6** roots had positive parity differences. Descriptive across-root standard deviation of paired parity deltas was **0.8644 pp** (standard error **0.3529 pp** at n=6, without a population-level causal inference). The fixed +10 pp / ≥4-root parity gate therefore **FAILED**.

The **six-root macro alternate-branch difference was +3.1812 pp**, so the registered −5 pp retention floor **PASSED**. Whole-task validation means were **67.9036% vanilla** and **69.3576% curriculum**, a paired mean **+1.4540 pp**, driven by alternate-branch behavior while parity remained near chance. Root cal-008 lost 5.77 pp on its alternate branch; this per-root heterogeneity stays in the original six-world denominator.

At the 240-update checkpoint, curriculum models had received only the first 240 source-parity-focused steps: their root-macro parity validation accuracy was **49.2898%** versus vanilla **50.3631%**; full-task means were **48.7630% curriculum** versus **64.0191% vanilla**. The early checkpoint is a preregistered diagnostic; the single primary endpoint remains 900 updates.

The pooled raw five-bit parity validation count at 900 was **1,072/2,144 vanilla** and **1,071/2,144 curriculum**; this pooled count is descriptive and does not replace the paired **n=6 world-root** unit. Original branch priors, balance/majority baselines, root-specific conditional accuracy, both checkpoint budgets, unrounded scores and native checkpoint SHA receipts are in final aggregate artifact 14742039872.

## Scientific interpretation and stop

A change to source sampling that emphasizes exact five-bit parity for 240 updates, followed by 660 context-selector-balanced full-target updates, **did not produce meaningful parity learning** in the tested native 32D twelve-loop Looper under the registered 900-update budget. The improved full-task score in some roots arose from the alternate Boolean branch. This result isolates a concrete failure of this fixed curriculum and model/training recipe; it alone does not distinguish representational inability from optimization difficulty, nor does it establish that parity is unlearnable by a different architecture or budget.

**Machine-recorded decision:** `STOP_PARITY_CURRICULUM_CALIBRATION_PILOT`. **No native headroom budget selected. No source SAE fitted. No physically matched sham, source-specific necessity, independent development-world outcome or confirmation outcome scored.** This is native training evidence within six calibration world roots, with no population extrapolation from the correlated masks/contexts.

A separately preregistered follow-up could test an explicitly constructive XOR composition/decomposition or memory-state readout under controlled capacity and equal compute, preserving a new calibration-world allocation and freezing the baseline+power/stop conditions first. The original NEC04B native-headroom gate remains failed. The older NEC03P pool's 69/88 source-eligible recruited worlds have **zero measured causal-sham outcomes**, with ≥59 independently eligible causal-sham roots still required for a later 95/95 directional tolerance bound after validating a physically and functionally inert null.

**FSI-01: `REDESIGN_OR_PILOT_ONLY`.**
