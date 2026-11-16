# RAM-16 technical report: source-certified structural tactic supervision and fresh Lean proof search

**16 November 2026 · Completed original primary · Separate corrected secondary and ablation · Independent Lean 4.35 third audit passed**

## Research question and result

RAM-14 found source-derived Nat-induction and earlier-lemma rewrite actions that solved two previously exposed `GramianTransfer` theorems. RAM-15 gave these actions to the old 20-family recurrent/flat models through frozen semantic proxies and found no recurrent-only advantage. RAM-16 trained **new 23-family models** with genuine Lean-source-certified structural actions and compared their complete-theorem search against task-only controls, matched flat models and symbolic policies.

**The registered primary criterion failed.** On ten module-disjoint original MathCan theorems, recurrent task-only certified **6/10** and recurrent structural-curriculum certified **5/10**. Native ordering and global frequency each certified 6/10; both parameter-matched flat models certified 5/10. All primary model checkpoints and candidate-generation rules were frozen before TEST source access.

The protocol and cohort are preserved in [PREREGISTRATION.md](PREREGISTRATION.md), [AMENDMENT_A.md](AMENDMENT_A.md), [AMENDMENT_C.md](AMENDMENT_C.md), [FROZEN_SELECTED_MODELS.json](FROZEN_SELECTED_MODELS.json), and [LOCKED_TEST_MANIFEST.json](LOCKED_TEST_MANIFEST.json). The new objective produced a learned action ranker, but did **not** produce a verified additional original theorem in the preregistered primary search.

## 1. Native source-certified training corpus

The TRAIN/DEV root source is `MathCan@f427e0ef4a61093304d12ca2c0dc943d8d2429e5`. The collector physically excluded all four pre-existing protected TEST module files from its source checkout. The chronological TEST source module did not exist at that earlier commit.

| Native TRAIN/DEV measurement | Result |
|---|---:|
| Source-root theorems | 68 (50 TRAIN, 18 DEV) |
| Authentic source-derived tactic candidates compiled | 574 |
| Two-time full-original-source Lean proof successes | 39 |
| Independently triple-certified positives | **39/39** |
| Certified TRAIN positives | **27** |
| Certified DEV positives | 12 |
| Distinct positive TRAIN theorem roots | 14 |
| TRAIN Nat-induction roots | 2 |
| TRAIN earlier-lemma rewrite roots | 1 |

The registered admission gate passed in [native independent run 43803934041](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43803934041). All 39 positives have three complete original-source compilations with accepted axioms and genuine original Lean goals. Unknown or nonclosing compiler attempts remain censored. The specific new-family TRAIN counts are **2 Nat induction, 2 prior-lemma rewrites and 23 prior-definition simplification positives**. Both induction roots and the earlier-lemma rewrite root occur in the previously exposed `GramianTransfer` module. That module contributes **8/27 (29.6%)** of TRAIN positives.

Original collection [run 43802741798](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43802741798) retains every attempted compiler source and failure. Initial dependency-resolution and audit-import failures are documented in [AMENDMENT_B.md](AMENDMENT_B.md) and [AMENDMENT_D.md](AMENDMENT_D.md); the original successful source experiment is unchanged.

## 2. Matched model training and development selection

Twelve CPU fits crossed two architectures, two objectives and seeds 17, 29 and 43. The recurrent model has **207,365 parameters** and the flat model **208,390**, a 0.49% difference. Each fit used 400 AdamW updates and the two objectives began from identical weights for each architecture/seed pair. Structural training allocated exactly **2,560 of 25,600 sampled action examples** to independently certified macro positives.

| Architecture | Objective | DEV-selected seed | Selected optimizer update | Original DEV loss |
|---|---|---:|---:|---:|
| Looped64 | Task-only | 17 | 200 | 0.49784 |
| Looped64 | 10% verified macros | 43 | 400 | 0.52704 |
| Flat71 | Task-only | 29 | 200 | 0.52638 |
| Flat71 | 10% verified macros | 29 | 200 | 0.48409 |

The twelve-fit [private run 43804250337](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43804250337) and independent repeat [43804203924](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43804203924) completed successfully. Selected checkpoints are byte-pinned in [FROZEN_SELECTED_MODELS.json](FROZEN_SELECTED_MODELS.json) and [LOCKED_CHECKPOINTS.json](LOCKED_CHECKPOINTS.json).

On original root-disjoint DEV observations, structural training increased recurrent loss for all three paired seeds (17: 0.49784→0.53450; 29: 0.53847→0.57132; 43: 0.52429→0.52704). Flat structural training improved DEV loss for one of three seeds, at 29 (0.52638→0.48409), while the other two worsened. Seed selection was frozen using DEV only. The selected recurrent models use different seeds, so their direct 6/10 versus 5/10 theorem comparison combines the training objective with the preregistered DEV-selection procedure. A causal objective claim would benefit from complete proof searches for every paired seed.

An independent [original TRAIN-overlap audit 43804593091](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43804593091) found **zero** historical RAM-08 TRAIN or DEV observations from any of the five protected TEST source modules.

## 3. First original TEST result and strong controls

The TEST source revision was frozen at `MathCan@fdbe1581e95427016699169fd19c9e915a40bcf9`. Ten original declaration roots were selected by taking the first two theorem/lemma statements from each of five registered modules after the training checkpoints were frozen. `NOR18RReciprocalReuse` is chronologically new relative to the training source. Four other modules were protected complete module holdouts, even though their files existed in the earlier public source revision.

| Policy | Original primary, 30s / 48 tactics | Corrected secondary, 60s / 96 tactics | Corrected original-20-family ablation |
|---|---:|---:|---:|
| Native order | **6/10** | **6/10** | **6/10** |
| TRAIN-only global frequency | **6/10** | | |
| TRAIN-only shape frequency (prior 8) | 5/10 | 5/10 | |
| Looped64, task-only | **6/10** | | |
| Looped64, verified structural macros | 5/10 | 5/10 | 5/10 |
| Flat71, task-only | 5/10 | | |
| Flat71, verified structural macros | 5/10 | 5/10 | |
| Frozen RAM-08 Looped64 proxy | **6/10** | | |

The eight-controller first primary consists of **80 policy/theorem cases** in [run 43805791781](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43805791781). Forty registered secondary and twenty original-family ablation cases were separately repaired postscore under their original budgets, with all sixty invalid original phase receipts preserved. The final auditable composite contains **140 valid original or corrected cases** and sixty retained invalid records.

### Per-theorem first primary coverage

| Original MathCan theorem | Certified policies /8 |
|---|---:|
| `NOR18RReciprocalReuse.sound_prefix_filter_preserves_full_acceptance` | 0 |
| `NOR18RReciprocalReuse.reciprocal_calls_per_demo_exact_saving` | 8 |
| `TemporalCutCapacity.tied_step_increments_phase` | 8 |
| `TemporalCutCapacity.one_head_four_visits_separates` | 4 |
| `PhaseLatch.iterStep_add` | 0 |
| `PhaseLatch.shifted_state_remains_shifted` | 0 |
| `ObserverLabelAlignment.k3_complete_edge_separation` | 8 |
| `ObserverLabelAlignment.k3_total_label_failure` | 8 |
| `ContractionAliasing.anchor_cancels` | 8 |
| `ContractionAliasing.affine_increment` | 0 |

The one four-policy target, `one_head_four_visits_separates`, was proved by native, global frequency, task-only Looped64 and the original frozen Looped64. It was missed by both structural-curriculum models, the task-only flat model and shape frequency, within the shared 48-call budget.

### Corrected phase provenance

**Original secondary defect:** the first implementation did not inherit `request_seconds` and `max_goals` from the primary budget. All forty first-run secondary receipts therefore returned a zero-call executor error. [AMENDMENT_E.md](AMENDMENT_E.md) identifies the failure and records the fixed-budget secondary-only [run 43806362281](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43806362281).

**Original ablation defect:** RAM-16's wrapper set the old twenty-family grammar flag, then called the RAM-15 searcher without forwarding it. The base search reset the flag to the expanded grammar, making the original nominal ablation's twenty scores invalid as a grammar comparison. [AMENDMENT_F.md](AMENDMENT_F.md) records the mechanism and the corrected original-grammar ablation-only [run 43807259096](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43807259096).

Both repairs used the original model bytes and same theorem source, with separate output directories and unchanged primary scores. Their findings are **postscore operational follow-ups**, reported as descriptive; the original primary remains the first untouched prospective evaluation.

## 4. Independent original-source proof certification

A separate [audit-only run 43807604461](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43807604461) fetched exactly the original case artifacts and the two corrected-phase runs. It verified the source/phase separation and source-byte hashes, preserved every invalid original phase receipt, reconstructed the original complete theorem statements and compiled every accepted proof in a fresh Lean 4.35 process with permitted transitive axiom inventory.

| Independent proof trust measure | Value |
|---|---:|
| Valid original-primary and corrected phase cases | 140 |
| Additional preserved invalid original phase cases | 60 |
| Accepted policy/theorem cases | **76** |
| Independently recompiled full-source proofs | **76/76** |
| Three-compilation total for accepted sources | **228** |
| Accepted cases using structural tactics | 36 |
| Invalid source certificates detected | **0** |

Full original source/compiler logs and the three-phase composite are archived as private GitHub Actions artifact **14198892808**, SHA256 `466515a1d0b6b3921a7bff4b8aeb2fd88df0fbed8b996c39e8367b4b637754e4`. They were never uploaded to public MathCan alongside private checkpoint weights.

## 5. Mathematical interpretation and follow-up

Adding a structural action family expands the candidate grammar. Let `C_G(s)` be the original candidate pool at Lean proof state `s`, and `C_{G^+}(s)` the extended pool; then `C_G(s) \subseteq C_{G^+}(s)` under the source-visible generator. With a fixed query cap, the set of **visited** actions still depends on the controller's ranking and can shift when new candidates consume query slots. More candidate expressivity does not imply monotone proof completion under a bounded search policy.

RAM-16 supplies a measured example: thirty-six accepted case certificates used structural macro scripts, while the corrected old-grammar ablation and expanded grammar certify the same number of original theorems for native ordering and Looped64-structural. The learned recurrent curriculum also loses one original theorem against task-only training. That is consistent with **supervision coverage and policy selection**, rather than parameter count alone, limiting transfer.

The new-family TRAIN positives remain concentrated in 27 one-shot source proofs, with the only verified Nat-induction and prior-lemma rewrite positives occurring in the Gramian source. A follow-up should expand truly independent compiler-supported macro positives over disjoint original theorem families, register family-balanced sampling before fitting, and preserve action-budget-matched symbolic controls. Evaluation should include all paired seeds on newly unopened theorem roots to isolate the training objective from DEV-selected seed variation.

The ten TEST theorem sources remain publicly visible, and only one module is chronologically new. A bounded proof-search failure is an empirical budget outcome. This study gives no recurrence-only capability result and does not modify RAM-08 PEF or RAM-10 original registered evaluations.

## Reproducibility ledger

- [Registered study](PREREGISTRATION.md), [original source-only macro gate](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43803934041), [original 12-fit models](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43804250337).
- [Machine result and certificate trust receipt](RAM16_RESULT_20261116.json).
- [First primary plus original invalid-phase case matrix](RAM16_ORIGINAL_FIRST_RUN_CASES_20261116.jsonl), [secondary-only repaired case matrix](RAM16_CORRECTED_SECONDARY_CASES_20261116.jsonl), [original-family ablation-only matrix](RAM16_CORRECTED_ORIGINAL20_ABLATION_20261116.jsonl).
- [First native TEST run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43805791781) and [independent complete-source audit](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43807604461).
