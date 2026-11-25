# ROC-06A: train-only counterfactual-gradient observer raises hidden steering, while equal-dose input remains stronger

**25 November 2026 · preregistered primary FAIL; independent hosted full-receipt replay PASS.**

The scientific protocol was frozen in commit `1c209f6db0f56dd0bb4d4677d1fb22a8cd462a8c` before any ROC-06A causal measurement. All source/model checkpoint hashes and the six original first-run model seeds are frozen from ROC-05A/ROC-05B. The completed [GitHub Actions run #45569749676](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45569749676) passed numerical-gradient preflight, twelve first-run checkpoint identities, all six model-seed causal jobs and the independent receipt audit.

## Hypothesis, architecture and frozen primary

ROC-05B's train-only rank-7 supervised state/label observer failed to outperform a same-hidden-dose continuous input edit under legal source-rule changes (mean observer minus input −0.06805338). ROC-06A instead constructs a **rank-7 counterfactual-gradient observer** using only each model arm's 1,024 training examples. The legal early source-A table-row swap changes the true target from y to y′; the train-only gradient is the four-step suffix derivative at the original hidden midpoint of the **native model's signed changed-target logit margin** z(y′)−z(y). Its reverse-mode analytic VJP was independently tested against PyTorch autograd and directional finite differences. Each nonzero train gradient was L2 normalized, its 1,024-row matrix centered, and the top seven right singular directions selected. No exposed development labels, gradients or interventions were used in the observer fit.

The same frozen QueryConditionedTableMemory models (width 128, eight tied steps, 51,520 parameters) receive a legal early A-rule binding edit on each of 256 exposed development cases. All hidden routes are inserted after the first four recurrent steps and scored with the original four-step suffix, original table head and unchanged final query. The physically comparable hidden and input routes receive the same midpoint L2 radius 0.05, 0.10, or **primary 0.20**. The softened input is continuous and off the discrete source manifold. Its logit gain is a **physical sensitivity control**, while the full legal discrete edit's task accuracy is reported independently.

The registered primary is the gradient observer's mean correct-new-target signed logit margin gain minus the **larger of the two precommitted seed-mean reference gains** (supervised rank-7 steering and softened input) at radius 0.20, on the identical triple-eligible cases for each frozen high-512 seed. Acceptance requires ≥205/256 triple-eligible dev cases per seed, ≥5/6 positive seed effects, positive six-seed mean and one-sided exact six-sign p≤0.05.

## Six-seed causal result

All six high-512 seeds had **256/256 eligible three-way comparisons**.

| Original frozen seed | Gradient rank-7 | Supervised rank-7 | Same-L2 soft input | Gradient minus best reference |
|---|---:|---:|---:|---:|
| 20261117231 | +0.130745 | +0.117114 | +0.197126 | **-0.066381** |
| 20261117233 | +0.125644 | +0.107219 | +0.180734 | **-0.055090** |
| 20261117239 | +0.135669 | +0.130885 | +0.186401 | **-0.050732** |
| 20261117241 | +0.131419 | +0.120991 | +0.186984 | **-0.055565** |
| 20261117251 | +0.130900 | +0.113885 | +0.185136 | **-0.054236** |
| 20261117257 | +0.121885 | +0.117713 | +0.179745 | **-0.057860** |
| **Mean** | **+0.129377** | **+0.117968** | **+0.186021** | **-0.056644** |

The train-only gradient observer improves the supervised hidden route by **+0.01140925 original-logit margin units on average**, with **6/6** positive paired seed differences. Its mean gain is approximately **9.67% larger** than the supervised gain. This predefined secondary comparator is *descriptive*: if evaluated as a one-sided sign-flip test it has p=1/64, but the single preregistered primary requires beating the input comparator too.

The softened-input control is the **larger seed-mean comparator in every frozen high-512 model**. The registered primary mean gradient-minus-best effect is **−0.0566440431951681**, with **0/6 positive paired seed differences** and an exact one-sided 64-sign p-value **1.0**. Accordingly, **ROC-06A primary gate FAIL**. The original-model legal changed-rule task check remains **1,157/1,536 = 75.33%** correct in the high-512 arm, with **1,147/1,536 = 74.67%** correct on the original unedited cases. These legitimate symbolic inputs give context for the off-manifold same-dose sensitivity control.

## Gradient geometry and rank-matched nulls

Every one of the twelve original frozen models contributes two explicitly indexed exposed shadow states. Each shadow has 128 SHA-keyed rank-7 Gaussian/QR random projectors, with complete computed-change D=T−I block A/B/C and original fixed-query output Gram H. ROC-06A publishes **3,072 raw null records**, **72 gradient/supervised/PCA observer cards**, and **432 one-sided tail/Holm values**. The independent audit checks all of these against the published original receipts and per-case routes.

For the twelve *high-512* model shadow cases per basis, representative mean ratios and fixed-query future-logit sensitivity capture are:

| Train-only rank-7 observer | Incoming B / rank-matched random median | Future-logit Jacobian H capture |
|---|---:|---:|
| Gradient basis | **1.780×** | **31.51%** |
| Supervised cross-covariance basis | 1.749× | 26.31% |
| PCA basis | 1.673× | 25.58% |

The gradient basis captures **34.13%** of the centered, individually normalized train-gradient variation energy in its top seven singular directions, averaged across the six high-512 seeds. The gradient and supervised bases retain partial overlap, but their projection geometry differs. **All 72/72 local observer cards remain unclassified** under the six-tail/ Holm ROC-01 diagnostic predicates. Gradient capture of native output sensitivity increased while the incoming computed-change coupling remains above the same-rank random median, so this basis does not constitute a nearly closed recurrent quotient.

A separate [mathematical journal note](GRADIENT_GEOMETRY_JOURNAL.md) records the exact Ky Fan gradient-energy objective and the first-order signed-margin formula. Its central limitation is structural: maximizing ||Pg||² over training gradients controls a norm bound on steering. The measured signed effect depends on the orientation of the true legal source-induced displacement δ, through gᵀPδ/||Pδ||, and nonlinear suffix dynamics.

## Reproducibility and claim boundary

- Original checkpoint/source lock: ROC-05A first training [run #44101721734](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44101721734), frozen by ROC-05B preregistration and revalidated here.
- Frozen ROC-06A protocol and exactly two comparator seed means: `configs/roc06a_train_counterfactual_gradient_observer_v1.json`.
- Analytic gradient, train-only observer fit, equal-radius inputs and original native logits: `research_gym/analysis/roc06a_counterfactual_gradient.py`.
- Independent full census: `research_gym/analysis/roc06a_independent_audit.py`, checking **92,160** model/case/radius/route entries, **3,072** null measurements, **72** cards, input radius validity and all **64** seed sign assignments.
- Complete [hosted CI run #45569749676](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45569749676), artifact `ROC06A-complete-frozen-gradient-causal-replay`, contains all 12 original first-run frozen model checkpoint bytes, train/dev source inputs, 12 per-case compressed CSVs, gradient singular receipts, geometry nulls, scripts, independent summary and **69 SHA-256-verified files**. Standalone archive SHA-256: `f0ca87809dd3675dab7baee4e7d02743e97301cffb96dc772e2cb7d2700f9093`.

The first attempt to run the experiment preflight failed at a workflow config-key lookup before any ROC-06A causal result. The next hosted attempt computed all six seed effects but failed the wrapper's check because its receipt study label had an outdated identifier. That metadata field was corrected without changing the frozen model, source, basis, intervention, rank, radius, or endpoint. The successful final run independently verifies the complete outcomes. These are numerical repeats of the same six original model seeds, not independent extra trials.

ROC-06A remains a **development-family mechanism diagnostic**, since ROC-05A task outcomes and ROC-05B causal effects were available when the gradient observer was registered. The 384 ROC-05 reserved rule identities remain ungenerated as input examples, and the original ROC-02 family-6/7 source examples remain sealed. No new model training occurred here.