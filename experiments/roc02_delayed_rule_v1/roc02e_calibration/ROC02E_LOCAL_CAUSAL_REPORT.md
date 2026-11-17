# ROC-02E: 256-pair rule-vs-cue causal intervention, calibration-only result

**Date:** 2026-11-17. **Status:** local CPU outcome completed and independently audited; hosted GitHub Actions rerun tracked separately. **Preregistration:** \`configs/roc02e_rule_vs_cue_causal_dev_v1.json\` at commit \`8fc97c97d9ed37a6a312933bc2040f5cc603d0c6\`, before ROC-02E measurements. All fitted weights, inputs, model identities and selected observer gamma values are inherited from ROC-02/ROC-02C, rather than changed for this experiment. The sealed families 6–7 were not opened.

## Frozen models and replay

The twelve original ROC-02 tied-residual checkpoints (six initializations each at depths 8 and 16; 240 training steps) are SHA-verified against merged original training receipts. Only training-family 0–3 and calibration-family 4–5 source files are loaded, under exact source digests. All 24 (model, rank) rank-7/8 observer cells reconstruct their gamma-0 and selected-gamma bases from the 80-update ROC-02C PR #144 fitting procedure, verifying task MSE, rank-null-normalized B², future-logit Gram capture, train MSE and calibration accuracy against the prior raw receipts to numerical tolerance 1e-10. Selected gamma remains fixed from the prior calibration selection.

For each of the 256 calibration records, an initial-A binding-table replacement (family 4 ↔ family 5) and a final-query-value shift by four modulo eight are both legal independent counterfactuals. The exact symbolic oracle certifies that both edits change the correct target to the same \`(y+4) mod 8\`, preserving all other events. The rule edit changes eight 8-way table one-hots (encoded-feature L2=4), whereas the cue edit changes one 8-way query one-hot (encoded-feature L2=√2). Full legal interventions retain their natural, unequal physical costs; separate soft input-feature interpolations match the midpoint hidden-state L2 to radii 0.05, 0.10 and 0.20. These soft vectors are continuous, off the discrete source manifold, and have no independent symbolic task labels.

Each model uses two observer ranks, two frozen observer arms, two legal edit kinds, three radii and seven physical/input routes: 256 × 2 × 2 × 2 × 3 × 7 = **43,008 per-case effect records per model**, 516,096 in total. All are reported as calibration-only. Checkpoint/output routes, physical midpoint displacements, valid/invalid masks and input-families are recorded.

## Registered primary result

At **depth 16, rank 7, radius 0.20**, the registered endpoint is the signed gain in the *original frozen eight-logit model's* counterfactual target-vs-original margin under the normalized projection of the true legal **early-rule** midpoint difference onto the candidate observer's span. Both tasks and joint observers use the same original model readout after the same suffix loops. The six independent model initializations are the inference unit; the 256 within-model cases are paired.

| Statistic | Result |
|---|---:|
| Mean task-only observed-rule steering gain | +0.001606 |
| Mean joint/leakage-aware observed-rule steering gain | +0.001748 |
| Mean paired joint − task-only | **+0.00014148** |
| Seeds with positive paired difference | **3 of 6** |
| One-sided exact sign-flip p over 64 signs | **0.375** |
| Descriptive t-based 95% six-seed interval for paired mean | **[-0.001011, +0.001294]** |
| Family-4 within-calibration mean paired effect | +0.000254 |
| Family-5 within-calibration mean paired effect | +0.0000467 |

### Six paired primary effects

| Seed | Joint minus task-only |
|---|---:|
| 2026111601 | +0.001742165 |
| 2026111603 | −0.001148760 |
| 2026111607 | −0.000597024 |
| 2026111609 | −0.000604106 |
| 2026111613 | +0.000659989 |
| 2026111619 | +0.000796637 |

**Primary registered differentiation gate: FAIL.** The direction is split 3/6 with small gain and p=0.375. This is a direct result on the same already-exposed calibration families used to select ROC-02C gamma. The sign-flip calculation describes six reproducible paired model effects and is *not* a prospective independent-source-generalization p-value.

## Input and causal comparators

At depth 16, rank 7, radius 0.20, the mean original-model margin gains are:

| Edit / intervention | Task-only | Joint |
|---|---:|---:|
| Full legal early-rule edit (unmatched dose) | +0.012010 | +0.012010 |
| Soft rule-input interpolation, hidden radius 0.20 | +0.001036 | +0.001036 |
| Rule-induced full-space hidden steering | +0.001037 | +0.001037 |
| Observed rule-induced steering | +0.001606 | +0.001748 |
| Dark rule-induced steering | +0.000757 | +0.000604 |
| Full legal late-query edit (unmatched dose) | −0.000485 | −0.000485 |
| Soft cue-input interpolation, hidden radius 0.20 | −0.000293 | −0.000293 |
| Observed cue-induced steering | −0.000391 | −0.000510 |

Equal-dosed input and full-space controls have identical original-model logits across observer arms by construction; that invariance is a routing check. The full legal early-rule edit elicits a larger directional change than the legal late-query edit, but the two legal edits have unequal encoded-input and physical midpoint costs. The soft controls are equal-radius, but off-manifold and have no source label.

An additional conservative **post-hoc per-case best-control upper envelope** chooses, for each case at radius 0.20, the largest directional gain among the soft early-rule input edit, soft late-query edit, task-only observed-rule steering, joint isotropic hidden steering and full-space input-induced hidden steering. Both soft edits are eligible on all 256 cases for every depth-16 seed. The joint observer's observed-rule gain falls below this upper envelope by a mean **−0.016509**, with **0/6** seeds beating it. Because this envelope chooses the best control after seeing each labeled case, it is a descriptive oracle upper bound rather than an implementable input-control policy. The individual components and all source cases are published for fairer fixed-policy comparisons.

## Interpretation and next gate

ROC-02C PR #144's 24/24 improved within-calibration task MSE and reduced null-normalized B remain independently replayable. This new intervention shows that the same leakage-aware bases do not offer consistent rule-targeted gains under a controlled hidden-state budget. The model's original delayed-rule calibration accuracy is only about 22% at depth 16, so output-Jacobian capture remains a sensitivity diagnostic on weak task predictions. The earlier rank-eight ROC-02D PR #130 negative source-readout sensitivity finding was obtained using different PR #127 observers and is preserved as a separate evidence line.

These findings do not satisfy the ROC-01 Amendment C complete five-mechanism classification, input-level prospective differentiation, or Chernoff allocation gates. Full label-preserving distractor perturbations and complete per-state six-direction A/B/C/H Holm cards remain additional work. Sealed source families 6–7 remain unopened. Follow-up should prioritize learning the delayed-rule task reliably and freezing a new independent source-family evaluation before seeking a mechanistic positive.

## Reproduction and provenance

- \`research_gym/analysis/roc02e_matched_causal.py\`: exact checkpoint and ROC-02C observer replay, legal dual counterfactuals, 64-grid+24-bisection radius matching, seven causal routes and per-case receipt writer.
- \`research_gym/analysis/roc02e_independent_audit.py\`: standard-library hash, 516,096-row census, target identity, dose, provenance, exact sign-flip and per-case best-input-envelope audit.
- \`research_gym/analysis/roc02e_source_pairs.py\`: independent frozen split SHA and legal-pair generator.
- Hosted workflow: \`.github/workflows/roc02e-causal-replay.yml\`.
- Per-model raw CSV-like JSONL gzip, 12 model JSON score cards, complete model/rank/arm/radius summaries and calibration-family strata and independent audit will be attached as a GitHub Actions artifact and made available in the sandbox. Use exact hashes in the generated artifact manifest rather than guessed receipt identifiers.

**Evidence scope:** local PyTorch CPU 2.10 / Python 3.13, frozen checkpoint/source bytes; no paid GPU and no sealed-test evaluation.
