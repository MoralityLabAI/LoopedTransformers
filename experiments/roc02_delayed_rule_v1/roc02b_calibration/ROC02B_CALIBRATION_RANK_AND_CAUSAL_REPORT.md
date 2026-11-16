# ROC-02B: calibration-only observer rank, output capture and causal-tangent pilot

**Status:** frozen-source, calibration-only pilot. Registered 16 November 2026 in `configs/roc02b_calibration_observer_card_v1.json` at commit `f9cc046` **before** the new rank-grid measurements. The previously merged ROC-02 source and twelve 240-AdamW-step checkpoints are unchanged. The prospective ROC-02 held-out family (6–7) remains sealed. This is an intermediate model/observer audit, not ROC-02's final registered causal discrimination or full task outcome.

## Source and measurements

- Models: six paired initialization seeds `2026111601, 2026111603, 2026111607, 2026111609, 2026111613, 2026111619`, trained depths 8 and 16, ambient recurrent width 128.
- Observations: after all input events and half of the post-sequence residual steps. Complete the remaining 4 or 8 updates and differentiate that suffix: `T = D F_suffix(h_mid)`, `D = T - I`.
- Basis: train-only task-supervised cross-covariance rank-k singular vectors, with PCA fill above the task-output rank, or train-only PCA. Every rank uses the precommitted grid `{1,2,3,4,7,8,12,16,32}`. The task-supervised rank-eight direction contains an arbitrary null singular direction because centered eight-class one-hot outputs have rank at most seven; subspace SVD/QR is numerically deterministic in the original CPU runtime but null directions have no canonical interpretation.
- Output-sensitivity Gram `H = (R T)^T(R T)` uses the **original trained eight-logit readout** `R`. It measures local future-logit Jacobian capture rather than verified delayed-rule recall.
- Compute `A=U^T D U`, `B=U^T D (I-UU^T)`, `C=(I-UU^T) D U`, their full `||D||F` ratios, `||B||F/sqrt(k)`, and output capture `s=tr(U^T H U)/tr H`.
- Two fixed calibration shadow states per model, indices 0 and 1, each compared with **128 independent SHA256-keyed rank-matched Gaussian QR subspaces**. Per-state empirical one-sided p values use inclusive ties and denominators 129. Six-direction Holm adjustments and diagnostic tags follow ROC-01 Amendment C, abstaining where undefined.
- Matched physical perturbations: hidden L2 radius 0.2, inside observed span, inside dark complement and isotropic; report original-model eight-logit displacement, within/dark terminal displacement, task margin change and tangent prediction residual. Readout ablation is reported separately because its intervention norm differs.
- All source model bytes are SHA verified against merged ROC-02 training receipts, as are input train/calibration files. **No sealed-test file is read by the analysis**.

## Results

All twelve checkpoints completed, giving 216 observer-model-rank summaries, 432 observer-by-shadow-state rows and 27,648 rank-matched random-null rows. All 216 null families have 128 distinct subspace fingerprints.

| Depth | Train-only observer, k=8 | Calibration accuracy | B / rank-matched null median | Future-logit Jacobian capture s |
|---|---|---:|---:|---:|
| 8 | supervised covariance | 24.61% | 1.6755 | 0.5895 |
| 8 | train-only PCA | 22.07% | 1.6878 | 0.5645 |
| 16 | supervised covariance | 22.53% | 2.1887 | 0.4998 |
| 16 | train-only PCA | 23.96% | 2.2413 | 0.4726 |

The original trained network's mean calibration accuracy is 21.55% (R8) and 22.40% (R16). The low accuracy and synthetic source-family shift matter: capturing the model's future logits **does not imply capturing the correct rule answer**.

All 432 observer-state cards remain `unclassified` under the full ROC-01 Amendment C diagnostic-tag predicates, despite individually significant upper-tail B and s for the rank-eight arms in 12/12 states at each depth. The observed range of rank and task/error/relative coupling outcomes defines a calibration-only three-coordinate Pareto surface. It has not been used to choose the prospective sealed-test rank or penalty.

Finite-radius experiments indicate output dependence on observed perturbations, but this is **not** an equal-cost input augmentation comparison, nor an observed neural intervention signature cover or a completed four-arm Chernoff program. They are local diagnostics on two already accessible dev states.

## Reproduction

- `research_gym/analysis/roc02b_calibration_card.py`: loads frozen original checkpoint SHA, does train/dev-only state extraction, full dense suffix Jacobian, 128 rank-matched nulls at each registered rank and state, six Holm probabilities and norm-matched physical interventions.
- `research_gym/analysis/roc02b_summarize.py`: verifies cardinalities and distinctness, publishes rank-calibration/causal means and true three-coordinate Pareto records.
- `tests/test_roc02b_calibration_card.py`: five local PyTorch geometry, native state, exact derivative, deterministic null and rank-p tests.
- `tests/test_roc02b_public_receipts.py`: five standalone public-data checks including all 27,648 random nulls, observer states and Pareto membership.
- `experiments/roc02_delayed_rule_v1/roc02b_calibration/`: CSVs, compressed null table, SHA-256 receipt manifest, full source-script replay instructions and six-seed outcome summary.
- Source original model bytes and optimizer states: `experiments/roc02_delayed_rule_v1/checkpoints/` from merged ROC-02 PR #114. Exact symbolic train/calibration sources from merged ROC-02 source archive.

## Remaining registered gates

The **80-update equal-budget task-only and leakage-penalized** candidate fits, train-calibration-only model adaptation, additional causal observer selection, rank-matched norm-controlled input augmentation, actual T-C intervention type-identification/Chernoff calibration, and the sealed held-out family all remain untested. The existing low calibration accuracy may limit the interpretability of output-sensitivity measurements. A successful neural differentiation claim still requires evidence beyond input controls with at least five of six seed directions and the declared multiple-testing family. ROC-01 synthetic identifiability mathematics and retrospective NRF-02A results are separate evidence lines.