# ROC-02D: input-binding interventions and frozen rank-eight observer responses

**16 November 2026. Registered before measurement at commit `714f149d6dabff24a3ef4bcec566ba7d93f4cf0d`.** The experiment is calibration-only on the original twelve short-trained ROC-02 models. Both ROC-02C rank-eight adapters are frozen from their earlier 80-update fits. The held-out source families 6–7 remain sealed.

Six calibration examples with indices 2–7 per model receive a legal source-A binding replacement from rule family 4 to family 5, or the reverse. The exact independent symbolic oracle confirms that every replacement changes the correct rule answer. All other source events and the query remain fixed. The **full legal source edit has a larger, separately reported hidden-state norm**.

A continuous interpolation of the first 79D encoded binding features finds a midpoint displacement of exactly 0.2 by a fixed 64-cell grid and 24 bisection steps. **This interpolated feature is not a valid symbolic binding and carries no exact label.** Further control directions are induced by the full legal input edit, or sampled deterministically within the selected observed subspace, its complement, and an isotropic null. They are all scaled to physical hidden-state L2 radius 0.2. Model and observer weights remain unchanged.

| Depth | Intervention measured by frozen eight-score observer | Task-only mean L2 change | Leakage-aware mean L2 change | Seeds with greater leakage-aware change |
|---|---|---:|---:|---:|
| 8 | Full legal input rule edit, unmatched radius | 0.258555 | 0.118625 | 1/6 |
| 8 | Soft input interpolation at radius 0.2 | 0.049076 | 0.023566 | 1/6 |
| 8 | Observed-subspace random hidden direction at radius 0.2 | 0.136995 | 0.102272 | 0/6 |
| 16 | Full legal input rule edit, unmatched radius | 0.230912 | 0.079755 | 0/6 |
| 16 | Soft input interpolation at radius 0.2 | 0.035255 | 0.011562 | 0/6 |
| 16 | Observed-subspace random hidden direction at radius 0.2 | 0.130482 | 0.074826 | 0/6 |

Across 72 unique model-case pairs, the full legal input flip moves the midpoint by mean 1.069 at depth 8 and 1.449 at depth 16. All can be interpolated to radius 0.2; average interpolation fractions are 0.202 and 0.165. The original trained model's future logits change by 0.338 and 1.191 under the full legal flip, and by 0.0587 and 0.1407 for the matched soft-input perturbation. The original model's output is unchanged by which observer is selected.

**Interpretation.** The leakage-aware observer is less sensitive to authentic changes in the source rule, even though ROC-02C found lower incoming Jacobian coupling and greater frozen future-logit Gram capture on its two calibration states. Its eight-score readout is not a verified rule-solving mechanism. These source families were already visible during gamma selection, and the original model has low calibration accuracy. This is a negative dev-split mechanism diagnostic; it neither opens the sealed test nor satisfies the full ROC-01 Amendment C causal-type or input-augmentation gates.

## Reproducibility

The public source `research_gym/analysis/roc02d_matched_input_causal.py` checks saved model and adapter SHA-256 against original training receipts, checks both input split hashes, and verifies each legal source-label change. `roc02d_summarize.py` writes 864 per-case effect rows, 12 paired-seed rows, SHA-256 receipt manifest and a ZIP containing the twelve complete JSON receipts. Stdlib tests audit every row and local tests check exact symbolic changes and deterministic orthogonal perturbations. The soft input control is a continuous feature edit, while the full legal change is a different-cost discrete edit; both labels appear explicitly in the output.
