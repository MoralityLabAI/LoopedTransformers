# ROC-06A: train-only counterfactual-gradient observer

**Frozen registration:** 1c209f6db0f56dd0bb4d4677d1fb22a8cd462a8c, 25 November 2026, before ROC-06A causal outcomes.

## Scientific question

ROC-05A showed strong rule-transfer accuracy from 512 rather than four permutation tables in six fresh model seeds. ROC-05B tested supervised rank-7 hidden-state projection against an input-feature interpolation with equal hidden midpoint L2 cost, and the supervised route underperformed in all six stronger models (mean −0.06805338, exact one-sided p=1). The present experiment tests a **train-only rank-7 gradient basis** that incorporates the output consequences of legal source-rule changes.

## Frozen models and data

Load exactly the original twelve ROC-05A first-run models (six seeds × high512/low4), with full SHA-256 hashes in the registration JSON. Use 1,024 arm-matched training examples to build each observer, and exactly 256 previously exposed ROC-05A development cases for causal interventions. The original ROC-02 sealed family-6/7 source examples and ROC-05's 384 reserved-rule test examples are excluded from every stage.

For each training row, swap the values at the queried A-table row q and row (q+4) mod 8 to construct a legal new label yprime. The intervention preserves intermediate events and query. Compute g_i = ∇_{h_i}[z_{yprime}(h_i,q_i)-z_{y}(h_i,q_i)] at the original unedited train midpoint, after the initial four of eight tied residual reasoning steps. The native remaining four steps and original table head define z. Use an analytically evaluated four-step reverse VJP, checked against autograd before causal evaluation. Normalize each finite nonzero 128-dimensional gradient to unit L2, center the 1024×128 train-only matrix, and use its top seven right singular vectors. The seventh singular value must exceed 1e-9. No development outcomes, gradients or labels select the subspace.

## Registered causal comparison

Keep the ROC-05B supervised rank-7 training-state/label cross-covariance subspace, with train-only PCA and random rank-7 controls. Reproduce the same source-A row-swap counterfactual on all 256 dev examples. Compare hidden gradient-projection, supervised-projection, dark-component, PCA, full induced source direction, random and isotropic directions at midpoint L2 radii 0.05, 0.10 and primary 0.20. The soft-input comparator changes only initial rule-table input features along a linear path, selected by the first radius crossing on a fixed 64-cell grid plus 24 bisections. It lies off the valid symbolic input manifold. Report the full legal discrete input at its natural cost as a separate task control.

The output endpoint is signed new-target-versus-old-target logit margin gain using the unchanged native model. The **single primary** compares the mean gradient-projected gain against the larger of the mean supervised-projected gain and mean softened-input gain, evaluated on the exact same triple-eligible cases in each high512 seed at r=0.20. Each seed must have at least 205/256 triple-eligible cases; otherwise report ABSTAIN. A PASS requires at least five positive seed differences, a positive unweighted six-seed mean, and one-sided exact six-seed sign-flip p≤0.05 over 64 assignments. Low-four and secondary radii are descriptive.

On the first two exposed development shadow cases per model, compute D=T-I, A/B/C blocks, native output Gram and 128 rank-matched random subspaces, with six empirical tails and Holm correction for gradient, supervised and PCA bases. Retain the complete 3072 raw null observations and 72 observer cards, including all unclassified/abstained cases.

## Interpretation boundary

The ROC-05 development accuracies and ROC-05B causal outcomes were known before this registration, so this is a development-family mechanism diagnostic. Even a PASS supports only this train-derived rank-7 observer and physical comparison, and it cannot establish a complete ROC-01 five-mechanism classification. No checkpoint refitting, test-set opening or training updates are authorized.