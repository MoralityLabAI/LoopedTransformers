# ROC-02C: equal-budget readout optimization on frozen delayed-rule models

**Status:** calibration-only follow-up. **Date:** 2026-11-17. **Preregistration:** `configs/roc02c_calibration_equal_budget_v1.json` at commit `a6501465438e8226107341710c4463726428a35e`. The twelve ROC-02 model checkpoints have exactly 240 original AdamW training updates and remain frozen throughout this study. The ROC-01 Amendment C rank-relative objective is used. No source-family-6/7 sealed-test input or label was evaluated.

## Experiment

All six original model seeds were crossed with recurrent depths 8 and 16, and observer ranks 7 and 8. Within each model-rank pair, the task-only and gamma-penalized observers start from identical, train-only supervised cross-covariance bases. Each gets 80 Adam steps at learning rate 0.02, QR-retracted to an orthonormal 128-by-k basis. At each step, an 8-class ridge head with regularization 1 is refit on the 1,024 source-family-0/3 training cases. The exact quadratic form of the centered train MSE avoids repeated materialization of the 1,024-by-128 state matrix.

The rank-adjusted computed-change loss follows ROC-01 Amendment C, with `D=T-I` for the frozen last `depth/2` residual updates. For each state i, `b_i(U)=||U^T D_i (I-UU^T)||_F/||D_i||_F`; its denominator is the median `b_i` over 128 SHA256-keyed rank-matched Gaussian/QR subspaces. The training objective is `train_MSE + gamma * mean_{i=0,1} (b_i(U)/null_median_i)^2`, for gamma in `{0,0.05,0.2,1}`. All model checkpoints and training/calibration source files are SHA-256 checked before loading.

For each model-rank cell, gamma is chosen using **calibration source families 4/5** and shadow indices 2/3, requiring calibration MSE at most 1.10 times the equally optimized gamma=0 task-only arm. Shadow indices 4/5 provide an additional within-calibration computed-change diagnostic. This distinction is a split across states within the calibration source family; all task MSE values still refer to the same 256 calibration labels. The gamma-selection values therefore estimate *calibration* performance and are exposed to selection optimism. No prospective test claim is made.

## Results

| Recurrent depth | Rank | Task-only MSE | Selected joint MSE | Task-only B²/null² | Selected joint B²/null² | Task-only future-logit capture | Selected capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7 | 0.1055 | 0.0972 | 2.4655 | 0.0238 | 0.397 | 0.630 |
| 8 | 8 | 0.1059 | 0.0979 | 1.7761 | 0.0226 | 0.367 | 0.632 |
| 16 | 7 | 0.1058 | 0.0976 | 2.7831 | 0.0385 | 0.348 | 0.842 |
| 16 | 8 | 0.1067 | 0.0981 | 2.5833 | 0.0346 | 0.394 | 0.846 |

All 24 model-rank cells have lower calibration MSE and rank-normalized incoming squared coupling for the selected joint arm than for the 80-step task-only arm. The mean joint `J_lambda=1` is also lower in 24/24 comparisons. On separately indexed calibration shadow states 4/5, the penalty's smaller normalized B persists: mean task-only vs joint `B²/null²` is 2.553 vs 0.0218 (R8,k7), 1.809 vs 0.0204 (R8,k8), 2.701 vs 0.0332 (R16,k7), and 2.536 vs 0.0303 (R16,k8).

The selected joint observer's calibration accuracy is only about 23.9–24.9% across these four groups, and the original model's calibration accuracy is about 21.5% or 22.4% at the two depths. The model's **future-logit Jacobian** `H=(R T)^T (R T)` uses its fixed, weak eight-way readout R. Capturing more of that output derivative is a local sensitivity measurement and is not proof of correct delayed-rule recall or learned causal source binding.

## Verification

- `roc02c_equal_budget.py` reproduces rank-matched null median generation, differentiable ridge loss, fixed 80-step optimization, calibration selection, and per-model JSON receipts.
- `roc02c_summarize.py` derives the 96 candidate-arm rows, 144 model/rank/state null-median rows, six-seed group means, and claim-boundary status from the twelve raw model receipts.
- `test_roc02c_equal_budget.py`: six CPU/stdlib tests passed, including exact quadratic train-MSE parity, direct projector B identity, 128-null determinism, QR rank orthogonality, frozen initialization, and complete original-model grid.
- The 48 previously published ROC-02B null medians at shadow indices 0/1 for the two ranks were independently recomputed from the original public gzip CSV; maximum absolute difference **2.78e-17**.
- `roc02c_all_arms.csv` contains all gamma candidate scores and selected/fallback flags. `roc02c_null_medians.csv` provides the six recorded null medians for each original model/rank pair. `roc02c_summary.json` includes paired six-seed data and selected gamma per seed.

## Inference boundary and next gate

This calibration family was already accessible when the study was designed. The penalty grid and selection rule were committed before fitting, but full task accuracy was optimized and assessed within the same family. The strong within-calibration null-relative B reduction can reflect structure of this particular short-trained recurrent model, and may fail on the untouched source-family-6/7 test set. An appropriate next confirmatory design will freeze a single rank/gamma choice per seed from calibration, include the independent input/cue augmentation and matched physical causal interventions, and only then open the sealed family. No result here certifies a mechanism type in ROC-01's five-type classification; the calibrated tags and Chernoff discrimination require their own testing.
