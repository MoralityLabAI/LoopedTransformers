# ROC-02C: equal-budget rank-eight leakage-aware observer fit on calibration states

**Study date:** 16 November 2026. **Status:** calibration-only, source-locked follow-up to ROC-02B.

**Registered before fitting:** `configs/roc02c_equal_budget_rank8_v1.json`, commit `006cdce9de6eaf529fc03fd37b71b448cbe4b063`. The frozen ROC-02 trained models were published in merged PR #114 and the ROC-02B rank-matched nulls in draft PR #124. Their SHA-gated original weights and source-family split are unchanged. The held-out source-family labels and inputs remain sealed.

## Protocol

Six paired original model seeds at recurrent depths 8 and 16 completed the original **240 model optimizer updates** before ROC-02C. At midpoint states after input events and half of the post-sequence recurrent updates, fit a rank-eight orthonormal basis to train source families 0–3. Refit only the orthogonal observer and a closed-form ridge head, using 80 Adam updates with learning rate 0.02 per arm. Every arm begins at the exact same train-only rank-eight supervised-covariance basis. Evaluation uses source-family-disjoint *calibration* families 4–5 (256 examples), and the two preregistered calibration shadow states.

For T_i the frozen suffix Jacobian and D_i=T_i−I, let b_i(U)=||U^T D_i (I−UU^T)||_F/||D_i||_F. The m_i are the median b_i values of the **128 previously frozen rank-matched random bases** at rank eight and the same model/state. The observer optimizer uses the registered *mean squared* null-relative penalty:

    E_train(U) + gamma * (1/2) sum_{i=0,1} (b_i(U)/m_i)^2.

Candidates `gamma={0,0.05,0.2,1,2}` receive equal 80-step budgets. The calibration comparison fixes lambda=1 and uses `E_cal + mean_i(b_i/m_i)^2`. A positive gamma is eligible only if its calibration task MSE is no more than 1.10 times the *equally optimized task-only* MSE. The lowest-J eligible candidate is selected, or gamma=0 if no positive arm improves it. The training source head, original trained output readout, and model weights remain frozen. Report original model accuracy and the exact symbolic interpreter separately in the full ROC-02 eventual assessment.

## Results

| Frozen recurrent depth | Arm | Mean calibration task MSE | Mean accuracy | Mean B / rank-null median | Mean future-logit H capture | Mean calibration J(lambda=1) |
|---|---|---:|---:|---:|---:|---:|
| 8 | Task-only, 80 updates | 0.105935 | 22.33% | 1.329596 | 0.370236 | 1.905300 |
| 8 | Selected penalty, 80 updates | 0.098223 | 23.70% | 0.138132 | 0.643560 | 0.117334 |
| 16 | Task-only, 80 updates | 0.106727 | 25.65% | 1.582499 | 0.400604 | 2.650466 |
| 16 | Selected penalty, 80 updates | 0.098295 | 25.65% | 0.158434 | 0.857428 | 0.123550 |

**Registered depth-16 calibration gate passes:** 6/6 seeds select positive gamma, improve the calibration J over equal-budget gamma=0, and satisfy the task-only-anchored MSE ceiling. Task MSE improves 12/12, and model-logit Jacobian capture increases 12/12. Five of six depth-8 models choose gamma=2; one chooses gamma=0.05. All six depth-16 models choose gamma=2.

The originally trained classifier's mean calibration accuracy was **21.55% at depth 8** and **22.40% at depth 16**. The new ridge observers remain weak predictors on source-family-disjoint calibration inputs. The large drop in B/m is a model-state geometric measurement under an unusually favorable loss landscape, not proof of useful source-rule retention. Calibration selection and calibration assessment share the same examples, so the reported improvements contain selection bias. No future-source-family test outcome is claimed.

## Reproducibility and corrections

- `roc02c_equal_budget_rank8.py` reconstructs all 12 original weights from pinned checkpoints and checks the source and ROC-02B null-receipt hashes before fitting. Each selected basis and train-only ridge head is serialized as an independent SHA-256-pinned adapter.
- `roc02c_summarize.py` checks the exact optimizer-step grid, full candidate arithmetic, selected gamma, task ceiling, 12 adapter bytes and provenance; it publishes 72 complete arm rows and 12 paired model outcomes.
- `roc02c_receipts_adapters.zip` contains the twelve original saved observer adapters and twelve complete arm-level JSON receipts, with no held-out source-family records.
- `tests/test_roc02c_public_receipts.py` uses only Python standard library for objective, cardinality, 80-step protocol, ZIP integrity and 12 original adapter SHA checks. `tests/test_roc02c_equal_budget_rank8.py` additionally audits the orthogonal block formula, train-prior restoration and equal-budget training with PyTorch.
- The first diagnostic fit briefly computed accuracy from centered logits. The final published result restores the train-label mean before argmax. All 12 adapters reproduced **byte-for-byte**, and all 72 candidate task-MSE, B, H-capture, and J values were identical on refitting. Class-accuracy reporting alone changed in 64 of 72 arm rows; every chosen gamma and primary gate was unchanged.

## Remaining ROC-02 gates

The study is an **in-sample calibration diagnostic** on an existing, low-performing synthetic family. It does not train the original model further or justify a mechanistic discovery label. The registered nine-rank equal-budget comparison, norm-matched **input augmentation** baseline, full causal T-C discrimination with real neural outcome distributions, rank-dependent Holm inference and **independent sealed source-family test** remain outstanding. Completing those gates requires a separate frozen design and new independent held-out evidence; use ROC-01 Amendment C and keep the original held-out families untouched.
