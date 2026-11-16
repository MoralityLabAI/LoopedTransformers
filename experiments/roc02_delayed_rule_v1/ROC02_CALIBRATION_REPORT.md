# ROC-02: six-seed tied-model training and calibration, sealed test untouched

**Date:** November 16, 2026. **Status:** first prospective **train/calibration-only** cohort; this is not a ROC observer-capture, rank-sweep, or intervention result. The original MathCan ROC-01 Part 4/Amendment C registration is unchanged. A dated source/optimizer implementation supplement was committed at `fbb67cc`, the 1,664-case exact source suite at `babd0c2`, and the tied model implementation at `a4f912b` **before the full twelve-model training cohort**. An earlier unreported one-model code smoke was run before the model source lock; it was excluded and every final model was retrained afterward.

## Model/source contract

The study uses 128D parameter-tied tanh residual MLPs with a 79-dimensional event encoder and eight task logits. Each event updates the same state once, followed by either 8 or 16 additional tied reasoning passes. Train and calibration rule families are disjoint: families 0–3 supply 1,024 training examples; families 4–5 supply 256 calibration examples. The test families 6–7 are stored as 384 **unlabeled** input records and were never read by this trainer. The symbolic reference interpreter verifies all 1,664 valid examples and rejects 24 invalid fixtures.

Six new initializations `2026111601/03/07/09/13/19` were trained at both depths under exactly 240 AdamW updates (batch 8, lr=0.001, zero weight decay). Same seed means identical initial model parameters and identical minibatch index streams at depths 8/16. The model sees 1,920 examples with replacement, drawn from the fixed 1,024 training cases, then reports calibration scores for the separate 256-case family holdout.

| Depth | Mean train accuracy | Mean calibration accuracy | Calibration cross-entropy | Mean step-zero calibration accuracy (retrospective) |
|---|---:|---:|---:|---:|
| 8 | 0.241536 | 0.215495 | 1.827791 | 0.116536 |
| 16 | 0.266764 | 0.223958 | 1.638484 | 0.114583 |

The exact source oracle is 100% on both labeled splits; an eight-way untrained random-class benchmark is 12.5% by symmetry. All 12 trained models finish above that numerical chance level on calibration accuracy, although **absolute accuracies remain low**. These are descriptive training/cross-family calibration statistics. No held-out task metric, causal gain, or meaningful rule generalization is established here. The step-zero comparison is retrospective, computed after the training receipt and not used in selection.

## Checkpoint provenance and future gate

Each original `.pt` contains the frozen 128D state weights and optimizer state after the exact last update, with per-model SHA-256 and raw source-file SHA-256. There are twelve model JSON receipts and six original-byte ZIP parts, each pinning two model/receipt pairs. `roc02_checkpoint_archive_index.json` pins SHA-256 and Git blob SHA-1 for each package. The [binary verifier](../../research_gym/analysis/roc02_checkpoint_archive.py) rejects missing or corrupted archives and checks every model against its receipt and score record. A rerun of the first R8/seed01 model reproduced the original `.pt` SHA-256 byte-for-byte in the same CPU environment.

The next analysis stage must freeze any model/rank/γ choice on calibration data and publish the complete prospective ROC-01 Amendment-C observer card: nine ranks, 128 rank-matched null subspaces per state, six Holm-adjusted one-sided tests, complete A/B/C/H/task columns, physically norm-matched intervention controls and an exact symbolic baseline. Given the current low task accuracy, source-vs-carried-state causal separation may fail; clean negative outcomes are registered and must be reported. The sealed test remains closed until all observer decisions are locked.
