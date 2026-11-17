# ROC-02B calibration-only rank/null receipts

Study preregistration: `configs/roc02b_calibration_observer_card_v1.json` at `f9cc0462e376175afe7901b6ffaa1d9e726c8fca`.

Six compressed rank-null partitions preserve all **27,648 original random-basis rows**. Splits by training depth and three seed pairs:

| File | Git blob SHA-1 |
|---|---|
| `roc02b_nulls_R8_a.csv.gz` | `283b5ae58f833f168e3fcaa32556fc1bcdc8364c` |
| `roc02b_nulls_R8_b.csv.gz` | `13047e8b25f23490403463ab020c5142ec8f5c52` |
| `roc02b_nulls_R8_c.csv.gz` | `d863cbe11c0555300059625f74be33c49c7bb604` |
| `roc02b_nulls_R16_a.csv.gz` | `597aee28fa5a6e70487c06cc2a85b94f34f9059b` |
| `roc02b_nulls_R16_b.csv.gz` | `56e2ff769ac1516d012ebee37cb6dbf42d3cfe02` |
| `roc02b_nulls_R16_c.csv.gz` | `83365a496a0af6bdfcf3948f478f9adbc284e021` |

**Run audit:** `python -m unittest discover -s tests -p 'test_roc02b_public_receipts.py' -v`. The geometry suite requires PyTorch and `roc02_tied_train.py` from the same repo. The source and checkpoint archive are already merged into main.

A whole-table `roc02b_nulls.csv.gz` SHA is in the original manifest for external archive replays; the six GitHub partitions are the equivalent public distribution. The python verifier consumes the six partitions directly. SHA claims for that whole-table file are retained for provenance, rather than applied to the parts.

The published code reads ROC-02 train/calibration input files and frozen checkpoints only. It does not run final rank selection or open held-out source-family labels.
