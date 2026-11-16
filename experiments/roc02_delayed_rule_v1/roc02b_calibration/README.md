# ROC-02B calibration-only reproducibility

**Warning:** this study uses the training and calibration source families only. It does not access or report the separately sealed ROC-02 test.

To run from this archive:

```bash
PYTHONPATH=roc02b_replay python roc02b_replay/roc02b_calibration_card.py --source-dir roc02b_model_data --checkpoint-dir roc02b_model_data/checkpoints --out-dir recreated --seed 2026111601 --depth 8
python roc02b_replay/roc02b_summarize.py --input-dir recreated --output-dir recreated_public
PYTHONPATH=roc02b_replay ROC02B_RECEIPT_DIR=roc02b_public python -m unittest discover -s roc02b_replay -p 'test_roc02b_*py' -v
```

Run the first command across all six seeds and both trained depths before summarizing. The source script verifies each original `.pt` hash and train/calibration split hashes. It creates dense suffix Jacobians, the exact registered rank grid and 128 rank-matched nulls per (model, state, rank). Compression uses fixed `mtime=0` for reproducibility.

The public `roc02b_public` folder has complete rows of 27,648 rank-matched random bases, 432 observer states, six-seed rank summaries, and the manifest.

The next required study stage is equal-budget 80-update task-only vs penalty-trained observer fits and input-level causal controls. None was claimed or unsealed by this calibration-only pilot.