# ROC-02C: calibrated rank-eight readout fit and public replay

*Calibration-only source-locked pilot, 2026-11-16. Original ROC-02 delayed-rule heldout labels remain sealed.*

## What is published

- `roc02c_candidates.csv`: 72 arms for six model seeds at each of depths 8 and 16, with all five optimizer penalties and one initial baseline; each fitted arm received exactly 80 Adam steps.
- `roc02c_paired_by_seed.csv`: twelve original model-seed task-only and selected-penalty contrasts, with raw task MSE, accuracy, null-relative coupling, future-logit Gram capture and `lambda=1` calibration objective.
- `roc02c_summary.json` and `roc02c_manifest.json`: six-seed summary, split hashes and checkpoint/observer-adapter SHA receipts.
- `roc02c_receipts_adapters.zip`: complete six-arm per-model JSON (baseline + gamma0 + four positive gammas), original twelve rank-eight saved projector/head `.pt` adapters, source code and tests. Every entry has a fixed time stamp and the ZIP SHA is published in the manifest.

## Replay

From LoopedTransformers repository root, first retrieve and verify the original ROC-02 model/source archives in `experiments/roc02_delayed_rule_v1/` (merged PR #114), and the ROC-02B rank-eight original null receipts (PR #124; complete original per-model ROC-02B JSON receipts also appear in the conversation's ROC-02B reproduction bundle).

For one model:

```bash
python -m research_gym.analysis.roc02c_equal_budget_rank8 \
  --seed 2026111601 --depth 8 \
  --source-dir /path/to/extracted/roc02 \
  --checkpoint-dir /path/to/checkpoints \
  --prior-roc02b-dir /path/to/roc02b_full_receipts \
  --prior-roc02b-manifest experiments/roc02_delayed_rule_v1/roc02b_calibration/roc02b_manifest.json \
  --out-dir /path/to/reproduced
```

Then repeat all six seeds and both depths. The script refuses checkpoint or ROC-02B null-receipt SHA mismatches. The source code never reads sealed test inputs or labels. The original `.pt` adapters in the ZIP are the byte-for-byte reproduced tensors after correcting the printed accuracy computation. The source includes exact train-calibration data checks.

Validation without PyTorch:

```bash
python -m unittest discover -s tests -p 'test_roc02c_public_receipts.py' -v
```

Optional PyTorch geometry and optimizer check:

```bash
python -m unittest discover -s tests -p 'test_roc02c_equal_budget_rank8.py' -v
```

The calibration penalty uses the **mean of squared** rank-normalized incoming computed-change blocks, with each denominator the median of 128 rank-matched random bases at that exact checkpoint and calibration state. The task-only anchor is 80 steps on exactly the same initial basis and training labels. The selected penalty is chosen on calibration labels and evaluated on the same calibration split, so the selection criterion is descriptive and cannot establish prospective generalization. Its output-Grad H refers to the *trained model's logits* on weak original model predictions; it is not an independent guarantee of true-rule representation.
