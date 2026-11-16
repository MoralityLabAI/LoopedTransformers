# ROC-02D equal-radius input/hidden diagnostic

**Status:** calibration-only retrospective mechanism diagnostic on the twelve original frozen ROC-02 model checkpoints and preselected rank-eight ROC-02C observer adapters. Original registration at `714f149d6dabff24a3ef4bcec566ba7d93f4cf0d`. No held-out ROC-02 source-family 6/7 inputs or labels have been opened.

Read [the report](ROC02D_MATCHED_INPUT_CAUSAL_REPORT.md), the complete per-case gzip CSV, the 12-row paired-seed CSV, and the full 12-receipt ZIP with checksum index.

The source script is `research_gym/analysis/roc02d_matched_input_causal.py` and the aggregator `research_gym/analysis/roc02d_summarize.py`. A full replay additionally requires the source data and original checkpoint ZIPs from merged ROC-02 PR #114, and the frozen observer adapter ZIP from ROC-02C PR #127. All are SHA-checked before evaluation.

Run from a repository root with the source train/calibration data already extracted, then for each seed and depth:

```bash
python -m research_gym.analysis.roc02d_matched_input_causal \
  --seed 2026111601 --depth 8 --source-dir path/to/source \
  --checkpoint-dir path/to/model_checkpoints \
  --adapter-dir path/to/roc02c_observer_adapters \
  --adapter-index experiments/roc02_delayed_rule_v1/roc02c_calibration/roc02c_summary.json \
  --out-dir recreated_results
python -m research_gym.analysis.roc02d_summarize \
  --input-dir recreated_results --output-dir recreated_public
python -m unittest discover -s tests -p test_roc02d_public_receipts.py -v
```

The direct legal binding flip changes the true symbolic label and has its own measured midpoint displacement. The softened input feature is a continuous interpolation with no symbolic legal label; it is the **radius-matched** input comparator. These are separate results. Selection of the ROC-02C adapters happened on the same calibration population, and the trained model's original accuracy was low. This data cannot certify independent rule-binding mechanistic recall.
