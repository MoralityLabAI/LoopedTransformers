# ROC-02E: complete calibration-only replay

This folder contains the prospective protocol and the measured negative early-rule causal result for the frozen 2026-11-16 ROC-02 models. Original test families 6–7 remain sealed.

## Prerequisites

Python 3.13 and CPU PyTorch 2.10.0 are the checked environment. Obtain the separately published original ROC-02C PR #144 complete calibration-only reproducibility archive, which includes all twelve SHA-pinned original .pt checkpoints, training/calibration source files, ROC-02B null data and full ROC-02C observer fitting receipts. Extract its contents into a directory named replay.

The active code and test files are:

- research_gym/analysis/roc02e_matched_causal.py
- research_gym/analysis/roc02e_independent_audit.py
- research_gym/analysis/roc02e_source_pairs.py
- tests/test_roc02e_source_pairs.py

Use original files. The code rejects changed checkpoint and source SHA and mismatched observer replay metrics. The intervention runner deliberately accepts only the existing train/calibration files and must never generate or read sealed family-6/7 source inputs, labels, or derivatives.

## One model, from repository root

~~~bash
python -m research_gym.analysis.roc02e_matched_causal \
  --seed 2026111601 --depth 16 \
  --source-dir replay/original_models \
  --checkpoint-dir replay/original_models/checkpoints \
  --receipt-file replay/experiments/roc02_delayed_rule_v1/roc02c_calibration/roc02c_full_receipts.jsonl \
  --out-dir roc02e_replayed
~~~

Before each neural measurement, the script verifies source and checkpoint hashes. It reproduces the rank-7 and rank-8 task-only/selected-gamma U adapters under the original 80-step Adam/QR budget, checking calibration accuracy, MSE, incoming B and future-logit capture against prior complete ROC-02C receipts to tolerance 1e-10. It writes a 43,008-row per-model JSONL gzip and a model JSON summary.

For the full cohort, run the command for seeds 2026111601, 2026111603, 2026111607, 2026111609, 2026111613, and 2026111619 at depths 8 and 16, using a shared output directory. Then:

~~~bash
python -m research_gym.analysis.roc02e_matched_causal \
  --summarize --out-dir roc02e_replayed

python -m research_gym.analysis.roc02e_independent_audit \
  --receipt-dir roc02e_replayed \
  --out roc02e_replayed/roc02e_independent_audit.json
~~~

The audit demands 516,096 complete unique matched per-case intervention rows, verifies source/cue target changes and source-family restrictions, checks valid physical radii, verifies compressed and **uncompressed JSONL SHA-256 hashes**, and independently reconstructs the six-seed primary and conservative best-input-control upper bound. Stable gzip uses mtime=0 and an empty filename header so archive hashes are repeatable on a fixed runtime.

## Hosted verification

The workflow .github/workflows/roc02e-causal-replay.yml runs the full CPU campaign on GitHub Actions, checks source and checkpoint archives, reproduces the twelve models and all 24 model/rank cells, and uploads complete receipts with SHA-256 manifest as artifact ROC02E-full-calibration-causal-replay.

The reported depth-16/rank-7/radius-0.20 early-rule intervention improved the original model's counterfactual target-logit gain by +0.00014148 on average, with only 3/6 positive seeds and one-sided exact sign-flip p=0.375. This calibration-only result fails the registered ROC-01 differentiation gate.

The per-case best-control envelope is a post-hoc oracle upper bound. It selects the best control after seeing each case's label, and does not constitute a deployable input policy. Continuous input interpolations have no symbolic task label. Full legal edits have unequal natural physical costs. The original ROC-01 Amendment-C full mechanism diagnostics, Chernoff allocations and prospective sealed-family evaluation remain outstanding.
