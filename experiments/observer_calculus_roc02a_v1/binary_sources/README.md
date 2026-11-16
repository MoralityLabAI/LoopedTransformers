# Frozen NRF-09 checkpoints and NRF-11 rank-eight adapters for ROC-02A

These four original-byte ZIP packages contain **twelve 128D tied ToyLoop model checkpoints**, **twelve frozen rank-eight observer adapters**, and **twelve NRF-11 source receipts**. The sources come from the six already known NRF-09/11 model seeds, at 16 and 32 recurrent iterations, after only 32 network optimizer updates. Publishing the bytes permits independent replay of the retrospective ROC-02A 128-null/rank-grid measurements. All checksums are recorded in [`roc02a_binary_sources_index.json`](roc02a_binary_sources_index.json).

Download all four `.zip` parts from this directory, and from repository root run:

```sh
python -m research_gym.analysis.roc02a_binary_source_verify \
  --archive-dir experiments/observer_calculus_roc02a_v1/binary_sources \
  --index experiments/observer_calculus_roc02a_v1/binary_sources/roc02a_binary_sources_index.json \
  --destination /tmp/roc02a_verified_originals
```

The verifier checks **all four ZIP SHA-256 digests**, validates archive member paths, and compares every original `.pt` checkpoint/adapter SHA-256 against its respective saved NRF-11 receipt. It checks 36 distinct files. The extracted layout matches `--archive-root` required by `roc02a_frozen_rank_causal.py`. The CPU neural replay also requires PyTorch and the source-compatible NRF-09/10 input generators.

The original dataset and model family are the same synthetic signed-permutation tasks exposed in NRF-09–11. This is **retrospective source material** and should not be treated as evidence from a prospective ROC-02 held-out task, external model family, or general reasoning evaluation.

The distinct full shadow/Jacobian rank-null measurements remain in `roc02a_rank_nulls_compact.csv.gz`, with registered source scripts and row-level statistical tests in the parent directory and `tests/`.
