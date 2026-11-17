# ROC-02C gamma-through-2 and ROC-02D exposed calibration pilots (archived)

This directory preserves 25 original Git blobs, including the saved rank-eight adapter ZIP and the earlier ROC-02D input-rule experiment ZIP. The bytes are identical to the former stacked draft [ROC-02C #127](https://github.com/MoralityLabAI/LoopedTransformers/pull/127) and [ROC-02D #130](https://github.com/MoralityLabAI/LoopedTransformers/pull/130).

The older ROC-02C experiment selected gamma from {0, .05, .2, 1, 2} for rank 8. ROC-02D tested six exposed calibration cases per model with those adapters and observed less rule-edit readout sensitivity for the penalized arm in 11/12 model-seed/depth cells.

The canonical later experiment is [ROC-02C #144](https://github.com/MoralityLabAI/LoopedTransformers/pull/144), using gamma through 1 with ranks 7 and 8. Its companion causal test [ROC-02E #147](https://github.com/MoralityLabAI/LoopedTransformers/pull/147) evaluates 256 exposed calibration cases per model; the registered primary rank-7 six-seed comparison fails (p=0.375). The observer fits and endpoints differ. The older lineages are retained as historically separate calibration evidence.

Archive structure preserves the source and tests in `roc02c/scripts`, `roc02d/scripts`, and their respective `tests` directories. Scripts are **original content snapshots**, with their original import paths. To run the old scripts directly, extract or check out the PR #127/#130 original tree structure and use its historic matching config and artifacts. The archive is for provenance and independent comparison; it does not redirect main's ROC-02C imports.

`LEGACY_GIT_BLOB_MANIFEST.json` pins every archived path to its original Git blob SHA-1; the CI verifies exact Git hashes and full compressed member validity. The new archive commits introduce **no model training or additional test unsealing**. ROC-02's separately sealed source families 6 and 7 remain unopened.
