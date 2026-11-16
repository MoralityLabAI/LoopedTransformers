# RAM-16 operational amendment D: independent audit module-name collision

Dated 16 November 2026, following successful native TRAIN/DEV collection and before any RAM-16 model fitting, DEV model selection or prospective TEST access.

The source-only training corpus is immutable: original collector [run 43802741798](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43802741798), artifact 14197417060, 574 candidate source compiler attempts, 39 twice-compiled positives across 68 original TRAIN/DEV roots, source effects SHA256 `af400fd5fd8c5b99431a3e5a3022326c53d807b338fb7e3bc9f32530aafbebe9`.

An independent third-audit attempt [run 43803476738](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43803476738) failed during Python import, before independent proof compilation. The `third_audit.py` script inserted the historical `experiments/lean_ram08` directory ahead of its own, so `from collect import ...` incorrectly loaded RAM-08's collector. The source-only fix explicitly inserts `experiments/lean_ram16` at the front of `sys.path` before importing the RAM-16 collector. A contract test asserts that the two imported functions refer to the RAM-16 module.

This dependency fix changes neither the original first-run candidate scripts/labels nor the preregistered diversity gate, TRAIN/DEV module split, parameter-matched architectures, optimizer budgets, or unopened TEST cohort. Original failed audit runs remain visible and are treated as infrastructure errors. Third-replay evidence is required before model fitting.
