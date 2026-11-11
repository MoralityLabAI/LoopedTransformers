# Recurrent-state SAE study

Sparse autoencoders, CKA and activation interventions on looped / LaCo-compressed / loop-recovered transformers, plus a
test of whether SAE-derived signals improve the HRM-style stopping controller.

- Report: [reports/REPORT.md](reports/REPORT.md); tables: [reports/tables.md](reports/tables.md); plots: `figs/`
- Saved SAE configs + weights: `out/sae/`; per-run result JSON: `out/res/`
- Code: `src/recsae`, `scripts/` (t* looped models, m* mini-LaCo replica, q1 Qwen3 path, r1 aggregation), `run_*.sh`
- `vendor/` holds frozen snapshots of the crrbench / looprec code from the sibling sessions (see `PROVENANCE.*`)

Scope: small-model evidence only (d=128 automaton tasks, plus a mini LaCo replica). The Qwen3 path (`scripts/q1_collect.py`,
`run_qwen.sh`) was smoke-tested on a tiny checkpoint but not run on real Qwen3 weights.

Paths in `src/recsae/__init__.py` and `scripts/sib_models.py` are machine-specific (`C:\Users\patri\.cache\recsae_*`,
sibling scratch folders); override with `RECSAE_DATA`, `RECSAE_LIBS`, `CRR_RESULTS`. Activations and checkpoints are not committed.
