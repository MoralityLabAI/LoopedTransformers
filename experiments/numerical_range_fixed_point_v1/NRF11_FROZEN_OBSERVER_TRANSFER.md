# NRF-11: frozen learned-observer transfer to new input folds

Registered 2026-11-15 in `configs/nrf11_frozen_observer_transfer_v1.json`, after completing NRF-10 and **before opening folds 12/13**. This is a within-family input transfer study using the same twelve NRF-09 frozen model checkpoints, six seeds per trained recurrent depth, and the exact previously selected NRF-10 rank-eight observation maps.

## Sealed selection and adapter replay

For each source-hash-verified 32-update NRF-09 checkpoint, the NRF-10 projection optimizer is replayed using fold 1 task training and fold 9 calibration. Its three gamma candidate results and chosen gamma are compared against the original NRF-10 receipt to 1e-11. The optimizer receives the original 80 Adam updates, learning rate 0.02, and QR orthogonalization. The selected task-only, supervised and joint observation bases and their ridge heads are frozen **before** new fold 12 task or fold 13 Jacobian data are generated.

The three rank-eight bases, three 8×8 ridge heads, training means, source checkpoint hash and chosen gamma are serialized to separate `.pt` observer adapter files. Their own SHA-256 checksums are recorded in the receipts. An independent replay test loads the saved adapter and verifies scores on the new folds without doing any optimization.

The evaluation protocol retains the exact NRF-10 objective J=MSE+(mean local leakage)^2 and the median of eight complete random-arm J scores.

## Fresh-fold results

| Trained depth | Arm | Mean MSE | Mean leakage | Mean J |
|---|---|---:|---:|---:|
| 16 | Supervised baseline | 0.245731 | 0.416464 | 0.419216 |
| 16 | Task-only 80 Adam updates | | | 0.269816 |
| 16 | Frozen leakage-aware observer | 0.153759 | 0.315401 | **0.253968** |
| 32 | Supervised baseline | 0.284454 | 0.836354 | 0.984138 |
| 32 | Task-only 80 Adam updates | | | 0.420226 |
| 32 | Frozen leakage-aware observer | 0.247049 | 0.317590 | **0.351471** |

The mean eight-arm complete random objective is 0.568205 at R16 and 0.916216 at R32. The registered compound R32 gate passes **5/6** against the required 4/6, with the same seed 149 failing task MSE 1.10× baseline (fresh observed ratio 1.1139). R16 passes 6/6. Every joint observer wins against the equal-budget task-only J, the baseline J and the random median J on these new folds (12/12 each).

Mean R32 joint objective changed from 0.352556 (NRF-10 folds 10/11) to 0.351471 (NRF-11 folds 12/13). R16 changed from 0.254382 to 0.253968. On the new R32 shadow fold, mean per-seed joint-to-baseline leakage ratio is 0.379314, essentially the same as 0.379612 in NRF-10.

## Reproducibility and inference boundary

- `nrf11_frozen_observer_transfer.py`: recreates the original selected projector and freezes it, verifies source hashes and NRF-10 calibration metrics, opens fresh folds, exports adapter files.
- `nrf11_compact_receipts.csv`: all twelve selected-gamma values, checkpoint hashes, adapter hashes, paired metrics, and original NRF-10 metrics.
- `nrf11_summary.json`: exact statistics and primary gate.
- Twelve `nrf11_observer_R{depth}_S{seed}.pt` files, twelve complete eleven-arm receipt JSONs, source-checkpoint binaries and reproducibility dependencies are packaged in the conversation ZIP.
- A native CPU independent adapter replay checks all artifact SHA-256s, basis orthogonality and completeness, and exactly recomputes two frozen-adapter held-out evaluations without fitting.

The two evaluation folds are independent new draws from the **same synthetic Gaussian task distribution**. This is a transfer of observer performance across input samples, and has no independent model/task seed replication beyond NRF-09 or claim about real LLM reasoning, global quotient invariance or safety.
