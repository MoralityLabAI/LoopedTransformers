# PEF-10 | Local frozen-checkpoint ablation-angle results

Local experiment: 2026-11-15. Preregistered at standalone commit `de6666c71b65287df56b24cb0d57e0e89ff7553c` before PEF-10 heldout scoring. Python 3.13.5 / PyTorch 2.10.0+cpu; single-thread CPU with oneDNN max ISA AVX2; the Python 3.12 hosted run is separate.

Every original PEF-07 checkpoint byte digest and weight digest was checked against its training ledger, and every baseline reproduced 768/768 at h12 native and h14 frozen ridge. The PEF-09 frozen feature ledger SHA256 is `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157`. All 504 condition directions and their individual newly fitted h12/h14 probe ridge readouts were written to the frozen ledger **before any PEF-10 heldout traces were made**. No parameters were retrained.

## Preregistered decisions

- **Geometric recovery**: `criterion_met`, 3/3 coalition seeds: both pair-derived tangents recovered >=720/768 at an angle <=4 degrees, and theta=0 failed that threshold.
- **Pair specificity**: failed, 0/3 coalition seeds. One probe-frozen random tangent met the same accuracy threshold at 0.25 degrees for every coalition and threshold seed.
- **Older native head**: for each coalition model, the original h12 native head stayed at its 480/768 majority-class result across all small angles shown below. Recoverability refers to the **newly fitted** h12 probe ridge readout.

## Earliest fixed-angle recovery at >=720/768

| Seed | Task | Zero-angle new h12 | Pair01 | Pair23 | PCA-2 | Random |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 103 | coalition | 611 | 2° | 4° | 16° | 0.25° |
| 103 | threshold | 574 | 8° | 8° | 16° | 0.25° |
| 131 | coalition | 671 | 1° | 2° | 1° | 0.25° |
| 131 | threshold | 534 | 4° | 4° | 8° | 0.25° |
| 167 | coalition | 632 | 4° | 4° | 4° | 0.25° |
| 167 | threshold | 585 | 8° | 8° | 8° | 0.25° |

The angle grid is 0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 90 degrees, with both +/- orientations for interior angles. Exact PEF-09 pair-SVD anchor angles were inserted in addition to the fixed grid for each intended pair. The positive-sign pair-anchor results reproduced **12/12** PEF-09R intervention results in all four accuracy measures (original native h12, old frozen h14 ridge, new h12 ridge and new h14 ridge).

## Exact probe-frozen pair angles

| Seed | Task | Pair01 phi | Pair23 phi |
| --- | --- | ---: | ---: |
| 103 | coalition | 2.5636° | 4.3928° |
| 103 | threshold | 15.1128° | 13.6027° |
| 131 | coalition | 3.1409° | 6.5193° |
| 131 | threshold | 6.1413° | 5.2394° |
| 167 | coalition | 9.1900° | 10.2743° |
| 167 | threshold | 11.8758° | 11.0373° |

## Interpretation and limits

When the rank-one ablation direction rotates from global PC g by a small angle theta along an orthogonal tangent, the formerly removed PC component reappears in the retained tangent at order sin(theta). Ridge refitting on the ablated probe split can amplify that weak residual. The random-tangent control recovered earlier than either coalition pair tangent in all seeds, pointing to generic geometric leakage rather than coalition-specific sensitivity.

The zero-angle new-ridge recovery scores are above the 480/768 majority baseline in coalition models for visit-8 ablation (611, 671, 632). Thus a PC ablation at h8 need not remove all decodable task signal. The original **fixed** head remained at majority accuracy in that setting. H12 projection geometry is treated separately in PEF-09R.

This study is confined to the PEF-07 12-bit synthetic operator with context-disjoint holdout groups. It does not establish a unique natural circuit, transfer to a novel Boolean operator, SAE interpretability, or improved Lean reasoning. Any causal claim about persistence under realistic perturbations should repeat the frontier under a fixed explicit noise floor or model quantization and matched-energy controls.

## Evidence inventory

- `frozen_angle_and_ridge_ledger.json`: six-model source/probe-only directions, projected energies and 1008 fitted ridge weights (84 conditions x 6 models x 2 horizons).
- `summary.json`: 504 complete heldout condition rows with native h12, original frozen h14, newly fitted h12 and h14 scores and per-focal-mask errors.
- `frontier.csv`: flat 504-row numeric frontiers suitable for plotting and independent audits.
- `test_pef10.py`: five deterministic mathematical/projection invariants (5 passed locally).
- `PEF10_recovery_frontier.png`: descriptive mean coalition frontier across 3 seeds, using the + orientation and deduplicated endpoints.

Full local reproducibility requires the original PEF-07 checkpoints and PEF-09 sealed selection ledger; this experiment makes no claim of green hosted Python 3.12 CI.
