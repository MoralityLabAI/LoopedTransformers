# PEF-15 | Independent-context conformal calibration, local results

**Registered design:** November 15, 2026, standalone commit [`20e7269`](https://github.com/MoralityLabAI/LoopedTransformers/commit/20e72699aa9717bc7de3bb395302d48f9e5a0043). Local run: Python 3.13, CPU PyTorch 2.10.0+cpu, single-thread AVX2 setting. PEF-14 PR #101 already existed when this experiment was designed, and its inadequate normal-interval coverage was known. PEF-15 re-trains every model from scratch with new seeds and a new split.

## Frozen registration and new source

- Independent new SHA256 salt `pef15-splitconformal-v1-20261115:`, 256 complete context families partitioned 160/32/16/48 into trunk train / fit / calibration / heldout (2560/512/256/768 distinct histories).
- Fresh model seeds 251,263,277 for coalition `(a AND b) XOR (c AND d)` and threshold `1[a+b+c+d>=2]`. All six trunks use the PEF-07/P14 exact architecture and 450 CPU optimizer updates.
- All six checkpoint bytes and final parameter digests were saved **before** fitting, all four-direction/angle probe ridge and mixture predictors were SHA256-sealed **before** calibration, and the calibration interval ledger was SHA256-sealed **before** heldout states were computed. No trained weights, readouts or quantizer scales were adjusted after fitting.
- Fit uses 32 complete source-context groups (512 histories), and calibration uses a disjoint 16 groups (256 histories). Each of the 96 frozen conditions retains both focal-mask and group closure.

**Content-addressed local receipts:**

| Artifact | SHA256 |
| --- | --- |
| Source manifest | `8470a602dbcb227740a00dcb105bb13b43cd8d21c17faf8837637f14e75e9f22` |
| Training ledger | `ee652d2dc8b72adb19587b75d52d09f96e6f95cf46c7281db642f1417afebab1` |
| Fit-only forecast ledger | `3c98b06d021a5814eacae95067b4ba3603f2322f4e6e3d769c91f896ea229a26` |
| Calibration interval ledger | `b3246f0e12257868db8a0a9a94e02fbc47ac79788d8caadc9002cccd78932bdf` |
| Heldout summary | `65d323c7cf9d81d1fdca91b18d9291236d703d85fe11b5e4b9efc0bb1bc6288d` |

## Preregistered primary result: **criterion_met, 3/3 coalition seeds**

Every model has 16 conditions (two tangents, two angles, four previously unused quantizer steps). Each condition evaluates 48 heldout context-group accuracies, with each accuracy computed from all 16 focal masks. Group coverage below is the observed proportion of group-condition pairs lying inside the frozen interval. The 16 conditions share models and context IDs, and are *not* independent statistical replications.

| Seed | Task | Covered /768 context-condition cases | Coverage | Mean interval width | PEF-14 style point-forecast MAE | Native h12 /768 | Serfling 48-group means covered /16 | Criterion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 251 | Coalition | 721 | **93.88%** | **0.2178** | 0.02355 | 768 | 16 | Pass |
| 263 | Coalition | 761 | **99.09%** | **0.2394** | 0.01594 | 768 | 16 | Pass |
| 277 | Coalition | 762 | **99.22%** | **0.2796** | 0.02031 | 768 | 16 | Pass |
| 251 | Threshold | 767 | 99.87% | 0.2342 | 0.01981 | 768 | 16 | Control |
| 263 | Threshold | 757 | 98.57% | 0.1067 | 0.00693 | 768 | 16 | Control |
| 277 | Threshold | 747 | 97.27% | 0.2375 | 0.01329 | 768 | 16 | Control |

The registered coalition rule requires at least two seeds with group-level coverage >=0.88 and average width <=0.85, and **all three** passed. The one-context conformal guarantee under exchangeability is at least 16/17 (94.12%) because there are 16 calibration contexts. Seed 251's observed 93.88% falls slightly below 94.12% on this finite, correlated panel; that is compatible with a marginal probability statement, and should not be treated as a guarantee violation. Without independent context replication, 768 correlated outcomes per model do not yield a calibrated confidence interval for the underlying coverage probability.

Every aggregate Serfling interval (different estimand: 48-context *mean*) covered. Its fixed pre-clipping radius is about **0.356963**, making it much wider than the average per-context conformal intervals. These serve separate purposes, and the all-covered result shows conservativeness rather than a uniquely informative risk estimate.

## Independent audit and reproducibility

The independent Python audit verified the 256-context owner map, six fresh 450-update optimizer states and checkpoint SHA256s. Using a separate scalar implementation of nearest-even rounding and phase mixing, it recomputed all 96 fit forecasts, all 96 group-level conformal intervals, the 96 Serfling average bounds, and the registered 3/3 decision from the saved 48 group-level scores. The 65-test PEF-07..PEF-15 local suite passed.

All full precision and context-level individual scores are in the sealed `summary.json` and the full source/receipt evidence archive. Hosted Python 3.12 verification, if later available, must report its own checkpoint and run hashes; CPU differences can change parameter hashes across machines.

**Boundaries:** Exchangeability of future contexts conditional on the fit data, or genuinely random context-role assignment within the 64 left-over source contexts, is the mathematical premise for marginal conformal coverage. A precommitted salted SHA256 sort makes the split reproducible and plausibly pseudorandom, but does not itself prove these probability assumptions. The models share the known two Boolean task functions, so this is a synthetic context transfer result, not generalization to new logic operators or natural-language reasoning.
