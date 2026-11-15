# PEF-12 | Probe-Jacobian Gaussian forecast results

**Local CPU, November 15, 2026.** The [prospective PEF-12 protocol](PROTOCOL.md) was committed alone at `600f686` before any PEF-12 heldout noise scoring. Earlier PEF-07 through PEF-11 heldout results were already known. This is a conditional calibration experiment using six frozen small 12-bit models.

The original PEF-07 six 450-update checkpoints and the PEF-09/10/11 immutable probe ledgers passed exact SHA256 checks. The input split remained 160/48/48 context groups with all 16 focal interventions inside the same source group. All six unmodified native-head h12 scores reproduced 768/768. No new model training or readout fitting.

The 24 probe-only Jacobian forecasts, each comprising 768 clean margins and directional derivatives, were **sealed before any heldout scoring**: SHA256 `fa9a3f3727b9551be721ca08e307082b81d1d2da2e359a34d88d26c9e7185b49`. New Gaussian levels were 0.002, 0.006, 0.02 times probe PC scale, 16 independent noise seeds per condition, with transverse and matched off-plane controls. A separate deterministic retained-coordinate quantization diagnostic used steps 0.005, 0.02, 0.05 times probe PC scale.

**Preregistered Gaussian criterion: met in 3/3 coalition seeds.** The new forecast's mean absolute prediction error across the random tangent at both angles and all three Gaussian levels is shown below. The baseline simply predicts clean probe accuracy regardless of noise.

| Seed | Coalition Jacobian MAE | Coalition constant-clean MAE | Threshold Jacobian MAE |
|---:|---:|---:|---:|
| 103 | 0.536 percentage points | 10.211 points | 0.182 points |
| 131 | 0.266 points | 8.614 points | 0.186 points |
| 167 | 0.536 points | 10.087 points | 0.067 points |

At **0.25°**, the random-tangent forecast predicted and observed the following correct counts per 768 heldout cases. Observations average 16 new Gaussian draws:

| Seed | Noise relative to probe PC scale | Forecast | Observed |
|---:|---:|---:|---:|
| 103 | .002 | 744.9 | 736.3 |
| 103 | .006 | 619.8 | 608.7 |
| 103 | .020 | 466.5 | 461.6 |
| 131 | .002 | 763.4 | 761.4 |
| 131 | .006 | 666.4 | 657.1 |
| 131 | .020 | 489.4 | 488.5 |
| 167 | .002 | 737.0 | 727.1 |
| 167 | .006 | 626.3 | 615.4 |
| 167 | .020 | 473.7 | 469.9 |

At 4° the same predictor and frozen decoder both remained near 768/768 across all three registered noise levels. The matched off-plane forecasts had 0.06–0.54 percentage-point mean errors for coalition random tangents. The pair01-tangent Gaussian forecasts were less precise, with 1.38–2.52 points mean error.

**Quantizer warning:** The uniform-phase approximation describes an independent uniform scalar quantization error, whereas actual coordinate quantization uses deterministic nearest-even rounding. At the 0.25°, 0.05-step setting for coalition seed 103, it predicted 468 correct versus an observed 531. Its worst error was 63/768 (8.2 percentage points). The quantizer scores were diagnostic only and did not affect the registered Gaussian criterion.

An **independent standard-library auditor** recomputed the CDF predictions from the saved probe margins and derivatives, verified all six original checkpoint byte hashes, 1,152 distinct Gaussian seeds, all 2,304 Gaussian and 72 quantization score records (including 16-mask error totals), and the 3/3 decision. Combined local PEF-08 through PEF-12 unit tests: **35/35 passed**. Runtime: Python 3.13/PyTorch 2.10 CPU with deterministic algorithms and one thread. Hosted Python 3.12 CI is a separate verification boundary.

The experiment tests conditional, reused-context heldout calibration on previously studied synthetic Boolean tasks. Results do not establish independent operator transfer, actual quantized device performance, unique learned circuits, Lean reasoner capability, or pretrained-model behavior. The complete reproducible checkpoint and evidence ZIP is distributed separately from the GitHub source to keep the repo free of binary weights.