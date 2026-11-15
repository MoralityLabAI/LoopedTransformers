# PEF-11 | Frozen-readout precision/noise frontier: local evidence

Date: 2026-11-15. This experiment was registered at standalone Git commit `8d04153c2c7328ec2998c4edb14f9ef96b4f250c` **before any PEF-11 heldout noise or quantization scoring**. PEF-10 observations were known and motivated the test. Python 3.13 / PyTorch 2.10 CPU, AVX2, deterministic algorithms and one math thread. Hosted Python 3.12 verification is a separate status.

## Input and freeze integrity

The PEF-07 six *original 450-step* checkpoints (3 seeds x coalition/threshold), source manifest, PEF-09 probe selection and PEF-10 probe ridge ledger were checked against their frozen SHA256s. The 120 retained PEF-10 conditions cover 4 tangents x {0, 0.25, 1, 4, 16} degrees x 6 models, positive sign for nonzero angles. Their original direction and fitted-readout hashes were validated. The 120 condition selectors, probe-only scales and all deterministic random seeds were written to `frozen_precision_ledger.json` with SHA256 `f0bd80d8dc7b4af99dbc24be2723828bcf85be5b6f60f46452581e4a53661741` **before generating the heldout traces**. The clean ablation results reproduced the corresponding PEF-10 reported native and fixed-ridge scores exactly for every one of the 120 conditions; the original natural models still scored 768/768.

The experiment scored 7,560 condition/precision rows, with all three fixed Gaussian replicates, matched off-plane perturbations, natural-state controls, retained-coordinate scalar quantization and per-channel quantization. All 16 focal-mask error counts are included in each stored score. No parameters or readouts were modified or refitted.

## Registered decision: met in three of three coalition seeds

Each row below uses the **previously probe-fitted** h12 ridge for the PEF-10 random tangent at a positive rotation of 0.25 degrees, except natural-control scores which use the original native h12 head. Noise level 0.01 means sigma equals **1% of the frozen h8 probe global-PC standard deviation**. Quantization step 0.03 means 3% of that same scale for one retained-coordinate scalar. Gaussian scores average three fixed independent noise seeds.

| Coalition seed | Clean 0.25° | Transverse Gaussian 0.01 | Matched off-plane Gaussian 0.01 | Natural-native control 0.01 | Retained-coordinate Q 0.03 | Transverse Gaussian 0.01 at 4° | Retained-coordinate Q 0.03 at 4° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 103 | 752 | 538.67 | 752.00 | 768 | 503 | 768 | 768 |
| 131 | 767 | 583.33 | 767.67 | 768 | 569 | 768 | 768 |
| 167 | 751 | 541.00 | 742.00 | 768 | 546 | 768 | 768 |

The combined preregistered small-angle fragility and larger-angle robustness criterion is **criterion_met, 3/3 coalition seeds**. The off-plane comparison also showed transverse noise caused more errors in 3/3 seeds at 0.25 degrees; the off-plane control has exactly matched per-example noise *norm* to the transverse condition and shares the same Gaussian z values. For context, the one-dimensional natural-state native head maintained full 768/768 at the specified noise level in all three seeds.

The threshold-control arms also show 0.25-degree fragility, so the observed precision threshold is consistent with a broadly shared representation geometry in these architectures. The full result includes Gaussian levels {0,0.001,0.003,0.01,0.03,0.1}, retained-coordinate quantization and per-channel quantization levels {0,0.003,0.01,0.03,0.1}, and all 5 angles for all 4 tangents and 6 models.

## Additional controls and numerical interpretation

The per-channel quantization at relative step 0.03 on the coalition random tangent at 0.25 degrees scored **721, 760 and 682** across seeds 103/131/167. The retained scalar coordinate quantizer scored **503, 569 and 546** at the same angle and relative scalar step. These quantizers use different, explicitly specified scales and coordinate systems, so their percentage levels should not be interpreted as identical underlying noise energy.

For a probe-centered state `x=a*g+b*u+r`, removing the normalized direction `v(theta)=cos(theta)g+sin(theta)u` retains scalar `q_theta=-a sin(theta)+b cos(theta)`. The leading-component leakage amplitude is `s * sin(theta)` (with `s` the global PC standard deviation). A transverse Gaussian perturbation at `sigma=0.01*s` therefore has a leading-component leakage-to-noise ratio of ~0.436 at 0.25° and ~6.98 at 4°, before including the preexisting tangent component and nonlinear recurrent amplification. This ratio predicts the direction of vulnerability and gives a quantitative interpretation of the observed finite-noise frontier. It does not uniquely predict classification accuracy.

The original fixed h12 head and previously frozen h14 ridge results are recorded for every condition. New-ridge recovery refers to the **PEF-10 h12 ridge fitted on unperturbed projected probe states**. Noise is applied to heldout memory states at h8 prior to natural replay; the readout never sees noisy or quantized probe examples during its original fit. Noise-sweep averages over three registered pseudo-random draws, not many independently trained models.

## Reproduction and limitations

```
python -m pytest -q research/pef11/test_pef11.py
python -m research.pef11.pef11 \
  --evidence /path/to/original/registered \
  --selection-ledger /path/to/frozen_selection_ledger.json \
  --angle-ledger /path/to/frozen_angle_and_ridge_ledger.json \
  --previous-results /path/to/pef10/summary.json --out /tmp/pef11
```

Combined PEF-08/09/09R/10/11 local test suite: **26/26 tests passed**. No new gradients, checkpoints or changes to PEF-07 through PEF-10. This is a finite 12-bit synthetic model-family experiment and does not measure actual low-bit GPU/NPU kernels, novel Boolean operators, Lean theorem proving or pretrained language models. The feature selection and its hyperparameters were informed by earlier heldout PEF campaigns; PEF-11's split and criterion are frozen conditional on that earlier knowledge. The full local evidence archive includes model checkpoints, every prior source ledger, all 120 PEF-11 selections, 7,560 scored precision-condition rows, tests, logs and a SHA256 manifest.
