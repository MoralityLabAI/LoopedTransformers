# PEF-14 | Independently trained context-phase calibration

Local experiment and evidence date: **November 15, 2026**. The protocol was committed before training and heldout inference at [`e63ad7c`](https://github.com/MoralityLabAI/LoopedTransformers/commit/e63ad7c93109713decdd69ad662c52e916d0a8b3). New branch is based on PEF-13 PR #94 head `2bb8ec38edfadaac1c6927e6577bf21563d5af9c`. The known PEF-13 negative result is retained. Python 3.13, PyTorch 2.10.0+cpu, single-thread deterministic CPU with oneDNN AVX2 selected. Python 3.12 hosted unit invariants passed (Actions run 43721689601); the separate full six-model hosted replay also passed (Actions run 43722118391), with six byte-identical checkpoint files, all 96 heldout integer scores identical, and an independent auditor pass. Full details are in HOSTED_REPLICATION.md.

## New source and frozen checkpoint provenance

- The **new** SHA256 partition salt is `pef14-independent-v1:2026-11-15:`. All 256 eight-bit contexts receive exactly one owner: 160 train / 48 probe / 48 heldout. Their corresponding 4096 complete twelve-bit words are disjoint at the context level. Every 16-pattern focal hypercube stays in one split.
- Six **freshly initialized** matched 32-channel recurrent attention models were trained with the original 450-optimizer-update PEF-07 schedule, seeds **211,223,239**, separately on coalition and additive-threshold tasks. The original six PEF-07 checkpoints were *not* reused.
- After an execution interruption, the first four complete checkpoints were recovered without retraining. Each recovered file passed the full 450-step optimizer-state audit, final parameter SHA256 check and source ownership check. The other two were trained and sealed normally. All six were complete and sealed **before probe fitting**.
- Each frozen model subsequently scored **768/768** native h12 accuracy on its 48 previously unseen heldout source contexts. There were no failed/truncated seed exclusions.
- Complete provenance: source SHA256 `68ee1776e613514e56050d187d15bf6595138db866493bcddc237504d057b5b8`; training ledger SHA256 `cf60408cd8dbf07083b3146bc4dc7e63e9444d79fabff5877e47a98e55586dd4`; frozen *probe-only* forecast ledger SHA256 `6f7ff6ae664679a742e8f15cb58ea3f63e5f489641ab78decf90c29ad58a5f53`.
- Every h8 principal direction, pair-derived and random tangent, new h12 ridge, Jacobian, and quantization forecast was selected or fitted **solely on the 768 probe examples**. All 96 forecasts were written and hashed before computing heldout h8 states. The same unmodified h12 ridge per condition was used to score the 768 heldout inputs under each step.

## Registered primary result: **criterion_met, 2/3 fresh coalition seeds**

MAE is absolute error in *fraction correct*, averaged over the fixed sixteen quantizer conditions for each seed: two tangent directions (random and pair01), positive rotations at 0.25° and 4°, and four new relative nearest-even lattice steps `0.009,0.018,0.045,0.07`.

| Seed | Task | Conditional mixture MAE | Uniform phase MAE | Exact probe phase MAE | 90%-interval coverage | Registered seed criterion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 211 | Coalition | **0.02769** | 0.04018 | 0.02620 | 5/16 | Pass |
| 223 | Coalition | 0.03381 | 0.03417 | 0.03426 | 4/16 | **Fail** |
| 239 | Coalition | **0.01482** | 0.01669 | 0.01497 | 11/16 | Pass |
| 211 | Threshold | **0.00910** | 0.02236 | 0.00724 | 9/16 | Pass (control only) |
| 223 | Threshold | **0.01791** | 0.02236 | 0.01921 | 8/16 | Pass (control only) |
| 239 | Threshold | **0.01284** | 0.02223 | 0.01310 | 8/16 | Pass (control only) |

The preregistered coalition rule requires mixture MAE <=0.030 and <=90% of uniform-phase MAE for at least two of the three fresh seeds. The independent auditor recomputed **all 96** forecasts from the stored (margin, Jacobian, phase, label) probe records and verified all six checkpoint byte digests, model parameter digests and 450-step optimizer states. It confirmed the 2/3 count.

Pooled descriptive MAE across coalition seeds is **2.544%** for the mixture, **3.035%** for uniform phase, and **2.515%** for exact probe phase. The mixture therefore improves the uniform approximation on this new training/split cohort; it does not improve over the known source-phase point estimator overall. The fresh threshold arm's mixture/uniform MAEs are 1.328%/2.232%. 

The registered 90% context-normal prediction intervals covered just **20/48 coalition** and **25/48 threshold** conditions, far below nominal coverage. They underestimate the distribution shift between distinct context groups; do not interpret them as calibrated uncertainty statements. A cluster-level/hierarchical predictive study must use new data to address this deficiency.

## Posthoc source of forecast error (excluded from decision)

After registering the 2/3 result, a separate analysis computed the *heldout* samplewise Jacobian and exact heldout lattice phase, as an explanatory upper-bound diagnostic. This cannot be used in deployment or the probe-only forecasts.

| Mean absolute difference, percentage points | Coalition (48) | Threshold (48) |
| --- | ---: | ---: |
| Heldout local-linear prediction versus actually quantized model, same inputs | **0.0298** | **0.0136** |
| Probe phase-mixture forecast versus heldout *linearized* score | **2.5360** | **1.3254** |
| Probe uniform-phase forecast versus heldout *linearized* score | 3.0264 | 2.2290 |

Thus the main residual error comes from cross-context differences in the joint distribution of clean margin, directional slope and quantizer phase. A local Jacobian remains a strong same-input approximation; the challenge is transferring distributional predictions to new context groups and correctly quantifying that transfer uncertainty.

## Scope and reproducibility

Run `python -m pytest -q research/pef14/test_pef14.py` and `python -m research.pef14.pef14 --out /tmp/pef14`. The stand-alone auditor is `python -m research.pef14.audit_pef14 --out /tmp/pef14`; posthoc diagnostics require the complete frozen evidence. Re-run on a fresh path to avoid checkpoint cache. The archived full source, evidence and SHA256 manifest allow an independent verification of the local outputs. Under other processors the SHA256 parameter state may differ because CPU floating-point kernels need not be bit-exact across hosts.

Results concern six tiny models trained on two already known Boolean operators. Context generalization on a new hashed split and fresh seeds provides independent model training replication, though all 16 focal masks for each task are still shared between training and testing. No new-operator, natural-language, SAE, Lean theorem-proving or safety capability is established. Earlier PRs remain unmodified and unmerged.
