# PEF-09 and PEF-09R | Local intervention evidence

Executed 2026-11-15 on PyTorch 2.10.0+cpu / Python 3.13.5, single CPU math thread with `DNNL_MAX_CPU_ISA=AVX2`. The exact original PEF-07 six 450-step checkpoints were independently verified against the source/byte/parameter ledger and left unchanged. Context-disjoint probe and heldout groups were preserved. This is **local CPU evidence**, not successful Python 3.12 Actions validation.

## Separate registration boundaries

- PEF-09 registration commit `2bbd53ec098ed5c256fe9b015599cc076c877a4f`, committed before running rank-one ablations on heldout. PEF-07/08 results were already known.
- PEF-09R recovery registration commit `4ea181286bea7d3e935f2585a2280f14e1b90117`, **after** PEF-09 results were inspected and before any PEF-09R heldout ridge recovery was scored. This explicitly conditional follow-up cannot change PEF-09's negative criterion.

PEF-09 probe-direction frozen selection ledger SHA256 `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157`; PEF-09R probe-only frozen ridge ledger SHA256 `df0c0090b04471b0bb32e396cf92ba770266d29b8de1c7b7e7a5d71429d89dae`.

## PEF-09: Ungated rank-one causal ablation

Each pair direction is the leading right singular vector of the 192 probe-only four-corner hidden differences, covering **all four complementary-bit assignments** across 48 probe contexts. The same rank-one projector is applied globally to *every* input mask, using a global probe-state center, with no bit-conditioned edit and no classifier-head weights used for selection.

At visit 8, **both pair directions for each seed** cause the frozen h12 head to predict the majority class for every input:

| Task / all 3 seeds | Original h12 | h12 after removing either pair-derived rank-one direction |
| --- | ---: | ---: |
| COALITION | 768/768 | 480/768 (all class 0) |
| THRESHOLD | 768/768 | 528/768 (all class 1) |

For coalition seeds 103,131,167 the absolute probe cosine of the pair 01/23 directions at visit 8 is 0.99679, 0.99722, 0.99911 respectively. Target directions are also close to the global top PCA direction. Ungated ablation of the *other intended pair* direction or the global top PCA direction produces the same majority-class prediction in every seed.

The preregistered loop-8 target-pair selectivity-excess criterion is **criterion_not_met: 0/3 coalition seeds**. PEF-09 targeted selectivity excesses at loop 8 are -0.00120 / +0.00007 for seed 103; -0.01497 / +0.00660 for seed 131; -0.00030 / -0.00371 for seed 167. There is a general high-variance task axis, but these results do not isolate the intended pair computation.

**Control limitation:** All eight closest-energy directions from 256 isotropic random candidates preserve 768/768 original accuracy on this task at visit 8. Yet their probe projection energies reach only 11.7%–38.7% of the target direction across the six tasks and both pairs. The random controls are therefore *not truly energy matched*, despite the predetermined closest-of-256 procedure. The comparison is an ablation magnitude contrast, not a calibrated demonstration that the pair directions are uniquely causal. The very high top eigenvalue makes independent isotropic directions with equal energy geometrically rare.

## PEF-09R: Post-ablation linear recovery

PEF-09R froze the previously fitted PEF-09 probe features and trained the same fixed-penalty ridge on **post-ablation probe states only**. Readouts were sealed in JSON before scoring the 48 heldout contexts. This tests whether task information remains linearly accessible, separately from the native frozen head.

| Coalition seed | Frozen h12 after pair ablation | New h12 ridge, pair 01 | New h12 ridge, pair 23 |
| --- | ---: | ---: | ---: |
| 103 | 480/768 | 762/768 | 745/768 |
| 131 | 480/768 | 768/768 | 768/768 |
| 167 | 480/768 | 768/768 | 768/768 |

The independently registered PEF-09R recovery gate was met in **3/3 coalition seeds**, requiring both pair-derived directions at t8 to yield >=576/768 with new, probe-trained h12 ridge. Threshold controls also recovered 755–768/768 for the two target directions in all three seeds.

The *natural h12 probe covariance* has extremely low effective rank:

| Seed | Coalition top-PC variance | Coalition effective rank | Threshold top-PC variance |
| --- | ---: | ---: | ---: |
| 103 | 97.61% | 1.049 | 96.26% |
| 131 | 97.78% | 1.046 | 99.24% |
| 167 | 95.07% | 1.104 | 98.58% |

**Crucial geometric distinction:** Removing the exact top PCA direction at h12, then retraining the same ridge on the residual probe states, yields 480/768 for all three coalition seeds and 528/768 for all three threshold seeds. Removing the *nearly collinear but not identical* pair-derived directions at h8 permits 745–768/768 recovery. This sensitivity to angle must be studied before treating individual direction ablations as evidence of exclusive feature ownership. h12 PCA rank sweep at k={1,2,4,8} and h14 readout recovery are fully preserved in the JSON receipts.

## Methods, provenance and limits

The paired differences in PEF-09 select directions from known candidate bit pairs, so this is a source-conditional, outcome-label-free feature selection, **not** blind feature discovery. All actual patches are ungated and use the same global operator on every focal mask. PEF-09R refits an external decoder and does not show recovery by the original h12 head.

The 12-bit synthetic inputs share the same task function across all contexts. Context transfer, rank-one causal dependence of the trained readout, and post-ablation decoder recovery are supported within this finite family. A uniquely identified natural Boolean circuit, SAE concept identity, language-model generalization and Lean-theorem reasoning improvements remain untested.

Source and regression tests: `pef09.py`, `pef09r.py`, `test_pef09.py`, and `test_pef09r.py`. Nine local invariant tests passed. The full six-checkpoint run is reproducible with the external PEF-07 archive and frozen PEF-09 selection ledger; the complete raw per-condition score matrices and every fitted readout are preserved in the linked local evidence ZIP.
