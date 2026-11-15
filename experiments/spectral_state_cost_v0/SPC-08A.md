# SPC-08A: phase alignment transfers, deferred-output alignment largely does not

This frozen-checkpoint extension of SPC-08 tests geometric transfer under
exactly the same orthogonal alignment fitted on the 256 paired, unlabeled
SPC-07 training prompts. The 48-channel independently trained models are
never fine-tuned. The top-eight subspaces are fitted on the SPC-08 training
cohort and compared by `trace(P_source P_target)/8`, equivalent to the
mean squared canonical cosine.

| Subspace | Raw overlap | Paired alignment | Shuffled-pair alignment |
| --- | ---: | ---: | ---: |
| Computational phase | 0.2004 | **0.8761** | 0.6932 |
| Immediate readout | 0.1865 | 0.7674 | 0.5973 |
| Ordinary temporal observability | 0.1873 | 0.4530 | 0.3493 |
| Deferred-only observability | 0.1989 | **0.2404** | 0.1864 |

The phase code aligns substantially better under the frozen
phase-derived gauge. The delayed-only sensitivity basis remains largely
misaligned despite containing future-output information in each model.
This says that the **chosen phase-calibrated coordinate alignment** does
not align deferred-only modes; a gauge fitted separately on delayed
measurements could behave differently. This is not an identification of
unique semantic features, and the results are limited to two synthetic
trained pointer models.

Reproduce after supplying the original SHA-verified SPC-04 weights:

```bash
python -m experiments.spectral_state_cost_v0.spc08_alignment_geometry \
  --checkpoint-dir /path/to/spc04/checkpoints \
  --out /tmp/spc08_alignment.json
```

The pinned numbers and alignment calibration summary live in
`receipts/spc08_alignment.json`. The unit tests verify gauge covariance
and exact equality for a known orthogonal coordinate conjugacy.
