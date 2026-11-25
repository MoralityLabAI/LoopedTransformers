# NBR engineering pilot report and fixed-direction control

Run 45515740820, parent branch commit 412e32cfd9d52e4f2d61067cbf0d69bf8ebe9145.
All four jobs succeeded. The three checkpoint artifacts were downloaded and
reviewed; per-board NPZ SHA256 hashes, clean/sham predictions and every
saved VDD, signed schedule improvement and final-correct count were
independently checked against raw prediction arrays and solver answers.
The original CI also ran its independent receipt audit.

Artifacts: gaoxin492 FP32 14977742490; gaoxin492 W4 14979040594;
yagizdevre FP32 14979105405. Shared six-board roster SHA256
bcf9b3c47f5703d8e33222bbb66f34e057277476c44a555b896dc5325dd50a5b.
Two boards per split; no scientific gate. All six boards are blank-correct
at ACT6 and ACT16 on all three checkpoint/precision arms. ACT3: 6/6 on
Gaoxin FP32 and W4, 5/6 on Yagizdevre. No eligible median-site recovery
case in either held-out family; no failure class. All detection AUROCs
undefined. Both small signed edit doses yield zero verified deduction
displacement across every held-out board and all direction arms.

## NBR-01 primary pilot statistic, ACT3 -> ACT6 L0

| Family | Mean native cosine | Shuffled null | Advantage | Residualized advantage |
|---|---:|---:|---:|---:|
| cal | .70998 | .00436 | .70561 | .70393 |
| eval | .79869 | .00336 | .79533 | .79763 |
| rep | .85364 | .00126 | .85238 | .84116 |

All 1536 columns valid per board at this site. ACT6 -> ACT12 raw advantage
.98171 eval, .98950 rep. Recurrence reuses a fixed decoder matrix; large
same-ID over shuffled-ID advantage can arise for any fixed residual
direction, even after static head residualization. These results establish
a measurement on six easy boards and cannot identify computational
primitives.

## NBR-03 local edge verification

| Checkpoint | Mean graph Jacobian vs central finite difference cosine |
|---|---:|
| Gaoxin FP32 | .9995523 |
| Gaoxin W4 | .9998466 |
| Yagiz FP32 | .9999209 |

Same-batch sham passed for all 12 held-out checkpoint-board instances.
Batch clean predictions agree with single-board runs. No full recurrent
circuit inference.

## NBR-07 synthetic adapters

| Checkpoint | Dynamic guess | Static guess | Output-only guess | Held target wins (24) | Negative control wins (24) |
|---|---:|---:|---:|---:|---:|
| Gaoxin FP32 | 6/6 | 1/6 | 6/6 | 11 | 2 |
| Gaoxin W4 | 6/6 | 2/6 | 6/6 | 6 | 3 |
| Yagiz FP32 | 3/6 | 2/6 | 3/6 | 11 | 1 |

Planted directions were generated using calibration gradients. This is a
construction-dependent engineering sanity check; output-only matches
dynamic target identification. No evidence of hidden finetuned behavior.
Pruning ablations on Yagizdevre delayed some deductions but left final
correctness unchanged. Equal-count ablations do not establish compute
savings or equal removed energy.

## NBR-01B: fixed shared direction control

Let W have normalized columns u_i, and let G_t be the transported logit
Jacobian at ACT t. The native score compares G_3^T u_i with G_6^T u_i.
The shuffled-ID null compares distinct directions at the two times.
Introduce fixed Gaussian mixtures v_j proportional to sum_i c_ji u_i,
using the same coefficients at both times and across boards. Before
renormalization, their covariance in expectation is proportional to the
native unit-column covariance. Positive scalar normalization leaves cosine
unchanged. Eight independent draws of 256 directions, seed 202611251031.
Compute J_t for mixtures by linearity from the saved per-neuron J_t and
decoder norms, without replaying the model forward. Apply the same
static-head projection removal to each mixture. Report native minus
mean fixed-mixture cosine at the board level, with the raw and residualized
forms, and the digit-change count. Toy identity check must give zero
advantage when the two transported maps coincide exactly.

Pilot reanalysis is exploratory, since this control was motivated by the
observed pilot results. Run on the original saved receipts. Prospective
proposal for separate registration: primary residualized L0 ACT3->ACT6
native-minus-fixed-mixture mean >=0.05, paired board bootstrap lower bound
>0 in both independent families; require >=40 valid boards per family
and separately report the preassigned stalled stratum. Treat a failed
control as lack of evidence of native-basis specificity even if the
original shuffled-ID gate passes. Do not use pilot outcomes to adjust the
registered threshold. Additional source-only PR #352 must pass and be
reviewed before prospective inference.
