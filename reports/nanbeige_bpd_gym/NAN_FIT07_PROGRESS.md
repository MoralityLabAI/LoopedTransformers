# NAN-FIT-07 fresh-source VPD/BPD progress

Updated 2026-11-20T09:52:02.201778+00:00

The source-only pilot was frozen and pushed at `561394b` before its first pretrained inference. Sixteen fresh source roots plus one explicit copy control per root give 32 clean contexts/backend. No source or answer repair, prompt truncation, result filtering or parser fitting is allowed.

The fixed screen requires at least 6/8 main and 8/8 copy answers in EACH CAL/DEV split at loop 2, separately per backend. It grades a first free token, distinct from native prefix-stop and Prime complete-reply endpoints. Passing this screen permits fitting only; the original algebraic gate remains failed.

| Backend | Worker evidence status | Clean screen | Audited learned cells | SAE cross-references |
|---|---|---|---:|---:|
| NF4 | competence_gate_failed | fail | 0/2 | 0/2 |
| BF16 | baseline_complete_or_fits_in_progress | pass | 1/2 | 1/2 |

Partial worker evidence is reported as partial; a state file alone does not prove that a process is live. The serialized GPU queue waits on the actual Prime supervisor and checks known model processes before loading a new model.

| Backend | Split | Condition | Native loop | First-token correct | Format valid | Roots |
|---|---|---|---:|---:|---:|---:|
| NF4 | calibration | main | 1 | 2/8 | 7/8 | 8 |
| NF4 | calibration | main | 2 | 7/8 | 8/8 | 8 |
| NF4 | calibration | copy | 1 | 0/8 | 0/8 | 8 |
| NF4 | calibration | copy | 2 | 8/8 | 8/8 | 8 |
| NF4 | development | main | 1 | 2/8 | 7/8 | 8 |
| NF4 | development | main | 2 | 5/8 | 8/8 | 8 |
| NF4 | development | copy | 1 | 1/8 | 1/8 | 8 |
| NF4 | development | copy | 2 | 8/8 | 8/8 | 8 |
| BF16 | calibration | main | 1 | 2/8 | 6/8 | 8 |
| BF16 | calibration | main | 2 | 7/8 | 8/8 | 8 |
| BF16 | calibration | copy | 1 | 0/8 | 0/8 | 8 |
| BF16 | calibration | copy | 2 | 8/8 | 8/8 | 8 |
| BF16 | development | main | 1 | 2/8 | 7/8 | 8 |
| BF16 | development | main | 2 | 7/8 | 8/8 | 8 |
| BF16 | development | copy | 1 | 1/8 | 1/8 | 8 |
| BF16 | development | copy | 2 | 8/8 | 8/8 | 8 |

| Backend | Fit depth | Cell status | Steps | Factor update L2 | Controller update L2 |
|---|---:|---|---:|---:|---:|
| NF4 | 1 | pending | 0 | — | — |
| NF4 | 2 | pending | 0 | — | — |
| BF16 | 1 | independently checked | 8 | 0.145263 | 0.191444 |
| BF16 | 2 | failed | 0 | — | — |

| Backend | Fit depth | Split | Mode | Readout loop | Correct | KL root mean | Local physical dose root mean | Reached-input change root mean |
|---|---:|---|---|---:|---:|---:|---:|---:|
| BF16 | 1 | calibration | all_one | 1 | 2/8 | 0 | 0 | 0 |
| BF16 | 1 | calibration | clean_informed | 1 | 2/8 | 0.0891782 | 0.40748 | 0 |
| BF16 | 1 | calibration | component_only | 1 | 2/8 | 0.078362 | 0.135311 | 0 |
| BF16 | 1 | calibration | reached_state | 1 | 2/8 | 0.0891782 | 0.40748 | 0 |
| BF16 | 1 | calibration | static_CAL_mean | 1 | 2/8 | 0.0842419 | 0.333289 | 0 |
| BF16 | 1 | development | all_one | 1 | 2/8 | 0 | 0 | 0 |
| BF16 | 1 | development | clean_informed | 1 | 3/8 | 0.0891364 | 0.406976 | 0 |
| BF16 | 1 | development | component_only | 1 | 2/8 | 0.0780369 | 0.13536 | 0 |
| BF16 | 1 | development | reached_state | 1 | 3/8 | 0.0891364 | 0.406976 | 0 |
| BF16 | 1 | development | static_CAL_mean | 1 | 2/8 | 0.0841692 | 0.332471 | 0 |

Every completed cell uses eight actual stock four-term BPDTrainer steps on CAL main roots only, shared rank-8 factors at layer-21 down_proj, and a fixed final checkpoint. The auditor checks complete source coverage, host grades, exact source/step order, four-term loss accounting, initial SVD identity, finite checkpoint/optimizer state, nonzero factor/controller changes, controller replay, all-one readouts and physical doses, means of saved CAL tolerance estimates and all registered diagnostic modes. It does not independently rerun full model gradients or recompute full-token CAL activation means.

T=1 uses zero ablation. T=2 uses the framework pass-mean residual replacement and sphere rescaling. Component-only therefore denotes the residual-neutralized framework pathway, not a complete rank-8 reconstruction of the native operator. Static, clean-informed and reached-state routes have unmatched physical doses and are fidelity diagnostics.

The JSON retains every initial/final diagnostic group, finite gauge diagnostic, physical factor-subspace comparison and completed SAE/initial-SVD/norm-matched-Haar cross-reference. SAE observations precede final loop RMSNorm; only down_proj has the registered immediate identity-addition bridge. T=1 application at loop 2 is outside its fit depth.

Engineering evidence is separate: random width-64 worker paths passed BF16 T=1/T=2 and NF4 T=2 with eight optimizer steps each; these are not pretrained results.

There are zero formal inferential comparisons, no calibrated matched causal-dose/sham correctness floor and no donor transfer. Eight independent DEV roots cannot support mechanism confirmation or equivalence. Rank 16, other sites/layers, the o_proj intervening-MLP bridge and genuine donor-specific causal controls remain later stages. The original 168-case test stays sealed.

[Frozen protocol](../../experiments/nan_fit07/PROTOCOL.md) · [Source registration](../../experiments/nan_fit07/SOURCE_MANIFEST.json) · [Full aggregate evidence](nan_fit07_progress.json) · [Engineering receipts](../../experiments/nan_fit07/receipts)
