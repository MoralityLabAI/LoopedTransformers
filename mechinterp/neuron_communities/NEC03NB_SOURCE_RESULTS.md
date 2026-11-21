# NEC-03N-B: Checkpoint-anchored reset-AdamW source learning

**Executed November 21, 2026. Evidence class: source-training-only competence calibration on four already examined five-input Boolean worlds.** Every new checkpoint began with the **exact original 900-step NEC-03M native model weights**, followed by 900 additional source updates under a **new AdamW optimizer**. This is a distinct recipe from the strict, continuous-optimizer NEC-03N experiment.

## Source model ancestry and failed continuous-replay investigation

The [merged NEC-03M native/SAE source pilot PR #255](https://github.com/MoralityLabAI/LoopedTransformers/pull/255) trained eight models (four composed five-input Boolean teachers and four independently seeded class-frequency-matched additive controls), each for 900 native AdamW updates on their complete 80 source-training context groups ×32 focal states. It fitted eight source-only TopK16 SAEs and found 8/8 feasible source angular controls, while only 6/8 native heads met the fixed 90% **in-sample source** learning gate.

The original [strict NEC-03N 900-to-1800 optimizer replay, PR #257](https://github.com/MoralityLabAI/LoopedTransformers/pull/257), failed its registered exact step-900 ancestry gate on two CPU workers, so that PR was closed **unmerged**. A separate [source-only diagnostic run 44897647242](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44897647242) verified a bit-for-bit match on both 900-step checkpoints in dev5-002 and material differences in the dev5-003 checkpoint weights. Dev5-003's composed model produced 1,920 in-sample source-correct outputs under this diagnostic versus the original 1,567, despite matching initial weight digests. The reason for different training trajectories across hosted workers remains unresolved. The registered continuous-optimizer ancestry result remains failed.

NEC-03N-B uses the **immutable original checkpoint bytes** restored by artifact name and run ID **44894980345**. It checks the original model and checkpoint SHA256, loads each original model with all parameter tensors, validates its original 2,560 source-training class predictions, initializes *fresh* AdamW moments, skips 900 historical minibatch draws of the separate fixed source generator, and runs exactly 900 new source-only updates. The originally trained models, SAE dictionaries and PEF/NEC inference gates remain unchanged.

The training contract and exact Git-source blob identities were frozen in [NEC03NB_RESET_LOCK.json](NEC03NB_RESET_LOCK.json) at commit **`48f2fc26d7f5883f1731132e0c9b25cd2e61da90`** before the first new weights. [Registered protocol](NEC03NB_SOURCE_PROTOCOL.md).

## Complete source learning table

Every entry is **correct predictions / 2,560 already used source-training inputs**, grouped as 80 complete seven-bit contexts of 32 correlated focal masks per world. The competency criterion was fixed at 90% before this follow-up.

| Five-input world | Composed original 900 | Composed frozen900 + reset AdamW 900 | Additive original 900 | Additive frozen900 + reset AdamW 900 |
|---|---:|---:|---:|---:|
| dev5-000 | 2242 (87.58%) | **2357 (92.07%)** | 2553 (99.73%) | 2557 (99.88%) |
| dev5-001 | 2560 (100%) | **2560 (100%)** | 2560 (100%) | 2560 (100%) |
| dev5-002 | 2474 (96.64%) | **2547 (99.49%)** | 2551 (99.65%) | 2560 (100%) |
| dev5-003 | 1567 (61.21%) | **2387 (93.24%)** | 2560 (100%) | 2560 (100%) |

**Original 900-step native source competence: 6/8 model arms and 2/4 complete teacher worlds.** Under the separate checkpoint-anchored reset-optimizer condition, **8/8 native model arms and 4/4 complete worlds** passed the 90% original-source threshold. Previously failed composed roots dev5-000 and dev5-003 gained **115 and 820 additional correct source predictions**, respectively. No per-world training length, seed, model architecture or source membership was changed in response to these outcomes.

## Independent receipts and hosted CI

[GitHub Actions run 44898361742](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44898361742) passed its locked-source tests and all **four separate CPU teacher-model training and original/new-checkpoint independent audit jobs**. Those jobs archived eight new frozen native model checkpoints, all 2,560 original source predictions for each old/new model, old/new full class confusion matrices, source model weight and file SHA256 digests and the first/last continuation minibatch-index hashes.

The run's final aggregation job failed on a lightweight worker because it imported the PyTorch-based neural auditor without installing PyTorch. The original four trained root artifacts were intact and all four of their independent source audits passed.

A separate **read-only, no-retraining standard-library replay** [Actions run 44898742318](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44898742318) retrieved those exact four original root artifacts by immutable run ID. It verified all eight new checkpoint byte SHA256 digests and independent neural-audit receipts, recomputed each of the original and new training-source class predictions against the **registered 32-bit Boolean truth functions**, and aggregated four distinct root-level source competence outcomes.

The completed aggregate is artifact **`nec03nb-four-root-source-checkpoint-reset-900vs1800-replay`**, ID **14705515343**, ZIP SHA256 **`3359231c9aac309d6c22d3c5ce6baee7002f8e6a57fa28912a639048e15c11af`**. Each separately trained source model and per-arm evidence artifact remains in run 44898361742, named `nec03nb-warm-source-root-0` through `root-3`.

## Interpretation and next gate

The result shows that these small tied recurrent models can reach the **in-sample five-input source competence** criterion on all four already examined roots when trained for 900 additional updates from the original frozen 900-step weights with a reset AdamW optimizer. It does not establish a bitwise reproducible continuous-AdamW 1800-step training trajectory or any generalization to independently held five-input context groups.

The previously fitted 900-step TopK16 dictionaries, source-attribution ranks and direction-sham controls belonged to the **old model weights**. Applying those unchanged SAEs to the new checkpoint-anchored weights would break model/SAE source provenance. A separate fixed source-only SAE dictionary and rank fit on these new weights must pass reconstruction and angle/dose gates before new causal behavior is measured.

All **24 five-input development-evaluation context groups**, 24 five-input reserved groups, 32 new embargoed world roots, and the original 64 four-input confirmation worlds remain untouched by this experiment. The original NEC-03H and NEC-03K behavioral ITT estimates and their two-percentage-point meaningful-effect threshold remain unchanged. These eight model heads share only **four** teacher worlds and all their scored inputs are training-source histories.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
