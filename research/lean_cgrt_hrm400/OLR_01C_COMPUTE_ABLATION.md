# OLR-01C: compute-depth, cross-message, and reproducibility audit

**Date:** 2026-11-23  
**Status:** Completed native Lean exploratory audit. No architecture promotion or direction-specific mechanism claim.  
**Parent:** [OLR-01C protocol](OLR_01C_GUARD_LEARNING.md), [initial result](OLR_01C_RESULTS.md).  
**Reproducibility run:** [45168959420](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45168959420).  
**One-pass and message audit:** [45169236940](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45169236940).  
**Latest audited ZIP:** [olr-01c-native-guard-pilot.zip (artifact 14815748042)](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45169236940/artifacts/14815748042). Both CI workflows passed.

## Why this extension was required

The original 3-pass recurrent cross-message arm saw *both* candidate guards at each iteration; the direct and isolated arms read candidates independently. An OOD gain against them could come from the extra **candidate-pair information** rather than from performing more recurrent steps.

Added `cross_onepass`: one cross-candidate update, using exactly the same public numeric features and **17,985 stored parameters** as the 3-pass cross arm. Its training data, minibatch sequence, model seeds and 80-step optimizer schedule match the primary. It computes fewer model passes. The comparison isolates whether three visits appear useful beyond a one-pass cross reader on this developmental grammar. It does not match model FLOPs or training-stage compute.

Also added inference-time interventions on each *frozen* 3-pass cross checkpoint:
- `cross_zero_message`: replace the opposite-candidate signal with zeros in the same tied block, without retraining or modifying source evidence.
- `cross_margin_sham`: preserve the natural checkpoint, perturb its selected-action logit margin at the **same absolute dose** as message removal, and assign an independently hashed random sign per root and seed. This is a margin-magnitude control, not a universal causal-sham certificate.

Each policy selection is replayed on a native Lean 4.19 proof root under the same two-tactic limit, with independent standalone proof compilation. No query label enters the model scorer.

## Results: complete paired worlds

Both worlds must close under Lean to count one complete pair. The 12 development root pairs and eight OOD root pairs are re-used for all model seeds, so seed totals are descriptive repeated observations.

| Policy | Development, seeds 17/29/43 (of 12 each) | Sum dev (of 36) | OOD, seeds 17/29/43 (of 8 each) | Sum OOD (of 24) |
| --- | --- | ---: | --- | ---: |
| **3-pass cross recurrence** | **5 / 3 / 11** | **19** | **3 / 1 / 8** | **12** |
| **1-pass cross reader** | 2 / 9 / 12 | **23** | 0 / 3 / 3 | **6** |
| Cross checkpoint, messages zeroed | 5 / 2 / 12 | 19 | 3 / 1 / 5 | 9 |
| Cross checkpoint, matched-margin sham | 5 / 2 / 10 | 17 | 3 / 1 / 7 | 11 |
| Independently trained isolated recurrence | 4 / 2 / 11 | 17 | 0 / 3 / 3 | 6 |
| Direct per-candidate reader | 4 / 3 / 11 | 18 | 0 / 3 / 4 | 7 |
| Frozen recurrent body, fitted embed/readout | 5 / 5 / 10 | 20 | 2 / 3 / 3 | 8 |
| Exact host arithmetic oracle | 12 / 12 / 12 | 36 | 8 / 8 / 8 | 24 |

On development roots, the single-pass cross reader completes four more pairs across seeded evaluations than the 3-pass recurrent model. The 3-pass checkpoint completes six more pairs across seeded evaluations on the out-of-range panel. This sign change reveals distribution-sensitive depth effects and is **insufficient evidence that added recurrence is broadly useful**.

The OOD cross-vs-one-pass paired differences across seeds are **+3, -2 and +5** of eight; the seed-29 one-pass reader beats the longer loop. The effect across three seeds therefore has substantial variance. These seed outcomes share the same eight source-independent OOD graph roots, and their observed range precludes a reliable mechanism claim.

In the OOD panel, zeroing the frozen message loses **three** complete pairs across seed observations, while the matched-margin sham loses **one**. The seed-43 checkpoint is responsible for those differences: natural 8/8, zeroed 5/8, sham 7/8. The seed-17 and seed-29 OOD pair counts are unchanged by message removal. This is a **seed-specific causal effect on policy choices** and supplies no prospective matched-sham improvement across model seeds.

## Reproducibility audit

The first two native CI runs used identical registered training/dev/OOD bytes, seeds, optimizer steps and model definition, yet produced differing fitted weights and some different development choices:

- [45167635953](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167635953): model archive SHA-256 `23a702fa7a634cff50b407588aa78d4022bea700587fdc60cc15cce33a56b1e9`.
- [45168290451](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45168290451): model archive SHA-256 `bd45278f4d1da3275e5166774585c1c7940a3cca7a0dc6f7c3d4b1d984b8b7b9`.

The source shard digests agreed bit-for-bit in those runs. The exact numerical cause of the earlier drift has not been isolated. The follow-up enabled `torch.use_deterministic_algorithms(True)`, disabled CPU MKLDNN and fixed one CPU thread. Within a single runner, repeated source fitting at the same seed now gives identical tensors.

**Cross-run verified checkpoint equality:** the source train/dev/OOD files are bit-identical across [45168959420](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45168959420) and [45169236940](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45169236940). All **12 common model state dictionaries** (`cross`, `isolated`, `direct` and `frozen_cross` at seeds 17/29/43) were independently downloaded from their artifact ZIPs and compared tensor by tensor. **Every tensor is bit-for-bit equal.** The latter run adds three distinct one-pass checkpoints, so comparing whole ZIP or `models.pt` file hashes would be misleading.

Runner environment in the deterministic audit: Python with PyTorch `2.5.1+cpu`, x86_64 Linux, single CPU thread, MKLDNN disabled, deterministic algorithms enabled. The native Lean 4.19 verifier, exact arithmetic source bank and independent axiom-free certifier were unchanged.

## What this suggests about OLR

The current neural model learns to rank **two simple numeric equality guards**. It carries two recurrent slots with trainable cross-candidate messages but does not yet maintain a full proof hypergraph across live tactic steps, plan on multiple simultaneous obligations, or learn a new tool-feedback update rule. A fixed integer-adder oracle already solves every target in two Lean calls.

The experiment supports the following narrow observations:

- The pinned native data pipeline, source/query split checks, independent certificate compiler, and deterministic model checkpoint protocol operate end-to-end.
- Three recurrent visits can affect out-of-range policy choices even with the same input and parameterization as a one-pass reader, but this advantage lacks stability across seeds and flips direction on development.
- A single checkpoint can depend causally on its cross-message input, while a magnitude-matched logit sham can reproduce some of the same outcome changes.

The next research gate should **move away from static equality classification** toward a multi-step, shared-premise Lean proof where exactly exposed symbolic structure is insufficient and native feedback changes the proper action. Register fresh root families, all verifier-call and FLOP constraints, matched direct/one-pass controls, an independent structural planner, the message intervention, and an EvalMath FSI-01 sham/power floor before model scoring.

Do not promote this pilot as a learned novel-recurrence success or scale to 5M/20M parameters based on the current data.
