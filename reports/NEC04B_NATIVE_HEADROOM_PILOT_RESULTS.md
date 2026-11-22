# NEC-04B: six-root Looper native-headroom checkpoint pilot

**Completed 22 November 2026. Evidence class: original-new calibration-world model training and within-calibration-world validation only.** The deterministic [NEC04A world gym](NEC04A_BOOLEAN_WORLD_GENERATOR_RESULTS.md) supplied a fixed new population of 24 calibration, 24 developmental and 16 confirmation world-root commitments. This pilot trained **exactly six calibration roots** at four sequential optimizer budgets and measured native model headroom on context groups owned by those same calibration worlds. The new developmental/confirmation **world roots remain uninstantiated**.

The full source/teacher-oracle/preregistration, Looper architecture and dependencies, exact six-root set, four budgets, checkpoint sealing rule, independent replay auditor, root-level headroom ranges and stopping criterion were committed before any new pilot training at [**NEC04B frozen commit e43dcfb**](https://github.com/MoralityLabAI/LoopedTransformers/commit/e43dcfbbf5102e3ab6b2265fe9c5e0abbadc2c0b).

**[GitHub Actions run #44992324562](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44992324562) succeeded**: 7/7 prospective code and oracle tests; independent pretraining teacher-label verification; **six out of six** original pilot world-root workers, each with independently audited original frozen eight native checkpoints; and the final pure-stdlib six-root headroom aggregate. There were **48/48** independently verified native model checkpoint byte hashes, optimizer-state step counters, original 12-step source+owned-validation logits and target-label correctness receipts. The eight checkpoints comprise four budgets in each of two model arms, per root.

Final aggregate: GitHub artifact ID **14742233649**, named **nec04b-six-root-native-headroom-independently-audited-aggregate**. Independently audited pretraining source/validation teacher inputs: artifact **14742611264**, named **nec04b-frozen-six-root-pretraining-oracles-independently-audited**. The six individual checkpoint/output artifacts are NEC04B pilot root0–5, respectively IDs **14742213234**, **14741843160**, **14742630522**, **14741954595**, **14742138779**, **14741818638**. They preserve all model+optimizer checkpoint files and native source/validation full-logit receipts for reproduction.

## Root and data isolation

The six pilot world IDs are **nec04a-cal-000..005**, giving exactly two original calibration worlds from each of the NEC04A easy/medium/hard symbolic circuit tiers. The source programs and first **80 complete 32-mask context groups per root**, or **2,560 training histories each**, came directly from the original NEC04A calibration-source artifact. That artifact's SHA256 remains **82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0**. The original 64-world root registry SHA256 remains **88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e**.

Each root's **24 owned developmental-context groups**, 768 previously untrained 12-bit histories in the *same calibration-world root*, were separated from the 80 source groups by the frozen original NEC04A SHA256 context ownership. The new pilot opened these owned validation-context target labels only under the committed pilot protocol; all **24 reserved contexts** in each calibration root were left without model scores, as were all *new development-world root* and *new confirmation-world root* specifications. No original NEC03Z group56–63 or NEC03P recruited-world data was read or scored.

**Composed** native teachers use the original committed root-specific 12-bit Boolean programs. The new **additive/linear** native teachers use an independently registered root-specific nonzero signed 12-bit linear score, deterministic SHA256 weight selection and a half-integer threshold chosen to balance the full symbolic truth table. The independent pretraining auditor recomputed the exact additive full-4096-case symbolic truth table per root and checked both task teachers' source and validation labels. It verified **30,720 source case-arm labels** and **9,216 owned-validation case-arm labels**, or **39,936 original teacher-label receipts** before any native training. Additive teachers were made independently from the old saturated NEC03 models.

## Frozen training and neural audit

Use the original repository **PEF07 Looper** network: 32D recurrent hidden state, original binary embedding, shared two-token attention/MLP residual block and a 2-logit head after twelve chronological looped updates. Each original root and teacher arm was initialized by the preregistered SHA256-derived seed and trained with **one continuous AdamW trajectory for 900 steps**, LR=0.003, no weight decay, gradient-norm clipping at 1.0, minibatch size 64 sampled only from its original 2,560 source training rows. The CPU stack uses Torch 2.10.0, a single CPU thread, deterministic algorithms, and an independently seeded source sampler.

Native model weights and complete optimizer state were frozen after **120, 240, 480 and 900** steps, without resetting optimizer moments. Each arm's four model/checkpoint SHA256 digests were saved before its owned validation truth labels were consulted. The native auditor independently restored all 48 original checkpoint files, compared their exact byte SHA256, Looper model weight hashes and optimizer state update counts, independently regenerated composed and additive true labels, and replayed every original checkpoint's full **2,560 training and 768 owned-validation logits**. This is an independent **checkpoint and model-prediction replay**, *not a second independent from-scratch optimizer-trajectory reconstruction*. Every original root, arm and checkpoint passed.

Across all twelve teacher models, the pilot performed **10,800 original native optimizer updates** (900 each), with **zero source SAE fit steps** and **zero causal-sham outcomes**.

## Validation accuracy by independent world root

Accuracy in each cell is the original native binary classification accuracy on all **768 owned calibration-world validation histories**. The four budgets belong to one continuous optimizer trajectory for each original root and teacher arm. Each model root is the statistical unit for any cross-root comparison.

| Root | Original symbolic tier | Teacher | 120 steps | 240 steps | 480 steps | 900 steps |
|---|---|---|---:|---:|---:|---:|
| cal-000 | Easy | Composed | 70.31% | 69.27% | 57.29% | **75.00%** |
| cal-000 | Easy | Additive | 76.56% | 83.85% | 91.15% | **93.36%** |
| cal-001 | Medium | Composed | 50.00% | 50.00% | 50.39% | **51.04%** |
| cal-001 | Medium | Additive | 74.74% | 81.38% | 90.49% | **97.40%** |
| cal-002 | Hard | Composed | 45.31% | 49.61% | 48.44% | **57.94%** |
| cal-002 | Hard | Additive | 73.44% | 90.76% | 97.66% | **91.28%** |
| cal-003 | Easy | Composed | 55.99% | 50.39% | 46.61% | **58.98%** |
| cal-003 | Easy | Additive | 62.76% | 81.64% | 85.42% | **92.06%** |
| cal-004 | Medium | Composed | 55.73% | 57.55% | 54.69% | **51.69%** |
| cal-004 | Medium | Additive | 70.83% | 74.74% | 75.39% | **86.33%** |
| cal-005 | Hard | Composed | 66.02% | 66.28% | 65.49% | **69.14%** |
| cal-005 | Hard | Additive | 83.98% | 87.24% | 92.84% | **95.96%** |

Original six-root mean per teacher family, equally weighting world roots:

| Updates | Composed mean validation accuracy | Additive mean validation accuracy |
|---|---:|---:|
| **120** | **57.23%** | **73.72%** |
| **240** | **57.18%** | **83.27%** |
| **480** | **53.82%** | **88.82%** |
| **900** | **60.63%** | **92.73%** |

This is substantial empirical **task-family separation** under identical Looper architecture, source-case count and training-step schedule. Native models learn the signed linear-threshold additive teacher much more easily than the context-switched Boolean composed teacher on this six-root calibration pilot.

At 900 steps, the original composed native models on **cal-001, cal-002, cal-003 and cal-004** remain below 60% validation accuracy. The composed cal-001 model scored 392/768 (51.04%) while its training accuracy was 1266/2560 (49.45%), indicating failure to learn the training function. By contrast, composed cal-004 reached 1600/2560 (62.5%) on source training but only **397/768 (51.69%)** owned validation, a **10.81-percentage-point source/validation gap**. That combination shows both representational/task-learning difficulty and calibration-context generalization limitations. The additive cal-001 model reached 748/768 (97.40%) validation at 900 steps.

## Registered root-level accuracy-band stopping gate

The prospective gate required **at least two of six independent world roots in EACH of three validation accuracy bands per teacher family, under one common checkpoint budget**: moderate [60%,75%), intermediate [75%,90%) and high [90%,98%], with high inclusive of 98%. Only after satisfying that for both teacher families could the lowest eligible budget be promoted to the next full-24-calibration-root stage.

| Updates | Composed moderate / intermediate / high | Additive moderate / intermediate / high | Both-teacher band gate |
|---|---|---|---|
| **120** | 2 / 0 / 0 | 4 / 2 / 0 | FAIL |
| **240** | 2 / 0 / 0 | 1 / 4 / 1 | FAIL |
| **480** | 1 / 0 / 0 | 0 / 2 / 4 | FAIL |
| **900** | 1 / 1 / 0 | 0 / 1 / 5 | FAIL |

The registered lower/upper band boundary and all six original worlds remain fixed. The composed family never has any root in the high-accuracy band and retains **four or five below-60% roots at every tested budget**. Increasing the optimizer budget up to 900 steps improves average additive training substantially but shifts additive roots toward the upper high band rather than producing two world roots in every band.

**Machine-verified decision: STOP_NATIVE_HEADROOM_SELECTION_CALIBRATION_ONLY.** The six-root aggregate recorded **selected training budget = null**, exactly **6/6** independent root audits and **48/48** original checkpoint replays, **zero source SAE fitting**, and **zero physical sham/causal mechanism scoring**. No root IDs, teacher functions, training budgets or accuracy bands were changed based on results.

## Posthoc diagnostic: the five-bit parity branch is near chance

This part is explicitly **retrospective descriptive analysis**, performed after the native validation results were available, using only the original frozen programs and archived model validation logits. It is a target for a *future preregistered curriculum or memory experiment*, not a new efficacy endpoint or population significance result.

Each NEC04A composed target program has an outer context-conditioned Boolean MUX. When its selector context bit is **1**, the executed expression is five-focal XOR parity with further context XORs. When the selector is **0**, the deterministic secondary XOR/AND/NOT/MUX expression runs. The six original validation root programs and logits therefore permit a fixed branch split without re-running a model.

| Original root | Parity-branch correct / cases at 900 updates | Alternate branch correct / cases |
|---|---:|---:|
| cal-000 | **192/384 (50.0%)** | **384/384 (100.0%)** |
| cal-001 | **226/448 (50.4%)** | **166/320 (51.9%)** |
| cal-002 | **160/320 (50.0%)** | **285/448 (63.6%)** |
| cal-003 | **177/352 (50.3%)** | **276/416 (66.3%)** |
| cal-004 | **256/512 (50.0%)** | **141/256 (55.1%)** |
| cal-005 | **128/256 (50.0%)** | **403/512 (78.7%)** |
| **Pooled descriptive cases** | **1139/2272 (50.13%)** | **1655/2336 (70.85%)** |

Each original context-specific parity branch is balanced across all 32 focal masks. Across all **four** registered optimizer checkpoints on **all six roots**, parity-branch validation stays near 50%. Several models predict one class for nearly every parity-branch case, while the alternate branch improves on some roots. This indicates a concrete **five-bit XOR parity and context-switching learning bottleneck** in the existing recurrent model. It does not establish a neuron-community mechanism or independent model-root causal claim.

Composed cal-005's overall 69.14% native validation accuracy has balanced accuracy only **62.58%**, because the alternate branch has an imbalanced target prior. This further motivates keeping balanced accuracy and conditional branch accuracy alongside raw accuracy in future headroom calibration.

## Scientific boundary and next experiment

NEC04B supplies actual Looper learning curves on six precommitted new calibration world roots and an exact independently replayed checkpoint/optimizer record. The new benchmark still lacks a qualified per-teacher native baseline headroom distribution, an SAE reconstruction audit and a physically/semantically calibrated sham negative control. The observed parity-branch bottleneck calls for a separately frozen **calibration-root-only parity curriculum or recurrent-memory intervention** rather than new tuning against sealed development or confirmation roots.

The next study may compare the same initial six roots under frozen task-aligned context-balanced minibatches, a progressive focal-bit parity curriculum or recurrent depth changes. Such experiments need their own full source-scoring and independent checkpoint audit protocol, with the entire original world-root cohort retained. The result of any such pilot remains in-sample world-root calibration.

All original NEC03Z source groups **56–63**, old NEC03P independent world-root contexts, original 32 five-input embargoed and old 64 four-input confirmation cohorts remain protected. **All 24 new NEC04A developmental world roots and 16 new confirmation world roots remain uninstantiated**. The original NEC03P independent pool of 69/88 source-qualified roots still has **zero measured causal-sham outcomes**; ≥59 eligible independent causally measured sham-root outcomes are needed for any eventual nonparametric 95/95 tolerance calibration under a scientifically meaningful negative control.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
