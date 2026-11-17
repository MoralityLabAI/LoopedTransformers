# LS-SAE06: high-gain recurrent modes move all-81 digit logits without changing solved Sudoku predictions

**Executed November 16, 2026.** [LoopedTransformers draft PR #128](https://github.com/MoralityLabAI/LoopedTransformers/pull/128). The prospective [protocol](../native_task_transfer_v1/PROTOCOL.md) was committed at **3b1bec465b89c552b93cd413593045bf61d4cec0**, before any LS-SAE06 heldout model state. [Original source-locked GitHub Actions run #43827999742](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43827999742) completed the CPU regressions, original pretrained checkpoint pilot, the complete 8+8 fresh Sudoku test and independent standard-library numerical audit successfully.

**Registered scientific verdict: FAIL in BOTH fresh Sudoku families.** The frozen primary endpoint was a *discrete missing-cell digit prediction flip* relative to a unit-norm random source intervention at ACT6, not raw continuous digit-logit sensitivity. No prediction flipped under any original signed physical edit.

## Primary preregistered result

Each unique new full Sudoku solution was generated outside the canonical solution orbit and then dug to exactly 30 or 34 uniquely solvable missing cells. One median visible blank selected the injection site. The 512D source z_H state was taken after ACT1; each selected direction received symmetric ±0.20 physical edits and was propagated through ACT6. The symmetric missing-cell flip score F was half the sum of the number of changed predicted blank digits under positive and negative edits.

| Preregistered ACT6 measure | Fresh 30-hole | Fresh 34-hole |
|---|---:|---:|
| Fresh unique-solution puzzles | 8 | 8 |
| Eligible structural STATE_POWER directions | 8 | 8 |
| STATE_POWER symmetric blank prediction flips | **0 in all 8** | **0 in all 8** |
| Equal-dose RANDOM symmetric blank prediction flips | 0 in all 8 | 0 in all 8 |
| Puzzles with F(power)−F(random) ≥1 | **0 / 8** | **0 / 8** |
| Mean power-minus-random flip score | 0.0 | 0.0 |
| Original 2,000-draw puzzle bootstrap 95% | **[0.0, 0.0]** | **[0.0, 0.0]** |
| Registered family gate | **FAIL** | **FAIL** |

The unchanged family gate required at least six of eight positive flip contrasts and a strictly positive paired puzzle-cluster bootstrap lower bound. Exact **blank-cell correctness gain was also zero** for every direction, sign and radius. No original puzzle was dropped for geometry or answer correctness, and the stronger failure is that **every one of the four intervention arms had zero predicted-digit flips across all 81 cells**, for both ±0.05 and ±0.20 doses at ACT3 and ACT6.

## Whole-board verified correctness, including one early recovery

| Verified clean full-board metric | Fresh 30-hole | Fresh 34-hole |
|---|---:|---:|
| ACT3 exact complete solved boards | **8 / 8** | **7 / 8** |
| ACT6 exact complete solved boards | **8 / 8** | **8 / 8** |
| ACT6 clean correct missing digits | 240 / 240 | 272 / 272 |
| ACT6 original clue disagreements | 0 | 0 |
| ACT6 27-unit Sudoku conflicts | 0 | 0 |
| ACT6 incorrect predicted digits caused by tested edits | **0** | **0** |

The one ACT3 exception in the 34-hole family predicted **32/34** blank digits correctly and recovered to **34/34** by ACT6 without an intervention. The full board validity check requires all 27 row, column and box units to contain all nine digits and every original clue to agree. The source generator verifies exactly one completed solution independently, so a valid predicted completion agrees with that unique registered target.

These 30/34-hole fresh families were **easy for this pretrained checkpoint at ACT6**, creating a ceiling in the task-accuracy endpoint. The negative causal flip result should be interpreted with this difficulty and confidence context, rather than as evidence of complete functional irrelevance of the singular mode.

## Continuous native digit-logit response is substantial

Both power methods were source-label-blind. STATE_POWER used exactly four native full-97×512 Jacobian JᵀJ iterations, inherited from LS-SAE05. LOGIT_POWER used an equal four forward-JVP/four reverse-VJP search on the **centered 81×9 digit-logit Jacobian** at ACT3, removing per-cell constant-logit shifts that do not change softmax. SAE was the original frozen fitted decoder direction and RANDOM was a separately seeded isotropic unit vector.

The following are **mean centered digit-logit Frobenius changes divided by the physical source edit radius**, symmetric over both signs of ±0.20, across all 81 cells.

| Native output logit response per input radius | 30-hole ACT3 | 30-hole ACT6 | 34-hole ACT3 | 34-hole ACT6 |
|---|---:|---:|---:|---:|
| **STATE_POWER** | **0.734** | **0.399** | **2.465** | **2.128** |
| **LOGIT_POWER** | **0.738** | **0.398** | **2.472** | **2.132** |
| Frozen source SAE decoder | 0.057 | 0.021 | 0.237 | 0.151 |
| Source isotropic random | 0.047 | 0.025 | 0.055 | 0.037 |

These are continuous logit changes in the native neural model, measured by actual finite perturbations. They coexist with **zero** altered argmax predictions, exact solved Sudoku outcomes and unchanged blank-cell correct counts at both physical doses.

The absolute cosine between the two independently sourced 512-dimensional structural and logit-power source directions has median **0.99484** across 30-hole puzzles and **0.99754** across 34-hole puzzles. Minimum individual absolute cosines were **0.95820** and **0.95630**. The high-gain structural mode is therefore also close to the dominant all-81 **centered output-logit sensitivity direction** at these clean states. LS-SAE05's low correct-digit task slope at the *original edited goal token* did not imply that the singular direction was absent from other digit logits. In this new experiment the stronger coupling to all-cell output is measured directly.

The median source STATE_POWER full-state Jacobian gain at ACT3 was approximately **0.775** and **1.291** on the two new families, while the median all-81 centered-logit JVP gain was **0.680** and **1.386**. Original actual ±0.05 finite-logit symmetric secants agreed with autograd JVPs, with median relative discrepancy approximately **0.00111** and **0.00077** for STATE_POWER (30/34-hole), measured against 1 plus the JVP norm.

## Exact finite decision-margin robustness explains the zero flips

The clean source's winning-digit logit gaps were enormous relative to the physical source edits. Use the proved [centered-logit winner-gap barrier](LS_SAE06_TASK_GAP_BOUNDS.md): a change of argmax digit at a given cell requires its original winner gap Δ to be ≤sqrt(2) times the L2 norm of the cell's **centered finite logit displacement**. This is an exact finite inequality based on the saved 81×9 logits.

Across every fresh source puzzle, all four direction classes, both ± signs and both fixed radii:

| Original finite barrier measurement | 30-hole | 34-hole |
|---|---:|---:|
| Smallest clean blank-cell winner gap at ACT3 | **44.74** | **12.40** |
| Largest observed sqrt(2)×centered cell displacement at ACT3 | **0.203** | **1.337** |
| Largest across cells/arms per-cell displacement-to-gap ratio at ACT3 | **0.00454** | **0.02542** |
| Smallest clean blank-cell winner gap at ACT6 | **62.13** | **61.39** |
| Largest sqrt(2)×centered cell displacement at ACT6 | **0.0481** | **0.5410** |
| Largest per-cell displacement-to-gap ratio at ACT6 | **0.000740** | **0.00811** |

Every recorded blank cell satisfies the strict margin barrier, often by orders of magnitude. Thus the raw logits imply unchanged blank-cell argmax predictions for these finite edits independently of directly checking argmax; the independent receipt auditor confirms the integer digit predictions as well.

The same absence of flips across all 81 cells also follows directly from the exhaustive saved argmax receipts, including given clues. This local robustness certificate is specific to the **tested physical radii, actual finite native outputs and these clean test states**. It does not certify model robustness to arbitrary source perturbations, larger state edits, a different checkpoint, or harder Sudoku inputs.

## Original source and reproducibility

- Frozen original pretrained native model: gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att step_21700 SHA-256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**, original LoopScope02 PyTorch FP32 implementation.
- Byte-sealed calibration-only PCA/SAE from LS-SAE04 source job, original NPZ SHA-256 **073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25**, source fit-array SHA-256 **83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec**, calibration manifest SHA-256 **cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a**.
- New pilot30 and pilot34 canonical source roster SHA-256 **518e4e75017892c8232e3605d715789cb197661f04d0e669d316b26e615e2f50**, **6a40d87b3efb6d7e07435c24a037655d6b5d75a82e99dee90662bca4c9babaee**.
- New registered full 30-hole and full 34-hole roster SHA-256 **f393291b10efd3c2245c07895ac3d2082bf16878e6acf3c56d0628866cca25e7**, **d325b263af36cdaafde90cabeb5e03cedd0962c3b51f8a6711b8db043872de6c**.
- Original [full 3-stage Actions #43827999742](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43827999742), archive [artifact #14209000375](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43827999742/artifacts/14209000375).
- Original **32,468,443-byte** full JSON SHA-256 **f40c75474bfe2c7e01dd7b5170263c6e0a16434687b094e9cc4aee9a45328d68**; independent stdlib audit JSON SHA-256 **f82cde95e6d64e29e43bcca576af1cdd55770961a1230eb194f1de8aad400b9d**; Actions ZIP artifact SHA-256 **7d2fc8f0491101a08eebd111d7c3970a272f16cc80c731ec36e6f58a08906e56**.

The independent audit recomputes all new unique Sudoku solution roster hashes, source goal locations, both four-step JᵀJ power transcripts, the four matched edit arms, every 81×9 clean and finite output array at both horizons and radii, centered-logit Frobenius effects, exact blank/given correctness, 27-unit Sudoku validity, frozen clean true-digit rivals and all 2,000 puzzle-cluster paired bootstrap decisions. It replays **saved numerical values** without independently executing the neural model or interval-certifying PyTorch arithmetic.

## Interpretation and next discriminating test

LS-SAE05 found narrow source-state directions with large finite-horizon **hidden-state** gain. LS-SAE06 finds that, on a fresh independent noncanonical Sudoku family, those same source-blind directions align closely with the dominant **all-board centered output-logit** singular directions. Their causal logit effects are substantially larger than equal-norm random source edits; the trained model's solved-digit output is strongly protected by large existing logit margins at these doses.

To test correctness-relevant causal transfer, a separate experiment should use source-locked, exactly unique **harder Sudoku families with clean model errors or smaller prediction margins**, selected using only puzzle structural difficulty before native evaluation. The original full source fit, heldout rosters and negative LS-SAE06 gate remain untouched. A dose defined by a frozen input-independent rule, or a prospectively registered clean-margin scale, must be evaluated for OOD hidden-state displacement before treating digit flips as mechanism evidence.
