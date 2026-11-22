# NEC-04F: matched recurrent XOR-state supervision on six untouched calibration worlds

**Prospectively frozen before new native training**, 22 November 2026. Research task [issue #305](https://github.com/MoralityLabAI/LoopedTransformers/issues/305). The research question concerns whether exact intermediate XOR supervision produces meaningful generalization on the original composed Boolean parity task, beyond the optimization effects of a matched meaningless auxiliary objective.

## Original model and world cohort

NEC04E's previous source-only 12-visit readouts yielded 100%, 95%, 78.34%, 53.60%, 50.94% accuracy on XOR of 1,2,3,4,5 observed focal bits. The previous random-initialized hidden control scored nearly as well. The 3-to-4-input transition motivates the new intervention.

Use **original NEC04A CALIBRATION world roots 12–17**, two per easy/medium/hard circuit tier. Restore original 24-calibration-world generator/source artifacts from [Actions 44983509977](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44983509977). Original 64-root registry SHA256 is 88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e and source corpus SHA256 is 82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0. Only the first 80 complete source context groups ×32 focal masks (2,560 examples/root) can enter training. The previously owned 24 evaluation context groups (768 examples/root), within the *same calibration WORLD root*, are scored after all model/optimizer checkpoints are frozen. Reserved contexts (24 per calibration root), separate 24 new developmental WORLD roots, separate 16 confirmation WORLD roots and all NEC03Z/NEC03P sealed cohorts remain protected.

Independently verify the original full 12-bit histories, immutable composed-program AST teacher labels and each XOR prefix at the chronological arrival of focal bits k=2,3,4,5. Generate deterministic source-only pseudo labels from the original root seed and source context group, using SHA256 to rank the 32 focal masks. Exactly the first 16 masks are class1 for each group and stage; remaining 16 are class0. The label construction uses only original root/group/mask/stage identifiers and has no dependence on any model score or true label. The independent scalar auditor must rederive all 16/16 pseudo labels. Pseudo labels exist on source contexts only.

## Original native network and three registered arms

Preserve the original PEF07 Looper: 32D recurrent memory and attention/MLP block shared across 12 chronological binary inputs, with the original 2-logit head at the end. Add **one shared 32D-to-2-logit auxiliary linear head**, evaluated after the 2nd, 3rd, 4th and 5th focal bits arrive. All three training arms have the same native architecture and identical root-specific model+auxiliary initial weights. Exact initial seed is derived from SHA256 of the original calibration-world root seed under the frozen NEC04F domain.

Train all arms on **the identical original source-minibatch sequence**: batch 64 uniformly with replacement from 2,560 source rows, separate Torch Generator with original root seed XOR 0x5DEECE66D. One continuous AdamW optimizer per arm, LR 0.003, no weight decay, gradient clip norm 1, CPU deterministic Torch 2.10.0, for exactly **900 optimizer updates**, checkpoints at 240 and 900 without moment reset. Save complete model+optimizer SHA256, original model-weight SHA256 and minibatch sequence SHA256 before opening owned-validation target scores.

Define the original final composed-task cross entropy as Lfinal and mean of four equal intermediate CE losses as Laux. Three fixed arms:

| Arm | Source-only training objective |
|---|---|
| **aux_shuffled** | Lfinal + 0.5 × Laux against the balanced-hash pseudo targets |
| **aux_prefix** | Lfinal + 0.5 × Laux against exact 2-,3-,4-,5-focal original observed-prefix XOR |
| **vanilla** | Lfinal + 0.0 × Laux, with the same auxiliary-head forward and parameters |

The **one primary comparison** is aux_prefix minus aux_shuffled. Vanilla is a separately registered secondary native baseline. Equal initial weights and source sample order must verify by SHA for all arms.

## Frozen correctness endpoint and root-level stop

The primary endpoint is the original native final-task composed Boolean accuracy on exactly those owned calibration-world validation cases where the original world program's outer MUX selects the five-focal XOR parity branch. It is **scored at 900 optimizer updates only**. The experimental unit is one original calibration WORLD root (n=6), with 32 masks/contexts correlated within root. Compute paired root-level accuracy difference aux_prefix minus aux_shuffled, equally weighted across all six roots. A positive calibration-only pilot must simultaneously satisfy:

- At least **+10 percentage points** mean paired parity accuracy difference;
- At least **4/6** strictly positive root differences;
- At least **60%** six-root mean original native parity correctness in the aux_prefix arm, above the exact 50% class-majority baseline;
- At least **−5 percentage points** alternate-MUX-branch retention (aux_prefix minus aux_shuffled, six-root mean).

Report every original failing root and arm, unrounded source/validation and branch correctness, 240-step diagnostic checkpoint, original native full-task accuracy and balanced accuracy, original k2/k3/k4/k5 true auxiliary prefix accuracy, and fifth-order final-logit Walsh spectral power. The intermediate auxiliary and spectral measurements are diagnostics and cannot substitute for native final-task correctness.

## Auditing and scientific limits

Before training, freeze original source/circuit SHA256 identities, original root cohorts and context allocations, both types of source auxiliary teachers, network/aux-head architecture, optimizer/source minibatch seed, checkpoint budgets, full primary family and stopping floor, and code file SHA256s. An independent scalar preflight verifies all original source and owned validation task labels, 32-mask group ownership and source-only balanced pseudo labels.

After both checkpoints for all arms are frozen, independently restore all original model+optimizer states and verify exact byte hashes, optimizer step counts, shared initial-weight SHA and uniform source minibatch trace SHA. Independently replay all original 12-step native final logits and four intermediate auxiliary logits on the 2,560 source and 768 owned calibration-world validation cases per root, re-score all original labels and retained successes/failures, and aggregate only when **all 36 checkpoints and all six root audits** are complete. The audit replays model predictions; it does not independently re-train the original optimizer.

The balanced-source-label arm is an **optimization control**, and does not qualify as a physically/semantically inert sham for neuron-community necessity. This is source-only training/within-calibration-world evaluation; no independent development/confirmation WORLD roots, reserved contexts, source SAE, or actual causal-sham outcomes may be scored. Even a positive pilot requires separately qualified native headroom and a new prospective SAE/physical-null instrument protocol. Old NEC04B–E outcomes remain unchanged.

**FSI-01 remains REDESIGN_OR_PILOT_ONLY.** 
