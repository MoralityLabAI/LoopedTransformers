# PEF-07 local CPU results and reproducibility boundary

## Protocol and execution

[Original preregistration commit a3b441c](https://github.com/MoralityLabAI/LoopedTransformers/commit/a3b441ce5fbc8f8d33914a1a34c883875ea5873c) was created before any PEF-07 training or scoring.

A full six-model local CPU run was completed, using fixed training seeds 103/131/167 for each of the two supervised tasks and 450 AdamW steps. The source consisted of 4,096 twelve-bit histories. The 256 eight-bit distractor contexts were partitioned, with all sixteen combinations of the oldest four bits allocated together: train 2,560; probe 768; heldout 768. Sealed split SHA256:

849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1

The heldout exact context-only ceiling is 480/768 for the coalition task (62.5%) and 528/768 for the additive-threshold task (68.75%).

**Execution boundary:** This is a local Python **3.13.5**, PyTorch **2.10.0+cpu** run, with AVX512 CPU capability, DNNL_MAX_CPU_ISA=AVX2 and one CPU math thread. The separately committed GitHub PR implementation uses the same frozen architecture, data split and training protocol, but is a distinct readable source file. GitHub Actions failed before runner steps and did not produce a hosted trial result. The local scripts, source manifest, checkpoint bytes and complete receipts are preserved in a separately linked conversation artifact.

## Results

Both original registered decision rules evaluated to **criterion_met** in the local run, each with **3/3** coalition seeds passing. Every model was frozen before its downstream probe was fitted.

| Seed | Task | Trained native h12 /768 | New fitted h14 head /768 | Random-initialized h14 head /768 | Intended-to-cross h12 mixed-logit ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| 103 | Coalition | 768 | 768 | 673 | 291.55 |
| 103 | Threshold | 768 | 768 | 728 | 0.98 |
| 131 | Coalition | 768 | 768 | 679 | 119.46 |
| 131 | Threshold | 768 | 768 | 746 | 1.00 |
| 167 | Coalition | 768 | 768 | 666 | 77.05 |
| 167 | Threshold | 768 | 768 | 727 | 0.98 |

The native task-head criterion required two coalition seeds with at least 615/768 h12 correct and at least 576/768 h14 frozen-trunk downstream predictions correct. The separate interaction diagnostic required the mixed-logit intended-pair ratio to exceed 1.25 and exceed the seed-matched threshold-arm ratio by 0.20 in at least two seeds.

The coalition teacher at focal old-bit source 0000 activates only two complete pairs, (0,1) and (2,3). Each coalition trained model reproduced these exact two pair-only class activations over 48/48 heldout contexts. Their actual continuous four-corner logit margin was much larger for intended than for crossed pairs.

The additive THRESHOLD teacher activates **all six** two-bit pairs from old-bit zero, although its **teacher prethreshold score is additive**. All three neural THRESHOLD models also reproduced the six pair-only class effects across 48/48 contexts. Their trained *logits*, however, are nonlinear and have nonzero four-corner responses, so the additive-logit surrogate cannot be assumed to describe the learned network merely because the teacher prethreshold score is additive.

These results establish learned pair-specific responses on a new twelve-bit synthetic family, with actual task supervision of the relevant four-bit Boolean operations. They establish context generalization rather than extrapolation to unseen Boolean operators. Random-initialized trunk decoding was already high and is included as a necessary capacity control.

## Independent frozen replay

A separate local read-only replay recomputed all six saved checkpoint SHA256 values and the complete source partition, rescored native h12 predictions, and fitted new h14 ridge heads using **augmented least squares** rather than the original normal equations. Every one of the six native and six downstream scores reproduced exactly at **768/768**. Independently recomputed intended-to-cross mixed-logit ratios also matched.

Checkpoint SHA256s:

- seed103 coalition: dc0cecc46b52904476d8e3ede3d5d9d1e2b0d9c8061d6e9b72222679e58e33d8
- seed103 threshold: 7290b63158af42302a777ae47487009e8afbb401511ec8d78cae8dbd5fed33b6
- seed131 coalition: ac6ee6819096957dc0f3c7f28f7d9eb9e67e9a2f68b9141f1fee99640a06685e
- seed131 threshold: f8f2f3cbb822ed7d501af92ebe8836db0f49a81e4005efd4acce281e506eb2ed
- seed167 coalition: ca637fdb461f0ebae6bfb255c5c4b283039f472434266f165bbafd6736157fdb
- seed167 threshold: 232e586c8c307ead403ff36641eda4aaf537c43efc3b1b31362862bfb474e412

The next prospective experiment should hold out **novel Boolean gate compositions or interacting input positions**. No pretrained language-model, Lean reasoner or safety-capability result is claimed. The PEF-04/05 registered negative results remain unchanged.
