# ROC-05A: fresh source-pool and independent six-seed training replication

**17 November 2026. Primary preregistered gate PASS; separate whole-table competence gate FAIL.** Source and outcome protocol was frozen in `configs/roc05_fresh_permutation_replication_v1.json` at `a56c177bbc956274fbf65018b5724d291e25ef3c`, before producing ROC-05 examples or training. The first complete hosted six-seed training run was [GitHub Actions #44101721734](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44101721734). Its six independent paired seed jobs succeeded and the standalone six-seed audit succeeded. That run's final archive-packaging step failed on a misnamed protocol path. The six original model artifact ZIPs were retained and independently reconstructed into one SHA-256-audited first-run archive. The fixed archive workflow subsequently passed in [full CPU repeat #44103113334](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44103113334).

## Frozen comparison

All 1,152 rule-table identities used anywhere in ROC-04A's train/dev or ROC-04B's consumed test pool and all eight original ROC-02 affine tables were excluded from ROC-05's newly shuffled permutation universe. ROC-05A created 1,024 examples for each train arm on identical event schedules, queries, distractors and delays. The four-rule arm repeats the four already-exposed source-A affine families; the high arm uses **512 entirely new permutation tables**, exactly two cases per table. The exposed ROC-05 development set contains **256 further disjoint, newly generated permutation tables**, one case per table. All original and legally edited targets pass the exact symbolic interpreter.

Six fresh initialization seeds `20261117231/233/239/241/251/257` give twelve trained 51,520-parameter, depth-8 query-conditioned recurrent models. Within each seed the two models begin with exactly the same weights, see the same minibatch indices, and run 1,920 AdamW steps at batch 32, lr 0.001, with equal task+0.5 eight-row table losses. Fixed step 1,920 is the registered outcome, regardless of development performance.

## Original complete training cohort (primary evidence)

| Initialization seed | Four-rule answer accuracy | 512-rule answer accuracy | Paired gain | High model exact 8-row table |
|---|---:|---:|---:|---:|
| 20261117231 | 47/256 = 18.36% | 189/256 = 73.83% | +55.47 pp | 42/256 = 16.41% |
| 20261117233 | 50/256 = 19.53% | 181/256 = 70.70% | +51.17 pp | 15/256 = 5.86% |
| 20261117239 | 55/256 = 21.48% | 203/256 = 79.30% | +57.81 pp | 40/256 = 15.63% |
| 20261117241 | 50/256 = 19.53% | 191/256 = 74.61% | +55.08 pp | 29/256 = 11.33% |
| 20261117251 | 56/256 = 21.88% | 201/256 = 78.52% | +56.64 pp | 44/256 = 17.19% |
| 20261117257 | 57/256 = 22.27% | 182/256 = 71.09% | +48.83 pp | 36/256 = 14.06% |
| **Mean** | **315/1,536 = 20.51%** | **1,147/1,536 = 74.67%** | **+54.17 pp** | **206/1,536 = 13.41%** |

The original-run six model-seed differences are **all positive**, yielding the exact one-sided six-sign randomization count **1/64 = p 0.015625**. The preregistered primary success criterion required at least 5/6 positive directions, mean gain ≥0.20, and one-sided p≤0.05; **PASS**. The separate competence gate required ≥70% answer accuracy and ≥50% complete eight-row table exactness in at least five of six high-diversity seeds. All **6/6** clear 70% answer accuracy, while **0/6** reach the complete-table threshold; **FAIL**.

On the original exposure-matched development tasks, high-diversity models average **75.33%** accuracy on legal source-A table-row-swap counterfactuals which change the exact answer, while four-rule models average 21.22%. Predictions from high-diversity models remain invariant under eligible label-preserving B/C distractor substitutions on approximately 94.99% of cases, averaged over seeds. These natural-dose edits are task specificity controls, not norm-matched hidden interventions.

## Numerical replay and checkpoint provenance

The complete, independently checked **original-run** archive is `ROC05A_original_first_six_seed_reproducibility.zip`, SHA-256 **`7e21395a51eb5a6db6d51a0e88a04a099b078627eb9a649e818911940a582f10`**. It preserves the original twelve `.pt` checkpoint and optimizer blobs, six source-hash-checked model receipts, twelve per-case prediction/control CSVs, source JSONLs, full scripts and a 48-entry SHA-256 manifest. The separately verified independent seed-only auditor recomputes all primary statistics directly from the original first-run CSVs. The source JSONL SHA-256 checks are:

| Frozen source | SHA-256 |
|---|---|
| Low-four train | `b590a3bcc8b1018c430f5eb6617144caaa9a87899a1d45110ee1804a9a2c9642` |
| High-512 train | `c687a129a0d8e5edda26195c22bd116732409a07fcff6511c6766fb998f12bf4` |
| New development | `342d9d811f8bdcaa6a8b4e86d4c02b66f772a3e2c1546cd0c521939f88f054ee` |

The second hosted training from the same source and seeds, [run #44103113334](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44103113334), independently passed all twelve model/receipt checks and archived a complete 45-file package, SHA-256 **`caa6c3b7253779124171698b599aa39d368371c5c39d41547af333d085fa366d`**. Its mean high-dev accuracy is **75.13%**, low-dev accuracy **20.51%**, and mean paired gain **+54.62 pp**, also 6/6 positive with exact p=1/64. The mean complete-table recovery is **11.98%**, and the separate competence gate fails.

**Numerical portability limitation:** Only **2/12** CPU model checkpoints are byte-identical between the independent full training runs. The maximum absolute per-seed gain difference is **0.02734375**, corresponding to seven of 256 task cases. Both runs confirm the preregistered effect, but they share model initializations and source cases, so they are **repeat computational fits**, not twelve independent model-seed replications. The *first* training run remains the primary selected cohort. The second run must not be substituted selectively for a higher score; its outcome and checkpoint hashes are disclosed. The causal ROC-05B protocol pins the **twelve original first-run checkpoint SHA-256 identities** rather than assuming GPU/CPU byte parity.

## Claim boundary and next experiment

ROC-05A repeats ROC-04's large source-rule-diversity transfer advantage on **fresh train/dev permutation populations and fresh independent model seeds** at the same per-model compute budget. It establishes a robust within-study causal effect of training source-table diversity on this explicitly query-conditioned architecture. The network's final query-to-table-row route is supplied by design, and its complete rule-table accuracy remains far below an exact table-lookup oracle. This is a behavioral rule-transfer result, **not** a closed hidden-state quotient or ROC-01 mechanism classification.

[ROC-05B's frozen analysis protocol](../../configs/roc05b_matched_rule_causal_v1.json) pins the original first-run models and registers physically same-dose source-A rule edits, hidden rank-seven observed/dark steering, and equal-radius soft input comparators on the already exposed ROC-05 development examples. No ROC-05B causal intervention has been measured by this ROC-05A report. **The 384 newly allocated ROC-05 reserved rule identities remain only identities: no input cases or labels generated or scored.** The separately historical ROC-02 sealed families 6–7 remain untouched.
