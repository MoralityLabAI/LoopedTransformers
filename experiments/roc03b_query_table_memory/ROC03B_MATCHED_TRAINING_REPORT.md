# ROC-03B: query-conditioned recurrent table memory memorizes training families

**17 November 2026. Status: preregistered single-seed competence gate FAIL.** Frozen protocol: `configs/roc03b_query_conditioned_table_memory_v1.json`, committed at `750e6a9e14fb5116082bb0ffdf967ed173e69e94` before training. Hosted CPU run: [GitHub Actions #44069373243](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44069373243). Training source families 0–3 and calibration families 4–5 are exactly the original SHA-pinned ROC-02 splits; families 6–7 remain sealed.

## Design and precommitted gate

ROC-03A failed to learn the delayed-rule task with its original linear readout. ROC-03B tests a deliberately stronger readout bias: the 128D tied recurrent prequery state receives the original A binding plus distractors and carry events; a linear head decodes an 8×8 table of row logits, and the final *visible* query one-hot selects one of those rows. The final query is excluded from the recurrent memory update, making the memory prediction query-independent by construction.

Two otherwise identical-start arms use seed `2026111703`, 1,920 updates, batch 32, AdamW lr 0.001, zero weight decay, and the same train-example stream. One minimizes task cross-entropy. The second adds `0.5 ×` eight-row table reconstruction cross-entropy, supervised only by the original training inputs' A table. The table query route is a supplied inductive bias, so its training success would not itself prove spontaneous recurrent symbolic discovery.

The registered gate requires at least 90% train and 70% calibration answer accuracy for the table-supervised arm at the fixed final update.

## Task result

| Fixed updates | Task-only train | Task-only calibration | Table-supervised train | Table-supervised calibration |
|---:|---:|---:|---:|---:|
| 240 | 99.71% | 29.69% | 99.90% | 28.91% |
| 480 | 100% | 29.69% | 100% | 29.30% |
| 960 | 100% | 29.69% | 100% | 29.30% |
| **1,920** | **100%** | **30.47% (78/256)** | **100%** | **28.91% (74/256)** |

At the last update, task-only calibration cross-entropy is **9.0072** and table-supervised calibration cross-entropy is **11.5879**. Both arms attain full 8-row table reconstruction accuracy on the 1,024 training cases, while both score **0/256 complete-table exact reconstructions on calibration**. The table-supervised per-row calibration accuracy is **25.59%**, and task-only is **25.05%**. Strong train performance and large calibration error identify family memorization rather than a competence gain on held-out rule permutations.

### Post-hoc decoding diagnostic (not part of the registered primary gate)

I loaded the two hosted checkpoint files and decoded all eight table rows for each of the 256 calibration inputs, comparing the resulting eight predicted row labels against the **four original training family rule tables** (families 0–3). In spite of the true rules belonging to calibration families 4–5, the decoded table exactly matched one of those four *training* rules on:

- **244/256** calibration examples (table-supervised arm).
- **216/256** calibration examples (task-only arm).

The table-supervised model generated only **six distinct** complete row-label vectors across 256 calibration inputs, versus 22 for task-only. Its mean best rowwise agreement with any one training-family table was 98.83%, versus 95.90% for task-only. These post-hoc diagnostics are computed on the already exposed calibration set; they establish a mechanistic-looking *memorized-table output pattern* under the particular learned readout, with no claim about a fresh rule-family test.

## Reproducibility

The original model starts from identical parameters in both arms; source split hashes, independent scorer, optimizer budget, and no-sealed-access rule are checked by the hosted workflow. The exact saved model checkpoint SHA-256 values are:

| Arm | Frozen 1,920-step checkpoint SHA-256 |
|---|---|
| Task-only | `1250659158b0f55805c5309fe9ef4fe185a1bb73c27f244ac931774cbd592c79` |
| Table-supervised | `f233754efb6cb4538306c457e6ecc54dc14bc871327af20fbd213ddd87e1725c` |

The hosted artifact `ROC03B-query-conditioned-table-memory-1920-step-pilot` contains both actual `.pt` checkpoints and the eight complete milestone receipts. Reloading each model reproduced the final task accuracy, task CE, per-row accuracy, and full-table exact accuracy for both train and calibration. The script `research_gym/analysis/roc03b_query_table_memory.py` regenerates the experiment using only the published train and calibration source files.

## Scientific consequence and next registered need

Increasing optimizer exposure was inadequate under the original model (ROC-03A). Query-conditioning plus explicit table supervision completely solves its *training* families, but fails on unseen calibration tables. This isolates **source-rule permutation transfer** as a distinct failure mode. A future source-locked experiment should train on a diverse distribution of held-out-by-permutation tables, while preserving a separate set of fresh never-seen rules, a matched model budget, exact oracle checks, and the original ROC-02 sealed source-family gate. No causal observer superiority or general-language-model statement follows from ROC-03B.
