# ROC-04A: rule-permutation diversity restores development transfer, while complete table recovery remains weak

**17 November 2026 · completed six-model paired source-locked CPU campaign.** Original preregistration was committed at `5dbf46e428e495b36d3d50470197540ca39a9a4e` before any ROC-04 source generation or fitting. Full replay [GitHub Actions run #44085250022](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44085250022) completed all six independent seed jobs and the standalone six-seed audit successfully. Source-only integrity [CI run #44085353463](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44085353463) passed. The sealed ROC-02 original families 6–7 and ROC-04 reserved independent-test cases remain unopened.

## Registered comparison and source checks

Both train arms inherit ROC-03B's exact 51,520-parameter query-conditioned recurrent table-memory architecture, depth eight, width 128, with a final visible onehot query selecting a predicted 8×8 rule-table row. Across six identical-start paired model seeds, each receives exactly 1,920 AdamW updates, batch size 32, learning rate 0.001, zero weight decay, full eight-row auxiliary table-CE weight 0.5, and identical minibatch **index** draws. The training contexts, distractor rules, delay and query are identical across arms. The initial source-A rule table and thus the exact label differ.

- `low_affine4`: 1,024 cases, four previously exposed original ROC-02 train rules, 256 examples per rule.
- `high_permutation512`: 1,024 cases, 512 distinct newly drawn permutation rules, two examples per table.
- Development: **256 distinct A-table permutations never used in either training arm**, one record per rule. Only this exposed development set is scored in ROC-04A.
- Reserved prospective test: 384 additional disjoint permutation identities, with **zero input records or labels generated or evaluated** at this stage. All eight prior ROC-02 affine tables were removed from the new rule-table pool.

The source generator independently checks the symbolic interpreter against literal first-event A-table row lookup, training-pair schedule equality and SHA-256 disjoint table IDs. Each source file was fully hashed before fitting; all replay runs verified those hashes.

| Source file | SHA-256 |
|---|---|
| Low-four training JSONL | `a62b966872b92897395700b25c44652cff9f0a0fbd9697773e60b360d13c8111` |
| High-512 training JSONL | `eeb7c4429ff51470d01c261d7e913f0d5a4843660922377cc2934e13a4137bf5` |
| Unseen development JSONL | `13007b528f455891447f4b766d67e0e19b6d5d6ad11061bd77ccc5eecbed864e` |

## Preregistered primary endpoint

For each seed, compare last-step accuracy on the **same 256 development records**, high-diversity minus low-diversity. The protocol required at least five positive seeds, at least 0.20 absolute mean improvement, and one-sided exact six-seed sign-flip p≤0.05 over 64 sign assignments.

| Seed | Four-rule accuracy | 512-rule accuracy | Improvement | Full eight-row table exact, high-diversity |
|---|---:|---:|---:|---:|
| 2026111705 | 22.66% | 71.09% | +48.44 pp | 15.23% |
| 2026111707 | 22.66% | 71.48% | +48.83 pp | 9.77% |
| 2026111711 | 23.83% | 59.77% | +35.94 pp | 5.08% |
| 2026111713 | 24.22% | 75.39% | +51.17 pp | 10.16% |
| 2026111717 | 23.83% | 77.34% | +53.52 pp | 16.41% |
| 2026111719 | 20.31% | 69.14% | +48.83 pp | 10.16% |
| **Six-seed mean** | **22.92%** | **70.70%** | **+47.79 pp** | **11.13%** |

**Primary registered gate: PASS.** All **6/6 seeds** favor high diversity. The exact one-sided seed-sign-flip test has one extreme assignment among 64: **p=1/64=0.015625**. The absolute six-seed mean improvement is **0.4778645833**. Because all models were evaluated on the same exposed development pool, the test describes reproducible seed variation at this dev source, rather than a sealed independent rule-family test. Individual cases within a seed are not independent model replicates.

**Separate competence gate: FAIL.** The criterion required at least five seeds with accuracy ≥70% and complete table reconstruction ≥50%. Only **4/6** seeds achieved ≥70% answer accuracy and **0/6** achieved ≥50% complete table exact recovery. The high arm's mean per-row calibration reconstruction accuracy is **74.10%**, but all eight table rows are simultaneously correct on only **11.13%** of development cases. As a result, increased table diversity improved the queried answer much more than it recovered an entire faithful carried rule representation.

## Post-training legal task controls

A legal rule-edit control swaps the values of the original A binding table at queried row `q` and row `(q+4) mod 8`. It leaves the query, distractors, and event timing intact while changing the exact symbolic answer. Across the six high-diversity models, mean accuracy on these 256 legal changed-rule examples is **73.76%**, compared with **23.96%** in the low-four arm. The high arm is correct on both original and changed-rule inputs in roughly 41–59% of examples across seeds, showing partial but incomplete semantic dependence.

For cases with an intervening B/C distractor, the code replaces one distractor rule by another permitted previously exposed low-four rule. This edit leaves the exact A query answer unchanged. There are **221/256 eligible development cases**, and the high-diversity models retain the same predicted answer under the distractor edit on **92.53%** of eligible cases on average. These are dev-only task controls, with natural unequal input-edit costs. They do not establish ROC-01 Amendment C's norm-matched low-rank causal classification.

## Relation to ROC-03B and mechanistic claims

ROC-03B's same architecture and optimizer on just four affine training rules achieved 100% training accuracy but 28.91% development accuracy and 0/256 exact full tables under the table-supervised arm. Its post-hoc table decoding copied one of the four training table outputs on 244/256 examples. ROC-04A shows that **broad exposure to new literal A-table permutations changes the error mode**, using the same model family, loss and budget. The four-rule ROC-04A comparator continues to memorize training rules while transferring poorly, whereas high rule diversity produces a much stronger unseen-table queried answer.

The observed transfer gain alone is not a mechanism classification. The model receives table bits explicitly, and the final query-to-row routing is supplied by construction. Complete table recovery is still scarce. A fixed deterministic direct-row lookup solves the symbolic task at 100% for any valid permutation, so the learned recurrent network retains a sizable competence gap.

## Replay, provenance and future gate

- Frozen source and analysis registration: [protocol](PROTOCOL.md), `configs/roc04_permutation_diversity_transfer_v1.json`, original preregistration commit `5dbf46e`.
- Source and training scripts: `research_gym/analysis/roc04_permutation_source.py` and `roc04_permutation_train.py`.
- Independent summary code: `research_gym/analysis/roc04_summarize.py`; it reopens all twelve dev-prediction CSVs, checks exact six-seed row and label parity, recomputes the entire primary contrast and enumerates 64 sign flips.
- [Full hosted six-seed run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44085250022), artifact `ROC04A-complete-6-seed-permutation-transfer-replay`, includes all twelve actual 1,920-step `.pt` model checkpoints and optimizer states, the three train/dev source JSONLs, all raw model receipts and per-case controls, a source-only replay, independent six-seed summary, and **44 SHA-256-indexed files**. Standalone reproducibility ZIP SHA-256: `e16cfca1feae0a495cf86a93a2ff669ab1c699c1b61a6d550a68c2f7de5e12e6`.

The next scientific gate is the **independent ROC-04 reserved permutation test**, with all twelve trained checkpoints and this primary analysis protocol frozen first. Its 384 reserved rule identities are distinct from both high-diversity training and exposed development pools. Opening that test requires a separately committed prospective ROC-04B protocol, with one fixed final checkpoint per model and no new model selection. The original ROC-02 sealed families 6/7 remain separate and unopened.
