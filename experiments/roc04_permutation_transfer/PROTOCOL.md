# ROC-04A: source-locked permutation diversity and carried-rule transfer

**Frozen 17 November 2026** at commit `5dbf46e428e495b36d3d50470197540ca39a9a4e`, before generating any new ROC-04 example or fitting its models.

## Rationale

The earlier 1,920-update [ROC-03B](https://github.com/MoralityLabAI/LoopedTransformers/pull/150) query-conditioned 128D recurrent table-memory model attained 100% training accuracy under both task-only and table-supervised arms. On family-disjoint calibration rules it attained 30.47% or 28.91%, respectively, with **zero complete 8-row table reconstructions**. Post-hoc, 244/256 table-supervised predictions reproduced one of four original training rules. Those training rules all used one affine multiplier, giving a narrow family and a plausible memorization route.

ROC-04A changes the diversity of the early source-A rule binding while fixing the recurrent architecture, query readout, optimizer cost, paired example schedules and task supervision. This tests whether broad literal-table exposure supports **generalization to previously unseen rule permutations**. It does not measure ROC-02's sealed families 6 and 7 and does not reuse those original examples. The final query onehot remains an explicitly supplied readout route.

## Source and data lock

Construct the 40,320 bijections on eight values. Exclude the eight original ROC-02 affine rules entirely from the new-permutation universe. Shuffle deterministically with Python `random.Random(20261117079)`. Partition the first 512 permutations into the high-diversity training rule pool, the next 256 into an exposed development pool, and reserve the next 384 table identities for a future *unopened* source-family test. ROC-04A builds only 1,024 training examples per arm and 256 development examples. Reserved future examples and labels will not be generated.

Two training corpora use exactly the same 1,024 event schedules. Each schedule fixes a delay from 2 to 16, source-B/C distractors from four already exposed ROC-02 training rules, and a final source-A query. In the **low-affine4** arm, source-A binds one of the four original ROC-02 training rules 0–3. In the **high-permutation512** arm, it binds one of 512 newly selected bijections, with each table appearing exactly twice. Thus only the first source-A table and its exact answer differ. All late queries and nuisance distractors are matched.

Development consists of 256 distinct, held-out high-diversity tables, one table per case. Every case label is verified both by the independent source-binding interpreter and literal row lookup. Publish byte-exact SHA-256 hashes of both train files and development file, table-ID disjointness, and source row IDs before any model training. The code is required to throw an error if asked to construct the reserved test split. The original ROC-02 family-6/7 files may never be loaded.

The source design intentionally creates a table-identity holdout while preserving the input coding. It does not create a syntactic out-of-distribution challenge. The correct symbolic behavior is available by a direct, deterministic table-lookup oracle; its 100% correctness is the reference level.

## Model and paired training

Use the exact `QueryConditionedTableMemory` module from ROC-03B, with 51,520 parameters, width 128, recurrent depth 8 and a 128-to-64 table head. All event updates except the last query are processed by the tied residual block. The visible query chooses a row from the 8-by-8 table logits.

For each of six fresh model seeds `2026111705, 07, 11, 13, 17, 19` (as complete integer suffixes), clone the same initialization for two arms. Each uses AdamW at learning rate 0.001, zero weight decay, batch size 32, fixed gradient clip 5, **exactly 1,920** optimizer steps, and identical per-seed sampled training *indices*. The task is eight-class cross-entropy plus 0.5 times full-eight-row cross-entropy for both arms. Checkpoint reports at steps 240, 480, 960 and 1,920 are descriptive. Model selection is *always* step 1,920, irrespective of dev scores.

Both arms use 1,024 training examples, 32 training examples per optimizer update, and the same model size. The experiment changes table diversity, with the targets necessarily varying because literal rule tables vary.

## Primary registered gate

Let `a_high,s` and `a_low,s` denote unseen-development 256-case task accuracy at the final frozen update for paired seed `s`. Record `d_s=a_high,s-a_low,s`. Predeclare **one** primary contrast, with acceptance requiring all three:

* At least five of six seeds improve: `d_s>0`.
* Mean six-seed gain `mean_s d_s >= 0.20` absolute accuracy.
* One-sided exact sign-flip probability `p = 2^-6 * #{epsilon in {-1,+1}^6 : mean(epsilon*d) >= mean(d)} <= 0.05`, with equality counted extreme.

An additional competence gate requires at least five of six independent seeds to attain **70%** unseen-dev answer accuracy and **50%** complete eight-row table reconstruction accuracy. Each gate has a fixed failure outcome. Report actual numerator/denominator counts, CE, full dev confusion rows and the complete per-seed vector. Do not use a pooled case-level p-value as a six-seed inference substitute.

## Source-level causal and specificity measurements

For each dev record, create a valid permutation edit by swapping the initial source-A binding table's values at queried row `q` and row `(q+4)%8`. The final query remains exactly the same, and the exact output label changes. After training is frozen, score each arm on that entire 256-case counterfactual set, and report correct-new-label fraction and the fraction correct on **both** the original and edited case. The edit is a legal discrete table change and carries its own physical input and latent costs, which must be measured before any equal-radius mechanism interpretation.

On each case with an intervening B/C distractor event, replace one distractor's table with a different low-four training rule while keeping the source-A binding and query unchanged. Verify equality of the exact correct label. Report prediction invariance and correctness under this nuisance edit, and account for ineligible no-distractor cases.

These are additional dev measurements, with no inferential claim beyond the original source family. ROC-01 Amendment C's full A/B/C/H null tests, rank sweep and matched physical interventions are future experiments and are **not** claimed as part of ROC-04A.

## Evidence standards and stopping

The reproducibility archive must contain generator source and tests, both train JSONL sources, the dev JSONL source, exact source and model SHA digests, all twelve original model checkpoint receipts (six seeds x two arms), and per-case final dev predictions for original and both legal controls. Record missing/invalid rows and every source hash mismatch. If any arm fails to train, the corresponding seed remains a failure in the six-seed report. The report must identify the full optimizer and model-input budget.

The independent reserved ROC-04 table-identity test remains closed until after source, architectures, metrics, seed/rank choices and trained checkpoint hashes have been frozen. A preliminary dev result cannot be called prospective source-family generalization. The original ROC-02 family-6/7 holdout remains unopened for this entire stage.
