# PPT gyms: runnable CPU reference, checked Lean menus, and DRAM learning

**Scope:** Executable research reference for [PPT Plan A](PPT_PLAN_A_DRAM_PLASTIC_PROOF_TRANSFORMER.md) and [PPT Plan B](PPT_PLAN_B_INKLING_BLOCK_MESSAGE_MEMORY.md). The gym ranks pre-collected tactics against previously verified Lean receipts. It cannot certify new Lean proofs, run a full 20M or 55M production model, or establish online plasticity gains from an engineering smoke test.

Code: [ppt_gym/](ppt_gym/). The real Mathlib collector and Lean kernel interface remain in lcgrt/. DataGoblin's checked candidate exporter supplies public tactic menus and private receipt labels.

## What each gym tests

| Gym | Recurrent architecture | Online learning | Main controls |
| --- | --- | --- | --- |
| PPT-A | Byte-local encoder, eight typed latent roles, one shared Transformer cell, action/outcome/continue heads | Independent CPU FP32 fast logistic head updated from support feedback | Frozen weights; retrieval-only; inverted feedback |
| PPT-B | One-pass prelude, two middle blocks tied across loop passes, gated previous-pass message bank, one-pass coda | Same CPU fast head; future message-adapter training | Message normal/zero/roll; depth sweep 1/2/3/5/8/10; fitted exit proxy |

The reference models use actual reported parameter counts and are much smaller than the future 20M/55M design targets at width 48. Plan B has extra parameters and FLOPs relative to Plan A, so direct raw A-vs-B differences do not isolate causal effects of messages. Compare normal-versus-zero/roll messages and later add a fully compute-/parameter-matched A arm.

Only independently checked full-goal-closing tactics are positive labels. An accepted tactic which creates subgoals is a different task and requires the later live Lean stepper.

## CPU synthetic smoke

From the repository root:

~~~sh
cd research/lean_cgrt_hrm400
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s ppt_gym/tests -v

python -m ppt_gym fixture --out runs/ppt-fixture --per-family 12 --seed 20261121

python -m ppt_gym train \
  --fixture runs/ppt-fixture --allow-synthetic \
  --plan a --width 48 --heads 4 --steps 24 --max-seconds 120 \
  --out runs/ppt-a-fixture

python -m ppt_gym train \
  --fixture runs/ppt-fixture --allow-synthetic \
  --plan b --width 48 --heads 4 --steps 24 --max-seconds 120 \
  --out runs/ppt-b-fixture

python -m ppt_gym compare \
  --fixture runs/ppt-fixture --allow-synthetic \
  --checkpoint runs/ppt-a-fixture/checkpoint.pt \
  --depth 3 --support 4 --out runs/ppt-a-dram-smoke

python -m ppt_gym sweep \
  --fixture runs/ppt-fixture --allow-synthetic \
  --checkpoint runs/ppt-b-fixture/checkpoint.pt \
  --split calibration --depths 1 2 3 5 8 10 \
  --out runs/ppt-b-depth-smoke

python -m ppt_gym fit-exit \
  --fixture runs/ppt-fixture --allow-synthetic \
  --checkpoint runs/ppt-b-fixture/checkpoint.pt \
  --depth 5 --steps 24 --out runs/ppt-b-exit-smoke
~~~

All of these outputs explicitly say synthetic_fixture, scientific_eligible=false. The fake fixture labels test execution, versioning, gradient flow and custody only.

## Import DataGoblin's checked Lean training pool

Run DataGoblin's rlt_lean_pool_export.py on an independently completed and audited *train* campaign. Its expected structure is:

~~~text
checked-pack/
  manifest.json                   # datagoblin.checked-candidate-pack.v1
  train.public.jsonl              # goal and tactic menu, without outcomes
  train.receipts.jsonl            # exact per-candidate checker receipts
  calibration.public.jsonl        # only if source campaign supplied it
  calibration.receipts.jsonl
~~~

The PPT importer verifies declared SHA-256 file digests, aligns ID and epoch, rejects invalid axiom/kernel positives, and preserves the source manifest and environment hashes in training manifests. File hashes establish integrity of the frozen data. Independent Lean replay of the source evidence remains necessary before scientific claims.

The 588 historical Mathlib c4 goals in DataGoblin PR #4 remain evaluation-only; keep them out of all PPT training, exit fitting and selection datasets. Keep original theorem-family lineage across splits.

~~~sh
python -m ppt_gym train \
  --pack /path/to/checked-pack --plan a \
  --width 48 --heads 4 --steps 64 \
  --max-cases 256 --max-seconds 120 \
  --out runs/ppt-a-checked

python -m ppt_gym train \
  --pack /path/to/checked-pack --plan b \
  --width 48 --heads 4 --steps 64 \
  --max-cases 256 --max-seconds 120 \
  --out runs/ppt-b-checked

python -m ppt_gym compare \
  --pack /path/to/checked-pack \
  --checkpoint runs/ppt-b-checked/checkpoint.pt \
  --support-split calibration --query-split calibration \
  --support 4 --depth 3 --out runs/ppt-b-checked-adapt

python -m ppt_gym sweep \
  --pack /path/to/checked-pack \
  --checkpoint runs/ppt-b-checked/checkpoint.pt \
  --split calibration --depths 1 2 3 5 8 10 \
  --out runs/ppt-b-checked-depth
~~~

For a same-split trial, each family is deterministically divided into disjoint support/query root lineages. Every update consumes the definitive outcome of only the selected support tactic. The online head resets to zero for each family; query labels are read for scoring and never used for online updates. Different support and query splits require nonoverlapping lineage and the same frozen source manifest.

The compare output reports four arms on identical query goals: frozen, dram_fast, retrieval and inverted_feedback. The support evidence and check allocation are shared. The stored Transformer checkpoint stays fixed. Retrieval-only is the direct test of whether local learning adds value beyond using past examples.

The sweep tests recurrence depths and, for Plan B, normal/zero/slot-rolled previous-pass messages. Its reported metric is offline *certified choice from a fixed menu*, using pre-existing receipts. This is different from online Lean proof-search success.

The fit-exit command updates just the continuation head on training examples where the deepest pass ranks a successful tactic after a shallower pass misses. The fitted checkpoint can then execute genuine early stopping using sweep --adaptive-threshold. The exit head's target is a training-only continuation-value proxy: threshold calibration on held-out development examples and measured savings versus the per-pass coda cost are still required before claiming an efficient adaptive policy.

## Persistent DRAM accretion and the DataGoblin handoff

The support/query comparison resets fast weights for each family, which is essential for causal evaluation. The **accrete** command implements a separate *persistent training* lane: it walks unique train lineages in a deterministic order, makes a decision using the current learned head, consumes one already-checked tactic outcome, and updates its FP32 weights in DRAM. The program writes a versioned memory snapshot, a hash-chained-by-state episode ledger, and train-only ALEPH-style row exports.

~~~sh
python -m ppt_gym accrete \
  --pack /path/to/checked-pack \
  --checkpoint runs/ppt-b-checked/checkpoint.pt \
  --max-cases 256 --depth 3 --fast-lr 1.0 --fast-scale 2.0 \
  --out runs/ppt-b-accreted

python -m ppt_gym replay-accretion \
  --pack /path/to/checked-pack \
  --checkpoint runs/ppt-b-checked/checkpoint.pt \
  --max-cases 256 --session runs/ppt-b-accreted \
  --out runs/ppt-b-accretion-audit
~~~

The source requires a train split, full source manifest identity and the checkpoint it was trained from. The replay recomputes every online decision and FP32 weight update, verifies the before/after weight digests and compares all retained JSONL rows. It rejects changed source cohorts, tampered ledgers and altered checkpoints. One tactic outcome is spent per processed root; unknown results remain censored and cannot update the head.

Output files:

- fast_memory.json: FP32 weights and a digest with checkpoint/source identity.
- ledger.jsonl: lineage, selected tactic hash, definitive/censored outcome, support check count and pre/post fast-weight digests.
- outcomes.jsonl: input/target/receipt pairs compatible in spirit with DataGoblin's response-only training corpus, including failed selections.
- sft.jsonl: certified successes selected by the online specialist, suitable for subsequent *deduplicated* DataGoblin admission on train-only roots.
- manifest.json: digest of each artifact, exact source and model hashes, accepted root counts and explicit new_lean_kernel_calls=0.

These exports provide **retrospective training selections over old checked choices**. The accretion runner performs no new Lean calls, and further collection must deduplicate these rows against the original DataGoblin source roots. Synthetic fixture exports have eligible=false, even when the synthetic labels say success. Accreted fast weights should be treated as a versioned specialist artifact, with separately frozen calibration and fresh evaluation before promotion.

An exit-head trial on the fitted Plan B checkpoint can be run as:

~~~sh
python -m ppt_gym sweep \
  --pack /path/to/checked-pack \
  --checkpoint runs/ppt-b-exit-checked/checkpoint.pt \
  --split calibration --depths 1 2 3 5 8 \
  --adaptive-threshold 0.5 \
  --out runs/ppt-b-adaptive-depth
~~~

Each adaptive prediction actually stops its recurrent computation when the learned continuation probability falls below the threshold. The report includes mean executed passes; the full output coda runs on every visited pass, so empirical speed/energy comparison with a fixed-depth policy is still necessary.

## Receipts and experiment hygiene

- Training outputs: checkpoint.pt, train_steps.jsonl and manifest.json, with source manifest hash, architecture parameters, observed step count, exact artifact hashes and evidence class.
- Online comparison: queries.jsonl and manifest.json, including decisions and labels for each query root, four aggregate arms and zero query-derived learning updates.
- Depth sweep and exit fit: separate manifest receipts.
- Run output directories must be fresh, outside Git. The CLI limits requests by maximum goals, optimization steps and wall clock. For a hard memory cap, wrap the entire process tree with a Windows Job Object or a Linux cgroup, as in the existing Lean-CGRT RAM-capped runner.
- Preserve the original Lean/Mathlib/compiler/certifier revision hashes, original source repository version, and full training data ledger when publishing a result.

## Next scientific gates

**PPT-G00:** CPU smoke, deterministic hashes, valid gradients, verified receipt importer and split leakage checks.  
**PPT-G01:** Train on new independently checked proof pools, evaluate unseen theorem-family queries after support feedback, and compare fast weights with retrieval-only.  
**PPT-G02:** Match parameter count and compute; test whether cross-pass messages improve outcomes through causal zero/roll/donor patches.  
**PPT-G03:** Add a live Lean action stepper with accepted-nonterminal, complete, rejected and inconclusive outcomes, and grammar-bounded out-of-pool tactic synthesis.  
**PPT-G04:** Train meta-learning across held-out proof families and implement actual computation-saving early exits. Transfer only successful components into ALEPH-1B.

Require a paired, kernel-certified improvement beyond frozen and retrieval controls with weight-reset ablation before claiming useful DRAM plasticity. Record every negative result.
