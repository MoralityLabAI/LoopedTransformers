# LS-SAE10: preregistered retrospective reproducibility forensics

**Registered 2026-11-17, before any LS-SAE10 model execution.** This is a *diagnostic of LS-SAE08/09 reproducibility*, with model-blind test inputs fixed by previous source registrations. It is not a redo of LS-SAE09, has no scientific success gate, and does not accept any LS-SAE09 previously rejected data. The LS-SAE08 source-locked negative scientific outcome and LS-SAE09 incomplete full-cohort record remain unchanged on main.

## Motivation, frozen evidence and source

- The prior LS-SAE09 implementation was merged with its explicit noncompletion receipt at `mechinterp/sae_causal_transfer/native_transport_v1/RESULT_20261117.md`. Full 16-puzzle native transport failed its exact original-logit replay during the fourth 54-hole case; two independent engineering replays also failed on different signed arms. Absolute maximum elementwise drifts from the historical original reached `0.000331878662109375` and `0.006816864013671875`. A1 relaxed a cross-run 0.0002 to 0.001; no further relaxation is authorized.
- **Frozen source original:** LS-SAE08 completed CI #43851165980, full.json SHA256 `69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9`, artifact #14220373987. Independently replay unchanged LS-SAE08 `audit.audit` first.
- **Frozen model:** original gaoxin492 Sudoku-Extreme step_21700 7M source file SHA256 `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`, `torch==2.7.1` CPU float32, `numpy==2.1.3`, `huggingface_hub==0.36.0`, MATH SDPA, 4 CPU threads unless the designated thread-control variant is explicitly named. No weights or SAE fit may change.
- **Two fixed diagnostic sentinels:** original LS-SAE08 `test_52[0]` (original seed 202628030000, expected easier case), and `test_54[3]` (seed 202628040300, the original LS-SAE09 full-replay failure site). This is an **outcome-informed retrospective selection**, not a held-out prospective scientific test. No puzzle substitutions.
- Use exact original LS-SAE08 `direction_metadata.geometry[arm].direction_512` for STATE_POWER, LOGIT_POWER, SAE, RANDOM, retaining ineligible arms and the native source median blank token. All eligible arms receive both signs of physical doses 0.20, 2.00 and 8.00 and continue to ACT3, ACT6 and ACT12. Only archived complete 81×9 digit logits are used as the historical output reference.

## Fixed within-run counterfactuals and output

For each sentinel, extract the original ACT1 `z_H,z_L,x_emb` native anchor from the **complete original 8-puzzle family** using `stage_one(batch_size=2)` as LS-SAE08 did. Independently extract the same puzzle from the same family with `stage_one(batch_size=1)`; log ACT1 `z_H,z_L,x_emb` hashes and maximum elementwise differences between the two source-extraction batch sizes.

Evaluate four named full-grid continuation conditions in this order:

1. `canonical_a`: invoke the exact historical LS-SAE08 `native_hard_task_v1.run_native.native_all_outputs` function with the archived vectors and paired-source anchor.
2. `canonical_b`: invoke the *same* function again with exactly the same tensors, batching and ordering. This directly tests fresh-fresh same-process reproducibility.
3. `instrumented`: perform the same ACT1 edit and full 25-variant-style batched ACT continuation while additionally collecting full zH/zL clean and edited ACT3/6/12 norms and SHA256 receipts. Instrumentation must leave trajectory values untouched.
4. `single_source`: invoke the original canonical function with the independently batch_size=1-extracted ACT1 anchor for that puzzle, leaving the continuation batch arrangement unchanged.

Record all real 81×9 digit logits per condition/horizon/arm/sign/dose, for an independent stdlib audit. Compute and save, using the *original frozen LS-SAE08 reference* as well as pairwise fresh outputs, per-condition `max |logit-original|`, number of all-81 predicted-digit argmax mismatches, missing-cell flip count, and centered output RMS deviation. The original archived logits may be read for diagnostics; they cannot select new vector signs, doses or sources. Report each comparison separately at ACT3/6/12 and retain all failures. Do not abort because a numerical drift exceeds any old threshold; drifting outputs are the research subject. Abort on original evidence hash mismatch, source roster mismatch, nonfinite logits, omitted variants, or violations of the experiment's frozen source/model/dose contract.

## Interpretation and limits

- If `canonical_a` differs from `canonical_b` given identical ACT1 tensors, investigate CPU numerical nondeterminism or hidden nondeterministic state.
- If the canonical repeats agree but `instrumented` differs, isolate the instrumentation/operation-order effect.
- If repeated canonical and instrumented agree but differ from archived LS-SAE08 logits, investigate cross-run numerical execution or checkpoint/source extraction differences.
- If `single_source` differs from paired-source output, quantify source extraction batch-size sensitivity.

The comparison identifies **observed numerical sensitivity**, not necessarily its unique root cause. All findings are descriptive on two retrospectively selected sentinels. An independent Python-stdlib auditor recomputes from the saved real logits and archived 81×9 reference the exact comparison statistics, source identity, expected variant coverage and original Sudoku prediction effects. It cannot independently execute the model or establish bitwise floating-point determinism across machines. CI includes CPU toy tests, fixed two-sentinel checkpoint run, and independent replay audit. This new diagnostic should remain a standalone PR until its own CI and results are inspected.
