# LSA-02D: deterministic same-worker CPU fits and public-field retention

**Date:** 2026-11-23. **Disposition:** engineering reproducibility **PASS** and field-retention audit **PASS**; scientific Plan B competence remains **BLOCKED_INCOMPETENT_CHECKPOINT**. This is a new, read-only analysis of the already verified training source and a same-worker fit duplicate check. It contains no native Lean re-evaluation and no sealed-root exposure.

- [Protocol and predeclared gates](../docs/lsa02d_same_worker_field_audit_protocol.md)
- [Source field auditor](../research/lean_cgrt_hrm400/ppt_gym/field_audit.py)
- [Deterministic CPU fit runner](../research/lean_cgrt_hrm400/ppt_gym/determinism_audit.py)
- [Successful GitHub Actions run 45179473540](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45179473540)
- [Run artifact 14820676296](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45179473540/artifacts/14820676296), containing all four duplicated checkpoint pairs, optimizer traces, per-frame field census, and environment receipts

The source is the original **48-lineage, 259-frame** native Lean shortest-continuation training teacher from LSA-02C. Its manifest SHA-256 is `a049c847c78a3278fb7599c20b59800ed72c7462b308bbb0ddffd3c6d3cde597`. The independent Lean 4.19 compiled proof attestations and source hashes were checked by the existing loader before either analysis. No proof labels were used to choose public evidence fields.

## 1. Same-worker CPU reproducibility

The single GitHub-hosted CPU job fitted seed 107 and seed 109, for raw and typed public-state inputs. For **each** seed/view pair it trained two fresh initialized copies sequentially with identical source sampling and **720 FP32 AdamW steps**. All eight fits used the existing **57,475-parameter Plan B**, width 32, 384-byte ByteEncoder, depth 2, and AB block schedule.

Execution controlled `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`, `torch.use_deterministic_algorithms(True)` and `torch.backends.mkldnn.enabled=False`. The host ran CPU PyTorch `2.5.1+cpu` with one intra-op thread. The artifact captures Python, platform and Torch-build fingerprints.

| Fit cell | Repeated checkpoint file bytes | Model tensor bitwise equality | Exact optimizer trace | Matched starting weights within seed |
| --- | --- | --- | --- | --- |
| 107 raw | PASS | PASS, all 74 state keys | PASS | PASS |
| 107 typed | PASS | PASS, all 74 state keys | PASS | PASS |
| 109 raw | PASS | PASS, all 74 state keys | PASS | PASS |
| 109 typed | PASS | PASS, all 74 state keys | PASS | PASS |

**All four duplicate tests passed.** The largest observed tensor displacement between duplicate fits was zero, and no optimizer trace hashes differed. The starting-state hash for seed 107 was `79c3ed76…`, common to raw and typed; seed 109 was `08491213…`, also common to both views. Exact SHA-256 files and model receipts are archived in the run artifact.

This establishes **same-host reproducibility for this pinned CPU configuration**. Earlier LSA-02C raw/typed fits differed when repeated on different Actions workers under the older two-thread/MKLDNN-enabled setup. The evidence supports changing that execution contract for future tests; it does not identify which host/kernel variation caused the earlier discrepancies, and it does not establish bitwise reproduction across different machines.

## 2. Exact public-state information reaching the ByteEncoder

The raw representation is the original canonical JSON packet. `ByteEncoder.forward` discards all but its **last 384 UTF-8 bytes**. The auditor traverses the exact canonical JSON byte sequence to locate complete nested-field ranges; it counts a raw field only when that **entire range** survives the actual tail crop. For typed, it counts the explicit goal/budget header or complete named hypothesis entries present in the typed byte string. These are representation-specific field-visibility tests, not equal-information encoding proofs.

| Public source information | Whole field survives raw tail | Present in typed view |
| --- | ---: | ---: |
| Current goal type | **175/259 frames** | **259/259** |
| Remaining tactic-check budget | **0/259** | **259/259** |
| Pretty-printed Lean goal text, complete | **99/259** | **0/259** |
| Candidate hypothesis type relation | **11/421** | **421/421** |
| Noncandidate background hypothesis type | **88/4,267** | **4,243/4,267** |
| Pending additional goal type | **84/105** | **105/105** |

Average raw source packet length was **2,826.86 bytes**, while the raw model sees exactly **384 bytes** per training frame. The typed input was **189.45 bytes on average** and stayed under the 384-byte budget throughout the 259 frames.

These counts separate three distinct facts:

1. The raw tail can truncate the structured proof goal: 84 frames lost at least part of the canonical current-target JSON field. It also omitted the full check-budget field in every training frame.
2. The compact typed view preserved every candidate's complete premise/head type relation in this training corpus and almost all background type relations, missing 24 of 4,267 background declarations.
3. The raw byte window kept the **complete Lean pretty-printed goal text in 99 frames**. The registered typed serializer omits `pp` entirely. This is an important uncontrolled *semantic-content* difference despite matching the byte budget and physical model.

The field auditor never consults native action success, chosen proof or shortest-path oracle to decide which public fields to keep. It is a **training-source structural audit** with 48 independent theorem lineages and repeated correlated frames. Root-weighted fractions and every per-frame boolean are archived separately.

## 3. Interpretation and next experimental gate

The [earlier LSA-02C native comparison](lsa02c_typed_public_state_cpu_20261123.md) failed to improve whole-proof completion with this compact typed view. Its raw control was severely truncated, while the typed representation preserved many more explicit type dependencies. The present audit documents a concrete reason the comparison cannot be reduced to a question about input length alone: the representations carry different **kinds of public context**, including raw pretty-printed Lean state syntax.

A future **prospectively registered** CPU comparison can add an exactly budgeted typed-plus-`pp` representation and a raw last-384-byte control. It should use the same-worker reproducibility contract, independently frozen train/calibration lineages, an exact public-memory budget, and native full-proof certification. An intermediate smoke should first confirm that the hybrid retains candidate premise/head structure and that extra text never includes private labels or proof results.

No new gym or larger architecture is warranted solely by these data. The previous 32-root competence threshold, registered uniform-menu baseline, and causal improvement gates remain outstanding.

## 4. CI and archiving

All following checks were green at the original LSA-02D PR head:

- Dedicated **same-worker deterministic CPU + exact field audit**.
- Existing PPT Gym CPU regression.
- Existing PPT-G03 native Lean protocol and verification.

The dedicated job uploaded **1,898,069 bytes** of source-field and checkpoint receipts under artifact ID `14820676296`. It has a 14-day retention period. The present Markdown report preserves the methods, denominators, and conclusions after those ephemeral archives expire.
