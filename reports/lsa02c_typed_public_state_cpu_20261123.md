# LSA-02C: matched CPU raw versus typed public-state results

**Date:** 2026-11-23. **Status:** completed exploratory CPU study, **registered development threshold missed**; formal competence remains **BLOCKED_INCOMPETENT_CHECKPOINT**. Implementation: [PR #324](https://github.com/MoralityLabAI/LoopedTransformers/pull/324) (squash merge `cff86d2103e1c1e698af3223ac54807b7c27ee86`). The two verified GitHub Actions campaign receipts are [first run 45171217881](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45171217881) and [compatibility-corrected run 45172467970](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45172467970). Artifacts contain model checkpoints, native Lean transcripts, independent proof certificates, replay receipts and root-level paired effects. Artifact retention is 14 days.

## Fixed source and training design

A new Lean-CGRT theorem suite was frozen under generator seed `20261124`, with disjoint train, calibration, ordinary test and OOD shards. The native Lean 4.19 `lgym` server certified **48 distinct training theorem roots** and extracted **259 source-labeled shortest-continuation frames**. All 48 shortest reference proofs independently compiled with the axiom gate; **43/48** were attainable within an eight-tactic budget. Models used 720 FP32 AdamW steps under the common source-root sampling schedule; the input was the only registered architectural difference. Both inputs used the same Plan B AB block schedule, tied parameter identities, **57,475 trainable parameters**, and a 384-byte `ByteEncoder` ceiling.

The new `lsa02c.typed_public_state.v1` encoder extracts only publicly available goal types, hypothesis premise/head structure, candidate menu, failure history and remaining budget. It allocates its byte budget by public dependency priority. It does not read proof outcomes or trainer labels. Source honesty checks found:

| Training packet property | Historical raw JSON | Typed view |
| --- | ---: | ---: |
| Average public packet size | 2,826.86 bytes | 189.45 bytes |
| Packet lengths exceeding 384-byte model ceiling | 259/259 | 0/259 |
| Conflicting-label input collisions on training frames | 0 | 0 |

Both seed-107 variants began with initial weight fingerprint `79c3ed76`; both seed-109 variants began with `08491213`. The actual full fingerprints are retained in the fitted manifests. `native_order` verified each checkpoint's typed-view schema and source hashes before evaluating the **same 32 independent new calibration roots**. Every model proof was checked by the actual Lean kernel and independently compiled; all 96 per-checkpoint action episodes were independently replayed without the model. The reference proof ceiling was **29/32** in every cell. Ordinary and OOD locked evaluation remain unopened.

## First completed CI campaign

Run [45171217881](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45171217881), code head `1f534c5ec5ceb831671c9f9961267ae4fffd13c9`:

| Seed | Raw native completed | Typed native completed | Auto8 public-order baseline | Lean reference attainable |
| --- | ---: | ---: | ---: | ---: |
| 107 | 7/32 | 4/32 | 5/32 | 29/32 |
| 109 | 5/32 | 7/32 | 5/32 | 29/32 |

The unique-theorem-root paired typed-minus-raw average across two seed-matched comparisons was **−1.5625 percentage points**. A descriptive root-cluster bootstrap 95% interval was **[−14.0625, +9.375] points**. The 32 theorem roots are reused across both seeds, so the study has 32 unique roots, not 64 independent samples.

On the original 102 discriminating *training* frames across 46 roots, the independent-root-weighted uniform-menu shortest-action baseline was **38.54%**. The raw/typed training-fork accuracies were 44.53%/39.71% (seed 107) and 52.10%/46.16% (seed 109). Those are in-sample diagnostics, not native proof certificates.

## Repeat after restoring old 32-byte fixture compatibility

The original PR also triggered older PPT workflows that use a `max_bytes=32` fixture. They initially failed because the new packet dispatcher unnecessarily required a whitelist of max byte lengths. The representation dispatcher was corrected to accept the existing supported 32–384-byte range; an error-message regression was also fixed. These corrections do **not** alter the registered model's 384-byte view logic. The latest PPT CPU, PPT-G03 native Lean and older native AB/BA smoke workflows passed.

The corrected branch re-ran the entire matched experiment: [45172467970](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45172467970). Its same-root native proof counts were:

| Seed | Raw native completed | Typed native completed | Auto8 | Attainable |
| --- | ---: | ---: | ---: | ---: |
| 107 | 7/32 | 4/32 | 5/32 | 29/32 |
| 109 | 7/32 | 4/32 | 5/32 | 29/32 |

The typed-minus-raw effect was **−9.375 percentage points**, with descriptive root-bootstrap interval **[−20.3125, 0] points**. The repeat also passed the independent 48-source-proof training gate, native proof kernel checks, source hashes, 32-root pairing, and all replay checks.

The two campaigns used **identical source-manifest hash `a049c847c78a3278fb7599c20b59800ed72c7462b308bbb0ddffd3c6d3cde597`**, identical initial weights within each seed, and the same nominal 720 training updates. Some end-of-training model tensors differed on replay, with measurable differences in final loss and seed-109 closure count. Exact CPU fit reproducibility across different GitHub runners was **not** established. Causes such as host CPU kernel variation or thread scheduling remain to be isolated rather than assumed. The two repeat campaigns share the same theorem roots and must not be pooled as independent tests.

The second-run in-sample fork accuracies were raw 43.26%, typed 39.71% at seed 107, and raw 45.87%, typed 49.13% at seed 109. Both were development measurements on exposed source roots.

## Decision and engineering boundary

Both campaigns missed the prespecified **+0.05 absolute typed-minus-raw native closure** development improvement screen. The strongest checkpoint closed 7/32 = 21.875% of calibration theorems, short of the previously registered 40% competence screen. This experiment did not construct a calibrated uniform-menu native performance baseline, so the full competence gate remains unmet independently of the closure rate.

This is evidence of an **input-budget bottleneck** in the historical raw JSON channel and a **valid negative comparison for the particular compact typed serializer, model and training recipe**. It does not establish that typed graph representations are intrinsically worse. The raw tail retains public pretty-printed proof-state context, which the compact encoder currently omits. Both policies may also suffer from limited model capacity, sparse branch supervision, or poor transfer from reference-path states to model-visited states.

**Before a new gym or model scale-up:** pin each matched view pair to the same physical CPU worker, use one PyTorch thread with deterministic algorithm settings, record CPU/Torch environment, and verify a duplicate source-only fit reproduces exact weights. Follow with a read-only audit comparing which public premise relations survived the historical raw tail versus typed encoding. A subsequent prospective fresh-root comparison can test additional representations, such as compact typed facts plus bounded Lean pretty-printed local context, on identical byte and compute budgets.

## Reproduction and evidence

- [LSA-02C registered protocol](../docs/lsa02c_public_state_cpu_protocol.md)
- [Public packet encoder](../research/lean_cgrt_hrm400/ppt_gym/typed_state.py)
- [Root-paired audit](../research/lean_cgrt_hrm400/ppt_gym/typed_compare.py)
- [Fresh CPU/Lean GitHub Actions workflow](../.github/workflows/lsa02c-typed-state-cpu.yml)
- [First-run receipts](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45171217881)
- [Corrected-run receipts](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45172467970)

Neither campaign opened any locked `ordinary_test` or `ood_family` answers. **No architecture promotion** is claimed.
