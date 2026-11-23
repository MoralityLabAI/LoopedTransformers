# LSA-02E: Typed Lean premise structure plus bounded public pretty text

**Registration:** November 23, 2026. **Scope:** CPU-only development pilot before any broader Lean gym. This experiment follows [LSA-02C's typed-state negative result](../reports/lsa02c_typed_public_state_cpu_20261123.md) and [LSA-02D's same-worker repeatability and field audit](../reports/lsa02d_same_worker_field_retention_20261123.md). The old development roots and all ordinary/OOD sealed splits are excluded.

## Motivating evidence

On 259 previous training frames, the existing raw JSON input had mean length 2,827 bytes and the model read only its final 384. That tail included a complete pretty-printed public goal in **99/259** frames. The original compact typed encoder kept current goal, budget and all candidate premise types, but preserved **zero** pretty-printed goal fields. It produced no native proof-completion gain on 32 developmental Lean roots, and a corrected CI repeat favored raw. The information-retention audit establishes a concrete public-feature difference for a new representation test.

LSA-02D also established byte-for-byte same-host CPU reruns on two separate GitHub workers, with **different final trained tensors across those workers**. The new matched source/seed experiments therefore train each competing view on **one physical CPU worker**, duplicate their fits on that worker, and treat the native evaluation as exploratory. Future cross-worker numerical sensitivity remains a separate engineering gate.

## Registered experimental intervention

The new view name is `typed_pp`, with schema `lsa02e.typed_pretty_public_state.v1`. It starts with the **same existing LSA-02C typed grammar** for budget, current and pending goal types, failed tactic choices, constructor flag, and all publicly declared candidate premise types. Then it appends the final **at most 96 valid UTF-8 bytes** of the first goal's Lean public `pp` field under a `P:` marker. Whitespace is normalized; it favors the end of the pretty-printed text, which typically contains the displayed current goal and local context. Remaining bytes go to background public hypothesis rules ordered by their dependency distance to the active goal. No part of an output exceeds the actual **384-byte ByteEncoder budget**.

The candidate types and budget are mandatory. If the source packet's public goal and candidate-premise entries alone cannot fit, the encoder fails closed before scoring. A limited pretty excerpt and some background type relations can be omitted when space is exhausted. Every output derives solely from `state_input` public Lean goal, type declarations and candidate action strings. It never reads proof outcomes, reference solution lengths, generator `gen` metadata, private oracle labels, or held-out data.

Two existing views are controls: `raw`, which retains the exact canonical JSON and downstream last-384-byte truncation, and `typed`, which uses its original compact type-only encoding. The intervention is therefore **adding a bounded pretty-print channel and reallocating available bytes from background types**. The grammar for candidate premise entries, physical Plan B architecture, and readout stay unchanged.

## Fresh data and matched computation

The new theorem suite is frozen at seed `20261126`, 64 roots per split. Only the first **48 independent training root lineages** enter the native teacher, which uses Lean 4.19 `lgym` to extract shortest-continuation labels and independently compile each reference proof. The first **32 calibration lineages** from the separate development shard are used for scoring. The ordinary/OOD shards are loaded only for frozen file/hash and lineage checks, and their proof labels remain unopened.

| Dimension | Frozen pilot |
| --- | --- |
| Model | Plan B, width 32, 57,475 parameters, 4 heads, byte encoder 384 |
| Recurrence | AB order, 2 recurrent passes |
| Public input views | `raw`, `typed`, `typed_pp` |
| Seeds | 113 and 127 |
| Training | 720 FP32 AdamW updates, LR 0.001, gradient clipping 1, identical theorem/frame sample stream |
| Host | One GitHub Actions CPU runner for the entire 6-cell fit matrix |
| Repeatability | Each seed/view trained **twice sequentially**; 12 fits total |
| Determinism | one Torch thread, `torch.use_deterministic_algorithms(True)`, disabled MKLDNN and pinned Torch 2.5.1+cpu |
| Calibration | 32 independent theorem roots, fixed eight tactic checks, same roots and Lean toolchain in all cells |
| Proof authority | Native Lean 4.19 kernel, standalone axiom-free compilation, model-free action replay, independent shortest-proof attainability |
| CI concurrency | At most two simultaneous calibration jobs, no GPU or paid inference |
| Research status | Development-only and pre-gym; no architecture promotion |

The six model checkpoints have the same model parameter count. Within each seed, initialization and theorem/frame optimizer exposure must match across all public views exactly; the runner audits both. Each within-worker duplicate fit must match parameter tensors, saved checkpoint bytes and optimizer action/loss traces. The matched experiment fails if any prerequisite is absent or a run reaches its compute cap before all 720 updates. All checkpoints and manifests are archived in Actions artifacts.

## Endpoints and decision gate

The **primary endpoint** is the native kernel-and-standalone-compiled **full-proof closure fraction under the trained AB schedule** on 32 independent calibration theorem roots, comparing `typed_pp` against `raw`. The **key secondary comparator** is `typed_pp` against `typed`; both share the candidate-premise grammar. Hold the reference-reachable ceiling and public-order `auto8` baseline alongside each native closure count.

The descriptive effect for each root is averaged over the two matched seed differences, then clustered at the unique theorem lineage for the root bootstrap interval. The same roots at different seeds are paired measurements rather than independent trials. Other secondary measurements include strict **training-only** discriminating-fork accuracy, uniform-menu in-sample baseline, exact public byte lengths, candidate premise survival, percentage of public pretty fields providing an excerpt, dropped background facts, and 720-step training-loss receipts.

**Predeclared development screen:** `typed_pp` must exceed *both* `raw` and `typed` by at least **+0.05 absolute native closure fraction**, with `typed_pp` at least tying raw in both seeds. If the screen fails, record the result and retain all roots. This is a development screen, not a significance claim. Even a successful screen would require a new source generator/holdout confirmation and calibrated FSI-01 sham before any architecture promotion.

The established formal competence gate separately requires at least 32 eligible independent calibration roots, trained-order closure ≥0.40, and ≥0.10 over a registered independent **uniform-menu native proof baseline**. The present protocol has a public-order baseline and independent proof attainability, not the registered random-menu native baseline, so it **cannot certify full competence**. A negative result or any failed native proof check keeps the architecture at `BLOCKED_INCOMPETENT_CHECKPOINT`.

## Custody and resource limits

The [dedicated workflow](../.github/workflows/lsa02e-typed-pretty-cpu.yml) runs source/unit gates, native Lean teacher creation, **one-worker same-seed three-view training**, six native proof/replay jobs with two-way concurrency, and one root-paired source-custodied audit. Every trained model stores its input-view schema identifier. A typed+pretty inference run rejects any checkpoint whose schema does not match the declared view.

The aggregate output verifies all source SHA-256 manifests and checkpoint files, independent theorem lineages, identical root rosters, full native reference certificates, completed replay receipts, zero evaluation-root training and the one-worker repeatability manifest. A failed control is reported explicitly. Training and native evidence stay in the short-lived Actions archive, while a text research report is committed after verification.

This run uses no local GPU or cloud model inference, and introduces no gym interface until native proof competency improves.
