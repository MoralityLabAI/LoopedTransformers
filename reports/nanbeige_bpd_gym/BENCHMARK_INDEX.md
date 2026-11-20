# Nanbeige benchmark evidence index

This page indexes committed aggregate reports and their source seals. It is not a fresh grading pass or a live-process claim. The linked reports retain complete groups, controls, failures and missing coverage.

Evidence source commit: `3cf9a1eedc6e6230c286fabef755a8998d4c8cbe`. Signature: `954608bd17cefc84a09be95806fc70ec73c1e5927445fba56dd2d86df9aca642`.

| Study / endpoint | Backend or fit | Published status | Evidence |
|---|---|---|---|
| [GYM05](NAN_GYM05_PROGRESS.md): native bounded-prefix-stop wider suite | bf16 | completed | main 31/105; copy 57/57; 322 contexts; 89 root groups; 4 SAE layers |
| [GYM05](NAN_GYM05_PROGRESS.md): native bounded-prefix-stop wider suite | nf4 | completed | main 26/105; copy 57/57; 322 contexts; 89 root groups; 4 SAE layers |
| [PRIME06](NAN_PRIME06_PROGRESS.md): official SDK complete-final-reply wider suite | nf4 | completed | main 22/105; copy 55/57; 162 graded SDK episodes; 0 missing grades |
| [PRIME06](NAN_PRIME06_PROGRESS.md): official SDK complete-final-reply wider suite | bf16 | stopped_with_incomplete_coverage | main 13/70; copy 11/22; 92 graded SDK episodes; 70 missing grades |
| [FIT07](NAN_FIT07_PROGRESS.md): native free-first-token screen and eight-step fitting | nf4 | competence_gate_failed | T1/T2 unrun: fixed fitting gate closed |
| [FIT07](NAN_FIT07_PROGRESS.md): native free-first-token screen and eight-step fitting | bf16 | completed_with_failed_conditions | T1 passed (8 steps); T2 failed (0 steps) |
| [CTX08](NAN_CTX08_PROGRESS.md): native free-first-token 1K/2K/4K contexts | nf4 | completed | main 81/144; copy 120/144; 291 closed contexts |
| [CTX08](NAN_CTX08_PROGRESS.md): native free-first-token 1K/2K/4K contexts | bf16 | completed | main 90/144; copy 141/144; 291 closed contexts |
| [RECOMP09](NAN_RECOMP09_PROGRESS.md): exposed-source BF16 T2 learned BPD execution amendment | bf16 | completed | 8 optimizer steps |
| [PRIME-CTX10](NAN_PRIME_CTX10_PROGRESS.md): official SDK complete one-token final reply, context taskset | nf4 | terminal_with_missing_coverage | main 14/25; copy 20/25; 50 graded SDK episodes; 238 missing grades |
| [PRIME-CTX10](NAN_PRIME_CTX10_PROGRESS.md): official SDK complete one-token final reply, context taskset | bf16 | completed | main 90/144; copy 141/144; 288 graded SDK episodes; 0 missing grades |
| [KV11](NAN_KV11_PROGRESS.md): same-reference-history cache/full-prefix numerical diagnostic | nf4 | completed | 130 closed contexts; 130 exact prefills; 3 loop-2 argmax disagreements |
| [KV11](NAN_KV11_PROGRESS.md): same-reference-history cache/full-prefix numerical diagnostic | bf16 | completed | 130 closed contexts; 130 exact prefills; 12 loop-2 argmax disagreements |
| [KV12](../../experiments/nan_kv12/SOURCE_MANIFEST.json): explicit-effective NF4 arithmetic/cache numerical diagnostic | registered | registered_no_published_execution_report | No published execution score |
| [RANKSITE13](NAN_RANKSITE13_PROGRESS.md): six exposed-source BF16 learned rank/site cells and local finite-write probes | t1_layer21_down_proj_rank16 | queued | 0 optimizer steps |
| [RANKSITE13](NAN_RANKSITE13_PROGRESS.md): six exposed-source BF16 learned rank/site cells and local finite-write probes | t2_layer21_down_proj_rank16 | queued | 0 optimizer steps |
| [RANKSITE13](NAN_RANKSITE13_PROGRESS.md): six exposed-source BF16 learned rank/site cells and local finite-write probes | t1_layer21_o_proj_rank8 | queued | 0 optimizer steps |
| [RANKSITE13](NAN_RANKSITE13_PROGRESS.md): six exposed-source BF16 learned rank/site cells and local finite-write probes | t2_layer21_o_proj_rank8 | queued | 0 optimizer steps |
| [RANKSITE13](NAN_RANKSITE13_PROGRESS.md): six exposed-source BF16 learned rank/site cells and local finite-write probes | t1_layer21_o_proj_rank16 | queued | 0 optimizer steps |
| [RANKSITE13](NAN_RANKSITE13_PROGRESS.md): six exposed-source BF16 learned rank/site cells and local finite-write probes | t2_layer21_o_proj_rank16 | queued | 0 optimizer steps |

Accuracy columns use each endpoint's own scored denominator; partial SDK coverage remains partial. Successful SDK episodes denote valid recorded grades, not correct model answers. CTX08 accuracy is DEV loop-2 free-first-token; its 288 correlated DEV contexts arise from four source families. GYM05 uses native prefix stopping; Prime uses complete final replies. PRIME-CTX10 reuses exposed CTX08 DEV sources; agreement adds transport/endpoint evidence, not independent source roots. Cache fragments receive no complete-task accuracy score, and cache argmaxes use the full-prefix reference history rather than an independent reply after divergence.

BF16/NF4 stay distinct; explicit-effective NF4 changes full-prefix arithmetic as well as incremental execution. Both original cache backends have measured token-choice disagreements despite exact prefills. The original failed NF4 fitting screen, BF16 saved-tensor failure and incomplete Prime attempts remain preserved. There is no automatic retry, orphan-response rescue or reinterpretation of missing grades as wrong answers.

Eight CAL-only SAE dictionaries and actual learned BF16 VPD/BPD assets are separate from weight-SVD/random controls. Static CAL-mean routes retaining the same small-DEV correctness limit behavioral specificity. Rank/site finite-write probes include the real o_proj residual/norm/MLP path; its derivative is not assumed equivalent to rounded finite writes. No geometry/correlation or random-model engineering result establishes donor-specific mediation.

All studies are exposed development work on one checkpoint. Roots/families, cases, loops, tokens and features are not interchangeable sample sizes; no accuracy is pooled across studies and no formal confirmation/equivalence is claimed. Original algebraic gates remain failed. The original 168-case test stays sealed.

Resource policy remains one large model, one numerical CPU thread and Below Normal priority, with the separate math process untouched. NF4 decoder layers/head are GPU resident; embedding lookup stays on CPU to retain context/cache capacity. BF16 streams decoder layers. [Dedicated/shared device counters](NAN_PRIME_CTX10_BF16_MEMORY.md) remain separate from CUDA allocator peaks.

[Exact aggregate source hashes and fields](benchmark_index.json) · [Requirements and remaining work](ROADMAP.md) · [Original gated causal campaign](../../papers/shared_weights_dark_fibers/NAN_BPD01_ADDENDUM.md)
