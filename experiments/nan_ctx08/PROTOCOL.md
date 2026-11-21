# NAN-CTX-08: longer contexts with GPU-resident BF16 readout

Freeze and push source, fixtures, config, dependency hashes and engineering
evidence before this condition's first pretrained inference. This is a new
execution condition; no previous frozen worker or result is modified.
The user requests heavier context evaluations and reports a concurrent
CPU-hungry math process. Prioritize GPU execution and reduce host traffic.

The pinned checkpoint index contains 8,339,601,408 bytes (about 7.77 GiB).
Its BF16 embedding and vocabulary head each occupy 1,020,788,736 bytes.
Full BF16 GPU residency cannot fit a 4-GiB device. In this condition NF4 keeps
all 22 decoder layers and the unchanged BF16 head on CUDA; BF16 keeps four
decoder layers, one reusable staging layer and the unchanged BF16 head there.
NF4 retains only embedding lookup on CPU. The existing scorer/GEMM shapes and
full-vocabulary normalization remain, with GPU-to-GPU head chunk copies.
No head quantization, KV cache, token candidate restriction or model download.

## Source design and endpoints

CAL seed 2026112008 is a separate capacity-profile root. DEV seeds
2026112308..2026112311 provide four independent source-generator root families.
Generate directed cyclic key ledgers with labels A..P. For each root and
1024/2048/4096-token budget, choose the largest record count fitting ALL
registered variants, without model observations. Actual lengths must be within
96 tokens below the budget. All extra content consists of usable ledger facts.

Each DEV root/budget has early/middle/late query-record placement, direct label
lookup or a two-record next-key/label join, worlds A/B differing only in the
relevant label, and main/literal-copy variants: 24 contexts per root/budget,
288 DEV contexts/backend. CAL has one middle/lookup/world-A main at each length:
291 total. Query/destination placements and all labels are algorithmically fixed;
no model-result selection, source repair or held-out exclusion use. Independent
checks solve every source and verify one changed ledger fact per world pair.

Read out one FREE first token after native loops 1 and 2. The anchored finite
label grammar accepts a complete label (and its already registered wrappers),
never scans prose, forces candidates or repairs answers. Primary readout is loop
2; loop 1 is secondary. This differs from complete-final-reply Prime and native
prefix-stop generation. Report all cases, main/copy separately and worlds as
repeated measures. Count four DEV roots, not 288 independent trials. Wrong CAL
answers never block DEV. No task competence gate, BPD donor transfer or causal
confirmation is enabled here; original algebraic/other study gates remain as
recorded, including NAN-FIT-07's backend-specific clean screen.

## Memory, parity and scheduling

One large model at a time, after actual fitting supervisor AND model processes
exit. NF4 precedes BF16. Use the pinned environment, BF16 compute, B1, native
two-loop no-cache prefill and 8192-row vocabulary chunks. Limit the CUDA
allocator to .85, CPU threads to one and process priority to Below Normal.
The concurrent math workload makes observed latency a shared-machine measure,
not an isolated throughput benchmark.

On the first CAL prefix, check every vocabulary logit and every scorer field
for exact CPU-head/resident-head agreement at both loops before publishing any
case. Preserve that receipt. The small random 22-layer/width-64 engineering
fixture checks increased BF16 decoder residency, normalized boundaries, all 40
physical observations and full-head parity; it is not pretrained evidence.

At each increasing length run the CAL capacity profile, then every DEV variant
of that length. A CUDA OOM stops the backend at that level and all higher levels;
preserve attempted results and report missing coverage. Do not retry at a lower
precision or select easier cases. A 7200-second backend budget is checked between
contexts; a finite in-flight native prefill completes. Preserve any failed or
partial condition. No silent retry or rewriting its receipt as complete.

## Geometry, provenance and inference limits

Capture last-prefix-token residual_after_layer and down_proj/o_proj input/output
at layers 5/11/17/21 and both loops, with tokens, scores, byte hashes and physical
dimensions. Residuals precede final loop RMSNorm. Down_proj output has the
immediate identity-addition bridge; o_proj requires its intervening MLP bridge.

Subsequent CPU probes use ALL available fixed CAL-only SAE dictionaries from
NAN-GYM-05, including initial controls. Report every feature, group and loop in
physical coordinates. Do not subtract BF16/NF4 feature-index codes. If NAN-FIT-07
produces actual final VPD/BPD checkpoints, include both T=1/T=2 in the same backend
with source/receipt/checkpoint bindings; if a gate/fit fails, report those learned
assets as missing. Never substitute initial SVD as a trained result. Frozen
factors/controller may be applied to cached clean inputs as an OOD observational
probe; this is not reached-state intervention or donor transfer. T=1 at loop 2
is outside fit depth. No dictionary, factor or controller fitting on these DEV
contexts; no feature/rank selection from their observed effects.

Experimental unit: independent source-generator root family, n_DEV=4, n_CAL=1,
one checkpoint; lengths, worlds, positions, tasks, loops and backends are repeated
measures. Formal inferential comparisons=0. No matched causal-dose/sham effect
floor or powered sealed mechanism readout is calibrated. No causal/equivalence
claim. The original ARG-03 168-case test is never read, regenerated, used as an
exclusion set or inferred on. Publish aggregates/source hashes only; raw replies,
tokens, state vectors and per-record derivatives remain local.
