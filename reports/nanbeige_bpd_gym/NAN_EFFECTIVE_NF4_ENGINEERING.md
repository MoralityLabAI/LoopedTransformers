# Explicit-effective NF4: random cache engineering

The new explicit BF16 effective-weight condition passed the declared random
22-layer cache fixture. Every within-condition prefill and every tested
incremental boundary/logit matched byte for byte. This is a small random model,
not a real-checkpoint equivalence or competence result.

| Random policy | Max boundary relative L2 | Max vocabulary logit difference | Incremental argmax agreements |
|---|---:|---:|---:|
| BF16 | 0 | 0 | 16/16 |
| Default NF4 | .01507268 | .01220703125 | 16/16 |
| Explicit NF4 dequantization to BF16 | 0 | 0 | 16/16 |

Both NF4 models received identical packed parameters. Mounting the new modules
and inference preserved parameter bytes/names. All three arms, all 18 step/loop
readouts per arm and all 360 physical site vectors per arm are retained in the
[complete receipt](../../mechinterp/nanbeige_bpd_gym/receipts/effective_nf4_random_v1.json).
The fixture used width 64, MLP 128, vocabulary 166144, two native passes, a
19-token prefix and eight predetermined appended tokens, with no outcome-based
history selection. Peak CUDA allocated was 31,986,176 bytes; runtime 47.25 s.

The policy also changes full-prefix arithmetic: versus default NF4, maximum
full-prefix boundary relative L2 was .01626319 and maximum logit difference
.013671875. Installed bitsandbytes 0.50.2 selects fused kernels based on matrix
shape/row count; duplicating a row would not ensure the long-prefix path.
Explicit dequantization fixes the effective BF16 operator for all row counts,
while GEMM/attention shape-dependent rounding may still differ on real weights.
Only the current matrix is materialized, never the whole BF16 decoder.

This was registered and pushed at `65257dc` before execution; the
[plan](../../mechinterp/nanbeige_bpd_gym/EFFECTIVE_NF4_ENGINEERING_PLAN.md)
and thresholds were unchanged. CPU contract tests passed. A first local
historical-comparison test encountered the sandbox temporary-directory access
restriction; rerunning unchanged tests with a writable workspace temporary
directory passed all four KV12 contracts. This was not neural inference.

The separate [KV12 real-checkpoint protocol](../../experiments/nan_kv12/PROTOCOL.md)
is frozen before its inference. It covers every public GYM05 DEV context
(130 contexts/41 source families), compares its cache with its own full-prefix
policy, and compares common initial histories with both original KV11 backends.
The real-checkpoint worker waits for actual KV11 termination. All prior outcomes
and failures remain unchanged; no replay, donor transfer or full-task accuracy
claim. One large model at a time, one numerical CPU thread, Below Normal, fixed
CUDA cap; the user's separate math process is untouched.

Unit: one random engineering seed/model per policy, repeated tokens/loops/sites;
three policy arms. Primary contrast: within-policy cache/full-prefix numerical
execution. Physical dose zero; scientific causal and matched-sham floors and
causal power uncalibrated. Planned/actual formal comparisons zero. Verdict:
random engineering evidence only, with real-checkpoint validation pending.
The original 168-case test stays sealed and the algebraic gates remain failed.
