# NAN-KV-11: measured cache versus full-prefix execution

Updated 2026-11-20T13:52:07.088743+00:00

All 130 public GYM05 DEV contexts are registered, across 41 source families: pointer, ARC1/ARC2, Sudoku and storyworld, with every public copy control. Private routing and the protected original test are not read. The full-prefix model supplies a natural reference history of up to eight tokens/EOS; both paths receive exactly that history. This compares numerical execution and measured probe/readout latency, not complete task solving.

| Backend | Status | Closed contexts | Prefills byte exact | Token argmax disagreements | Max logit difference |
|---|---|---:|---:|---:|---:|
| NF4 | completed | 130/130 | 130 | 3 | 3.5 |
| BF16 | partial_evidence | 32/130 | 32 | 3 | 0.5 |

Independent CPU checks recompute paired differences from every archived vocabulary readout and physical site/boundary array. Prefill must be byte exact; a failure stops that backend with partial artifacts preserved. Incremental GEMM/GEMV/SDPA differences, argmax changes and all source groups remain visible. Approximate agreement does not authorize replacing any frozen full-prefix experiment. A cache argmax on a reference history is not an independent cached free-running reply after a divergence. No task accuracy or transfer score is claimed.

Latency includes both-loop observations and full-vocabulary readout, in fixed reference-then-cache order on a machine shared with the CPU math workload. It is not an isolated throughput benchmark. One CPU thread, Below Normal priority and an .85 CUDA cap are retained. All raw tokens, states and readouts stay local.

The original algebraic competence gates remain failed; this execution diagnostic does not repair them. No causal/equivalence claim or donor intervention. The original 168-case test stays sealed.

[Protocol](../../experiments/nan_kv11/PROTOCOL.md) · [Source seal](../../experiments/nan_kv11/SOURCE_MANIFEST.json) · [Aggregate evidence](nan_kv11_progress.json)
