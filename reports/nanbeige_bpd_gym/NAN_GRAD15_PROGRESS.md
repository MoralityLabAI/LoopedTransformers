# NAN-GRAD-15: real-checkpoint NF4/BF16 gradient preflight

Conditional on both COMP14 focused first-token screens passing. Four fixed CAL prefixes from two roots; native depth1/2, rank8 layer21 down_proj, fixed SVD initialization. Zero optimizer steps: this is not trained VPD/BPD or donor transfer.

| Backend | Status | Depth | Gradient cases passed | Registered cases |
|---|---|---:|---:|---:|
| nf4 | pending | — | — | 8 |
| bf16 | pending | — | — | 8 |

Exact native all-one closure, finite/nonzero V/U gradients, archived-logit KL and resource checks are separate from gradient correctness against a full backward replay. Fixed masks are not matched-dose causal interventions. Target weights and factor values must remain unchanged. Failed/unrun cases remain visible; no neural retry.

One large model at a time, one numerical CPU thread/Below Normal,2-GiB added saved-tensor CPU cap,85% CUDA cap. The separate math process is untouched. Original FIT07/algebraic failures remain unchanged. The168-case test stays sealed. The broader goal remains active for actual factor fitting and later registered stages.

Source seal: `61f37ab12d17ab1d3569b9a6ebeff59e25deac1cc9c8f00e6240e43d4cfe23c5`.

[Frozen protocol](../../experiments/nan_grad15/PROTOCOL.md) · [Source seal](../../experiments/nan_grad15/SOURCE_MANIFEST.json) · [Full aggregates](nan_grad15_progress.json)
