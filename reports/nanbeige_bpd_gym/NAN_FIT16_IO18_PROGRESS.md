# NAN-FIT-16-IO18: matched actual NF4/BF16 VPD/BPD development fitting

Execution-only repair: the original FIT16 NF4 partial updates and step-log FileExistsError are preserved; original BF16 and XREF17 were left unrun under the prospective supervision amendment. Only the repeated step-log writer changes; all sources/numerics/criteria remain identical. No failed checkpoint is warm-started. 

Conditional on both unchanged COMP14 focused first-token screens and every GRAD15 gradient case passing. Each rank8 layer21 down_proj depth1/2 fit starts independently from its backend-specific SVD and the same CI seed, with sixteen genuine stock optimizer steps on both worlds of all eight CAL roots. No DEV/control fitting or donor transfer. T1 is a depth/recipe control; only loop2 competence was qualified.

| Backend | Status | Depth | Cell audit/status | Optimizer steps | Factor update L2 | Controller update L2 |
|---|---|---:|---|---:|---:|---:|
| nf4 | completed_audited | 1 | passed | 16 | 0.26161908985152227 | 0.37792415363941395 |
| nf4 | completed_audited | 2 | passed | 16 | 0.15781058514490853 | 0.24150051682958631 |
| bf16 | pending | — | pending/unrun | — | — | — |

Scores appear only after independent terminal checkpoint/controller/first-token audits. Pending/missing cells are not zeros. Initial component-only routes and every baseline split/arm/loop remain in the JSON.

| Backend | Depth | Split | Route | Loop | Correct worlds | Both-correct roots |
|---|---:|---|---|---:|---:|---:|
| nf4 | 1 | calibration | all_one | 1 | 1/16 | 0/8 |
| nf4 | 1 | calibration | clean_informed | 1 | 1/16 | 0/8 |
| nf4 | 1 | calibration | component_only | 1 | 1/16 | 0/8 |
| nf4 | 1 | calibration | reached_state | 1 | 1/16 | 0/8 |
| nf4 | 1 | calibration | static_CAL_mean | 1 | 1/16 | 0/8 |
| nf4 | 1 | development | all_one | 1 | 3/32 | 0/16 |
| nf4 | 1 | development | clean_informed | 1 | 3/32 | 0/16 |
| nf4 | 1 | development | component_only | 1 | 2/32 | 0/16 |
| nf4 | 1 | development | reached_state | 1 | 3/32 | 0/16 |
| nf4 | 1 | development | static_CAL_mean | 1 | 2/32 | 0/16 |
| nf4 | 2 | calibration | all_one | 1 | 1/16 | 0/8 |
| nf4 | 2 | calibration | all_one | 2 | 15/16 | 7/8 |
| nf4 | 2 | calibration | clean_informed | 1 | 1/16 | 0/8 |
| nf4 | 2 | calibration | clean_informed | 2 | 15/16 | 7/8 |
| nf4 | 2 | calibration | component_only | 1 | 1/16 | 0/8 |
| nf4 | 2 | calibration | component_only | 2 | 15/16 | 7/8 |
| nf4 | 2 | calibration | reached_state | 1 | 1/16 | 0/8 |
| nf4 | 2 | calibration | reached_state | 2 | 15/16 | 7/8 |
| nf4 | 2 | calibration | static_CAL_mean | 1 | 1/16 | 0/8 |
| nf4 | 2 | calibration | static_CAL_mean | 2 | 15/16 | 7/8 |
| nf4 | 2 | development | all_one | 1 | 3/32 | 0/16 |
| nf4 | 2 | development | all_one | 2 | 29/32 | 13/16 |
| nf4 | 2 | development | clean_informed | 1 | 2/32 | 0/16 |
| nf4 | 2 | development | clean_informed | 2 | 25/32 | 12/16 |
| nf4 | 2 | development | component_only | 1 | 2/32 | 0/16 |
| nf4 | 2 | development | component_only | 2 | 26/32 | 12/16 |
| nf4 | 2 | development | reached_state | 1 | 2/32 | 0/16 |
| nf4 | 2 | development | reached_state | 2 | 25/32 | 12/16 |
| nf4 | 2 | development | static_CAL_mean | 1 | 2/32 | 0/16 |
| nf4 | 2 | development | static_CAL_mean | 2 | 26/32 | 12/16 |

## Physical factor operators

Compare V@U in shared physical input/output coordinates, never independently fitted component indices. All four fixed pairs are reported only after all four fit audits. Shared initialization, small updates and joint depth/recipe changes limit interpretation.

| Reference | Compared | Relative Frobenius L2 | Cosine |
|---|---|---:|---:|

One large model, one numerical CPU thread and Below Normal priority protect the separate math process. Existing offline checkpoint/software only; native NF4 kernels retained, GPU head unchanged, CPU embeddings. 2-GiB added saved-tensor CPU cap,85% CUDA allocator cap, four-hour between-stage budget per backend. Allocator and driver counters remain distinct.

Copy/erasure are clean-baseline controls; all five learned routing modes are evaluated on all48 main prefixes, not on copy/erasure prompts. These are unmatched-dose fidelity diagnostics, without a calibrated sham/effect floor, population power, mechanism specificity or transfer claim. Original algebraic/FIT07 failures remain unchanged and the168-case test stays sealed. The continuous goal remains active; learned SAE cross-references require subsequent registered analysis.

Synthetic prompts and host-only oracle fixtures are public frozen sources; host-only means isolated from the worker. Generated outputs, training checkpoints and per-record run grades stay local. Host audits replay controller/checkpoint arithmetic, not a second full neural backward or exact CAL-mean recomputation.

Source seal: `7bb76c5c895848cf12643312609a584ca0797bc87e9a05e320b8cfa2d04f4830`.

[Protocol](../../experiments/nan_fit16_io18/PROTOCOL.md) · [Source seal](../../experiments/nan_fit16_io18/SOURCE_MANIFEST.json) · [Full aggregates](nan_fit16_io18_progress.json)
