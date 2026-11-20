# NAN-CTX-08 resident-head long-context pilot

Updated 2026-11-20T10:08:54.120845+00:00

Fresh 1K/2K/4K-token ledgers test direct lookup and a two-record join, with query records early/middle/late, two worlds differing in one relevant label, and explicit copy controls. Four independent DEV root families generate 288 contexts; one separate CAL family supplies three capacity profiles. Longer variants and worlds are repeated measures.

NF4 retains every decoder layer on CUDA. BF16 retains four decoder layers plus a reusable staging layer. Both keep the unchanged BF16 vocabulary head on CUDA and use the same full-vocabulary chunked scorer; only the embedding lookup remains on CPU in NF4. This removes repeated CPU-to-GPU head-weight transfers. Real-checkpoint head parity must pass before cases can be reported. Old frozen conditions are preserved.

| Backend | Evidence status | Contexts | Physical vectors checked | CUDA allocated peak GiB |
|---|---|---:|---:|---:|
| NF4 | partial_evidence | 194/291 | 7760 | 2.708 |
| BF16 | queued | 0/291 | 0 | 0.000 |

| Backend | Context budget | Position | Task | Control | Native loop | Correct | Format valid | Independent roots |
|---|---:|---|---|---|---:|---:|---:|---:|
| NF4 | 1024 | early | join | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 1024 | early | join | main | 2 | 1/8 | 8/8 | 4 |
| NF4 | 1024 | early | join | copy | 1 | 5/8 | 8/8 | 4 |
| NF4 | 1024 | early | join | copy | 2 | 6/8 | 8/8 | 4 |
| NF4 | 1024 | early | lookup | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 1024 | early | lookup | main | 2 | 8/8 | 8/8 | 4 |
| NF4 | 1024 | early | lookup | copy | 1 | 5/8 | 6/8 | 4 |
| NF4 | 1024 | early | lookup | copy | 2 | 8/8 | 8/8 | 4 |
| NF4 | 1024 | late | join | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 1024 | late | join | main | 2 | 2/8 | 8/8 | 4 |
| NF4 | 1024 | late | join | copy | 1 | 7/8 | 8/8 | 4 |
| NF4 | 1024 | late | join | copy | 2 | 4/8 | 8/8 | 4 |
| NF4 | 1024 | late | lookup | main | 1 | 1/8 | 8/8 | 4 |
| NF4 | 1024 | late | lookup | main | 2 | 7/8 | 8/8 | 4 |
| NF4 | 1024 | late | lookup | copy | 1 | 5/8 | 6/8 | 4 |
| NF4 | 1024 | late | lookup | copy | 2 | 8/8 | 8/8 | 4 |
| NF4 | 1024 | middle | join | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 1024 | middle | join | main | 2 | 2/8 | 8/8 | 4 |
| NF4 | 1024 | middle | join | copy | 1 | 6/8 | 8/8 | 4 |
| NF4 | 1024 | middle | join | copy | 2 | 6/8 | 8/8 | 4 |
| NF4 | 1024 | middle | lookup | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 1024 | middle | lookup | main | 2 | 8/8 | 8/8 | 4 |
| NF4 | 1024 | middle | lookup | copy | 1 | 5/8 | 6/8 | 4 |
| NF4 | 1024 | middle | lookup | copy | 2 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | early | join | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 2048 | early | join | main | 2 | 3/8 | 8/8 | 4 |
| NF4 | 2048 | early | join | copy | 1 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | early | join | copy | 2 | 5/8 | 8/8 | 4 |
| NF4 | 2048 | early | lookup | main | 1 | 3/8 | 8/8 | 4 |
| NF4 | 2048 | early | lookup | main | 2 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | early | lookup | copy | 1 | 6/8 | 8/8 | 4 |
| NF4 | 2048 | early | lookup | copy | 2 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | late | join | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 2048 | late | join | main | 2 | 1/8 | 8/8 | 4 |
| NF4 | 2048 | late | join | copy | 1 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | late | join | copy | 2 | 5/8 | 8/8 | 4 |
| NF4 | 2048 | late | lookup | main | 1 | 1/8 | 8/8 | 4 |
| NF4 | 2048 | late | lookup | main | 2 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | late | lookup | copy | 1 | 6/8 | 6/8 | 4 |
| NF4 | 2048 | late | lookup | copy | 2 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | middle | join | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 2048 | middle | join | main | 2 | 0/8 | 8/8 | 4 |
| NF4 | 2048 | middle | join | copy | 1 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | middle | join | copy | 2 | 7/8 | 8/8 | 4 |
| NF4 | 2048 | middle | lookup | main | 1 | 0/8 | 8/8 | 4 |
| NF4 | 2048 | middle | lookup | main | 2 | 8/8 | 8/8 | 4 |
| NF4 | 2048 | middle | lookup | copy | 1 | 6/8 | 8/8 | 4 |
| NF4 | 2048 | middle | lookup | copy | 2 | 8/8 | 8/8 | 4 |

The endpoint is one free first token after each native loop, scored with a bounded anchored label grammar. It is distinct from native prefix-stop and Prime complete-final-reply generation. No forced candidates, answer-based filtering or long rationale generation is used. Wrong CAL answers do not block DEV: the CAL profile gates hardware capacity only.

A CUDA OOM stops that backend at the current level; all attempted outcomes remain, and unattempted contexts stay missing. Each backend has a 7200-second between-context budget and an 85% CUDA allocator limit. CPU threads are limited to one and priority is Below Normal to favor the concurrent math process. Observed latency includes shared-machine contention and is not an isolated throughput benchmark.

Layer-5/11/17/21 residuals and projection inputs/outputs are retained locally for SAE and learned-factor cross-reference. Residuals precede loop-final RMSNorm; the down_proj identity bridge does not extend to o_proj. No dictionaries or factors are fitted on these DEV contexts. No donor transfer is enabled. The original 168-case test stays sealed.

[Protocol](../../experiments/nan_ctx08/PROTOCOL.md) · [Frozen source](../../experiments/nan_ctx08/SOURCE_MANIFEST.json) · [Aggregate data](nan_ctx08_progress.json)
