# NAN-CTX-08: matched BF16/NF4 development comparison

Both registered backends completed all 291 contexts/profiles and passed independent token/score/state audits. The 288 DEV contexts come from four source families; one separate CAL family supplied three resource profiles. These are exposed development results on one checkpoint, not 288 independent trials.

| Context | Task | Control | BF16 correct | NF4 correct | BF16 only correct | NF4 only correct |
|---:|---|---|---:|---:|---:|---:|
| 1024 | join | main | 8/24 | 5/24 | 3 | 0 |
| 1024 | join | copy | 21/24 | 16/24 | 5 | 0 |
| 1024 | lookup | main | 24/24 | 23/24 | 1 | 0 |
| 1024 | lookup | copy | 24/24 | 24/24 | 0 | 0 |
| 2048 | join | main | 7/24 | 4/24 | 3 | 0 |
| 2048 | join | copy | 24/24 | 17/24 | 7 | 0 |
| 2048 | lookup | main | 24/24 | 24/24 | 0 | 0 |
| 2048 | lookup | copy | 24/24 | 24/24 | 0 | 0 |
| 4096 | join | main | 5/24 | 4/24 | 1 | 0 |
| 4096 | join | copy | 24/24 | 17/24 | 7 | 0 |
| 4096 | lookup | main | 22/24 | 21/24 | 2 | 1 |
| 4096 | lookup | copy | 24/24 | 22/24 | 2 | 0 |

The table uses the free first token after loop 2, parsed with the same bounded anchored label grammar. The JSON retains both loops, invalid outputs and paired discordance counts without dropping wrong cases. Direct lookup and two-record joins remain separate tasks; explicit answer-copy controls do not establish join competence.

| Layer | Loop-2 state relative L2 range | State cosine range | Recurrence-displacement relative L2 range |
|---:|---:|---:|---:|
| 5 | 0.147–0.170 | 0.986–0.989 | 0.244–0.273 |
| 11 | 0.153–0.192 | 0.981–0.988 | 0.340–0.372 |
| 17 | 0.209–0.305 | 0.950–0.978 | 0.305–0.373 |
| 21 | 0.144–0.184 | 0.984–0.991 | 0.065–0.086 |

Geometry ranges cover all length/position/task main-DEV strata. Each stratum averages within root before averaging its four roots. State differences are normalized by BF16 state norm; recurrence differences compare (loop 2 − loop 1) in physical residual coordinates and normalize by the BF16 displacement norm (denominator floor 1e-12). All position/control/loop groups are in the tower JSON. Residuals precede final loop RMSNorm. Independently fitted BF16/NF4 SAE feature indices are never subtracted.

Fixed CAL-only SAE dictionaries and their random initial controls were applied to every context at all four layers, with every feature/group retained. The eligible BF16 FIT07 T1 VPD asset includes SVD/Haar controls and cached-input controller/SAE correlations; loop 2 is outside fit depth. NF4 has no eligible learned asset. This is observational geometry, not online intervention or causal transfer.

| Backend | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| BF16 | 2.694 | 3.010 |
| NF4 | 2.875 | 3.201 |

Both conditions keep the unchanged BF16 head on GPU. NF4 retains every decoder layer there; BF16 retains four layers plus a reusable stage. One CPU thread and Below Normal priority favor the separate math process. Latency reflects contention and is not an isolated throughput benchmark.

Zero formal inferential comparisons; no equivalence, mechanism confirmation or donor transfer. Original failed algebraic gates remain failed. The original 168-case ARG-03 test stays sealed.

[All paired accuracy counts](nan_ctx08_comparison.json) · [All tower/physical geometry](NAN_CTX08_TOWER_PROBES.md) · [Behavior and capacity audit](NAN_CTX08_PROGRESS.md) · [Frozen protocol](../../experiments/nan_ctx08/PROTOCOL.md)
