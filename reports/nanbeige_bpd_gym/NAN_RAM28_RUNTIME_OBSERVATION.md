# RAM28: measured RAM recovery and active NF4 GPU traces

The frozen NF4 full gym is running. During a fixed60-second window (2026-11-21T14:48:38.123940+00:00 to 2026-11-21T14:49:38.138537+00:00),13 samples at five-second spacing measured **92.31% mean device busy time**, **at least 6.031 GiB available system RAM**, and **at most 2.144 GiB resident working set** for this model process.

| Measurement | Earlier interrupted BF16 snapshot | NF4 observation window |
|---|---:|---:|
| Model private commitment | 11.626 GiB | At most 5.416 GiB |
| Model resident working set | 4.947 GiB | At most 2.144 GiB |
| Available system RAM | 0.612 GiB | At least 6.031 GiB |
| GPU memory | 3,545 MiB snapshot | 3677 MiB of4,096 MiB |

The snapshots use different resource/execution conditions and contexts on a shared machine. This combines NF4 placement, file-backed loading, bounded MLP workspaces and serialized host audits; it does not isolate one optimization or precision. Private commitment and resident RAM differ. The window is not a full-run memory peak or throughput benchmark. Device busy time is not per-process SM occupancy; one sample was idle, and source rows/tokens are not independent replications.

All22 decoder layers and the BF16 vocabulary head are on GPU. CPU embedding storage is file-backed. Packed/native parameter bytes matched the original NF4 reference before the first forward; old quantization-state identity was not independently serialized or verified. One CPU thread, Below Normal priority, physical-RAM reserves and private-memory limits remain enforced. The separate math process was untouched.

## Actual closed CUDA preflight

B1 old/new-loop parity, full-head parity, B2 clean-row identity and B2 all-one closure passed. The original/chunked MLP differences are retained in the linked JSON; they do not select tasks. Both required capacity probes passed before any scored episode.

| Batch | Tokens | Right padding | Capacity result |
|---:|---:|---:|---|
| 1 | 4608 | 0 | passed |
| 1 | 9216 | 0 | passed |
| 2 | 4608 | 1 | passed |
| 2 | 9216 | 1 | capacity_failed |

Long B2 capacity failures remain explicit; admitted B1 handles long baseline inputs. All short A/B inputs remain admitted as actual batch two, with no serial A/B replacement. This is the separately registered chunked/batched execution condition, not an equivalence result against old arithmetic.

## Current trace capture

At the observation cutoff, the actor reported **96 captured row prefixes** across **48 decoder batches**, from the first two still-incomplete storyworld trajectories. These are repeated prefixes, not that many cases. Closed baseline trajectories: **0**; independent full episode/state/SAE audits remain pending in the serialized host lane. No model text, token IDs, states or individual grades are published here.

The driver snapshot reported **software thermal slowdown active**. That limits clock speed even while the device is busy; it is a driver observation, not a hardware setting change or a measured full-window throttle fraction.

The original CTX21 BF16 interruption preserves378 closed native cases, with its original terminal audit failed and no synthesized receipt. Further BF16 remains deferred. The complete NF4 workload retains1,549 baseline trajectories and216 batched clean/intervention pairs. The continuous goal remains active, and the168-case test stays sealed.

[Raw engineering samples, closed-preflight metrics and source bindings](nan_ram28_runtime_observation.json) · [Full-gym progress](NAN_RAM28_PROGRESS.md) · [Frozen resource protocol](../../experiments/nan_ram28/PROTOCOL.md)
