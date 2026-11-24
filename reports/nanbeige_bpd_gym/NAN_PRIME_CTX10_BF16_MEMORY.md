# CTX10 BF16: CUDA and driver memory measurements

The completed 288-episode BF16 worker used an unchanged BF16 head, four resident decoder layers and one reusable staging layer. The math process was untouched; this read-only sampler ran Below Normal priority.

| Measurement | Value |
|---|---:|
| CUDA allocated peak | 2.702 GiB |
| CUDA reserved peak | 3.010 GiB |
| Sampled worker dedicated range | 3.104–3.104 GiB |
| Sampled worker shared range | 78.00–78.00 MiB |
| Driver samples / errors | 33 / 0 |

Driver counters were sampled every 30 seconds over part of the worker lifetime. They are not exact full-lifetime peaks or per-tensor residency measurements. Shared usage may include runtime/transfer memory; it does not by itself show model weights spilling out of dedicated VRAM. No matching NF4 driver-counter window was captured for this condition.

The original 168-case test stays sealed. [Aggregate counters and provenance](nan_prime_ctx10_BF16_memory.json).
