# NAN-STREAM-41 guarded two-row READOUT34 execution

The driver binds the original 432 READOUT34 prompts and 672 slots per
precision to the STREAM36 schedule. The plan has 336 genuine two-row prefills
per precision, 120 at one physical pass and 216 at two. The runtime keeps
a single checkpoint on the 3050, prunes rows at their independent stop,
and preserves physical KV positions. The GPU and host stages have an
explicit journal and terminal handoff.

The host verifies the whole-batch hash, fixed row order, independently
replayed stop, emitted-token counts and physical row occupancy. It counts
each batch wall interval once. Sum of response latencies remains a separate
metric. The CUDA admission includes B1/B2 full-vocabulary parity, pruning,
maximum-position capacity and a short, matched B1/B2 timing run on CAL.
The worker records sampled host and device allocation, utilization,
temperature, SM clocks and Torch peak bytes.

## Local verification

- The pinned tokenizer reproduced all 432 prompt bindings and the EOS
  binding without Torch. The CAL pair contains 617 tokens per prompt.
- The production preflight and timing paths passed on tiny CPU DynamicCache
  weights, with 20 full-vocabulary timing comparisons at the two depths.
- The batch fixture independently closed two responses with a 2-to-1 row
  transition, counted 14 physical phase-token visits and rejected four
  altered records or replay attempts. An explicit device refusal yielded
  zero verified responses.
- The CPU cache and journal audit used at most 655.3 MiB private commitment
  and 396.0 MiB working set while retaining about 7.03 GiB available RAM.
- The first development fixtures exposed a zero-duration synthetic batch
  on Windows and an EOS check before prompt setup. The code now reports an
  unresolved rate for zero-duration windows and verifies EOS after the pinned
  prompt builder sets it. The final replay passed both paths.
- The cold GPU entrypoint refused the live RAM28 process before importing
  Torch or creating a STREAM41 artifact directory. Actual CUDA capacity,
  throughput and behavioral scores are pending a free lane and publication.

[Protocol](PROTOCOL.md) · [CPU verification](CPU_VERIFICATION.json) ·
[Pinned prompt replay](TOKENIZER_VERIFICATION.json)
