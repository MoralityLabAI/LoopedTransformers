# STREAM41 CUDA commissioning and paired-depth evaluation

Execute the frozen STREAM36 B2 packing of READOUT34 on one model at a time.
The roster has 672 slots per precision in 336 two-row prefills. It uses two
separate cache phases for each two-pass trajectory. It includes 120 B2 jobs
at one physical pass and 216 at two. The 4 GiB RTX 3050 runs NF4 first and
then staged BF16 as separate processes. Keep all required answer, literal-copy,
erased and trace controls. The previously registered B1 program remains an
independent condition.

Before loading Torch, authenticate the RAM28, BF31 and CACHE32 worker
terminations and host terminal, take the shared local GPU mutex, verify
the source and remote runtime manifests, check the pinned checkpoint and
adapter, and admit at 5 GiB of available RAM. Keep the READOUT34 soft/hard
process limits, one numerical thread, Below Normal priority and no live
progress JSON reads. Require 20 GiB available on the evidence volume.
The existing GPU cadence guard covers process-private and working-set memory
and sampled Windows dedicated/shared allocation. Record Torch peaks and
device telemetry separately.

The production preflight runs native CAL prompts at both depths and covers
B1/B2 full-vocabulary parity at prefill and incremental decode, padding
masks, a surviving row with a padding hole, and maximum B2 position 3,463.
A refusal yields zero scored slots. The workload binds every public prompt to
its verified tokenization and greedily stops each row at the original
READOUT34 first-stop rule. Live rows use the same physical position axis after
pruning. The scorer follows the first independently verified stop, the
hashed whole-batch journal and unchanged checkpoint/quantization receipts.
An incomplete integrity-qualified backend keeps unclosed slots unknown.

The GPU worker records one wall duration per physical batch and row-level
stops, emitted tokens, full-head row occupancy and physical phase-token
visits. Use these for per-depth throughput. A separately timed CAL diagnostic
runs up to four fixed decode steps after identical B1/B2 prefills at both
depths. It alternates ordering across depths and uses shared fixed decode
tokens; compare logits and greedy IDs before reporting the paired timings.
This is a short diagnostic on this device. The primary behavioral endpoints
are one-versus-two-pass repairs/regressions, exact answers and both-world
success per source root. Compare context length and format cells separately.

The twelve source roots are three CAL and nine exposed DEV roots. Repeated
worlds, queries, readouts, trajectories, physical passes and precisions
share roots. This development evaluation performs zero formal tests and no
causal intervention. Independently verified depth repairs can qualify a
readout for a later matched BPD/sham comparison. The sealed ARG-03 split
is outside this roster.

This protocol requires a publication receipt before pretrained execution.
Run the GPU worker from the canonical checkout with the pinned interpreter:

```powershell
python -B -m experiments.nan_stream41.run --backend nf4 --snapshot <checkpoint> --adapter <adapter> --tokenizer-receipt <receipt>
```

Run BF16 after the NF4 process exits. The post-model scorer uses the pinned
tokenizer on CPU and the independently replayed source oracle. Run it only
after both attempts have reached terminal state or have been checked closed.

```powershell
python -B -m experiments.nan_stream41.score --snapshot <checkpoint> --tokenizer-receipt <receipt> --output runs/nan_stream41/aggregate.json
```
