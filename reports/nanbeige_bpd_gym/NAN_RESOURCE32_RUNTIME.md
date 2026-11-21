# Nanbeige host RAM and GPU runtime check

At 2026-11-22 06:03 Taipei time (2026-11-21 22:03 UTC), five consecutive
NVIDIA samples reported **100% device activity**, with3,679/4,096 MiB allocated.
The authenticated RAM28 NF4 worker held all22 decoder layers and the BF16
output head on GPU; embeddings were CPU-backed. The latest completed prefix
reported23.538 s CUDA versus23.547 s wall time. These are short observations,
not continuous utilization or benchmark speedup claims.

Host pressure remains material: **5.467 GiB private commitment**,1.029 GiB
resident working set and3.704 GiB system available. Low residency does not
remove the commitment. Process GPU allocation counters reported3,678.715 MiB
dedicated and78 MiB shared; these counters do not establish physical residency
or continuous peaks. The three small waiting supervisors together held about
33 MiB private. No second model or numerical CPU audit was launched.

The active condition already uses read-only checkpoint mappings,22 NF4 GPU
decoder layers, a resident GPU head, one numerical thread, Below Normal
priority and its registered host guards. Its frozen numerical policy was not
changed. The separate math process was neither stopped nor reprioritized.

The [new frozen cached comparison](NAN_CACHE32_PLAN.md) removes unused legacy
raw-observer retention, avoids the discarded full-size CPU matrix in NF4
wrapper construction, and reuses two phase-specific KV caches. It retains all
1,549 baselines and216 genuine B2 A/B pairs per precision. Its streamed
full-vocabulary GPU head frees capacity for the caches; embeddings and head
source remain read-only mappings. Complete local CUDA capacity, correctness,
RAM and throughput checks are still required, so no memory or speedup gain
is claimed yet. Heavy CPU audits remain serialized after model termination.

The implementation passed remote CPU CI with small random parameters and
synthetic factors, keeping Torch validation off the constrained workstation.
Original outcomes and frozen runs are preserved. The continuous goal remains
active; the original168-case ARG03 test stays sealed.

[Measured resource samples](nan_resource32_runtime.json) ·
[Frozen source manifest](../../experiments/nan_cache32/SOURCE_MANIFEST.json)
