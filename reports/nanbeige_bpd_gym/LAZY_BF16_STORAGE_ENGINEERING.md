# BF16 storage prototype: keep offloaded decoder weights on disk

The source-only CPU prototype passed eight storage contracts and reproduced
the original BF16 ordered parameter checksum from the pinned checkpoint while
all **201 parameters remained on meta**. This is progress toward restoring
matched BF16 controls without the former full CPU heap. **It is not a registered
BF16 inference condition or evidence of CUDA capacity.** No BF16 model has been
started alongside the ongoing NF4 gym.

The current BF16 path retains all offloaded decoder tensors on CPU, in addition
to a resident GPU prefix and reusable stage. Merely mapping every CPU tensor
can avoid private heap commitment while still retaining large active mappings.
This candidate stores file extents and tensor metadata instead. It maps one
source tensor, performs the same synchronous parameter copy into an allocated
stage, then releases that temporary tensor normally. Selected CPU parameters
can remain read-only mappings; other decoder parameters stay on meta. It never
manually unmaps a live tensor or writes a mapped checkpoint parameter.

The digest preserves the original ordered `parameter name + parameter bytes`
serialization. Actual resident parameters are hashed from their actual bytes;
meta parameters stream bounded file chunks. A mutated resident clone changes
the digest. This binds logical weight bytes, not executed stage kernels, RoPE
buffers, CUDA residency or inference outputs. A future model integration must
verify those separately and retain the selected BPD target weight needed by
the existing component attachment code.

| CPU engineering check | Observed result |
|---|---|
| Tiny stage forwards | Bit-exact at every layer of two loops, BF16 and FP32 |
| Digest against original dense implementation | Exact for all-meta and mixed-storage toys |
| Actual resident-byte mutation | Detected by a changed digest |
| Partial binding | Only selected parameters materialized; other layers stayed meta |
| Source mapping ownership | Survived store scope; checkpoint bytes unchanged |
| Invalid layouts, extents, missing tensors and destinations | Rejected |
| Pinned architecture | 201 parameters, all meta before and after checksum |
| Pinned parameter bytes streamed | 8,339,601,408 bytes (7.766859 GiB) |
| Pinned logical checksum | `cf3db880c052137e7ff7a0dac1e498174f78e32fd3ae9a384aca341fb1d09186` |
| Reference | Original CTX21 BF16 before-forward parameter manifest |
| Pretrained forwards / CUDA initialization | Zero / false |

The tiny stage tests use a four-layer feed-forward CPU toy, not Nanbeige
inference. The pinned architecture check verifies checkpoint files against
their existing lock, validates shapes/dtypes and streams bytes; it does not
materialize the pretrained matrices as Torch parameters or run them. Torch
is used here for storage, meta architecture and stage-copy tests, unlike the
tokenizer-only audits that intentionally avoid it.

The entire process had **708.098 MiB maximum sampled private RAM**, with at
least **5.274 GiB system RAM available**. It used one numerical thread and
Below Normal priority, with 5 GiB admission, a 4 GiB system reserve and a
0.75 GiB private cap checked every two seconds. Samples are not continuous
peak measurements. These measurements cover CPU engineering only and do not
predict the memory of a CUDA BF16 worker. The process exited; the original
NF4 worker was independently verified live under the same creation-bound
handle afterward. No real generated response/state was read.

Remaining production work is explicit: integrate the lazy source into a new
native loop and logical/resident target verification, test the actual pinned
decoder class and BPD hooks, then freeze the complete future BF16 condition
and resource policy before any pretrained forward. Only after the active NF4
model and its serialized host lane finish may a newly admitted BF16 worker
attempt CUDA native/head/row/all-one closure and padded capacity gates. Its
matched workload must retain every registered source/control and all missing
denominators; the current 1,549-baseline / 216 actual A/B-pair scope is not
reduced by this prototype. No capacity or score failure licenses a silent
fallback or neural replay.

BF16 remains `deferred_host_RAM_policy`. This prototype changes no frozen
worker, grading rule, previous failure or causal gate, and supplies no model
accuracy, intervention, SAE or BPD result. The original 168-case test stays
sealed; the continuous research goal remains active.

[Storage primitives](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_storage.py) ·
[CPU contracts](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_storage_tests.py) ·
[Bounded local harness](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_engineering.py) ·
[Sanitized receipt](lazy_bf16_storage_engineering.json) ·
[Current NF4 full gym](NAN_RAM28_PROGRESS.md) ·
[Preserved BF16/NF4 cache comparison](NAN_CTX21_RESOURCE29_RESULTS.md)
