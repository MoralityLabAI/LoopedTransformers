# Lazy BF16 loop and BPD hooks: random small-class CPU integration

**MOCK_ONLY — no CUDA initialization, pretrained forwards or model grades.**
The candidate storage-backed loop passed a CPU integration check using the
actual pinned decoder class with random small weights: 22 layers, hidden
width 64, intermediate width 128, vocabulary 256, four resident CPU layers
and one reusable CPU stage. Of its 201 parameters, 161 remained on meta.
The real NF4 worker and all frozen methods remained unchanged.

| CPU integration check | Result |
|---|---|
| Original native B1 loop | Dense/file-backed state and readout bytes exact |
| Actual full-gym batched B1 and ragged B2 loops | State/readout bytes exact |
| Captures | 22 across dense and candidate loops; all 40 physical vectors per row checked |
| Actual pinned BPD hooks | All-one, component-only, static CAL-mean-mask and reached-state routes passed |
| Clean A row | Unchanged by every edited B route |
| All-one closure | Exact states, boundaries, logits and saved native tensors |
| Reduced B routes | Nonzero changes in the edited row; dense/candidate outputs still byte-exact |
| Candidate stage copies | 396; largest synthetic source tensor 16,384 bytes |
| Logical/resident target digest | Unchanged before/after; scoped dispatch restored on exit |
| Additional regression tests | Four passed: buffer admission, dispatch restoration, repeated stage copy and observer lifetime |

The BPD framework, row dispatch, pass-mean/sphere algebra, zero-block state-only
CI network and pointwise chunk adapter are the actual pinned implementations.
The factor bank, CI weights and their fixture receipt are explicitly synthetic;
there were zero optimizer updates and no real qualified factors were used.
This verifies plumbing, not learned quality, source competence or mediation.

## Preserved failed fixture and buffer admission

The first CPU attempt failed native B1 byte comparison because its partially
materialized fixture left four resident-prefix `rotary_emb.inv_freq` buffers
on meta. Matching parameter hashes alone did not cover these non-persistent
buffers. The failed attempt, resource receipt and all three original source
files are retained locally with their exact recorded byte hashes.

The constructor now rejects uninitialized resident or stage buffers before a
forward. The fixture then uses the same native RoPE initialization already
used by the pinned loaders. The second attempt passed the original exact-byte
criterion; no tolerance or frozen model protocol was relaxed. This candidate
still requires a complete prospective CUDA loader and capacity admission.

## Duplicate native observer retention

The genuine full-gym `RowInstrument` constructor also reproduced a retention
issue: three native captures left **2, 4, then 6** vectors in the legacy
`base.raw` list. The full-gym writer exports `raw_by_phase` instead. Native
contexts do not call the base context reset, so that duplicate list can grow
across captures. At pinned width 3072, two float32 vectors represent **24,576
raw bytes per native prefix**, excluding object overhead. This is a source-
derived estimate, not a measured growth rate in the live full model.

A narrowly scoped candidate cleanup clears only this unused list **after**
capture exports the authoritative native tensors. The test preserved exact
states, boundaries, full tiny-vocabulary logits, native outputs and all four
BPD routes while retained duplicate vectors returned to zero. The helper
rejects other instrument types; original STORY23 snapshot consumers still
need their own `base.raw`. This repair is not patched into any frozen live
worker. Its use belongs to a separately registered future condition. Existing
RAM28 memory stops remain authoritative if its unmodified condition reaches
its budget.

## Resources and remaining work

Maximum sampled private RAM was **711.809 MiB**; minimum sampled system
available RAM was **8.859 GiB**. One numerical thread, Below Normal, no CUDA,
with 5 GiB admission, a 4 GiB reserve and a 0.75 GiB private cap checked every
two seconds. Samples are not continuous peak measurements or a forecast of
full-model CUDA memory. The CPU process exited after the check.

BF16 inference remains deferred. A full worker/loader/resource/source freeze,
genuine factor qualification, actual checkpoint stage copies, CUDA native/head/
row/all-one parity, padded capacity and RAM admission are still required.
The full 1,549-baseline / 216 real A/B-pair scope remains intact; this is not a
replacement subset. Real storyworld scores and canonical SAE/BPD analysis
remain pending. No previous failure, gate or inferential endpoint changes.
The original 168-case test stays sealed and the continuous goal stays active.

[Candidate loop and scoped cleanup](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_loop.py) ·
[CPU integration harness](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_integration.py) ·
[Regression contracts](../../mechinterp/nanbeige_bpd_gym/lazy_bf16_loop_tests.py) ·
[Sanitized receipt](lazy_bf16_loop_integration.json) ·
[Prior storage/checksum proof](LAZY_BF16_STORAGE_ENGINEERING.md) ·
[Live NF4 condition](NAN_RAM28_PROGRESS.md)
