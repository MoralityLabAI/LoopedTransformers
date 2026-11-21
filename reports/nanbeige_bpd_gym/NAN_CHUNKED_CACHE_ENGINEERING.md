# Future chunked cache, BPD and streamed-head CPU engineering

**MOCK_ONLY:** these checks use the actual pinned decoder code with a small
random configuration, synthetic factors/CI weights and fixed synthetic token
histories. They make **zero pretrained forwards and no CUDA initialization**.
The full current NF4 run and queued BF31 control are unchanged. This prototype
is **unregistered for pretrained use** and starts no worker or supervisor.

The engineering target is a future condition that reuses causal past states,
keeps the unchanged vocabulary head on CPU and stages one row chunk for full
readout. It is a distinct cached/chunked numerical condition. Actual NF4/BF16
capacity, memory, complete-task behavior and throughput require a prospective
source freeze and real checkpoint run; no native equivalence or speedup is
supplied here. The [prior pretrained cache diagnostics](NAN_CACHE_EXECUTION_COMPARISON.md)
retain their errors and token disagreements.

The prototype uses two independent phase caches and layer-major query chunks.
A staged decoder layer is loaded once per phase, regardless of prefill chunk
count. Semantic RoPE positions are separate from padded physical cache slots.
Row pruning selects one cache row at a time and preserves the padded canvas;
last-valid-token observations use each row's own position. Projected K/V
storage must not alias the residual buffer before in-place chunk reuse.

BPD hooks run before output observers and retain the original pointwise
algebra. Each chunk explicitly binds calibrated native phase 1 or 2; chunk calls
are not additional recurrent passes. Native outputs, actual masks and reached
inputs are captured at the correct final valid token. Duplicate legacy raw
snapshots and controller observations are cleared in this new adapter.
The original frozen adapters/workers are not patched.

| CPU check | Evidence | Scope |
|---|---|---|
| Native full-prefix versus chunked/cache | 48 paired captures / 2,880 physical-vector comparisons | Actual random pinned class: 22 layers, hidden 64, intermediate 128, query width 128, KV width 32, vocabulary 256; B1/ragged B2, both buffer policies and row pruning |
| FP64 structural gate | Max abs 8.88178e-16 | Preset numerical gate 1e-10; not a scientific effect floor |
| BF16 numerical diagnostics | Max abs 7.62939e-06 | Every difference retained; no equivalence gate or claim |
| Actual BPD hooks | 40 paired captures / 2,800 physical-vector comparisons | Native/all-one/component/static-CAL/reached-state, duplicate A/B and ragged rows; synthetic factors/CI, zero training |
| Internal controls | Exact all-one and clean-A closure | Same cached execution/history/shape; edited B changes on nontrivial routes |
| Independent mask/write replay | Max mask abs 5.99438e-08; local-write relative L2 0 | Original mask rtol2e-5/atol1e-5 and local-write .03 tolerance retained; phase-specific static masks differ |
| Streamed tiny-vocabulary readout | 160 exact readouts of all 256 logits | Same row-chunk arithmetic; one tiny reusable stage of 896 bytes, unchanged head/target parameter bytes |
| Self-contained regression contracts | Four methods passed | Causal padding/semantic positions/row pruning/reset and full-vocabulary staging/source preservation/invalid inputs |

These are repeated numerical comparisons on untrained small models, not
independent trained checkpoints, source-world sample sizes or task grades.
The BPD fixture's measured full-prefix/cache state and logit errors were zero
in its synthetic cases; that does not remove differences in the separate
native fixture or the older pretrained experiments. The first native and BPD
checks, their code and receipts remain preserved; the later BPD run adds the
head reader to the same hook path. No missing neural reply or grade is rescued.

The real checkpoint's **shape arithmetic**, without tensor materialization:

| Buffer | Bytes | GiB |
|---|---:|---:|
| One row, 8K two-loop BF16 K/V | 1,476,395,008 | 1.375000 |
| One row, 9,216 two-loop BF16 K/V | 1,660,944,384 | 1.546875 |
| Two rows, 4,608 two-loop BF16 K/V | 1,660,944,384 | 1.546875 |
| Unchanged CPU head source | 1,020,788,736 | 0.950684 |
| One 8,192-row BF16 head stage | 50,331,648 | 0.046875 |

K/V costs 180,224 bytes/token/row across both phases. This explains why a cache
cannot simply be added to the current resident-head run. The table excludes
other weights, hidden/query buffers, allocator/runtime overhead and transient
copies. It is not proof that a future cached condition fits 4 GB; the required
B1x9216/padded B2x4608 CUDA closure/capacity and host/driver memory checks remain
unmeasured. Large mapped CPU heads must remain read-only and obey RAM reserves.

All three CPU lanes ran sequentially, with one numerical thread and Below
Normal priority, 5-GiB admission, 4-GiB system reserve and a .75-GiB private
cap checked every two seconds. These are sampled values, not continuous peaks:

| Lane | Max private MiB | Max resident MiB | Min available GiB |
|---|---:|---:|---:|
| Native | 731.348 | 460.496 | 4.295 |
| BPD plus head | 719.082 | 456.871 | 4.321 |
| Contracts | 640.887 | 378.809 | 4.331 |

The new cache rejects edited-route labels without an attached BPD instrument,
unknown routes and an explicitly empty mode list. Expanded negative checks ran
with the positive native/BPD/head fixtures against the current source. Two
additional attempts remained unrun at RAM admission: the first before Torch;
a unit wrapper also imported Torch before its inner start check and then
refused admission before running tests. That wrapper was corrected, and the
successful unit run installed its budget before Torch import. All failed/
unrun logs, original positive versions and source bytes remain preserved;
no limit or numerical criterion was relaxed and no neural reply was replayed.

The CPU processes exited and the separate math process was untouched. No real
reply, hidden-state array or task label was read. A future model condition
still needs complete sources/token/parser/budgets, full frozen scope, qualified
real factor bindings, target/stage checks, actual CUDA admission and independent
behavior/state audits. The full 1,549-baseline/216-pair goal is not replaced by
these CPU checks. Original competence failures and all scientific gates remain;
no faithful-reasoning, donor-transfer, mediation or equivalence claim. The
original 168-case test stays sealed and the continuous goal stays active.

[Native cache](../../mechinterp/nanbeige_bpd_gym/chunked_cache.py) ·
[Phase-bound BPD adapter](../../mechinterp/nanbeige_bpd_gym/chunked_instrument.py) ·
[Streamed full-vocabulary head](../../mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py) ·
[Native engineering harness](../../mechinterp/nanbeige_bpd_gym/chunked_cache_engineering.py) ·
[BPD/head harness](../../mechinterp/nanbeige_bpd_gym/chunked_bpd_engineering.py) ·
[Full synthetic receipts](nan_chunked_cache_engineering.json) ·
[Unchanged NF4 full gym](NAN_RAM28_PROGRESS.md) ·
[Separately frozen BF16 control](NAN_BF31_PLAN.md)
