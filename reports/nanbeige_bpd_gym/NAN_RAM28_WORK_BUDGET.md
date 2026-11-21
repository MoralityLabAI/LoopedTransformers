# RAM28 full-roster work and trace accounting

This is **source-only engineering accounting**, using the unchanged registered
roster, token bindings and already closed NF4 capacity preflight. No generated
reply, hidden-state array, grade or oracle answer was read. It makes no new
model call, changes no worker/order/budget, and supplies no competence or causal
result. The original 168-case test stays sealed and the continuous goal stays active.

The NF4 full gym registers **1,549 baselines plus 216 true A/B pairs**: **1,981
trajectory slots**. Both registered precisions together have 3,962 slots; these
are repeated trajectories, not independent evidence units. The roster has 150
distinct root identifiers on one checkpoint. The BF31 control has its own loader,
CUDA admission and eventual timing evidence; NF4 timing is not a BF16 prediction.

The exact frozen initial scheduler produces **931 baseline batches**: **618
B2 and 313 B1**. It tests the next prompt with the original +256 capacity margin.
Every A/B pair requires actual B2. Dynamic EOS/packet stops can leave a genuine
B1 tail, which is recorded rather than replaced with a serial A/B simulation.

| Full-cap allowance per backend | Count |
|---|---:|
| Generated row prefixes | 242,048 |
| NF4 capture batches with no early stop | 141,056 |
| Physical state vectors, 40 per row prefix | 9,681,920 |
| Forty-vector BF16 payload per row prefix | 245,760 bytes |
| Those vectors' payload at every token cap | 59,485,716,480 bytes (55.400 GiB) |
| Registered artifact budget per backend | 160 GiB |

These are token-cap scenarios, conditional on reaching every trajectory.
They are not collected-data counts or a guarantee that all slots finish.
EOS, complete packets, character caps, capacity/resource/time/disk failures
can stop earlier. The payload excludes boundary/native/mask arrays,
NPZ/JSON/filesystem overhead and later host artifacts; it is not a total
storage forecast or disk-safety certificate.

The closed reached-state capacity probes measured:

| Batch | Probe length | Padding in row B | CUDA interval |
|---:|---:|---:|---:|
| 1 | 4,608 | 0 | 10.795 s |
| 1 | 9,216 | 0 | 27.412 s |
| 2 | 4,608 | 1 | 28.357 s |

The B2x9216 probe failed capacity and stays failed. The B1x9216 and padded
B2x4608 required gates passed. Successful shape admission is not a fast-run
or full-roster completion guarantee. These are two native loops over a full
prefix, with reached-state intervention allocations; they are not cached-token
latency, per-layer kernel occupancy or a native-versus-edit equivalence test.

For transparent sensitivity accounting, the JSON also charges each full-cap
capture the measured enclosing B1/B2 probe interval. That arithmetic totals
**45.833 probe-days**. This is **neither a
runtime prediction nor a bound**: most shorter native prefixes differ from
those padded edited probe shapes; early stops, clock/thermal state, allocation,
head/I/O work and timing attribution also differ. No selection, admission or
execution change is derived from that scenario.

The frozen limit remains **seven loaded-model days per backend**, with all
baselines before A/B. If a fixed limit is reached first, later baselines/pairs
stay explicitly unknown and the corresponding scientific requirement remains
unfinished. No report may count mere registration as completed interventions.
Future scheduling changes require their own prospective condition; this
accounting does not reorder or retry the current job.

The metadata-only check itself used one CPU thread and Below Normal priority,
**17.781 MiB** maximum sampled private RAM and at least **5.460 GiB** system available RAM. It ran
under the separate 0.75-GiB private cap, 5-GiB admission and 4-GiB reserve,
then exited. Torch was never imported. The independent math process was untouched.

[Reproducible accounting tool](../../mechinterp/nanbeige_bpd_gym/ram28_work_budget.py) ·
[Complete arithmetic, groups and source hashes](nan_ram28_work_budget.json) ·
[Unchanged RAM28 protocol](../../experiments/nan_ram28/PROTOCOL.md) ·
[Full NF4 progress](NAN_RAM28_PROGRESS.md) ·
[Separate BF16 control](NAN_BF31_PLAN.md)
