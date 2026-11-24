# RAM28 full-roster work and trace accounting

**Correction:** the first publication used residual width3072 for every site
and incorrectly reported 245,760 bytes per row prefix / 55.400 GiB at all caps.
The checkpoint projection shapes and already-frozen state-audit definitions
show wider down-projection and attention-output inputs. The corrected payload
is **417,792 bytes per row prefix / 94.181 GiB**. The original publication and
private receipts remain preserved; vector counts, scope, timings, raw inference
files and frozen criteria did not change. The tool now derives per-site widths
from safetensors headers, with four metadata-only regression checks.

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
| Forty-vector BF16 payload per row prefix | 417,792 bytes |
| Those vectors' payload at every token cap | 101,125,718,016 bytes (94.181 GiB) |
| Registered artifact budget per backend | 160 GiB |

The five site widths for each of four observed layers, repeated at both loops:

| Site | BF16 elements |
|---|---:|
| residual_after_layer | 3,072 |
| down_proj_input | 10,752 |
| down_proj_output | 3,072 |
| o_proj_input | 6,144 |
| o_proj_output | 3,072 |

This is an on-disk raw-vector payload, not a host-RAM allocation or a measured
peak. The source-only tool reads the checkpoint's header/index/config metadata;
no pretrained tensor is materialized and no archived neural state is read.

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
**18.035 MiB** maximum sampled private RAM and at least **5.166 GiB** system available RAM. It ran
under the separate 0.75-GiB private cap, 5-GiB admission and 4-GiB reserve,
then exited. Torch was never imported. The independent math process was untouched.

[Metadata regression checks](../../mechinterp/nanbeige_bpd_gym/tests/test_ram28_work_budget.py) ·
[Reproducible accounting tool](../../mechinterp/nanbeige_bpd_gym/ram28_work_budget.py) ·
[Complete arithmetic, groups and source hashes](nan_ram28_work_budget.json) ·
[Unchanged RAM28 protocol](../../experiments/nan_ram28/PROTOCOL.md) ·
[Full NF4 progress](NAN_RAM28_PROGRESS.md) ·
[Separate BF16 control](NAN_BF31_PLAN.md)
