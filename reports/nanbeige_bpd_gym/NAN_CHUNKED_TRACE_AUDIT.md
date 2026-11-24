# Independent cached-trajectory audit engineering

**Synthetic engineering and pinned-header metadata only.** RAM28 NF4 continues;
BF31 remains queued, with both frozen conditions unchanged. The prospective
cached condition is unregistered: no pretrained worker, model grade or CUDA
capacity/speedup claim. The full 1,549-baseline/216-pair scope per precision
remains required, the original 168-case ARG-03 test stays sealed, and the
continuous goal stays active.

The [writer](../../mechinterp/nanbeige_bpd_gym/chunked_gym.py) now uses
`cached-prefix-v2` / `cached-episode-v2`. No real v1 cached replies existed;
earlier sources/CI outcomes remain preserved without reinterpretation/rescoring.
Each prefix hashes its preceding complete batch context. After B2-to-B1 row
pruning, matching current own inputs alone cannot establish identical cache
histories or earlier kernel shapes. The geometry key requires independently
audited companions and every ancestor; missing/inconsistent companions also
exclude their cache descendants. An audited own response can remain an own
response while unverified geometry/control outcomes stay explicitly unknown.

Binary attention masks now use canonical half-open ranges of ones, preserving
every bit and padding hole. The checksum still covers the full decoded matrix.
This changes metadata size, not tokens, vectors, cache bytes or precision:

| Synthetic context, writer indentation | Full-mask bytes | Lossless-range bytes |
|---|---:|---:|
| B1 x 9,216 | 83,655 | 807 |
| Padded B2 x 4,608 | 83,883 | 1,076 |
| Pruned B1, physical 4,609 / semantic 4,608 | 42,192 | 851 |

These isolate mask encoding, not complete file sizes, collected coverage,
inference speedup or disk admission. Pinned headers confirm unequal physical
widths: residual/down-output/o-output 3,072, down-input 10,752, o-input 6,144;
vocabulary 166,144 and K/V 180,224 bytes/token/row. Physical vectors remain
417,792 bytes/row-prefix. Boundaries plus native-output/instrument-input add
67,584 raw bytes. At the unchanged all-cap 242,048 row-prefix scenario their
combined payload is **109.416 GiB**, excluding masks, NPY/ZIP/JSON, closed
records and filesystem overhead. This is accounting, not collected data,
a time forecast or artifact-budget qualification.

The [independent reader](../../mechinterp/nanbeige_bpd_gym/chunked_audit.py)
imports neither the producer nor Torch, NumPy, model code or an oracle. It
checks immutable checksums, scheduled source/route/row identity, pinned prompt
tokens, every generated history, bounded stopping, cache extents/masks/ancestry,
all 40 BF16 vectors and required boundaries/native/input snapshots. The bounded
ZIP/NPY reader rejects object arrays, extra members, wrong dtype/shape and
nonfinite BF16. Instrument inputs must equal physical inputs; native/all-one
outputs must equal native snapshots. Edited masks need valid shape/range and
fixed nonadaptive masks are checked exactly.

**Seven local stdlib-only tests passed**, including successful ragged pruning,
all three genuine clean/edit structures, missing companions, a self-consistent
forged companion hash, broken ancestry/masks, wrong stops/vocabulary claims,
wrong widths/nonfinite arrays and native/input/output mismatches. The Torch CI
also connects actual writer outputs to the independent reader; its scripted
tokens are fixtures, not model answers. A no-package `python -S` CI job runs
the independent tests separately.

The separate tiny byte-fixture lane used one thread, Below Normal priority,
a frozen **128 MiB self cap, 4.25 GiB start admission and 4 GiB reserve**.
Maximum sampled private RAM was **14.629 MiB**, minimum available **4.580 GiB**.
A separate header check used **14.586 MiB** and left **4.559 GiB** available.
Neither imported Torch/NumPy or read real episodes/gold; both ended. These are
samples, not continuous peaks. The original Torch-engineering/real-host
budgets remain unchanged, and the separate math process was untouched.

The [NumPy-only projection processor](../../mechinterp/nanbeige_bpd_gym/chunked_probes.py)
now accepts audited episodes and the fixed CAL-only trained/random SAE towers
and qualified IO18 T2 bank. It uses the original encoding, state-only
controller, BF16 local-write rounding, rtol2e-5/atol1e-5 mask and .03 relative-L2
write gates, recurrence metrics and packet endpoints. Its loader checks fixed
asset hashes/qualification and backend target-byte binding; no fitting occurs.
Actual asset admission, source sealing and final target receipts remain the
future host's responsibility. Cold import needs no numerical package, and
numerical execution needs NumPy without Torch.

Three additional local tests passed against the **actual unchanged FULL26
inner processor AST**, executed with the same synthetic inputs. Eight mock
episodes across native/component/static/reached routes cover 16 row-prefixes,
640 physical vectors and 18 edited phase mask/write replays. Grades, SAE/
random reconstruction and recurrence metrics, full code arrays and shadow
masks matched exactly; maximum mask/write errors were zero in these fixtures.
Positive/wrong-path story packets retain all five endpoints, and accepted,
wrong and malformed bounded packets retain original grades across all five
source stages. Wrong target binding, stale immutable files and bad controller
masks are rejected before publishing a code/score result. The unit temporary
code arrays were compared before cleanup; synthetic source and closed resource
receipts are retained. No actual dictionaries, trained checkpoints, real
episodes or gold were read. Outputs retain an explicit `MOCK_ONLY` status.

This tiny NumPy lane used the same separately registered 128-MiB cap,
4.25-GiB start / 4-GiB reserve, one thread and Below Normal priority. Maximum
sampled private RAM was **25.945 MiB**, minimum available **4.505 GiB**, with
NumPy 2.5.3 and **no Torch import**. This is not a measurement of the future
full-size dictionary host or an end-to-end memory saving. CI also exercises
these three tests in a NumPy-only job without installing Torch.

The audit cannot reconstruct full-vocabulary argmax from top-logit/normalizer
metadata, certify actual K/V contents, or replace target/stage/terminal checks.
Independent numerical CI-mask/local-write replay with original tolerances,
fixed CAL-only SAE/random tower projections, original behavioral endpoints,
the complete worker/host freeze and real matched BF16/NF4 CUDA runs remain
required. No refitting, donor transfer, necessity, faithful-reasoning or
equivalence claim; formal comparisons 0 and `REDESIGN_OR_PILOT_ONLY` remain.

[Sanitized receipts](nan_chunked_trace_audit.json) ·
[Earlier cache/BPD/head checks](NAN_CHUNKED_CACHE_ENGINEERING.md) ·
[Work accounting](NAN_RAM28_WORK_BUDGET.md) ·
[Active NF4](NAN_RAM28_PROGRESS.md) · [Frozen BF16](NAN_BF31_PLAN.md)
