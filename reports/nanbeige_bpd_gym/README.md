# Nanbeige local gym reports

The continuous Nanbeige research goal is active on the local RTX 3050 4 GB.
This is the LoopedTransformers report collection; the original 168-case ARG-03
test stays sealed. Draft [PR235](https://github.com/MoralityLabAI/LoopedTransformers/pull/235)
remains stacked on234, with no merge requested.

**Host-RAM policy update (2026-11-21):** the RAM-heavy CTX21 BF16 worker was interrupted after 378 closed contexts, recovering system free RAM from0.6 to5.9 GiB. Its original terminal audit failed before a normal receipt; no BF16 grade or receipt was fabricated. Earlier unstarted model/host supervisors are deferred. The separate [RAM28 NF4 full gym](NAN_RAM28_PROGRESS.md) retains all1,549 baseline trajectories and216 batched A/B pairs with a file-backed loader and host-memory guards. Substantial SAE audits follow GPU termination. [Resource protocol](../../experiments/nan_ram28/PROTOCOL.md).

The [full BF16 control](NAN_BF31_PLAN.md) is frozen with storage-backed decoder stages, bounded host RAM and authenticated predecessor waits. It retains all 1,549 baselines and 216 true A/B pairs; real BF16 CUDA results remain pending.

The [full-roster work and trace accounting](NAN_RAM28_WORK_BUDGET.md) binds all token-cap allowances, physical-state payload and closed NF4 probe timings. The corrected payload uses the checkpoint's unequal physical site widths; real completion/grades are still required.

A [measured NF4 runtime window](NAN_RAM28_RUNTIME_OBSERVATION.md) now verifies the passed CUDA preflight, resident GPU decoder/head,92.31% sampled mean device activity and at least 6.031 GiB available RAM during initial trace generation. Independent episode/SAE grades remain pending.

The separate [CTX21 closed-prefix audit](NAN_CTX21_RESOURCE29_RESULTS.md) now
verifies all 378 preserved BF16 cases and 312 matched 4K prefixes. DEV loop2
main accuracy is BF16 90/96 versus NF4 92/96 across three profiles of the same
16 roots. All 96 accuracy rows, 48 paired rows and 576 geometry scalars are
published. The CPU audit used 194.1 MiB maximum sampled private RAM and left
at least 6.467 GiB available while NF4 inference continued. The original BF16
terminal gate remains failed, both final target hashes remain unavailable,
and 8K has no NF4 pair. These are descriptive exposed-development results.

A separate [early storyworld audit](STORYAUDIT30_PLAN.md) is frozen for the
entire first 72-trajectory story block, with the original five behavioral
endpoints and exact state/history checks. Its small metadata follower waits on
authenticated handles; it has not loaded another model or started grading.
Eight local tests passed. The one-shot CPU audit has a 0.75 GiB private cap
and 4 GiB system reserve; substantial SAE/BPD analysis stays after GPU exit.

The [unchanged story-audit main runner also passed a synthetic integration
check](NAN_STORYAUDIT30_ENGINEERING.md): six mock episodes, 261 prefixes,
10,440 synthetic vectors, explicit malformed/wrong-world outcomes and 66
unknown slots. The full check used 132 MiB maximum sampled private RAM.
This is engineering evidence, separate from real model scores.

The [storage-backed BF16 assembler CPU integration](LAZY_BF16_BACKEND_INTEGRATION.md) now passes the pinned small-class capture/BPD checks. It is engineering evidence only and preceded the separately frozen BF31 control.

The [future chunked-cache/BPD/head CPU engineering](NAN_CHUNKED_CACHE_ENGINEERING.md) now passes the actual small-class and hook checks. It is MOCK_ONLY, unregistered for pretrained use; the current NF4 run and queued BF31 control are unchanged.

The same prototype now includes a prospective meta-initialized NF4 assembler
and full cached trajectory writer, including complete companion-history/cache
metadata and failed-batch preservation. Its new CI contracts are distinct from
the earlier numerical receipts; local installed-bnb construction checks obey
RAM admission before importing Torch. No cached pretrained worker has started.

A [BF16 storage prototype](LAZY_BF16_STORAGE_ENGINEERING.md) passed eight
CPU contracts and matched the original BF16 ordered checksum while all 201
pinned parameters stayed on meta. It used 708 MiB maximum sampled private
RAM and initialized no CUDA. This is preparation for a separately frozen
control, not a BF16 inference/capacity result; BF16 remains deferred and the
current NF4 model continues.

The [random small-class lazy-loop integration](LAZY_BF16_LOOP_INTEGRATION.md)
also passed native B1/B2 state/readout byte checks and all four actual BPD
hook routes on CPU. Its initial missing-RoPE-buffer failure is preserved.
The test reproduced an unused legacy observer list growing by two vectors
per native capture, and checked a candidate cleanup without changing exported
states or logits. The frozen live worker is untouched; any repair requires
its own future registration. Four additional regression tests passed.

For a single current overview, use the [research synthesis](NAN_GYM_RESEARCH_REPORT.md)
and [delivery status and scientific gates](ROADMAP.md). Both separate completed
audits from frozen work still waiting to run; earlier failures and historical
score tables remain explicitly scoped to their original registrations.

A [180-second device observation](NAN_CTX21_DEVICE_OBSERVATION.md) measured
98.73% mean device utilization at 3,357 MiB, with NVIDIA software thermal
clock limiting active. It is a hardware-performance observation with raw
samples and counter caveats, separate from model scores or mechanistic claims.

A separate [complete CTX21 comparison presentation](CTX21_MATCHED_PRESENTATION.md)
was prepared before the BF16 interruption. Its metadata follower correctly
ended unrun when the original terminal audit failed. The new RESOURCE29 report
uses separate stopped-prefix provenance; it does not bypass that reader's guard.

The new [full-gym trace and batched A/B condition](NAN_FULL26_PROGRESS.md)
is frozen before inference: 1,549 baseline contexts and 216 clean/intervention
pairs per precision, followed by exact-history/shape BF16–NF4 geometry.
Its original unstarted supervisor is now deferred under the host-RAM policy;
RAM28 executes the scope as a separate frozen NF4 resource condition.
All 40 physical states are retained at every generated prefix; one CPU lane
was registered to stream the fixed SAE towers and independent local-write
audits alongside the model. RAM28 instead serializes the substantial host
analysis after GPU termination. See the [original protocol](../../experiments/nan_full26/PROTOCOL.md).
An additional [actual pinned-BPD dispatch check](NAN_FULL26_ENGINEERING.md)
passed24 synthetic physical calls, including clean-row/padding closure and
independent controller/local-write replay, without changing frozen sources.
The same [engineering report](NAN_FULL26_ENGINEERING.md#closed-artifact-host-path)
also verifies the actual frozen host/report path on 16 synthetic episodes,
including exact-history/shape exclusions and unequal-length root weighting.

Start with the [matched accuracy/dose/memory figure and tables](NAN_FIT16_IO18_RESULTS.md),
[complete SAE/factor comparison](NAN_XREF25_RESULTS.md), and
[current continuation evidence index](CONTINUATION_INDEX.md).
The index binds nine later stages14–22 to a named immutable snapshot; the
storyworld and separate repair reports below extend that scope explicitly.

| Condition | BF16 | NF4 | Evidence |
|---|---|---|---|
| COMP14 fresh two-world competence | DEV loop2 32/32 under each recipe | 29/32 under each recipe | [Matched results and recurrence](NAN_COMP14_RESULTS.md) |
| IO18 actual CAL-only fits | T1/T2 each16 updates, audited | T1/T2 each16 updates, audited | [Full route/fidelity tables](NAN_FIT16_IO18_RESULTS.md) |
| IO18 T2 DEV loop2 routes | All five routes32/32, including static | Native29/32, clean/reached25/32, component/static26/32 | [Verified figure](figures/nan_io18_matched_routes.png) |
| Separate XREF25 cache audit | Both depths passed1706 vectors/cell | Both depths passed1706 vectors/cell | [All layers, strata and controls](NAN_XREF25_RESULTS.md) |
| Official local Prime20 | 96/96 audited; main32/32, copy32/32, erased2/32 | 96/96 audited; main29/32, copy32/32, erased2/32 | [Matched official/native comparison](NAN_PRIME20_RESULTS.md) |
| Fresh CTX21 4K/8K challenge | Separate audit verifies 378 closed cases; DEV 4K main90/96, copy91/96, erased2/96. Original terminal gate remains failed; 246 missing | All312 4K contexts audited; DEV main92/96, copy94/96, erased4/96; first8K OOM,312 missing | [Separate stopped-prefix comparison](NAN_CTX21_RESOURCE29_RESULTS.md) · [Original terminal stream](NAN_CTX21_PROGRESS.md) |
| Fresh STORY23 state/path traces with online BPD and SAE towers | Frozen/deferred under RAM policy; no result yet | Frozen/deferred under RAM policy; no result yet | [Storyworld pilot](NAN_STORY23_PROGRESS.md) |

A [bounded MLP workspace candidate](NAN_MLP27_CPU_ENGINEERING.md) passed 12
synthetic CPU cases after the observed 8K allocation failure. It preserves
one physical down-projection hook per layer. It has not been used for a
pretrained forward or substituted into any frozen model condition.
The separately [frozen conditional MLP27 full gym](NAN_MLP27_PROGRESS.md)
retains all 1,549 baseline contexts and 216 A/B pairs per precision. It follows
actual FULL26 termination and runs both precisions fresh only if the closed
FULL26 receipts establish a CUDA/input-capacity failure. Main/copy/erased and
difficulty strata remain separate; every missing slot stays explicit. Ten
source contracts and the real new preflight/host/report path passed synthetic
CPU checks, including 22 closed episodes and all 3,962 trajectory coverage
slots. No MLP27 pretrained forward has run. See its
[protocol](../../experiments/nan_mlp27/PROTOCOL.md) and
[source seal](../../experiments/nan_mlp27/SOURCE_MANIFEST.json).

The SAE/factor presentation retains all 48 layer/dataset/depth range rows,
3,528 scalar strata, all 12,288 labelled physical weight-geometry entries and all72
registered repeated geometry-case bindings. The 12 labelled basis groups have 10 numerical matrix fingerprints; initial SVD controls also repeat across fitted depths. Some trained-SAE reconstruction
improvements are negative, and only 34,528 of 516,096 correlation entries are
defined. Dictionaries, roots, worlds, loops, layers and features repeat;
these counts are not independent mechanism replications. Every display row
was independently reconciled to the CSVs in the
[verification receipt](nan_xref25_results_verification.json). Full activity
vectors and correlation matrices remain in the linked immutable aggregate.

Original algebraic/FIT07 competence failures remain failed; later pointer
success does not open those gates. Original FIT16 NF4 writer failures retain
two real updates each. Original XREF19 dimension failures and XREF24 strict
mixed-precision geometry failures stay preserved; XREF25 is a distinct
three-transpose/one-FP64-cast audit, with original tolerances unchanged.
Its [supervision amendment](../../experiments/nan_xref25/SUPERVISION_AMENDMENT.json)
replaces only the verified idle old host supervisor and leaves all model/math
workers intact. Earlier partial official SDK runs retain their missing grades;
no response or loss was reconstructed.

- [Full research synthesis](NAN_GYM_RESEARCH_REPORT.md)
- [Objective, original phases and remaining scientific requirements](ROADMAP.md)
- [Historical nine-study benchmark index](BENCHMARK_INDEX.md)
- [Historical native wider-suite results](NAN_GYM05_NATIVE_REPORT.md)
- [Historical matched context recurrence and fixed tower probes](NAN_CTX08_COMPARISON.md)
- [Earlier learned rank/site extension](NAN_RANKSITE13_RESULTS.md)
- [Separate strict cache-audit provenance](NAN_XREF25_XREF19_PROGRESS.md)
- [Future long-context tower admission and original analysis](NAN_XREF22_PROGRESS.md)
- [Separate future long-context audit](NAN_XREF25_XREF22_PROGRESS.md)

One large model and one substantial host analysis maximum, one numerical CPU
thread each, Below Normal priority. The math process is unchanged. Live
supervision uses authenticated health/process handles/append-only journals;
no ordinary reads of live atomic progress or SDK trace files. Frozen methods,
source pins, budgets, failure gates and missing denominators remain intact.
Raw replies, tokens, hidden states, learned checkpoints and per-record grades
stay local in ignored runs. Synthetic public sources/host oracles and audited
aggregate reports have separate roles. This development evidence supplies no
calibrated causal/sham floor or confirmation power, donor transfer, isolated
adaptive-routing advantage or equivalence conclusion.
