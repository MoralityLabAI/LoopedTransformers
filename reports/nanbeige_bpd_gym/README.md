# Nanbeige local gym reports

The continuous Nanbeige research goal is active on the local RTX 3050 4 GB.
This is the LoopedTransformers report collection; the original 168-case ARG-03
test stays sealed. Draft [PR235](https://github.com/MoralityLabAI/LoopedTransformers/pull/235)
remains stacked on234, with no merge requested.

For a single current overview, use the [research synthesis](NAN_GYM_RESEARCH_REPORT.md)
and [delivery status and scientific gates](ROADMAP.md). Both separate completed
audits from frozen work still waiting to run; earlier failures and historical
score tables remain explicitly scoped to their original registrations.

A [180-second device observation](NAN_CTX21_DEVICE_OBSERVATION.md) measured
98.73% mean device utilization at 3,357 MiB, with NVIDIA software thermal
clock limiting active. It is a hardware-performance observation with raw
samples and counter caveats, separate from model scores or mechanistic claims.

The new [full-gym trace and batched A/B condition](NAN_FULL26_PROGRESS.md)
is frozen before inference: 1,549 baseline contexts and 216 clean/intervention
pairs per precision, followed by exact-history/shape BF16–NF4 geometry.
It follows the existing bounded context/storyworld jobs without stopping them.
All 40 physical states are retained at every generated prefix; one CPU lane
streams the fixed SAE towers and independent local-write audits alongside
the single resident model. See the [frozen protocol](../../experiments/nan_full26/PROTOCOL.md).
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
| Fresh CTX21 4K/8K challenge | Matched BF16 worker running; scores await terminal audit | All 312 4K contexts audited; DEV main 92/96, copy 94/96, erased 4/96. First 8K forward OOM; 312 outcomes missing | [Coverage/capacity stream](NAN_CTX21_PROGRESS.md) |
| Fresh STORY23 state/path traces with online BPD and SAE towers | Frozen/queued, no result yet | Frozen/queued, no result yet | [Storyworld pilot](NAN_STORY23_PROGRESS.md) |

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
