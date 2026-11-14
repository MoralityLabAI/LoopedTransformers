# AF-01 primary results, 14 November 2026

The registered funnel and resource-gain gates failed in every eligible seed.
Joint training preserved the task and provenance readouts across the full
held-out family with the original preview order. Reversing that order exposed
failures throughout the source and codec models.

This run used a newly trained shared GRU inside the isolated AF-01 experiment.
Its conclusions concern memory over a finite proof-preview interface. The
architecture, public features, and repeated codec insertion are described in
[the experiment README](../../README.md). [The frozen protocol](../../PROTOCOL.json)
sets the hypotheses and acceptance rules. Machine-readable values appear in
[metrics.json](metrics.json), and the complete retained run is in
[af01-primary.tar.gz](af01-primary.tar.gz).

## Registration and execution

Protocol version 1 was published in commit
`07dd51055e81006bd50a97ec950ada11a9014db2`. Version 2, published before neural
training, specified the generator seeds and matching on public depth for donor
interventions. Its SHA-256 is
`5d3880f00f73a2e5cb6fd733d5013889418db9b3fbf1eb08a6cc0cd67cc4b22a`.
The scoring source was published before held-out evaluation in
`a6630da632667b684bd7e99d8d587da1b3817de3`.

The fixed run used seeds 17, 29, and 43, PyTorch 2.5.1+cpu, float32, and two CPU
threads. Each source trained for 400 steps. Learned codecs and compact students
received 300 steps. All sources reached 100% task accuracy and 100% audit
accuracy for each branch over distinct train and development inputs, satisfying
the 95% eligibility threshold. The final weights were frozen in 24 checkpoint
files before the held-out process began. Their manifest SHA-256 is
`4edecadca1c2878c1f75942dcde3a23502c5bce9103a2b6f1292b52877a9ea47`.

An earlier partial training attempt was stopped during source review. Its
eligibility function weighted renamed copies unequally. The replacement
averages distinct full input patterns, as registered. The archive retains the
partial checkpoints and log under `excluded/row_weighted_qualification`.
Held-out evaluation began after that repair. Optimization settings and model
objectives stayed fixed throughout the primary run.

## What the measurements count

Each held-out family has 72 host theorems, comprising 18 distinct full input
patterns over nine ordered status pairs and two public chain depths. Metrics
weight those 18 patterns equally. Renamed theorems supply separate kernel
receipts; their repeated observations share one neural outcome.

The `depth_ood` family uses deeper certificate chains with the original preview
order. The `order_ood` family also reverses the order. The neural input carries
depth metadata and preview receipts. The Horn graph stays in the host, so
certificate depth transfer has the scope of interface memory.

Task accuracy requires the registered preferred action: the smallest valid
branch, or STOP when both branches lack a valid proof. Positive completion
accepts either valid branch and uses only provable patterns as its denominator.
Audit exact match requires both branch statuses to be recovered by the frozen
source audit head. A tainted submission is an attempted use of an external root;
the host rejects it.

All slash-separated entries below follow seed order **17 / 29 / 43**. Values
are percentages over distinct full inputs.

### Original preview order, deeper certificates

| Arm | Task accuracy | Audit exact match | Tainted attempts |
|---|---:|---:|---:|
| Teacher | 100 / 100 / 100 | 100 / 100 / 100 | 0 / 0 / 0 |
| Task codec | 100 / 100 / 100 | 33.3 / 33.3 / 33.3 | 0 / 0 / 0 |
| Reconstruction codec | 100 / 100 / 100 | 88.9 / 88.9 / 88.9 | 0 / 0 / 0 |
| Causal codec | 100 / 100 / 100 | 50.0 / 88.9 / 88.9 | 0 / 0 / 0 |
| Joint codec | 100 / 100 / 100 | 100 / 100 / 100 | 0 / 0 / 0 |
| PCA | 77.8 / 100 / 100 | 66.7 / 100 / 100 | 0 / 0 / 0 |
| Random projection | 44.4 / 22.2 / 66.7 | 11.1 / 11.1 / 33.3 | 0 / 33.3 / 0 |
| Compact transition | 100 / 100 / 100 | 77.8 / 88.9 / 100 | 0 / 0 / 0 |

The joint objective preserves the audit behavior that distinguishes dead from
tainted branches. Task success alone leaves that distinction unconstrained.
Every learned codec reached full task accuracy here. The audit readout then
separates their behavior: task-only compression reaches 33.3%, and joint
compression reaches 100% in each seed. This result concerns a fixed reader and
its downstream behavior. An independently fitted probe would answer a further
question about information recoverable by a different reader.

### Reversed preview order

| Arm | Task accuracy | Audit exact match | Tainted attempts |
|---|---:|---:|---:|
| Teacher | 55.6 / 55.6 / 55.6 | 22.2 / 33.3 / 33.3 | 22.2 / 22.2 / 22.2 |
| Task codec | 44.4 / 55.6 / 55.6 | 0 / 11.1 / 11.1 | 33.3 / 22.2 / 11.1 |
| Reconstruction codec | 44.4 / 55.6 / 55.6 | 11.1 / 22.2 / 33.3 | 33.3 / 22.2 / 22.2 |
| Causal codec | 33.3 / 55.6 / 55.6 | 33.3 / 22.2 / 22.2 | 33.3 / 22.2 / 22.2 |
| Joint codec | 44.4 / 55.6 / 55.6 | 22.2 / 33.3 / 33.3 | 33.3 / 22.2 / 22.2 |
| PCA | 55.6 / 55.6 / 55.6 | 33.3 / 33.3 / 33.3 | 22.2 / 22.2 / 22.2 |
| Random projection | 44.4 / 22.2 / 55.6 | 11.1 / 11.1 / 22.2 | 0 / 33.3 / 11.1 |
| Compact transition | 55.6 / 77.8 / 77.8 | 44.4 / 55.6 / 33.3 | 22.2 / 11.1 / 0 |

The joint-minus-task accuracy gain is exactly zero in each seed. Its audit
scores fall below the registered 95% threshold, and seed 43 also has additional
tainted submissions. The funnel gate therefore fails for all eligible seeds.
The uncompressed teacher's order-transfer failure places an additional limit on
what this source can teach through compression.

The compact transition achieves higher task accuracy in seeds 29 and 43.
Positive completion rises from the teacher's 20% to 60% in those seeds; seed 17
stays at 20%. Its provenance scores still fail the resource-gain requirement in
every seed on this family. The full gate fails despite its lower counted
arithmetic and its task improvement in two seeds.

## Interventions and finite state classes

Every arm was tested on 162 ordered donor/recipient pairs per family and seed.
Matching preserves the first-probed branch and public depth. Among these,
108 pairs change the first preview's status. Across the experiment this yields
7,776 counterfactual records, with 5,184 status-changing interventions.

The inserted prefix state agrees exactly with direct execution of the composed
history for every arm: task and audit agreement are 100%, and maximum logit
replay error is zero. This checks that the intervention resumes the intended
recurrence. Its semantic scores follow the clean results above. In particular,
joint compression reaches 100% task and audit on changed-prefix trials in
`depth_ood`; its order-transfer scores remain impaired.

The causal objective produces depth-family one-step closure MSE between
0.00273 and 0.00434. Joint compression lies between 0.00686 and 0.01354. The
audit outcomes above show why the declared readouts belong in the preservation
criterion alongside a closure residual. These secondary MSE diagnostics average
the raw 72-case batches. This differs from the exact-pattern averaging used for
the primary task, audit, and gate metrics. The MSE values serve as recorded
diagnostics; future comparisons should apply the same pattern weights to them.

Byte-identical float32 boundary codes with differing task/audit target signatures
occur zero times for every arm in the finite final-boundary check, including
random projections. Each code can remain distinct even when its readout produces
the wrong answer. The collision test
therefore has a narrower scope than successful task and audit continuation.

The exact reference computation partitions nine decision states at one fixed
public stratum into four task-continuation classes and nine joint task/audit
classes. Fixed-length class identifiers require two and four bits respectively.
The weaker preferred-action code has three classes. A retained witness shares
the original host target, visible goal, and preferred action while a submission
on the other branch distinguishes dead from tainted provenance. See
[exact_quotient.json](exact_quotient.json) for the exhaustive finite receipt.

## Arithmetic, timing, and memory

| Model | Stored trainable or frozen parameters | Dense matrix MACs per episode | Boundary coordinates |
|---|---:|---:|---:|
| Teacher | 4,425 | 23,904 | 32 |
| Teacher plus learned codec | 4,717 | 25,440 | 4 |
| Compact transition | 637 | 1,992 | 4 |

MACs include all recurrent calls, each visit's decoder, and the final task and
audit heads. The compact transition uses one twelfth of the teacher's counted
matrix arithmetic. The learned codec adds 1,536 MACs per episode because it
retains the source recurrence.

Four float32 coordinates hold 128 boundary bits. The codec executes through a
decoded 32-coordinate working state. This boundary budget and the exact
finite-class bit counts answer different accounting questions. Runtime storage
also includes parameters and intermediate tensors.

Total measured fitting time was 24.40 seconds on the local CPU run. Forward
timing uses one warm-up followed by one timed block of three full-batch forward
passes over 72 episodes.
Per-case batch averages span 4.02–6.60 microseconds for the teacher and
3.73–5.22 microseconds for the compact transition. The compact model is slower
in one of the six paired seed/family measurements. These short measurements
leave a measured speedup claim unresolved; the 12-fold ratio concerns dense
matrix arithmetic. Fitting, checkpoint loading, environment work, and Lean are
outside those forward measurements.

## Lean and receipt verification

The evaluator retained 4,320 original-host decisions across all clean arms and
controls. Independent host reconstruction accepted 1,262 rows. Their selected
proofs reduce to 96 unique declarations, all checked by Lean 4.19.0 with empty
axiom dependency lists. The host rejected 1,128 submissions, including 569
tainted attempts. The remaining 1,930 rows chose STOP. False host acceptances
were zero.

The fixed Lean-native reference solved all 80 provable theorems among 144 hosts,
with zero timeouts. The other 64 hosts lack a valid branch. Its separate compiler
process per theorem took 16.13 seconds in total. Batched checking of selected
proofs took 0.324 seconds. Different compilation policies and theorem access
make these separate cost measurements.

Kernel guards accepted the positive control and rejected an unauthorized root
and a changed target. The Lean executable SHA-256 was
`92c3d35b5bfaa5e0fea413a775d504cf46cd95e1345df61c2274f76779e7e023`.
The independent audit receipt is retained with the complete run and is required
by the CI workflow. It independently recomputes all primary outcome arithmetic,
resource counts, and registered gate decisions, then binds the original-host
proof sources and compiler logs. Train/development eligibility is checked from
the frozen metrics. Raw activation tensors, collision diagnostics, and elapsed
times remain evaluator records with that stated scope.

All 51 unit tests passed with actual Lean available, both normally and under
`python -O`. The compact [verification record](verification.json) binds the
full audit receipt and retained test logs. The archive includes the exact source
snapshot used for replay, every checkpoint, raw outcomes, counterfactuals,
generated Lean sources, compiler logs, and the excluded training attempt.

## Follow-up scope

A subsequent registration can test whether training across preview orders
supports a source that meets task and audit thresholds on that variation. The
existing `Trace` interface also provides a path to instrument the actual
TypedLoop recurrence. Reversible-layer work would add inverse-error and
semantic-cycle checks under that architecture. AF-01 fixes the observable
contract, preserves the failed transfer results, and supplies proof-bound
receipts for the measured behavior.
