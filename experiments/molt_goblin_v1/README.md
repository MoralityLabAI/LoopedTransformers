# MOLT-GOBLIN-01: strict-prefix recurrence pilot

Status: executable development fixture with a proposed path to verified research tasks.
The companion repositories use `research/molt-goblin-01-20261126`.

[Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923) motivates the prefix-memory
interface. The construction here uses one small tied PyTorch block with a zero null slot.
Its current training objective supervises the final loop only. Upstream MoLT also has
current-loop depth attention and a distinct initial pass; its loop-forcing objective gives
positive loss weight to every loop. Reproducing that system requires a separately identified arm.

## Questions and implementation boundary

| Question | Observable result | Present support |
| --- | --- | --- |
| Does the prefix recurrence implement its dependency contract? | Causality, finite stabilization and sequential parity | Executable tests |
| Does recurrence improve learned task performance for a fixed budget? | Paired verified correctness with the training and evaluation costs recorded | Small prefix-sum pilot; broader comparison proposed |
| Does retained memory save useful work as the RAM cap rises? | Verified success at fixed CPU quota, with cache and recomputation costs measured | Proposed cache adapter |
| Do selected BPD components causally improve memory use? | Recovery and clean-case preservation beyond physically matched controls | Proposed BPD protocol |

Passing the first row establishes an implementation property. Each later row needs its own
task competence check and evidence. Keep the exact scalar oracle, learned task results and
mechanistic intervention results in separate report sections.

## 1. Existing executable fixture

`pilot.py` defines a width-32 model over seven symbols, with one shared attention/MLP block
and a shared prediction head. Input sequences contain independent symbols; targets are their
cumulative sums modulo seven. Training uses batches of 32 sequences of length eight.

| Arm | Query on pass r+1 | Memory on pass r+1 | Passes |
| --- | --- | --- | --- |
| `prefix_kv` | Fixed input embedding | Previous-pass states at j < i | Requested loop count |
| `residual` | Previous state at the same position | Previous-pass states at j < i | Requested loop count |
| `one_pass` | Fixed input embedding | Input embeddings at j < i | One |

The first attention slot is a fixed zero key/value, which gives position zero a defined
softmax. The residual arm has a same-position path through its query. Only the prefix arm
has the strict triangular recurrence used by the finite-stabilization argument.

```sh
python -m pytest -q experiments/molt_goblin_v1/test_pilot.py
python -m experiments.molt_goblin_v1.pilot --steps 64 --loops 4 --threads 2 --seed 31
```

Use an external operator hard-cap wrapper. The CLI limits steps and threads but does not
enforce a process memory limit or CPU quota. CUDA is opt-in and the fixture downloads no weights.

The tests exercise prefix/sequential parity, future-token exclusion and finite gradients.
`sequential()` recomputes prefix projections at every position; its current implementation
serves as a correctness reference. An optimized decoding claim needs a separate append-only
projected-KV implementation, parity checks and measured timing.

Evaluation uses 64 examples at each of lengths eight and sixteen, at requested depths
one, the training loop count and sequence length. All depths for a given length see the
same examples. For `one_pass`, every requested depth still executes one block. The report's
`loops` field describes the request; readers must retain this distinction when plotting it.
For the prefix arm, `stabilization_max_abs` always compares a length-pass run with
`sequential()`, regardless of the row's requested depth. That extra check runs after the
timed forward call and is excluded from `elapsed_s`. Equal requested depths can produce
duplicate rows, such as training depth eight at evaluation length eight; deduplicate them
for reporting and keep their shared task identity.

## 2. Make the compute comparison explicit

With S updates, B examples per batch, T tokens per example and R loops:

```text
processed training tokens = S * B * T
forward block calls       = S * R
token-block visits        = S * B * T * R
```

Here `train_block_visits` in the existing JSON counts forward block calls for a whole batch.
It excludes backward work, optimizer work and evaluation. Attention cost also depends on
sequence length, so these counts alone cannot establish equal FLOPs or equal wall time.

For the illustrative S=256, B=32, T=8, R=4 configuration:

| Arm or proposed budget | Updates | Processed tokens | Forward block calls | Token-block visits |
| --- | ---: | ---: | ---: | ---: |
| Prefix-KV | 256 | 65,536 | 1,024 | 262,144 |
| Residual recurrence | 256 | 65,536 | 1,024 | 262,144 |
| Existing one-pass control | 256 | 65,536 | 256 | 65,536 |
| Proposed one-pass reinvestment control | 1,024 | 262,144 | 1,024 | 262,144 |

The first two arms pair initialization, batches, updates, optimizer and forward block calls.
The reinvestment control asks whether spending the saved recurrence work on extra training
improves the result. It changes data exposure and optimizer updates, which belong in the report.
The current CLI runs only the first three rows at the same S; the last row needs a study driver.

Register each budget analysis before opening its evaluation panel:

| Budget axis | Held fixed | Costs that remain visible |
| --- | --- | --- |
| Equal tokens | Processed non-padding training tokens | Additional loop work, optimizer updates and runtime |
| Equal estimated compute | A declared operation-count model | Kernel efficiency, memory traffic and verifier work |
| Equal measured time | Same machine, CPU quota, warmup and stopping clock | Actual tokens, updates, cache traffic and completed tasks |
| Equal verifier calls | Task roots and all charged calls | Generation, search and memory costs |

A timeout, malformed action or repeated proof attempt still consumes its declared budget.
Keep every arm and failure in the receipt. Hardware-specific measurements for RTX 3050
and RTX 5080 require separate runs and reports.

## 3. Study sequence

### P0: establish the computation

Replay the present tests and a bounded CLI smoke. Record commit, dependencies, precision,
seed, device and cap configuration. The exit condition is a functioning report with finite
values and parity within the registered floating-point tolerance. Preserve an explicit
`development_only` status regardless of task accuracy.

The next implementation checks should cover padding masks, prefix offsets, frozen-epoch
gating and append-only projected-KV parity. A malformed dependency or cache version must
fail before model evaluation. Keep any stochastic evaluation behavior fixed when testing
the deterministic recurrence.

### P1: establish learned competence

Start with fresh development task roots. Freeze a task-specific competence floor after
calibration and record it with the exact task generator. Inspect position-wise and whole
sequence performance so an easy early token cannot conceal later failure.

Use at least eight independent training seeds to discover development variance, as a pilot
design choice. Confirmation size comes from the observed paired signal, matched shams and
the registered effect floor through EvalMath. Repeated evaluation of one checkpoint remains
one checkpoint for claims about training reproducibility.

Add an explicit loss-schedule ablation before attributing results to upstream loop forcing:

```text
L_last = CE(readout(h[R]), y)
L_all  = sum_r w[r] * CE(readout(h[r]), y), with w[r] > 0 and sum_r w[r] = 1
```

`L_last` is implemented. `L_all` is proposed. Compare the schedule within each architecture,
then pair it across architectures. Preserve the recorded cross-loop detach policy and
supervision clock; detaching a cached tensor changes the gradient computation.

### P2: test recurrence under competing uses of the budget

Run the paired prefix/residual comparison first. Add the reinvested one-pass control and a
parameter-accounted untied-depth control before drawing an efficiency conclusion. A full
causal-attention arm and an upstream reproduction answer additional architectural questions;
introduce them with separate hypotheses and include them in the comparison count.

For each arm, compare finite unrolling with its supported inference schedule. Measure
extrapolation across task depth and input length. The existing fixture's homogeneous
transition starts from h[0]; upstream MoLT has a special first pass, so its stabilization
clock must be mapped explicitly before comparing bounds.

### P3: connect to verified research episodes

Build the DataGoblin adapter around a common task packet, strict action parser and external
verifier. Begin with frozen checkpoints so dataset changes can be audited before learning
changes. Add ordinary adapter training on verified TRAIN traces, then compare BPD-guided
training against an equal-budget ordinary-training control.

| Target | Episode must require | Primary outcome |
| --- | --- | --- |
| LBF / Lean search | Retain and revisit candidates under actual kernel calls | Verified closure within budget |
| Typed lattice chains | Propose refinements with evidence checked outside the model | Sound useful refinements and false authorization rate |
| Delayed memory tasks | Reuse an earlier fact after distractors or a local change | Verified recovery at controlled memory distance |
| Counterfactual repair | Distinguish edits with similar surface form and different consequences | Correct minimal repair with a valid witness |

The cumulative-sum task remains a plumbing fixture. Its exact solver is a structural ceiling.
Broader tasks need fresh world families and an independently specified verifier.

## 4. RAM as an experimental variable

The proposed sweep uses 2, 4, 8 and 16 GiB total process-tree envelopes at a fixed CPU quota and
thread count. Each cell must fit within its host's existing operator limits; this document
does not amend DataGoblin's registered 2,048 MiB mock envelope or authorize larger runs.
Fix model size, batch schedule, context limit and verifier-call budget while varying the
capacity available to the same registered cache policy.

For dense keys and values, a useful planning term is
`2 * bytes_per_element * batch * layers * tokens * kv_width`.
Add weights, optimizer state, activations, component dictionaries, receipt indexes and allocator
overhead to obtain the process-tree requirement. Include child workers, verifier/checker and
reporting processes, pinned buffers and both live epoch banks. Increasing the cap alone leaves memory use unchanged;
the receipt must state which additional entries the policy retained.

Separate cache contents by validity rules:

| Stored content | Required identity | Reuse rule |
| --- | --- | --- |
| Frozen-model K/V | Checkpoint, input, position, graph and epoch | Reuse only for compatible dependencies |
| BPD basis and component coefficients | Checkpoint, fitted TRAIN source, layer and basis hash | Recompute affected projections when any binding changes |
| Verifier receipts | Task, action, checker and environment hashes | Reuse only if the verifier's contract permits it |
| TRAIN gradients or live activations | Current computation graph and optimizer boundary | Keep the declared detach/checkpoint policy; cross-update reuse is a separate approximation |

A residency change may preserve the computation when a missing entry is recomputed exactly.
Dropping a semantic dependency changes the transition and starts a new version or epoch.
Pass-dependent eviction or gating cannot silently enter the fixed-map stabilization claim.

Monotone public-state updates stay within a fixed problem frame. If an external source change
retracts a premise or changes the candidate universe, the environment opens a new frame and
rebuilds its initial lattice from valid assumptions. Cache invalidation alone cannot undo an
earlier elimination. Link frame versions to the same task root, preserve spent budget and
recheck any reused certificate against the new premises.

Compare retained memory against fresh recomputation and random eviction at the same cap.
Measure total RSS, peak device allocation, retained bytes, hit rate, invalidations, bytes moved,
recomputation, runtime and verified yield. A larger cache that causes more scanning or transfer
work can lose under the fixed CPU budget; retain that outcome.

## 5. Receipts and inference

The current script emits aggregate token/sequence accuracy, timing, training tokens,
forward block calls and peak CUDA allocation during training. It does not yet emit a
root-level research receipt, peak process RSS or actual FLOPs.

The episode adapter should write one row for every case/arm combination and a run manifest:

| Record | Required fields |
| --- | --- |
| Task provenance | task_id, root_id, family, split, generator_digest, input_digest, verifier_digest, checker_version |
| Frame and epoch | frame_id, problem_version, parent_frame_id, premise_digest, epoch_id, public_state_digest |
| Model provenance | checkpoint_digest, architecture, training_seed, data_manifest_digest, loss_schedule, detach_policy |
| Episode | actions, all verifier outcomes, stop_reason, verified_success, corruption, budget_exhausted |
| Cost | processed_tokens, forward_calls, updates, verifier_calls, wall_s, cpu_s, peak_rss_bytes, device_bytes |
| Memory | cache_policy_hash, epoch, graph_version, retained_bytes, hits, misses, invalidations, recompute_work |
| Evidence | primary_contrast, independent_unit, comparison_count, selection_exposure, effect_floor, verdict |

These are proposed adapter fields. Content digests bind canonical bytes; `problem_version`
and `checker_version` are separate version labels. Existing fixture output keys remain intact.

Choose the independent unit to match the claim. World variants belong to their source root;
loop depths and intervention doses are repeated measures. For a fixed checkpoint, task-root
inference is conditional on that checkpoint. Claims across trained models need independent
training seeds and a plan for the crossed model/task structure, rather than pooling every
seed-by-case row as independent.

Follow the [EvalMath mechanistic-power skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power)
and [FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md).
Freeze the meaningful effect, matched physical dose, sham family, comparisons and readout before
confirmation. Export paired correctness through `research_gym/scripts/evalmath_fsi_gate.py`
where its endpoint assumptions apply. Continuous and crossed-model effects need their declared
analysis. Historical sealed cohorts, including ARG-03-N, remain outside this development work.

## 6. Repository handoffs and stopping decisions

| Repository | Owns | Next reviewable deliverable |
| --- | --- | --- |
| [DataGoblin #82](https://github.com/MoralityLabAI/DataGoblin/pull/82) | Task generation, independent labels, TRAIN exports | Episode adapter and raw outcome contract |
| [HybridTRMLDT #33](https://github.com/MoralityLabAI/HybridTRMLDT/pull/33) | Public-state membrane and prefix-memory epochs | Verifier-bound epoch adapter with invalidation tests |
| [LoopedTransformers #398](https://github.com/MoralityLabAI/LoopedTransformers/pull/398) | Paired models and runtime accounting | Study driver with reinvested baseline and explicit loss schedule |
| [BPD #163](https://github.com/MoralityLabAI/BPD/pull/163) | Component interventions and physical shams | Frozen development intervention manifest |
| [Compactification-Theory #307](https://github.com/MoralityLabAI/Compactification-Theory/pull/307) | Dependency and quotient obligations | Formalized recurrence and certified readout-preservation theorem |

Stop an architecture claim when competence remains below its frozen floor. A saturated
specialized task calls for a fresh discriminator. A memory improvement requires measured
useful work saved under the declared budget, and BPD proceeds only after full-head no-op
parity and dose calibration. Development selection stays recorded as development; a future
confirmation requires a prospectively registered and authorized readout.
