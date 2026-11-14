# FC-07: misleading-ancestry learned Horn proof selection

**First frozen experiment completed November 13, 2026.**
The [CI run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43422916239)
passed both train-freeze and evaluate-heldout jobs,
including seven train-only tests and Lean compilation
of **352 model-selected positive theorem terms**.

The first test result meets the preregistered **learning**
criterion against the static backward-goal-ancestry
control in both depths for both seeds. The distinct
preregistered **CPU-cost** criterion did not pass.

## Input and model provenance

- Registration frozen before either source generation
  or model evaluation:
  `d3216371ae480b7e1b42cb913a33847ed5ad47c2`.
- TRAIN-ONLY source checkout
  `f470a97502b853addf2b0f67c9f6225001074a90`,
  source hash
  `f725c31a18aa84b2b831b90d8d2f6ca62c2d20d40530b8f8284429d8c80e1359`.
- Frozen test generator
  `6cee4fdba91ba9c13ad47c492a127c7dc1884090`,
  source hash
  `f7481ed2f489c43eb95fe74b1544aad3ef59898b29d10fd1455874ad990d4cdb`.
- The training job checked out only the early
  source revision and asserted the absence
  of the test generator. It uploaded
  SHA256-pinned network weights and θ
  values before the test job read the new
  source and scored any held-out labels.

Two seeds trained 11,521-parameter LR-01
four-pass gating models on FC-02 train,
and both calibrated to θ=0.55 using FC-02
development only. Separate copied LR-01
actors with 4,705-parameter rule heads
(16,226 parameters per actor) were jointly
fine-tuned on **1,149 FC-07 training-only
positive proof-prefix states**. A first
128-prefix in-sample diagnostic yielded
100% for both seeds, excluded from
generalization claims.

## Main proof-selection scores

Every valid held-out graph admits a proof of
at most five actions, but extra legal
distractor rules share real goal ancestors.
The independent verifier accepts only
complete, valid five-rule source traces.
The same accepted action traces are
compiled into Lean 4.19.0 theorems.

| Seed | Test | Learned positive /96 | Backward slice /96 | Goal-first /96 | Random legal /96 | Exact negative /96 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | same-depth | **94** | 47 | 65 | 41 | 96 |
| 17 | deeper | **85** | 22 | 29 | 16 | 94 |
| 29 | same-depth | **93** | 47 | 65 | 41 | 95 |
| 29 | deeper | **80** | 21 | 28 | 15 | 94 |

Seed29's deeper model received 85 positive
eligibility signals from the frozen FC-04
gate, and proved 80. All other conditions
had 96 positive-eligible cases.
Across the four seed/split conditions,
the learned head produced **352 positive**
kernel-compiled terms and **379 negative**
exact Boolean certificates. It abstained
on **37** seed-case evaluations and
issued **zero false finite formal certificates**.
The positive count contains repeated
underlying problems across seeds.

Paired learned-only/baseline-only positive
certificate comparisons versus the
preregistered backward slice were
47/0, 65/2, 46/0 and 60/1 in
the same four condition order.
The registered *learning criterion*
therefore passed, with greater positive
certificate recall at both depths
for both trained seeds.

## Cost result

The learned actor consumes four extra
recurrent cell passes per proposed action.
Backward slicing is a zero-neural
graph traversal, with the same frozen
gate cost as the learner.

| Seed/test | Learned cells | Backward cells | Learned CPU ms / case | Backward CPU ms / case |
| --- | ---: | ---: | ---: | ---: |
| 17 / same | 2,198 | 1,026 | 8.173 | 3.731 |
| 17 / deep | 2,886 | 1,126 | 10.824 | 4.142 |
| 29 / same | 2,214 | 1,042 | 8.311 | 3.819 |
| 29 / deep | 2,712 | 1,204 | 10.267 | 4.470 |

Measured total CPU time per certified
positive was **16.69 vs 15.24 ms** for
seed17 same-depth and **17.16 vs
15.60 ms** for seed29 same-depth;
backward slice was cheaper.
On the deep set, the learned actor
was more cost-effective per verified
positive: **24.45 vs 36.15 ms**
(seed17), **24.64 vs 40.87 ms**
(seed29). The preregistered
compute criterion required a
consistent lower cost on all
four conditions and **failed**.

CI recorded all source-family results,
paired outcomes, original failed
traces and first fault reasons,
actual model and verifier time,
and neural cell invocations in
`FC07_report.json`. This is one
GitHub Actions CPU runner, with
no device-normalized timing claims.

## Post-hoc unary-path control

A subsequent [independent structural audit](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc07-ancestral-decoys/research/refutation_bench/fc07/POSTHOC_UNARY.md)
found a zero-parameter solution to all
192 positive source cases: restrict
the rules to **one-premise implications**
and run a shortest-path BFS from
initial facts to the goal. Source
alternative goal producers always
use conjunctions, so the source's
canonical goal backbone remains
the unique unary path.

[Post-hoc source CI #43424550518](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43424550518)
passed four tests and Lean-compiled
all 192 resulting positive theorem
terms with zero false certificates.
That method is a stronger symbolic
control than the originally frozen
backward-slice heuristic and costs
no neural recurrent cells.

It was discovered *after* the
registered first-run comparison,
whose results and hashes remain
unchanged. The trained LR-01 actor
therefore demonstrates transfer
beyond the preregistered static
backward slice, with **no established
advantage over the new unary-path
symbolic method**. A future
experiment must preregister unary
BFS and make the shortest useful
proof require actual conjunctions,
with competing alternative proofs
and shared subgoals.

## Reproduce

Pin separate public source checkouts
at the train and test commits above.
With Python 3.11 and CPU PyTorch 2.5.1,
from LoopedTransformers repository root:

~~~sh
export FC07_TRAIN_ROOT="$(realpath ../Compactification-FC07-train)"
python -m unittest discover -s experiments/refutation_transfer \
  -p 'test_fc07_ancestry.py' -v

python experiments/refutation_transfer/fc07_ancestry.py train \
  --source "$FC07_TRAIN_ROOT" --out /tmp/fc07_weights

python experiments/refutation_transfer/fc07_ancestry.py eval \
  --source "$(realpath ../Compactification-FC07-test)" \
  --weights /tmp/fc07_weights \
  --out /tmp/fc07_scores

lean /tmp/fc07_scores/FC07_model_proofs.lean
~~~

The GitHub Actions two-job workflow
enforces the stronger train/test
source-isolation protocol and uploads
`fc07-frozen-weights` and
`fc07-ancestry-43422916239` artifacts
(30-day retention). The test artifact
contains 16 complete per-case JSONL
prediction files, the generated Lean
theorems and kernel compilation log.

The validated result concerns finite
Horn proof dependencies. The exact
symbolic shortest-proof oracle still
solves 100% of these tasks, and
the software does not infer semantics
from arbitrary mathematical manuscripts.
The next useful comparison is a
compute-limited symbolic best-first
policy, preregistered before any
new test.
