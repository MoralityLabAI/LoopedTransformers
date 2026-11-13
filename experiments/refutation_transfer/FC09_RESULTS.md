# FC-09: measured CPU-budget-matched AND/OR search defeats LR-01

**First correctly pinned evaluation: November 13, 2026.**
[GitHub Actions #43449947846](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43449947846)
passed train-and-freeze and independent held-out
evaluation, with Lean 4.19.0 kernel compilation
of **324 original model-generated positive proof
terms**, and independent validation of **383**
negative countermodel certificates.

The [public permanent research receipt](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc09-cpu-matched-andor/research/refutation_bench/results/fc09_cpu_matched_andor_20261113.md)
contains complete source provenance, paired results,
symbolic search time allowances, actual wall times,
all exact checker counts and CPU efficiency outcomes.

## Frozen preregistration and sources

- Registration committed first:
  `37ceea8c6e5310fb9fb6fb43fd2aab1e9f3210a4`.
- Train/dev-only source commit:
  `315119ead7264fbfbf5e58ee1b7b5c8b512d475d`.
- Generated train/dev JSONL SHA-256:
  `680318ea334bba0777a048d3299a9e490207faa3a8ead549393616790a3761d5`.
- Test source frozen separately:
  `0b045fade18ca725ad9adc536ad48d6f58935fb3`.
- Generated test JSONL SHA-256:
  `89719bf1af2927fbd957f6735d15e4ec30598696b173a3a365bcd39bbfd5440d`.
- [Source training CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43448030483)
  verified 960 samples and compiled 480 valid Lean
  reference theorems.
- [Source held-out CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43449281226)
  verified 384 samples and compiled 192 positive
  reference theorems with 192 exact negative
  countermodels.

The first source-only run failed because an
alternate goal producer shortened the intended
minimum proof. It was corrected and the source
train-only commit frozen **before test generation
or model training**. An earlier model run with an
unfilled test-hash placeholder stopped before
valid test scoring; the correctly pinned
CI #43449947846 is the first complete run.

## Exact real training and held-out model

Two LR-01 four-pass gating networks of
**11,521 parameters** each were trained on
original FC-02 train and calibrated on
FC-02 dev only. Seed17 selected theta=0.70;
seed29 selected theta=0.55.

Each separate actor copied the tied trunk
and trained its original 11,521 parameters
jointly with a 4,705-parameter
goal-conditioned rule head. The actor
has **16,226 parameters**, supervised
on **1,344 FC-09 training-only positive
proof-prefix states** through 300 AdamW
updates. The first 128 in-sample next-rule
accuracies were 77.3438% and 79.6875%.

The training job checked out ONLY the old
source revision and confirmed that FC-09
test_data.py and controls.py were absent.
It wrote SHA256-identified weights and
a frozen calibration ledger. The test
job downloaded those immutable weights
before checking out and checking the
fresh test source commit.

## Registered primary comparison

For an eligible positive case, the learned
actor first attempts up to five original
source-rule applications, spending four
recurrent passes per action. The symbolic
planner receives only the **measured
rule-scoring milliseconds** from that
frozen actor as its time allowance.

The separate symbolic AND/OR search uses
the h-max relaxed conjunctive lower bound,
a best-first heap of known-atom sets and
an actual deadline check before graph
expansions. It never receives the learned
actor's proposed rule IDs, probabilities,
losses or source reference proof. Both
positive results must pass independent
exact source-rule replay; all learned
positives are compiled as original Lean
theorem terms. Negative claims require
complete verified Boolean countermodels.

Each test group has **96 positive** and
**96 negative** source claims:

| Seed / split | Learned positive /96 | CPU-budgeted symbolic positive /96 | Learned negatives /96 | Abstained |
| --- | ---: | ---: | ---: | ---: |
| 17 familiar | **96** | **96** | 95 | 1 |
| 17 deep | 64 | **96** | 96 | 32 |
| 29 familiar | **96** | **96** | 96 | 0 |
| 29 deep | 68 | **96** | 96 | 28 |

Across all 768 seed/case evaluations,
**324 positive model-selected proof terms**
compiled under Lean and **383** negative
countermodels passed exact checks.
There were **61 abstentions** and
**zero false finite formal certificates**.

Paired model-only / symbolic-only
positive success counts were:
0/0, 0/32, 0/0 and 0/28, in table order.
The preregistered learning criterion
therefore **FAILED**. The symbolic
method succeeded on every eligible
valid case.

### Actual compute rather than nominal equal runtime

The dynamic symbolic search consumed
only **9.2%–13.5%** of the actor's
available measured rule-scoring time
per test condition, recorded across
all eligible cases. It hit no
timeouts and overshot no deadlines.
Thus CPU time was an *upper allowance*,
rather than a requirement to expend
equal compute.

| Seed/split | Learned full CPU-runner ms/case | Symbolic full ms/case | Learned ms/positive proof | Symbolic ms/positive proof |
| --- | ---: | ---: | ---: | ---: |
| 17 familiar | 9.176 | **4.478** | 18.35 | **8.96** |
| 17 deep | 11.030 | **4.671** | 33.09 | **9.34** |
| 29 familiar | 9.370 | **4.622** | 18.74 | **9.24** |
| 29 deep | 11.328 | **4.767** | 31.99 | **9.53** |

The **registered CPU-efficiency
criterion FAILED** for every seed and
test depth. These are wall-clock
measurements on one CPU GitHub runner,
not hardware-neutral FLOP estimates.
The shared original FC-04 gate and
exact checkers are included in both
full times.

A fixed **24-pop** h-max source
control, measured *before* neural
evaluation, already certified 96/96
familiar and 70/96 deep positives.
Giving it the larger *time* allowance
let it expand enough states to certify
all 96 deep cases, much faster than
the actor consumed its allowance.

## Separate post-run certificate replay

The [independent audit receipt](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc09-cpu-matched-andor/research/refutation_bench/results/fc09_independent_artifact_audit.json)
records a second verification of the
original archived output against the
original source-test ZIP. It checked
**6,144 case responses**, independently
replayed **324** original positive
rule sequences, verified **383**
complete negative truth assignments,
and confirmed zero false formal
certificates. It also confirmed
the *exact* per-case time allowance
matched the actor-forward time
and no symbolic CPU deadline was
overrun. The Lean source has
324 theorem declarations matching
the CI receipt SHA256.

## Artifacts and audit

The successful job archived
`fc09-frozen-models` (pretest weights)
and `fc09-cpu-matched-43449947846`
(the full score ledger, 32 raw prediction
JSONLs, model-selected Lean sources
and kernel compilation log). The
latter contains **6,144 responses**
(192 cases per file), and all 324
model-generated theorem declarations
match the SHA256-referenced Lean
source recorded in `FC09_report.json`.

No neural model here exceeds the exact
finite symbolic solver's correctness.
The result is a negative finding
for neural compute advantage on
this *small already formalized*
12-atom conjunctive Horn family.
It does not establish that neural
proof search is useless on larger
Lean tactic-state spaces or in
natural-language mathematical papers.

A higher-value next step is a corpus
of actual Lean proof obligations with
larger branching action spaces, a
typed tactic budget and a fully
preregistered comparison to optimized
symbolic proof search. Maintain
independent Lean kernel release
gates for all positive claims and
full finite certificates for those
negative claims whose semantics
are already formalized.

## Reproduce the registered experiment

With separate Compactification-Theory
checkouts at the train/test commits
specified above:

~~~sh
export FC09_TRAIN_ROOT="$(realpath ../Compactification-FC09-train)"
python -m unittest discover -s experiments/refutation_transfer \
  -p 'test_fc09_trial.py' -v
python experiments/refutation_transfer/fc09_trial.py train \
  --source "$FC09_TRAIN_ROOT" \
  --out /tmp/fc09_weights
python experiments/refutation_transfer/fc09_trial.py eval \
  --source "$(realpath ../Compactification-FC09-test)" \
  --weights /tmp/fc09_weights \
  --out /tmp/fc09_results
lean /tmp/fc09_results/FC09_model_proofs.lean
~~~

The [GitHub Actions workflow](../../.github/workflows/refutation-fc09-matched.yml)
enforces training-versus-test access
with separate dependent jobs and
immutable checkpoint hashes.
