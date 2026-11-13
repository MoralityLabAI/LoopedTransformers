# FC-10 first real Lean tactic-state training result

**November 13, 2026. Completed.**
[Successful two-job model CI #43456144563](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43456144563)
compiled all 156 train/dev reference theorem
declarations and later all 48 held-out
reference theorems using pinned Lean 4.19.0.
The actual `trace_state` compiler output
became model input. The held-out evaluation
then independently compiled each selected
tactic on the **original theorem**, with
failed scripts treated as abstentions.

The preregistered criterion requiring a
four-pass LR-01 model to beat the train-only
lexical top-1 baseline **FAILED**.

## Frozen source and implementation

- [FC-10 preregistered before source or training](https://github.com/MoralityLabAI/Compactification-Theory/blob/32330e86366f3ea6378ceb1bf6a4b189610b4483/research/refutation_bench/fc10/PREREGISTRATION.md).
- Train/dev-only source commit
  `631ccbb1dd393d2608b5d2c7e154aeb3ce5099b0`,
  actual 156 goal-state SHA256
  `3d2c5b6055ffb0bf0118c69f9a6ea405a92a922f7f869b591e26ae5feb9cb03b`.
- Held-out source commit
  `171d23e84dfc5af3ed8723ac99f75badbaf6a619`,
  actual 48 goal-state SHA256
  `8e1792c737c47124b8962e3377ec4be12ec91c083b9d87644e5293fc892b0eed`.
- [Source train CI #43454981592](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43454981592)
  and [source test CI #43455352149](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43455352149)
  both passed.

The model imports the **actual original
`TypedLoop` implementation** from
`experiments/lean_reasoner/train.py`.
Its hashed goal-state tokens and twelve
candidate tactic slots form a small
typed graph, processed by one or four
shared cell passes. Every model has
**62,401 parameters**. Seeds 17/29
each trained two models for 250 AdamW
CPU steps, batch24, on 132 genuine
Lean-printed training goals.

The 24 development goals deliberately
introduce two action families absent
from train. All four models had
0% agreement with those dev
families' *reference tactic labels*;
this is distinct from proof success
and was recorded before test scoring.
The train job asserted that no test
generator existed, then uploaded
SHA256-identified model weights. A
dependent job subsequently accessed
the pinned held-out Lean source.

## Exact held-out compiler success

Each count is a number of original
theorem declarations actually accepted
by Lean after applying the policy's
chosen tactic. The test has 48 theorems.

| Tactic policy | One attempt | Up to three attempts |
| --- | ---: | ---: |
| Train-only lexical nearest-goal | **36** | **36** |
| Tied LR-01 one-pass seed17 | 33 | 36 |
| Tied LR-01 four-pass seed17 | 31 | 36 |
| Tied LR-01 one-pass seed29 | 35 | 36 |
| Tied LR-01 four-pass seed29 | 35 | 36 |
| Sequential registry search of all twelve | **48** | n/a |
| Lean-native `first | ...` tactic search | **48** | n/a |

A symbolic sequential full-registry
search required 204 separate actual
candidate compiler calls. A single
Lean-native `first | ...` file
compiled all 48 theorem declarations
in about 0.973 seconds, while each
48-candidate standalone policy
took around 18 seconds in repeated
compiler-process startups. The two
timing modes must not be interpreted
as an exact equal-compute comparison.

### Held-out family diagnosis

All methods proved **24/24 Nat arithmetic**
goals. For the 12 nested conjunction
projections, lexical top1 proved 12;
the four LR-01 arms proved 9,7,11,11.
For the separate twelve **List append
associativity** theorems every trained
arm and the lexical baseline proved
**zero**. The source's successful
`simp [List.append_assoc]` tactic
was absent from the 132 training
reference labels, even though it
was already available as a fixed
candidate output action. Full
Lean tactic search still proved
all twelve List goals.

This is concrete evidence that
the current trained model fails
to select an unseen tactic category.
Four-pass recurrence gave no
consistent advantage over one pass.

## Independent audit

[The public permanent report](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc10-lean-tactic-states/research/refutation_bench/results/fc10_real_lean_tactic_states_20261113.md)
documents every preregistered
condition and scope limit. A
separate [artifact consistency audit](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc10-lean-tactic-states/research/refutation_bench/results/fc10_independent_audit.json)
checked the original source and
model ZIPs, all **480 top1/top3
case-policy rows**, the **209
unique candidate-compiler sources**,
all four SHA-frozen checkpoint files,
and exact case-by-case compiler
acceptance consistency. The native
batch file contains 48 theorem
declarations and has the SHA256
recorded in the CI report.
There were zero conflicting
candidate acceptance records.
Only the actual successful Lean
CI is claimed as kernel verification.

The model Actions job retains two
30-day artifacts:
`fc10-frozen-real-lean-models`,
containing original trained weights,
their manifest and 156 source
goal states, and
`fc10-real-lean-eval-43456144563`,
containing the 48 held-out goals,
ten policy prediction JSONLs,
full symbolic search, selected
tactic compiler responses, and
Lean-native first tactic source.

## Separate successful imported theorem-state source probe

An additional, explicitly
**post-hoc** [source import](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc10-lean-tactic-states/research/refutation_bench/fc10/POSTHOC_MATHCAN_IMPORT.md)
harvested seven actual theorem
goal states from pinned
`patrickdugan/MathCan`, including
finite-horizon arithmetic,
deletion-memory proof obligations,
and cosmology proxy identities.
The separate [Lean 4.35.0 source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43458916540)
compiled all three instrumented
original source files and extracted
the seven compiler-printed goals.
Those research source goals
did not enter FC-10 model
training or its registered test.
The current model has no verified
performance claim on imported
MathCan declarations.

## What follows

The pilot proves that **actual Lean
goal-state acquisition, tied recurrent
tactic scoring and independent
compiler-accepted certificates**
can run autonomously in CI.
It does **not** show neural
transfer beyond simple train-only
lexical retrieval or a performance
advantage over Lean-native symbolic
tactic search.

The next source corpus should
pin real declarations from
OntologicalProofs, EvalMath
or MathCan, collect intermediate
multi-step Lean goal states,
ensure each held-out tactic class
also has valid train-family
examples, and match the CPU
cost of optimized Lean tactic
search. Numeric auxiliary
hypotheses in this pilot may
correlate with generator family
and warrant a masking ablation.

## Reproduce

Pin separate public train/test
checkouts to the commits above.
Within private LoopedTransformers,
with Python3.11, CPU torch2.5.1
and Lean4.19.0:

~~~sh
export FC10_TRAIN_ROOT="$(realpath ../Compactification-FC10-train)"

python -m unittest discover -s experiments/refutation_transfer \
  -p 'test_fc10_tactics.py' -v

python experiments/refutation_transfer/fc10_tactics.py train \
  --source "$FC10_TRAIN_ROOT" --out /tmp/fc10_weights

python experiments/refutation_transfer/fc10_tactics.py eval \
  --source "$(realpath ../Compactification-FC10-test)" \
  --weights /tmp/fc10_weights --out /tmp/fc10_results
~~~

The two-job
[Actions workflow](../../.github/workflows/refutation-fc10-real-lean.yml)
enforces source separation and
retains both real Lean compilation
receipts.
