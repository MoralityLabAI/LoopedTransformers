# FC-10: a real Lean tactic-state LR-01 policy

FC-09 found that a source-only AND/OR search was both more effective and
cheaper than the recurrent model on small, already formalized Horn graphs.
FC-10 evaluates the original `TypedLoop` cell on **actual Lean 4.19.0
tactic-state output**, using real compiler acceptance of candidate
tactic scripts as the sole release criterion.

The [public preregistration](https://github.com/MoralityLabAI/Compactification-Theory/blob/32330e86366f3ea6378ceb1bf6a4b189610b4483/research/refutation_bench/fc10/PREREGISTRATION.md)
was committed before building the source or looking at any model scores.

## Fixed provenance

| Source | Commit | Real Lean goal-state JSONL SHA256 |
| --- | --- | --- |
| Train/dev-only, no test generator | `631ccbb1dd393d2608b5d2c7e154aeb3ce5099b0` | `3d2c5b6055ffb0bf0118c69f9a6ea405a92a922f7f869b591e26ae5feb9cb03b` |
| Separate held-out source | `171d23e84dfc5af3ed8723ac99f75badbaf6a619` | `8e1792c737c47124b8962e3377ec4be12ec91c083b9d87644e5293fc892b0eed` |

The [train source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43454981592)
compiled 156 original source theorem declarations (132 train, 24 dev)
and parsed their actual `trace_state` outputs. The [test source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43455352149)
compiled 48 reference theorems from four new theorem-template families
and parsed the 48 actual elaborated goals.

## Model

`fc10_tactics.py` imports `TypedLoop` directly from the existing
`experiments/lean_reasoner/train.py`. Goal-state strings are tokenized
and deterministically hashed into at most 40 token nodes. The 12 tactic
action slots are typed rule nodes with incidence links to those goal
tokens. The recurrent cell is tied across one or four passes, and
a small learned head ranks the candidates.

- **62,401 trainable parameters**, identical for one/four passes.
- CPU PyTorch 2.5.1, two threads, deterministic algorithms.
- Seeds 17 and 29.
- 250 AdamW steps, batch 24, learning rate 0.002.
- Only 132 training goal states contribute gradients.
- Source dev contains 24 separate theorem families used for diagnosis.
- The learned input is the compiler-emitted goal text, not a generated
  source label, theorem ID or proof trace.

The registered four checkpoints were trained and SHA256-frozen in
[GitHub Actions #43456144563](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43456144563).
In-sample model accuracy is not used to select a checkpoint.
Its **reference-tactic label agreement on both dev families was 0%**
for all four arms; the dev families deliberately include tactic
templates absent from training. This is an out-of-vocabulary action
selection limitation.

## Exact held-out proof verification

A second, dependent CI job accesses the test source only after training
artifacts exist. It regenerates actual test goal states under Lean
4.19.0 and validates the SHA256 before reading frozen model checkpoints.
For every selected tactic the evaluator invokes a **fresh Lean compiler
process** on the original theorem with that exact script.

An accepted proof is determined solely by the compiler's success and
absence of `sorryAx`. Failure counts as abstention. Both one-tactic
and three-tactic adaptive search are reported. The independent
train-only lexical nearest-goal baseline has the same action registry.
A full registry search tries the twelve scripts with real Lean
compiler calls; a separate native `first | ...` control uses
Lean's own tactic combinator in one compiled source file.

No candidate tactic may be replaced with the source's reference
tactic after a model failure, and no proof term is manufactured
from the train metadata. The 12 scripts are fixed by the
preregistration. The CI archives each proposed tactic, original
Lean compile success/failure, trial wall time, model checkpoint
hashes and the exact goal-state inputs.

## Scope

This is the first experimental migration from finite Horn simulation
to real Lean elaboration states. It still uses **generated theorem
templates**, including Nat arithmetic, Prop introduction/elimination
and List append associativity. Label agreement with one chosen
reference tactic is stricter than actual proof success because
multiple tactics can prove a goal. Evaluation therefore uses the
compiler itself to determine real accepted proofs.

Numeric irrelevant source hypotheses provide unique source
signatures, but may leak the generator-family distribution
into the model's input. Additional tests should mask those
auxiliary numeric tokens and import existing Lean theorems
under exact repository revisions. No conclusion about
unformalized mathematical manuscripts follows from this pilot.
