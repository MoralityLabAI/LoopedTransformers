# RAM-07 pretest mining amendment A: natural named-intro traversal

**November 15, 2026; before any RAM-07 training fit or any new-source prospective theorem score.**

[Original RAM-07 preregistration](PREREGISTRATION.md), at original
commit `e2a717aca927221f21fd3df9b47520b6f74f1f7c`,
fixed the two old/new source commits, all twelve prospective theorem
declarations, 19-family candidate grammar, 47 old training roots,
12 states/theorem, depth5, max8 goals, nine models, seeds, and
all six primary/four secondary comparator budgets.

[The original train-only run #43682247680](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682247680)
completed **10,564 actual original MathCan compiler candidate
observations** across 47 source theorem roots and 370 unique
compiler-reported goal states. A fresh independent Lean4.35
session replayed every observation with **zero disagreements**.
Its original corpus SHA256 was
`7082339fb28bd700a394a1abde348a51934be2e60dd5843a5e64d0eec3892588`.
The model-training gate then deliberately stopped because the
original natural TRAIN split contained **zero independently verified
positive exact_typed or exact_local actions**. No model was
fitted, and the newer MathCan fca46 held-out source was never
checked out or evaluated. The complete original failed collection
and its audit were archived as
`ram07-train-failure-43682247680`,
artifact ID `14149740022` and ZIP SHA256
`d800dd9aadf866bf491d1be1931fcac4cfd77415f9cdb97bd2ad5fa2fc8e1de0`.

## Why valid natural exact actions were missed

Read-only inspection of the **original archived train-only** goal
paths found that the root `Intervention.obsEqOn_trans` generated
`intro ram07_arg0` successfully, then `intro ram07_arg1`
successfully, but the original breadth-first successor queue visited
list-induction and simplification siblings ahead of the
two-introduction prefix. The 12-state-per-theorem cap was
exhausted before the complete named-intro state was evaluated
for typed-exact actions.

A **separate independent training-only** real Lean smoke,
[CI #43682572642](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682572642),
demonstrated that the original 4.35 checker independently
certifies `Intervention.obsEqOn_trans` from the non-gold prefix:
`intro ram07_arg0`;
`intro ram07_arg1`;
`exact (hst ram07_arg0 ram07_arg1).trans (htr ram07_arg0 ram07_arg1)`.
The full original Lean proof compiled and was replayed byte-for-byte
with an empty axiom inventory (source hash
`b9f6d9cbb4fbaadd3e1a153c3685f55dc109fe432b2030657a6e55c8f01633f5`).
The existing development theorem `obsEqOn_symm` also passed using
the same independently generated two named introductions and
`exact (h ram07_arg0 ram07_arg1).symm`,
source SHA256
`0f1f595f09e37337150e53092f97d4a9ee1e130fa7297fd168480b7012e55e59`.

Both are *previously known original MathCan theorems* and are
training/development diagnostics. They are excluded from any
new held-out proof coverage. Neither original reference proof
prefix was read or copied into the training collector.

## Narrow pretest scheduling repair

The original queue code was:
`branches.extend(multi[:3]); ordinary.extend(single[:3])`.

The source-only **amended collector** retains the same source roots,
source code, 19-family candidate pool, 12-state and five-transition
limits, all labels/replay requirements, and same train/dev groups.
When the original compiler genuinely accepts a registered
`intro ram07_argN` candidate and yields a single remaining goal,
the collector places its resulting state at the **front of the
train-only exploration deque**, then uses up to two of the other
same-state successors, preserving their relative order. Other
ordinary states still use the original first-three expansion.
There is no new source lemma, reference action, hypothesis or
target-facing tactic. Every resulting dataset has a new SHA256
and is independently replayed in a fresh Lean session before fitting.

This changes **which 12 natural descendants** are inspected in
TRAIN/DEV and hence alters the new training distribution; disclose
the change as a preregistration amendment, rather than claiming
a fully untouched original miner. The new evaluation remains a
time-ordered source-family holdout because no heldout theorem goal
or score was inspected. Preserve the original failed attempt
as an independent research negative result.

The complete **new** source-only mining, nine-model training and
the already fixed 120-case prospective comparison must run as a
new named GitHub Actions run, from fresh original source checkouts.
The new original MathCan `fca46b...` source may enter only the
dependent heldout evaluation job after independent training audit
and complete checkpoint freeze. No model-only result is accepted
from a partial or failed train-only run.
