# RAM-07 evaluation amendment B | term-style original Lean theorem support

**November 15, 2026. Post-first-heldout inspection, before any complete RAM-07 original proof comparison.**

The original [RAM-07 preregistration](PREREGISTRATION.md)
fixed the 47 old original MathCan TRAIN roots, the four new
MathCan source modules, twelve original theorem names,
19-family action grammar, model architectures and seeds,
all three non-neural controls, and primary/secondary budgets.

[Pretest mining amendment A](PRETEST_MINING_AMENDMENT_A.md)
already disclosed the source-only queue repair made before
the new heldout source was ever opened. It fixed a
train-data scheduler issue while preserving all original
negative evidence and the new-family evaluation selector.

## Source data and model freeze before first heldout attempt

[Full CPU train-only CI `43682912918`](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682912918)
completed all training prerequisites:
- 10,644 independently audited actual original MathCan
  candidate actions, including a **single** positive
  `exact_typed` action on an original source TRAIN root,
  `Intervention.obsEqOn_trans`, reached by two context-driven
  named introductions with no original reference prefixes;
- nine complete tiny CPU fits, with original development-only
  checkpoint seed selection:
  loop32 seed29, loop64 seed17, flat64 seed29;
- exact independently verified best checkpoint SHA256s and
  no synthetic auxiliary data in this training set;
- full original model and train-effect artifact
  `ram07-frozen-audited-train-43682912918`, artifact ID
  `14148788243`, ZIP SHA256
  `c622367da9a7ced515df1db2869937b4edb61f3cc0736b4b51035c5442038db0`.

[Training receipt](TRAINING_RECEIPT_20261115.md) pins the
unchanged optimizer RNG states, model hashes and data SHA.
No model weight, seed or train-source record is eligible
for revision after this checkpoint freeze.

## The incomplete first prospective result

The separate heldout job in that same GitHub run checked
out the **newer original MathCan**
`fca46bfb3430ac85ffca16f6a6a83bb0b0231e41`,
built its complete import graph and the independent
`MathCan.OrderedJet` module, and compiled the original
source as the separate reference oracle.

The scorer completed exactly the first **three**
preregistered source-order theorem cases, with all
**six** primary policies per case, then stopped on the
fourth theorem
`MathCan/OrderedJet.lean:scalar_order_invisible`.

That original theorem uses a legitimate Lean **term-style**
`:= term` proof rather than a `:= by` tactic proof.
The original FC-11 `theorem_bounds` helper, reused through
RAM-03/04/05/06, recognized only the latter. It raised:
`AssertionError: no source tactic proof: scalar_order_invisible`
before any of the fourth case's registered controllers
could be scored. The original new MathCan module itself
compiled without error. This is a renderer coverage
failure, not a counterexample, failed source proof,
or evidence that a model's tactics are wrong.

The entire incomplete 18-case original result is **excluded**
from the registered prospective outcome. Its exact
unaltered compiler sources/logs and recorded partial
scores are preserved in artifact
`ram07-first-prospective-proofs-43682912918`,
ID `14149632474`, ZIP SHA256
`718e0ad50eb7985021a1e85cb791766af1c1ae840a2e779852256011b559e284`.
No model/baseline is credited for only the initially
completed cases.

## Narrow generic source-renderer repair

The RAM-07-only source core now calls
`original_theorem_bounds`, which preserves the existing
FC-11 bounds for every `:= by` proof. When the old
parser reports the precise missing tactic-proof error,
the fallback accepts an original, uniquely named
top-level `theorem` declaration whose original proof
marker is `:=`, identifies the full original theorem
statement, **deletes the entire original term proof**,
and inserts `:= by` followed by only the registered
candidate tactic script.

No theorem statement, assumption, import or axiom
changes. The fallback never uses an original proof
command as a candidate, training label or model feature.
The same complete original theorem must compile and
independently replay with only the same allowed axioms
to count as a solution. Unit tests cover term-style
source isolation and unchanged tactic-style handling.

This syntax-support repair was made after some heldout
outcomes were visible, so the correct analysis requires
a **new, complete all-policy comparison** with disclosure,
rather than treating earlier partial positives or
rerunning only the failed fourth case.

## Frozen full evaluation-02

Launch a separate GitHub Actions workflow using the
*original frozen training artifact by its run ID*.
It must verify exact original checkpoint SHA256s:
- Looped32 seed29:
  `15aefb27ef9dd6cb9219fcd1e2e826fa542159a68b2eb2dc2ac95a03af4f271c`.
- Looped64 seed17:
  `24be13905f50f111b2e3c285af48bfef003de97b1d1d857adbcd859781a04eda`.
- Flat64 seed29:
  `ba04242b82b3a08a6c96d74352a875beb40e468719bfa570b243b53c9d8b6777`.

The new run must score **all twelve original theorem
declarations** with **all six** original primary
controllers (30s/48 logical evaluations/8 transitions)
and all four original secondary controllers (60s/128/16),
total **120 complete cases**. The original current
MathCan source SHA `fca46...`, exact test names,
candidate grammar and local/contextual argument
generation are identical. Train and development labels
and weights remain byte-frozen.

Every accepted proof still requires complete original
Lean4.35 compilation, printed allowed-axiom gate,
a byte-identical independent in-budget source replay,
and a third independent post-score full-source compile.
If the complete run fails again, preserve that run
and report the new failure explicitly. No selective
case retry, score mutation or post-heldout model
selection is permitted.

The registered success condition remains **one novel
original new-family theorem** certified by a recurrent
model and missed by all three same-budget non-neural
controls, with zero false positives. Recurrence-specific
evidence additionally requires flat64 to miss that
theorem. Original previous RAM-07/06/05 source and
score receipts remain unchanged.
