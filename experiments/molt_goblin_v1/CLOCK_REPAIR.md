# MOLT-14: finite carry-targeted donor repair

Development follow-up to the measured [MOLT-13 results](CLOCK_CARRY_RESULTS.md).
The original addressed teacher is saturated on the tested sequence task.
A new learning comparison would need an independently specified headroom
condition. This campaign instead produces a frozen repair instrument and
paired receipts for future training data selection.

## Repair geometry

The teacher's actual answer residual is h at primitive cut 4. Let M(h)
denote the 16 legal logit differences to the last legal class after the
actual final LayerNorm and tied unembedding. Equality of M gives equality
of the 17-way legal softmax carry in real arithmetic. A donor is a clean
source-related sequence with the same future operands. Its observed
cut margins are the repair target, not a gold label or future answer.

A raw native fitted-factor swap is d=beta (x_donor-x_anchor) V U in
the actual post-residual coordinates. Fix the per-case dose limit R=||d||.
Optimize ||M(h+Q c)-M(h_donor)||^2 with ||c||<=R for three orthonormal
charts of equal dimension: native U span, seeded random span, and the
leading right singular span of the anchor legal-margin derivative.
All begin at zero and use eight damped Gauss-Newton steps, the same
scale-aware ridge rule and four fixed line-search proposals per step.
Reject loss-increasing steps. No answer-dependent stopping, no choice of
dose after observing results. The raw swap and intact model are references.
A full donor residual replacement is a separate, unbounded, privileged
ceiling; its dose is recorded and excluded from the dose-capped claim.

Use actual LayerNorm derivative, including centering, learned scale,
variance denominator and epsilon. Record the native chart's numerical
singular values and fixed-threshold rank. Local full rank does not
establish finite-dose reachability or good conditioning.

## Exact finite carry sham

Given the optimized native endpoint t=h+d*, define the rows A from
the legal unembedding contrasts multiplied elementwise by the LayerNorm
scale. Include the all-ones row and original residual h in the constraint
matrix. Let P be the orthogonal row-space projector. Reflect the
endpoint: t'=P t-(I-P)t. This preserves the norm, mean, A t and h dot t.
Consequently the normalized legal logit contrasts agree exactly in real
arithmetic, and ||t'-h||=||t-h||. The full future trajectory agrees by
the exact carry quotient. The current unrestricted logits may differ.

The sham has full ambient support; it is not a distinct match restricted
to the native span. It controls the finite carry output, not the signed
first-order displacement. At zero residual in the constraint nullspace,
or zero intervention dose, retain the no-distinct-match outcome. This
is an intentionally output-matched diagnostic: equal future outcomes
follow from the mathematical contract and supply no evidence of
donor-specific superiority. Every numerical constraint and complete
suffix output is checked in the actual dense native forward path.

## Fixed development roster and gates

Predeclared addressed seed 157 and native 16-component BPD engineering
fit from MOLT-12. 16 fresh depth-4 and 16 fresh depth-8 worlds, excluding
prior generated training and development tuples. Each root has two
oracle-checked source edits: a changed cut addition (+1 modulo 17),
and a same-target paired prefix edit (second addition +1, fourth
addition compensates for the intervening multiplication). Both preserve
future operands; the same-target edit preserves the true cut state.
Keep both source and anchor correctness, every arm including failures,
complete suffix/terminal correctness, clean-suffix preservation,
physical dose, legal margin/carry residual, objective and work ledger.

The primary contrast is native_solve versus observer_solve and random_solve
under equal rank, dose cap and solver evaluations. Their actual doses
can differ; this is a budget-matched control, not equality of physical
dose. The finite sham matches actual dose and endpoint carry. The panel
is descriptive, one frozen model, 32 independent world roots with two
dependent edits per root. No confirmation, protected evaluation, native
swap API parity claim, or model training. These are additive displacements
derived from actual fitted factors. Keep native factor reconstruction
quality separate from solver success. No mechanism claim from beating
a weaker random control. Any subsequent training experiment must use
fresh TRAIN-only data, headroom, ordinary/reinvested controls and the
existing EvalMath gate. The engineering teacher is saturated at ordinary
evaluation, and this campaign does not manufacture a new learning gain.

    python -m pytest -q tests/test_molt_clock_repair.py
    python -m research_gym.interp.molt_clock_repair --clock /tmp/molt14/clock \
      --factors /tmp/molt14/bpd --bpd-checkout .native-bpd \
      --bpd-source-receipt /tmp/molt14/bpd-source.json --out /tmp/molt14/repair

The worker is run under the unchanged child-free resource supervisor.
Hard address-space/CPU time and sampled RSS are distinct; no kernel
process-tree RAM guarantee.
