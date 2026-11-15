# RAM-06: first completed typed-exact prospective Lean experiment

**November 15, 2026. Registered learning result: negative.**
All 120 original MathCan policy/theorem searches completed under
their fixed primary or secondary budgets. The strongest
contemporaneous TRAIN-only symbolic policy, a plain global
frequency order, certified **8/12** original heldout theorems.
Both recurrent models, the shape-frequency control, the flat
network and fixed native search certified **7/12**.
No recurrent or flat network proved an original theorem that
the three registered non-neural comparators missed.

## Time-ordered protocol and independent source families

- [Frozen preregistration](PREREGISTRATION.md), original commit
  `f4f26ed48ed675cc7b80c73f151ba3927b46bafa`,
  before RAM-06 training/holdout access.
- Pinned original source
  `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189`,
  exact Lean4.35.0. All **12/12 original reference theorems**
  compiled as a separate source oracle, with original proof steps
  excluded from every model and comparator.
- Prospective modules: FundedHooks, TOEFLRW, TOECovariantGates,
  TOEMixedPrincipal; exactly their first three source-order top-level
  theorem declarations each. No selected module appeared in the
  preceding FC-11/12/13 or RAM-03/04/05 train/test family sets.
- The original completed
  [RAM-06 full CI run #43680136177](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680136177)
  trained and froze the models **before** the separate heldout job
  accessed the new source and compiled the original oracle.

## Original versus generated training supervision

[Measured training-only receipt](TRAINING_RECEIPT_20261115.md) and
[the exact machine-readable receipt](receipts/verified_train_only_20261115.json)
record the **6,293 actual compiler-verified candidate outcomes**.
The original genuine MathCan root corpus provided 5,916 new
compiler effect records, while a separate generated training/development
fixture provided 377 additional compiler observations. Each action
was independently replayed under original Lean4.35, with
**zero label disagreements**. The synthetic examples remained
separately tagged and never counted as original theorem proofs.

The 12 synthetic TRAIN propositions yielded **12 distinct verified
positive `exact_typed` actions**, and six disjoint synthetic
DEVELOPMENT propositions yielded six corresponding positives.
The **793 natural original MathCan typed-exact probes**
produced **zero positive `exact_typed` transitions**.
Thus the experiment demonstrates that the training pipeline can
learn from actual Lean-positive typed exact examples, while
the crucial new-domain exact transfer remains unsupported by
the prospective original theorem results.

The model saw only the printed local telescope, current/pending
goals, instantiated candidate action, 18-family ID and goal count.
Generated curriculum theorem names, source filenames, gold script
commands and future proof states were excluded from model features.

Nine low-memory CPU models were fitted on the combined
train-only corpus, with the original five-theorem MathCan
development split and six synthetic development declarations.
At fixed predeclared early stopping, checkpoint choices were:

| Architecture | Parameters | Seed | Development loss | Checkpoint SHA256 |
| --- | ---: | ---: | ---: | --- |
| Looped32 | 85,093 | 43 | 0.810591 | `70d1e481d0effe56bfc06c323e102b6ba305f01f716f3b5891c63c5758384258` |
| Looped64 | 207,045 | 29 | 0.867178 | `ac7d2fed61053cc1be9a293b5d057bb0c5d61a570889bc6fda1c0756f9865d2d` |
| Flat64 | 182,149 | 43 | 0.831433 | `748e18e6276af3f8a732d8cb640c310ee59593f371154f4ce21be4fcb3284ec0` |

All nine original best/last checkpoints and optimizer/RNG
receipts are in original training artifact
`ram06-frozen-audited-train-43680136177`, artifact ID
`14148087652`, ZIP SHA256
`17c21aa9faf600841d03209f61161afa355b6cade8f9510fbe62223dbc213265`.
The training memory peak was **432,898,048 bytes (~412.84 MiB)**
and the 30-step CPU pilot measured 51.47 updates/second.

## Frozen original theorem proof scores

The stronger baseline is the registered **global family
frequency from observed TRAIN-only rewards**, with no theorem
labels from development or heldout source in its ranking.
The new typed shape-frequency control uses printed goal shape and
number of pending goals with a fixed eight-observation prior.

| Budget | Controller | Original Lean proofs | Logical tactic evaluations | Search wall seconds |
| --- | --- | ---: | ---: | ---: |
| Primary 30s/48/8 | Fixed native ordering | 7/12 | 180 | 20.443 |
| Primary 30s/48/8 | Global TRAIN-only frequency | **8/12** | 153 | 22.100 |
| Primary 30s/48/8 | Goal-shape TRAIN-only frequency | 7/12 | 194 | 20.998 |
| Primary 30s/48/8 | Looped32 | 7/12 | 217 | 21.491 |
| Primary 30s/48/8 | Looped64 | 7/12 | 192 | 21.166 |
| Primary 30s/48/8 | Flat64 | 7/12 | 182 | 20.906 |
| Secondary 60s/128/16 | Fixed native ordering | 7/12 | 420 | 24.831 |
| Secondary 60s/128/16 | Global TRAIN-only frequency | **8/12** | 313 | 26.262 |
| Secondary 60s/128/16 | Goal-shape TRAIN-only frequency | 7/12 | 434 | 25.888 |
| Secondary 60s/128/16 | Looped64 | 7/12 | 432 | 28.681 |

The first number in a budget is the **maximum seconds per
theorem**, then max logical tactic candidate evaluations, then
max transitions per proof path. Search wall time in the table
is summed across twelve theorem cases. Cost from failed search
is included; one CPU-hosted run is insufficient for strong speed
comparisons. Peak evaluator process-tree memory was
**2,024,144,896 bytes (~1.885 GiB)** with no resource abort.

### Exact per-original-theorem primary coverage

| Original MathCan theorem | Native | Global frequency | Shape frequency | Looped32 | Looped64 | Flat64 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| FundedHooks.conserved_along | · | ✓ | · | · | · | · |
| FundedHooks.mark_to_market_conserved | · | · | · | · | · | · |
| FundedHooks.trader_gain_exact | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOEFLRW.form_degeneracies | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOEFLRW.ricci_eigen_trace | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOEFLRW.bivector_eigen_trace | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOECovariantGates.ricci_scalar_trace | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOECovariantGates.doubled_tracefree_temporal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOECovariantGates.doubled_tracefree_spatial | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOEMixedPrincipal.one_flavor_schur_cancel | · | · | · | · | · | · |
| TOEMixedPrincipal.one_flavor_schur_invertible | · | · | · | · | · | · |
| TOEMixedPrincipal.rank_one_square_expansion | · | · | · | · | · | · |

No completed original theorem proof uses `exact_typed`
or `exact_local` in its successful script. The new model
family was attempted, but the effective hypothesis-application
capability did not transfer into an additional original
MathCan proof in this frozen suite. The negative result is
preserved as the sole registered RAM-06 outcome.

### A substantive strong symbolic comparator witness

Only **global empirical TRAIN frequency** certified the
preexisting original theorem
`MathCan/FundedHooks.conserved_along` under the registered
primary and secondary budgets. Its original complete Lean
script was:

`induction word generalizing s`
`constructor`
`simp_all only [MathCan.HookState.endpoint]`

This script is instantiated entirely from the compiler-visible
root goal and the registered source-derived grammar. It completed
in **nine** logical tactic evaluations under the primary budget.
The exact full compiled Lean proof source SHA256 is
`5faeed214c2fd0c1c03082d504cc2fe60c13df7515a0d285e97e0dce9ad3283f`.
The final axiom inventory was `propext`.
Original native, shape frequency, flat and both recurrent policies
failed to certify the same theorem within their matched budgets.
The theorem had already been proved in MathCan. This is a
new successful compiler-verified script discovery under the
RAM-06 gym, not a new original mathematical result.

## Independent full-proof trust-boundary audit

The completed GitHub Actions run contains **120 scored
policy/theorem trajectories**, each with the registered
original source file, context-derived script, active and
pending goals, compiler source/log SHA, elapsed time and
actual process counts.

Of the 120 cases, **72 policy/theorem instances** earned
a full original sorry-free theorem certificate. A separate
independent audit recompiled every accepted source in
Lean4.35 and checked its allowed axiom inventory and
byte-identical source SHA. It reported **zero false-positive
certificates** and left every original score unchanged.

- Original scored report SHA256:
  `fd4ac266d63086e31eeba146a7ade30a3fed318d5feca8c89b48f6dcc13c4003`.
- Independent 72-source hash manifest SHA256:
  `10db1d6afe1811d758d4d2e8ba650b14049be6c698cd9e502579a4b7045629c1`.
- Original 120-case evidence ZIP:
  `ram06-first-prospective-proofs-43680136177`
  (artifact ID `14148381661`, SHA256
  `52bcf3b664292461e8de965f15fc8ca9f4db2c46a6de126a1916f05edcccc8f4`).
- Compact machine-readable permanent receipt:
  [first_prospective_20261115.json](receipts/first_prospective_20261115.json).

There are eight unique original theorems covered by the union of
all controllers, rather than 72 new theorems. The four original
uncertified statements have untouched reference proofs
compiling under Lean4.35. Failure here is search/grammar/budget
evidence, not a refutation.

## Consequences for the next campaign

The additional positive exact curriculum achieved its stated
training coverage and low RAM target. Its 18 positive
examples were **all synthetic** and supplied no new
original-domain success. The clear next experimental task
is to acquire independently replayed `exact` successes
from **natural MathCan proof-state descendants**, with
type-aware continuation through local-lemma application and
more expressive proof hypotheses.

The simple global frequency controller remained the strongest
contemporary baseline. A successor must preserve this
control and seek a uniquely model-certified original
heldout theorem, rather than claiming gains against
the weaker static native order alone.
