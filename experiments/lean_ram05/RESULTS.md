# RAM-05 | first complete bounded-unfolding prospective Lean result

**November 15, 2026. Preregistered primary outcome: model-only success against the fixed-order native control; recurrence-specific success: false.**

[Original preregistration](PREREGISTRATION.md) was committed as
`215de8648d6db73352a4118f5507b45f9eb4f814` before the
new training data were collected or the twelve original
MathCan theorem states were inspected. This RAM-05 campaign
is stacked on RAM-04 and changes none of the previous outcomes.

## Source, training and model freeze

Exact original source is
`patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189`
under Lean4.35. The new prospective suite is the first three
top-level theorem declarations in each of TOETwoMultiplet,
TOEYangMills, TOEConeMatching and TOETensorConstraints.
The untouched original MathCan reference proofs compiled 12/12,
separately from the candidate policy.

[Source-only training CI #43675801084](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43675801084)
completed the independent collection gate and all nine CPU
fits before any prospective test scoring:

- 27 original TRAIN theorem roots, 22 model training and five
  development-only roots.
- **5,123 new concrete Lean compiler-probed actions**, all replayed
  in 27 independent fresh sessions with zero label disagreement.
- **213 original compiler-derived goal states**: 171 train and
  42 development; 49 train and 19 development multigoal states.
- Frozen model-development selection: Looped32 seed29,
  Looped64 seed29, Flat64 seed29.
- 17 registered tactic families, including `simp only [D]`,
  `simp_all only [D]`, `dsimp only [D] at *`, where `D` is
  a single *compiler-visible* MathCan definition.

Exact new dataset SHA256
`45d2c7d7d29bf17c48a09c431a3cbe295f78690c10baf6c01edef69221ce7b62`.
Exact [training receipt](TRAINING_RECEIPT_20261115.md) and
all nine best/last CPU checkpoints are retained in
`ram05-frozen-audited-train-43675801084`.

A supervision limit must remain visible: the new dataset
contains **zero positive `exact_local` actions** among 718
attempts, while it has 49 successful train rewrites, two
successful train `rcases` transitions, 53 successful
list-induction transitions and 451 progressing single-definition
`only` actions. The earlier original FC-12 exact-positive example
is preserved as independent historical evidence; it was not
injected into this new model fit. Scratch progress is never
a full proof certificate.

## Fixed primary and secondary scores

The original 84-case model/native/frequency scorer completed
and saved all its source and compiler logs in the GitHub Actions
artifact `ram05-first-prospective-proofs-43675801084`.
The overall first workflow concluded failure only because its
*additional third audit* looked for a different report filename
and unsuffixed suite paths. The scored 84-case source report
was already complete, and its SHA256 is
`89dc5e33255fc0acfb820c344d9986f2e8e95cdc985fde9c35f5af284a94051c`.

[Audit-only amendment A](EVALUATION_AMENDMENT_A.md) documents
the report-path repair, retaining the full original artifact
and without rerunning theorem search, fitting new weights or
changing a single scored outcome.

| Search budget | Controller | Lean-certified originals | Logical tactic evaluations | Search wall seconds | Seconds/certificate |
| --- | --- | ---: | ---: | ---: | ---: |
| Primary: 30s / 48 / 8 | Fixed native | **6/12** | 328 | 24.089 | 4.015 |
| Primary: 30s / 48 / 8 | Train-frequency | **8/12** | 220 | 27.155 | 3.394 |
| Primary: 30s / 48 / 8 | Looped32 (85,061) | **7/12** | 232 | 25.844 | 3.692 |
| Primary: 30s / 48 / 8 | Looped64 (206,981) | **8/12** | 212 | 27.363 | 3.420 |
| Primary: 30s / 48 / 8 | Flat64 (182,085) | **8/12** | 179 | 26.870 | 3.359 |
| Secondary: 60s / 128 / 16 | Fixed native | **6/12** | 568 | 27.146 | 4.524 |
| Secondary: 60s / 128 / 16 | Looped64 | **8/12** | 292 | 28.527 | 3.566 |

Both models, native search and the frequency baseline
received the same registered candidate grammar, compiler
feedback and budgets. No controller used original reference
proof commands, future states or test labels to rank tactics.
The primary comparison gave each original theorem 30s,
48 total logical evaluations and eight transitions per path.
The secondary increased those limits to 60s, 128 evaluations
and 16 transitions without altering model weights or grammar.

### Exact first-suite original theorem coverage

| Preregistered theorem | Native | Frequency | Looped32 | Looped64 | Flat64 |
| --- | :---: | :---: | :---: | :---: | :---: |
| TOETwoMultiplet.two_multiplet_cone_matching | · | ✓ | ✓ | ✓ | ✓ |
| TOETwoMultiplet.compensation_positive | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOETwoMultiplet.aligned_flavor_wedge | · | · | · | · | · |
| TOEYangMills.opposed_electric_mixed_zero | · | · | · | · | · |
| TOEYangMills.opposite_electric_norm | · | · | · | · | · |
| TOEYangMills.diagonal_match | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOEConeMatching.matched_cones_have_root | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOEConeMatching.cone_gap_at_first_root | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOEConeMatching.mismatched_cones_lift_first_root | · | ✓ | · | ✓ | ✓ |
| TOETensorConstraints.source_cone_match_iff_compensation | · | · | · | · | · |
| TOETensorConstraints.rank_three_needs_two_multiplets | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOETensorConstraints.one_multiplet_cannot_have_rank_three | ✓ | ✓ | ✓ | ✓ | ✓ |

All four original statements unsolved by the best policy
have independently compiled reference proofs. An exhausted
search budget or no-progress report is evidence of the
fixed grammar/ranker limits, not a mathematical counterexample.

## Two proof trajectories beyond fixed native ordering

The first new original theorem beyond fixed native is
`TOETwoMultiplet.two_multiplet_cone_matching`. Looped64's
complete script uses bounded-only unfolding of the original
`electricTheta`, `magneticTheta` and `totalMagnetic` definitions,
then a constructor and `omega`, with 34 logical evaluations.
Its exact independent full-source SHA256 is
`3dcc80cb25232a246c9d995dfdb2ee9bb723fdb9ca1920c2233985060c1825ba`.

The second is
`TOEConeMatching.mismatched_cones_lift_first_root`.
The 207k Looped64 model chose
`simp_all only [MathCan.TOEConeMatching.conePolynomial]`,
`simp_all`, `omega` and certified the original complete
theorem in seven evaluations. Its full-source SHA256 is
`e312c5b05cbe14e685c3653392b5d94ab8a2a326da9a78fedc6d1c6e41bacdca`.

The fixed-order native controller exhausted its 48
evaluations on these two cases. Both the trivial
training-only frequency ordering and the flat64 scorer
also discovered and certified both. Therefore:

- The **registered model-only-vs-fixed-native criterion
  passed**, with two new original theorem certificates for
  Looped64 compared against the native 6/12 set.
- The **recurrence-specific criterion failed**.
  Neither the 83k recurrent policy nor 207k recurrence
  showed an improvement over the simple frequency/flat
  policies on this fixed new corpus.

The stronger family-frequency baseline is a consequential
control. A gain against a fixed enumeration order does not
establish that recurrent computation learned a stronger
general theorem-proving ability.

## Independent trust-boundary and resource audit

[Independent third compiler audit CI #43676926205](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43676926205)
passed. It downloaded the **unchanged original 84-case
scorer artifact** by run ID, checked its original recorded
SHA256 and all trained checkpoint hashes, then separately
recompiled **all 51 successful policy/theorem instances**
under the original Lean4.35 dependency graph. It verified
byte-identical complete original source, an explicitly
allowed dependency list from `#print axioms` and the two
original scored proof scripts. It found **zero false
positives**, with every source success/failure score unchanged.

Independent audit JSON SHA256:
`a96d8078718ed91199d0a9c4cccd068d1eb64dca613f6b9800404ed1395a8661`.
Complete 51-source SHA manifest:
`b5e6d247e2b045b933bf6c8af327db4637054e4a79186081324038d29d828b58`.

The original test run preserved 4,699 generated evidence
files, including compiler sources/logs, candidate choices,
failed attempts, positive source/replay witnesses and
whole-goal-list transitions. Every accepted proof was
independently checked twice inside its deadline and one
additional time outside scoring.

No fatal process-exit unknowns occurred in the new
12-case scoring suite. This is an observation on these
four source modules, not a controlled proof of crash
elimination on the original RAM-04 FoundryProbe failure.

Training CI measured peak process-tree RSS 398.39 MiB
and 54.74 CPU optimizer steps/s. Evaluation measured
1,988,030,464 bytes (~1.852 GiB) peak process-tree RSS,
within the fixed 6 GiB cap; no resource abort occurred.
This is CPU-only hosted evidence. Existing RTX3050
Windows reproduction remains available in
[run_windows.ps1](run_windows.ps1) without changes to the
original RAM-04 local runs.

The measured wall times come from one hosted CPU
comparison, with executor/cache effects and no repeated
interleaved latency trials. Proof coverage and source
certification carry greater evidentiary weight than
small timing differences.

## Next research question

RAM-05 demonstrates that bounded visible-definition
unfolding can expose two original complete proofs beyond
a naive fixed native tactic order. With frequency and
flat controls matching this result, the next original
research question is whether a parameterized, branch-
capable policy can outperform **the strongest contemporary
training-frequency or typed symbolic ranking baseline** on
a new separately frozen source-family suite.

That experiment should require genuine positive `exact`,
`rw`, `rcases` and `induction` support in the audited
*training* corpus before fitting, include a theorem-family
split with positive multigoal branch evidence, and score
new complete proofs against a strengthened baseline.
It belongs in RAM-06; RAM-05 weights and results are frozen.
