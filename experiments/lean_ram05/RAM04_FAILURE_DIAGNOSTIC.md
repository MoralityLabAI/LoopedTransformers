# RAM-05 design diagnostic from the already published RAM-04 result

This file records **post-hoc** analysis of the historical RAM-04
evaluation and frozen RAM-04 training artifact. It was written before
the RAM-05 held-out theorem families were opened for scoring.
RAM-04's experimental score and certified scripts remain unchanged.

## Crash evidence

The failed original theorem
`MathCan/FoundryProbe.lean:two_currency_zero_at_lock` caused
**21 fatal persistent Lean process exits** across the complete RAM-04
primary and secondary evaluation policies. The exact original
compiler error was `maximum recursion depth has been reached`
during simplification.

Archived candidate scripts responsible for the original failure
included `all_goals simp_all [MathCan.V4Transient.count, MathCan.V4Transient.nz]`,
`simp_all [MathCan.V4Transient.count, MathCan.V4Transient.nz]`,
`simp_all [MathCan.V4Transient.nz]` and plain `simp_all`.

RAM-04's other two uncertified cases,
`TOETwoFormTriad.two_fixed_axes_not_positive_isotropic` and
`TOETripletJordan.exact_source_factor`, exhausted their registered
tactic-evaluation budgets. Their original MathCan source-reference
proofs compile. Failure of the bounded new search is a limitation
of the particular action grammar/ranking/budget; it is not a
mathematical counterexample to the original statements.

The full RAM-04 original run evidence is archived at:
[CI #43594537281](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43594537281)
and retained in the RAM-04 parent branch.

## Safety-capability tradeoff in original training data

A read-only audit of the original RAM-04 independently checked
training artifact `ram04-frozen-audited-train-43593270279` found
4,973 compiler-observed actions from 209 visible goal states,
of which 184 states included at least one visible MathCan definition.

Legacy-family counts:

| Removed unbounded/bulk family | Compiler-tested actions | Progressing actions | Scratch closures |
| --- | ---: | ---: | ---: |
| `unfold_simp` | 536 | 348 | 48 |
| `arith_norm` | 209 | 69 | 21 |
| `dsimp_visible` | 184 | 129 | 2 |

These **546 recorded progressing actions** in the excluded legacy
families are a major potential capability loss. The fresh RAM-05
collector must measure replacement tactics independently; none
of those positive labels is reinterpreted as success of a new
`only` script.

For the 184 historical definition-visible states, the RAM-05
three-single-definition limit would have generated up to
**1,179** additional concrete `only` candidates if applied
to every archived state, before source-specific deduplication
and progress exploration.

RAM-05 retains generic `simp_all` for comparability with the
old safe grammar but stops generating unrestricted MathCan
definition-bulk simplification. Compiler fatal exits remain
unknown observations with bounded restarts and preserved logs.
No parameterized tactic family or fixed goal-search budget
is changed after the new prospective score.

The new RAM-05 theorem-family selector comprises four different
original MathCan modules, so the old FoundryProbe outcome is an
engineering diagnostic and excluded from RAM-05's primary proof
coverage and model-selection results.
