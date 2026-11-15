# RAM-04: completed first original family-heldout Lean experiment

**2026-11-14, full first valid evaluation.** The preregistered primary
learning criterion failed: each recurrent policy, flat feed-forward
control, training-only frequency control, and fixed native Lean search
independently certified the same **9 of 12 previously unused original
MathCan theorems**. Neither the primary 30-second/48-evaluation regime
nor the preregistered 60-second/128-evaluation secondary regime
produced a model-only complete proof.

## Frozen provenance

- [Preregistration](PREREGISTRATION.md) commit
  `ae7be99b6367434f0dd724ae232cfa218ed99342`, before training
  or inspecting heldout goal states.
- Exact original mathematical source:
  `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189`,
  Lean4.35.0. Original reference source compiled 12/12.
- Source families: OpcodeRefinement, FoundryProbe, TOETwoFormTriad,
  TOETripletJordan. Each contains three preselected original theorem
  declarations, selected by source order before scoring.
- Frozen nine CPU checkpoints, train-only dataset fingerprints and
  exact development seed selection are in
  [TRAINING_RECEIPT_20261114.md](TRAINING_RECEIPT_20261114.md).
- The first evaluation job in
  [CI #43593270279](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43593270279)
  failed after a fatal Lean simplifier process exit during
  `FoundryProbe.two_currency_zero_at_lock`. Its partial score is
  **excluded entirely**. All original attempt and compiler logs are
  preserved as artifact `ram04-first-prospective-evaluation-43593270279`.
  The pre-rerun [engineering amendment](EVALUATION_AMENDMENT_A.md)
  records the observed exception and fixed bounded recovery behavior.
- The **complete first valid run**,
  [CI #43594537281](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43594537281),
  reran all twelve theorems for every controller from the exact original
  checkpoint artifact, without weight, grammar, seed, source selector
  or search-budget modification. Recovery records a terminated Lean
  subprocess as an *unknown* action outcome and uses the original
  at-most-two server restarts.

## Full contemporaneous native comparison

Primary budget: 30 seconds, 48 logical Lean tactic evaluations,
eight successful transitions per path.

Secondary budget: 60 seconds, 128 logical evaluations,
sixteen transitions per path. Both budgets include session startup,
final original theorem compile and independent exact-source replay.

| Search phase | Controller | Theorems certified | Total search wall seconds | Logical tactic evaluations | Actual Lean processes |
| --- | --- | ---: | ---: | ---: | ---: |
| Primary | Native fixed ordering | **9/12** | 35.061 | 139 | 32 |
| Primary | Train-only frequency ordering | **9/12** | 35.663 | 147 | 32 |
| Primary | Looped32, 85,061 parameters | **9/12** | 35.066 | 134 | 32 |
| Primary | Looped64, 206,981 parameters | **9/12** | 36.556 | 171 | 32 |
| Primary | Flat64, 182,085 parameters | **9/12** | 35.446 | 148 | 32 |
| Secondary | Native fixed ordering | **9/12** | 44.088 | 299 | 32 |
| Secondary | Looped64 | **9/12** | 44.671 | 331 | 32 |

The same nine original theorems were certified by all seven
controller/budget combinations. The unique theorem count is nine.
No successful original theorem used the new parameterized `exact`,
`rw`, `rcases` or `induction` actions. The complete successes used
`omega`, `rfl`, locally visible definition simplifications and
occasionally `intro` or `all_goals simp_all`.

The cached model supervision and training outcomes remain genuine
compiler transitions. Their success does not automatically indicate
a new complete formal proof.

## Original theorem case matrix

| Original MathCan theorem | All seven policies |
| --- | --- |
| OpcodeRefinement.two_step_gross_budget | Certified |
| OpcodeRefinement.two_step_output_aggregation | Certified |
| OpcodeRefinement.boundary_liquidity_transition | Certified |
| FoundryProbe.two_nonzero_after_swap | Certified |
| FoundryProbe.two_currency_zero_at_lock | Uncertified: persistent fatal simp recursion |
| FoundryProbe.erc20_custody_neutral | Certified |
| TOETwoFormTriad.orthogonal_triad_isotropic | Certified |
| TOETwoFormTriad.two_axes_missing_third | Certified |
| TOETwoFormTriad.two_fixed_axes_not_positive_isotropic | Uncertified: evaluation budget |
| TOETripletJordan.exact_source_factor | Uncertified: evaluation budget |
| TOETripletJordan.exact_lightcone_null | Certified |
| TOETripletJordan.exact_lightcone_det_zero | Certified |

All seven policies encountered the same three unsuccessful theorems.
The simplification crash case incurred three recorded fatal Lean
process exits per controller, for **21 total** across primary and
secondary. They remained unknown observations and each controller
terminated within its registered two-restart cap. The failures reveal
a search/prover robustness limitation, not a certified impossibility
of proving those original statements. Their unchanged original MathCan
reference proofs compile.

## Independent proof and byte audit

The completed CI evaluation generated **84 scored
policy/theorem trajectories** and **63 accepted instances**. The
separate post-score audit freshly recompiled each accepted original
theorem using the exact source bytes, checked SHA256 agreement with
the scored final and prior replay sources, and verified the explicit
axiom gate. It found **zero false-positive certificates** and
preserved all original success/failure verdicts.

- Original complete scored JSON report SHA256:
  `c3f71628e953b4f978c25fb3815dfaded02d9b324fa98febd5aefab319e8c80a`.
- Independent audit receipt SHA256:
  `ad9edf01afeda844bb549b78be59dd8312a892359cb152405b5ab8cebd58fb0c`.
- Manifest of 63 independently verified original source hashes SHA256:
  `c225cba10492c6453a2c532647fc7e90514bd10a51b38e82754a41e90c9f3ee5`.
- All 4,973 original train-only source actions were independently
  replayed and frozen before test; dataset SHA256
  `a84b1264e2eee76195041efc81fbcd7ec60e48c22068b7eba3c45ce85838f0b3`.
- Complete machine-readable summary and receipt:
  [first_complete_eval02_20261114.json](receipts/first_complete_eval02_20261114.json).

The original GitHub Actions artifact
`ram04-full-prospective-eval02-43594537281` archives
**3,470 generated files**, including all positive and negative
compiler records, full source/scripts and repeated-source proofs.
The artifact expires after the standard retention period;
the compact reproducibility receipts remain committed.

## Interpretation and next experiment

The first frozen original source-family transfer shows that the
bounded RAM-03 recurrent policy did not improve complete proof
coverage over contemporaneous native search. Additional recurrent
capacity increased the sampled tactic-search count in the
207k model with no solved-set change. Measured wall times here are
single sequential run observations, insufficient for a robust
speed comparison. Extra independent compiler replay is unscored.

One source-family case exposes repeated simplifier recursion
through visible `MathCan.V4Transient.count` and `nz` definitions.
Other cases exhaust logical proof-search budgets. The follow-on
mathematical question concerns a safer controlled unfolding or
lemma retrieval policy and deeper branching search, with a fresh
held-out set preregistered before modifying tactics or model weights.
The present experiment remains negative even if such post-hoc
repairs later solve one of these original twelve theorems.
