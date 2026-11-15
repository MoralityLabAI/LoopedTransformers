# RAM-08 independently audited natural-exact CPU training | November 15, 2026

The source-only job of [GitHub Actions run 43689874169](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689874169)
**succeeded**. This is a measured training-only receipt, distinct from
the still-unscored 12-theorem new-PEF source evaluation.

## Exact source, protocol and audit

- [RAM-08 preregistration](PREREGISTRATION.md) originally committed
  at `b8001e50e89401b44c8acb276f4c15e02c4b5b05`,
  before any new RAM-08 compiler collection and test source access.
- Original TRAIN/DEV source MathCan
  `c84e2025334e9a967c1297425a490f98d5b3d189`,
  Lean `leanprover/lean4:v4.35.0`.
- 62 distinct original theorem roots, of which 57 belong to TRAIN
  and five belong to the original fixed DEVELOPMENT groups.
  The 47-root RAM-07 cohort was augmented by 18 specified source
  declarations; three overlapped existing roots and were
  transparently excluded:
  `Intervention.obsEqOn_action_subset`,
  `FiniteHorizonGap.epsilon_sixth_equiv`,
  `DeletionMemory.correct_decoder_separates`.
- 640 genuine original TRAIN compiler goal states, 304 of them
  multigoal; 74 genuine DEVELOPMENT states, 36 multigoal.
- Every one of **20,930 registered, concrete Lean tactic effects**
  was independently replayed in **62 fresh Lean4.35 sessions**,
  with **zero disagreements** and no rewritten legacy 961 labels.

| Original source split | Distinct goal states | Positive scratch closures | Additional progress | Failed or unchanged | Concrete action observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 640 | 347 | 3,162 | 15,149 | **18,658** |
| DEVELOPMENT | 74 | 29 | 397 | 1,846 | **2,272** |
| Combined | 714 | 376 | 3,559 | 16,995 | **20,930** |

The original TRAIN-only candidate dataset SHA256 is
`f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`.
The exact registered RAM-08 config SHA256 is
`2812d03012c3dd5f67c6b1f7c95989b9d376da3677701e74b18d74779fa577d9`.
The original independent source-only audit reports
`status=passed`, `checked=20930` and
`independent_fresh_lean_sessions=62`.

## Natural `exact` diversity: source-root minimum passed

The unmodified original MathCan source-root collector obtained
**eight** positive `exact_typed`/`exact_local` candidate occurrences,
which deduplicate to **five** unique
(source-file, theorem, visible-state-hash, script) combinations
across **three distinct original TRAIN theorems**.
This exceeds both registered training gates of
two source theorems and three unique actions.

| Previously proved original MathCan root | Compiler-checked positive candidate example | Source prefix |
| --- | --- | --- |
| `Intervention.obsEqOn_trans` | `exact (hst ram07_arg0 ram07_arg1).trans (htr ram07_arg0 ram07_arg1)` | `intro ram07_arg0`; `intro ram07_arg1` |
| `TOEMixedPrincipal.independent_kinetic_and_spectral_gates` | `exact hk` | `constructor` |
| `TOENonAbelian.zero_mixed_nonzero_source` | `exact h` | `constructor`; `omega` |
| `TOENonAbelian.zero_mixed_nonzero_source` | `exact ram07_arg0` | `constructor`; `omega`; `intro ram07_arg0`; `apply h` |

The repeated action-family rows `exact_local` and
`exact_typed` at the same proof-state and tactic text are
separate **family-label observations**, not independent
theorems. This distinction is preserved in the training receipt.
There are five positive typed-family TRAIN records but a smaller
number of distinct underlying semantic scripts, with no
assumption of broad `exact` generalization. The only
positive `exact` development instances remain on the separate
`Intervention.obsEqOn_symm` source-root group, excluded from training.

The new shared `decide` family was tested on 640 TRAIN states
and 74 DEVELOPMENT states. It recorded 54 positive TRAIN
action outcomes and zero positive DEV outcomes. A
successful scratch action is still **not** an independently
certified original theorem proof.

## Nine original low-memory CPU model fits

Training used Python3.11, PyTorch2.6.0 CPU,
original Lean4.35, two CPU optimizer threads and the
original RAM-03 resource guard. The fixed 30-update
loop64 CPU pilot measured **53.6123 updates/sec**;
no CUDA device was available in the GitHub runner.

| Architecture | Parameters | Dev-selected seed | Best optimizer update | Balanced dev loss | Selected original best.pt SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Four-pass Looped32 | 85,157 | 29 | 300 | 0.639684 | `4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f` |
| Four-pass Looped64 | 207,173 | 43 | 100 | 0.679183 | `5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac` |
| Flat64, zero recurrence | 182,277 | 29 | 200 | 0.589670 | `2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8` |

All nine architecture/seed fits and original best/last
optimizer, sampler and Torch RNG checkpoints were completed
and saved. Their hashes were verified **before** any
new-PEF source theorem could be submitted for scoring.

- Peak CPU model-training process-tree RSS:
  **575,410,176 bytes (~548.8 MiB)**.
- Peak source-collection/replay process-tree RSS:
  **1,921,900,544 bytes (~1.79 GiB)**.
- Zero resource-guard aborts.
- Minimum available runner system RAM
  approximately 6.9 GiB.

The exact, complete original artifact from the successful
TRAIN job is `ram08-frozen-audited-train-43689874169`,
GitHub Actions artifact ID `14152063191`,
20,496,574 compressed bytes and ZIP SHA256
`a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`.
It includes all nine best/last checkpoint states and every
original compiler observation, recorded train/dev split,
calibration and independent audit.

## PEF heldout evaluation status

The 12 original new-PEF theorems remain the registered first
three declarations from CausalInteraction, OrderedTangent,
PairedHistory, PrecisionEntropyRobust, pinned to unmerged
original MathCan PR #44 source commit
`83f61997948818b68da94429c2fe736d93ea3093`.

The dependent test job failed *before executing any steps*
while GitHub reported no assigned hosted runner. The
standalone evaluation-only replacement and one alternate
Ubuntu runner label also terminated with zero executed
steps. [Evaluation amendment A](EVALUATION_AMENDMENT_A.md)
records the infrastructure blocker. **No PEF original
theorem scores can be reported** from these failures.
A complete evaluation must retain these exact original
checkpoint bytes, all registered budgets and all three
symbolic controls.
