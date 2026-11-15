# RAM-07: authentic original MathCan natural exact mining and CPU training

**November 15, 2026. Source-only training was successful after disclosed, pre-heldout queue amendment A.**

## Original provenance and immutable first attempt

The first breadth-first train-only run `43682247680`
independently checked 10,564 original MathCan tactic attempts,
zero disagreements and **zero natural original TRAIN exact positives**.
The preregistered minimum shut down training. All attempts were
archived as `ram07-train-failure-43682247680`, SHA256
`d800dd9aadf866bf491d1be1931fcac4cfd77415f9cdb97bd2ad5fa2fc8e1de0`.
The new test source commit was never checked out by that run.

[The pretest scheduling amendment](PRETEST_MINING_AMENDMENT_A.md)
records the source-only investigation and queue repair:
genuine successful `intro ram07_argN` successors are processed
before competing simplification/induction siblings, while the
12-state, depth5, 8-goal and 19-family grammar limits,
original theorem roots, and heldout test cases remain fixed.

The independent [train-only smoke run `43682572642`](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682572642)
closed original MathCan `Intervention.obsEqOn_trans`
and original development `obsEqOn_symm` with only generated
named-intro prefixes and actual compiler-visible local
hypothesis exact applications. Both full source proofs passed
separate byte-identical Lean4.35 compilation with zero axioms.
No original teacher prefix commands were used.

## Full amended 47-root source-only data audit

[Successful amended run `43682912918`](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682912918)
collected the original 27 FC-12 roots plus 20 additional,
previously evaluated original MathCan training theorem roots,
using the **old original source commit**
`c84e2025334e9a967c1297425a490f98d5b3d189`,
Lean4.35. The dependent prospective test uses a different,
newly pinned MathCan revision.

| Split | Original root theorems | Real compiler goal states | States with multiple goals | Actual compiler-probed actions |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 42 | 320 | 143 | 9,163 |
| DEVELOPMENT | 5 | 50 | 27 | 1,481 |
| Combined | 47 | **370** | **170** | **10,644** |

All 10,644 before/after visible-goal and label observations
were **independently replayed in 47 new Lean sessions**, with
zero outcome discrepancies. The original 961 FC-12 recorded
candidate effects and original theorem sources remained
unchanged. No synthetic `RAM06/ExactCurriculum` rows entered
either RAM-07 training or development.

- Original new dataset SHA256:
  `7ac405fabdb559fb1ffa1baae54c29197177d24ace95a8fad7bb60df888cadd4`.
- Registered RAM-07 source/train config SHA256:
  `85799df5e2f18fcabbe4dca3193eb04c69ff97caa4cafa728568a7e823c40a84`.
- Original independent audit `status=passed`, `checked=10,644`
  with `independent_fresh_lean_sessions=47`.

### Natural exact-positive provenance

The original source-domain TRAIN subset contains **exactly
one distinct independently compiler-verified positive
`exact_typed` action**:

`MathCan/Intervention.lean:obsEqOn_trans`
- Natural prefix: `intro ram07_arg0`, then
  `intro ram07_arg1`, both accepted from the original
  theorem root without any reference proof commands.
- Actual positive action:
  `exact (hst ram07_arg0 ram07_arg1).trans (htr ram07_arg0 ram07_arg1)`.
- Lean confirms all goals close. The positive input
  visible-state SHA256 is
  `1c234139f67b5ae9e9f1fa51f3efbe61f3043766e50bcea5aad7d801af60a7d6`.

The separate development root
`MathCan/Intervention.lean:obsEqOn_symm` has an independently
verified positive application
`exact (h ram07_arg0 ram07_arg1).symm`
at visible-state SHA256
`fa4979b5457f9fb23b9ad9f3df35fa9b1ca678970b6fcbf10be8f1e611de9bc3`.
It appears twice in the new development candidate records,
and those repeated records represent **one source proof
context**, not independent distinct theorem examples.

These are previously proved original MathCan statements.
The experiment demonstrates a naturally obtained Lean
`exact` training label but remains **extremely sparse in
natural-source positive exact coverage**: a single positive
TRAIN action from a single source root. It should never be
misrepresented as strong statistical evidence of natural
exact generalization. The source-family heldout test must
establish any proof-coverage gain separately.

## Nine CPU-only checkpoints frozen before heldout access

All nine fits completed, with the fixed optimization
setup and 10% natural-positive exact oversampling within
the effective batch of64. **Model checkpoint selection was
entirely by the original five-source DEVELOPMENT loss**.

| Architecture | Parameters | Selected seed | Best optimizer update | Balanced development loss | Exact best-checkpoint SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Looped32 | 85,125 | 29 | 100 | 0.620855 | `15aefb27ef9dd6cb9219fcd1e2e826fa542159a68b2eb2dc2ac95a03af4f271c` |
| Looped64 | 207,109 | 17 | 200 | 0.641407 | `24be13905f50f111b2e3c285af48bfef003de97b1d1d857adbcd859781a04eda` |
| Flat64 | 182,213 | 29 | 100 | 0.661141 | `ba04242b82b3a08a6c96d74352a875beb40e468719bfa570b243b53c9d8b6777` |

All nine `best.pt` and `last.pt` checkpoints and their
optimizer/RNG state, full loss histories and every
compiler-observed TRAIN label are in immutable GitHub
artifact `ram07-frozen-audited-train-43682912918`,
artifact ID `14148788243`, 19,525,397 compressed bytes,
ZIP SHA256
`c622367da9a7ced515df1db2869937b4edb61f3cc0736b4b51035c5442038db0`.

## Memory and compute

Training on hosted CPU Linux with Python3.11,
CPU-only PyTorch2.6.0, two training threads, Lean4.35
and unchanged resource monitoring:

- CPU 30-update pilot measured **80.1501 optimizer
  updates per second** for loop64.
- Peak sampled whole-process tree RSS was
  **463,654,912 bytes (442.18 MiB)**.
- Lowest sampled available system RAM was
  **6,965,780,480 bytes**.
- Zero training memory aborts, zero new compiler
  replay failures, nine successful model checkpoint
  byte/hash checks.

These measurements concern the CPU GitHub runner, not
the user's local 4GB RTX3050 GPU. Source-only training
uses the **older MathCan checkout** and does not inspect
the new four-module original heldout revision
`fca46bfb3430ac85ffca16f6a6a83bb0b0231e41`.
The separate prospective proof score is measured only
by the dependent evaluation job.
