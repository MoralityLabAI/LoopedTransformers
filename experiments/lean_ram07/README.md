# RAM-07: natural source-derived exact learning with new-family transfer

**Branch:** `feat/lean-ram07-natural-exact-mining`, stacked on
RAM-06 without changing any previous paper, checkpoint or result.
The original [preregistration](PREREGISTRATION.md) froze the
new MathCan source-family test and genuine Lean-mining protocol before
any RAM-07 collector or model evaluation started.

## What is different

RAM-06 verified positive typed exact only in auxiliary synthetic
tautologies. All 793 exact candidates in its actual MathCan
TRAIN/DEV corpus failed. Inspection of the immutable old FC-12
source goal trace exposed a tractable representational defect:
the plain `intro` tactic gave local names such as `word✝` which
the contextual argument parser rejected, preventing later use
of valid original hypotheses.

RAM-07 adds one deterministic `intro ram07_argN` action per
original Lean-visible goal and a bounded source-derived
application, equality-symmetry and equality-transitivity
candidate pool. It does not insert premises, a duplicated
conclusion, theorem-reference scripts or any synthetic
auxiliary goal into the training corpus.

Every candidate is a genuine Lean action against an original
source-root proof state, and every new accepted/failed effect
is independently replayed in a separate Lean session.
The original reference proof prefix is never consulted by
the collector. A successful `exact`-based scratch closure
is a learning label; **only a fully compiled original new
MathCan theorem** is a scored result.

## Two immutable MathCan checkouts

RAM-07 uses different original source revisions, pinned in
`config.json` and [the preregistration](PREREGISTRATION.md):

- **Training only:** `patrickdugan/MathCan` at
  `c84e2025334e9a967c1297425a490f98d5b3d189`. Its
  original FC-12 train-only data artifact from run
  `43547762837` is required and hash checked.
- **First prospective evaluation only:** the newer MathCan
  commit `fca46bfb3430ac85ffca16f6a6a83bb0b0231e41`
  with four modules introduced after the old training pin:
  CoarseStateDependence, OrderedJet, TacticFibers and
  TreeReconstruction. The first three source-order original
  theorem declarations in each are frozen as twelve test
  cases. `OrderedJet` is explicitly built because it is not
  included by the top-level `MathCan.lean` at this commit.

The new test revision must be a **separate checkout** and
cannot be opened by the training job. Both revisions use
Lean4.35.0.

### Natural TRAIN roots and model selection

The original 27 FC-12 roots are retained, including five fixed
development-only theorem groups. The first five theorem
declarations each from four old, historically already
evaluated modules (Hodge, FundedHooks, TOEJordan,
TOETensorConstraints) add up to twenty new training roots.
All source branches are bounded to 12 actually observed
goal states, at most five steps, maximum eight simultaneous
goals. All source roots are original formal MathCan
theorems, rather than generated proof tasks.

A model may fit only if the original **TRAIN** split contains
at least one genuinely compiler-verified **positive natural
`exact_typed` or `exact_local` action**, independently
replayed by Lean, with its full original source theorem,
observed prefix and source SHA archived. Zero such positives
requires a hard stop, preserving all attempted evidence
and leaving the new test modules untouched.

Loop32, loop64 and flat64 have fixed seeds 17,29,43.
Training reserves 50% of the effective batch for actual
multi-goal contexts and 10% for positive *natural* exact
source contexts, with the remainder sampled from other
real training rows. There are no synthetic RAM-06 examples.
AdamW lr.002, decay.01, clip1.0, microbatch16×
accumulation4, at most1600 optimizer steps and fixed
development-only early stopping.

## Strong contemporaneous controls

Every proof search uses the same registered 19-family
compiler-visible candidate generator. Primary search:
native, train-only global family-frequency, train-only
smoothed goal-shape frequency (prior8), loop32, loop64,
flat64; 30s/48 logical evaluations/8 transitions per
original theorem. Secondary: native, global frequency,
shape frequency, loop64; 60s/128/16.

Each scored success requires all goal branches closed,
a sorry-free Lean4.35 original-source theorem compile,
`#print axioms` permitting only `propext`,
`Classical.choice` and `Quot.sound`, and an independent
byte-identical full-source replay inside its budget.
A third post-score independent Lean source compile checks
every accepted original theorem without changing the
score. The primary criterion requires an original new
theorem that **all three non-neural comparators** miss;
a recurrence-specific result must also exceed flat64.

## Reproduce on GitHub Actions

The [RAM-07 workflow](../../.github/workflows/lean-ram07-prospective.yml)
keeps both immutable MathCan commits separate. The
training job checks out only the old source, builds its
old import dependencies, mines the registered original
roots, verifies all compiler labels in independent
Lean sessions, and trains nine CPU models. It then
uploads source observations and frozen byte-hashed
best/last checkpoints.

The dependent heldout job checks out the new source
commit, compiles MathCan and OrderedJet at that commit,
verifies all original model artifact hashes, and starts
the new twelve-theorem original full proof comparisons.

CPU-only runtime: Python 3.11, PyTorch 2.6.0 CPU,
psutil7.1.3, two CPU training threads and original
Lean4.35. Total process-tree RAM cap 6GiB; all
compiler processes and RAM failure logs remain in
the named CI artifact. Model-positive training data
and failed original test searches are retained rather
than selected by outcome.

## Reproduce locally on 4GiB RTX3050 / ~10GiB free RAM

Prepare separate physical folders for old TRAIN MathCan
and new TEST MathCan; never change revision in place.
Prepare the exact frozen original FC-12 data using
the authenticated GitHub CLI:

`gh run download 43547762837 -R MoralityLabAI/LoopedTransformers --name fc12-frozen-typed-lean-models --dir "D:/Research Engine/RAM-07/old-fc12"`

The local PowerShell runner, on the RAM-07 branch,
will check source commits and Lean toolchain:

`powershell -ExecutionPolicy Bypass -File experiments/lean_ram07/run_windows.ps1 -MathCan "D:/Research Engine/MathCan-old" -TestMathCan "D:/Research Engine/MathCan-new" -Gym "D:/Research Engine/Compactification-Theory" -OriginalFC12 "D:/Research Engine/RAM-07/old-fc12" -Out "D:/Research Engine/RAM-07/runs/first" -Python "D:/Research Engine/ram03-env/Scripts/python.exe" -LeanBin "D:/Research Engine/lean435/bin"`

Use fresh output directories for every run. The CPU/GPU
calibration chooses the CPU when CUDA gives less than a
1.15× improvement on the identical micro-model step.
Whole-process memory is checked continuously, capped
at 6GiB with 2GiB free system RAM headroom and
2.5GiB allocated CUDA cap.

Original reference proofs and new target-module scripts
never enter the model's training or generated action
features. The complete test result is a measured
generalization outcome, including all negative cases.
No merge, paid runner or deployment.
