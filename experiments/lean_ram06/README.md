# RAM-06: compact Lean reasoner with independently verified typed exact

[Preregistration](PREREGISTRATION.md) was frozen at commit
`f4f26ed48ed675cc7b80c73f151ba3927b46bafa` before collecting
RAM-06 labels or testing any new original MathCan theorem.

RAM-06 keeps the original compiler-backed persistent Lean search from
RAM-05 and adds a **source-visible, bounded typed exact family**.
All generated candidates use names and hypothesis types obtained from
the printed Lean goal, with a strict identifier grammar and 18 typed
candidates per state. Legal forms include:

`exact h`, `exact h.symm`, `exact pair.left`, `exact pair.right`,
`exact f hp`, `exact f hp hq`, `exact bridge.mp hp`,
`exact bridge.mpr hq`.

Candidate scripts remain untrusted until the Lean interpreter accepts
the actual tactic. A progressed goal state is a **training signal**;
a solved original theorem requires a separate full original-source
Lean compilation, axiom audit and byte-identical replay.

## Different proof origins and honest labels

The original MathCan root corpus remains the frozen 27-theorem
FC-12 TRAIN selector, with original source SHA
`c84e2025334e9a967c1297425a490f98d5b3d189` and 51 source
states/961 original effects. RAM-06 independently recompiles any newly
generated candidate outcomes from those roots under Lean4.35.

[ExactCurriculum.lean](ExactCurriculum.lean) declares 12 separately
generated **TRAIN** and six separately generated **DEVELOPMENT**
synthetic Lean propositions, including exact hypotheses, implication
application, conjunction projections, symmetric equality, and iff.
These synthetic examples are **auxiliary training/debugging tasks**;
they are never scored as original MathCan formal theorems.
Every synthetic candidate effect and every negative is independently
replayed in a fresh Lean session. The data loader requires >=10
distinct synthetic train and >=4 development exact-tactic positives,
with negative controls and no train/development goal collision.
All original root observations are preserved separately.

The exact curriculum occupies a predeclared 20% of each 64-example
effective optimizer batch. Recurrent models remain width32/64,
four tied passes, and flat64 is an equal-protocol nonrecurrent control.
Fixed seeds17,29,43; original source-development groups plus six
synthetic-only development examples determine checkpoint selection.

## First new MathCan theorem-family test

Exactly the first three original theorem declarations from:

- `MathCan/FundedHooks.lean`
- `MathCan/TOEFLRW.lean`
- `MathCan/TOECovariantGates.lean`
- `MathCan/TOEMixedPrincipal.lean`

These four modules were excluded from all previously selected
FC-11/12/13 and RAM-03/04/05 train and prospective evaluations.
Original source-reference proofs compile separately as the 12/12
oracle. The author-made preexisting proofs never enter model
features, candidate generator, source ranking or training labels.

Fixed primary search: native, global training-only frequency,
smoothed goal-shape training-only frequency (prior weight8),
loop32, loop64 and flat64; 30s,48 actual tactic evaluations,
eight steps per theorem. The secondary runs native, frequency,
shape-frequency and loop64 under 60s,128 evaluations,16 steps.
Each controller gets the same source-visible 18-family grammar and
bounded multigoal persistent Lean proof search. Original full
certificate and same-source replay are within the time limit.
A further post-score independent Lean recompile checks the saved
positive source bytes and axiom inventory.

**Registered improvement** requires one source-family-held-out
original theorem certified by a recurrent model and missed by
all three non-neural contemporaneous baselines, with zero false
positive certificates. A success also missed by flat64 is
reported separately as recurrence-specific evidence.

## CPU GitHub Actions and artifact reproduction

[RAM-06 source-only collector, nine CPU fits, and prospective Lean
comparison](../../.github/workflows/lean-ram06-prospective.yml)
runs in two separate jobs. The first job checks the registered
grammar and synthetic source, independently verifies every MathCan
and generated curriculum label, and only afterward trains and
freezes nine CPU checkpoints with optimizer/RNG states and SHA256s.
The second job can open the new MathCan family goals only after
the first job succeeds and uploads its frozen artifact.

Runtime: Python3.11, pinned CPU PyTorch2.6.0, psutil7.1.3,
Lean4.35.0, two CPU threads. The existing RAM-03 resource monitor
limits whole-process RAM to 6GiB. The original previous RAM-05
checkpoint and result files are unchanged.

Reproduce locally on Windows RTX3050 (CPU fallback), after preparing
the exact pinned original MathCan and Compactification-Theory
checkouts and downloading the frozen FC-12 train-only CI artifact:

`gh run download 43547762837 -R MoralityLabAI/LoopedTransformers --name fc12-frozen-typed-lean-models --dir "D:/Research Engine/RAM-06/original-fc12"`

Then, on the RAM-06 feature branch:

`powershell -ExecutionPolicy Bypass -File experiments/lean_ram06/run_windows.ps1 -MathCan "D:/Research Engine/MathCan" -Gym "D:/Research Engine/Compactification-Theory" -OriginalFC12 "D:/Research Engine/RAM-06/original-fc12" -Out "D:/Research Engine/RAM-06/runs/first" -Python "D:/Research Engine/ram03-env/Scripts/python.exe" -LeanBin "D:/Research Engine/lean435/bin"`

Use actual directories appropriate to the laptop. The runner
redirects temporary files to the experiment drive, checks every
input/version, compiles the Lean helper, collects original states,
independently verifies synthetic exact examples, trains the models
under RAM/VRAM caps, and only afterward builds the MathCan project
and executes the untouched original theorem test. Complete
source/Lean logs and every negative candidate are retained in
the output directory.

To inspect the code in separate stages, use
`experiments/lean_ram06/collect.py`,
`exact_curriculum.py`,
`train_local.py`,
`evaluate.py` and `audit_results.py`.
All finished receipts distinguish natural MathCan compiler outcomes,
generated verified auxiliary actions and certified complete original
theorems. No auto-merge.
