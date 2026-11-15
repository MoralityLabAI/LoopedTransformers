# RAM-05 | low-memory compiler-supervised Lean tactic reasoner

**Status:** prospectively preregistered; original 30s/60s proof evaluation
is gated on independently replayed training labels and frozen checkpoints.
See [PREREGISTRATION.md](PREREGISTRATION.md). The initial protocol commit
is `215de8648d6db73352a4118f5507b45f9eb4f814`.

## Scientific question

RAM-04 solved the same nine original theorems as a fixed native tactic
baseline. Its one troublesome FoundryProbe declaration exhausted the
Lean simplifier through unbounded visible-definition expansion.
RAM-05 uses that *published* failure solely to design a bounded
definition-unfolding grammar; it measures generalization on a different
set of twelve original, previously unused MathCan theorem declarations.

The new registered 17-family grammar preserves the original concrete
context-instantiated `exact`, `rw`, `rcases`, `induction` and other
tactics, while replacing unrestricted MathCan definition batches with
up to three locally visible single-constant actions:
`simp only [MathCan.Definition]`,
`simp_all only [MathCan.Definition]` and
`dsimp only [MathCan.Definition] at *`.

All controllers use the same grammar. A terminated Lean subprocess is
recorded as an *unknown* observation with the exact candidate script,
and the existing two-restart bound applies. Training labels are accepted
only after an independent Lean replay, and the full theorem proof gate
requires separately compiled, sorry-free source with an explicit axiom
audit plus an exact-source replay.

## Fixed evaluation source

Original untouched MathCan revision
`c84e2025334e9a967c1297425a490f98d5b3d189`, Lean 4.35.0.
First three top-level theorem declarations in each new module:

- `MathCan/TOETwoMultiplet.lean`
- `MathCan/TOEYangMills.lean`
- `MathCan/TOEConeMatching.lean`
- `MathCan/TOETensorConstraints.lean`

The exact twelve names and their original source-order checks are in
`config.json` and the preregistration.

Training is confined to the old 27 original MathCan TRAIN theorem
roots and the same five-source theorem-development selector used in
RAM-03. There is no use of RAM-04 failed theorem test labels or new
prospective proof scripts to train the student or change the grammar.

## GitHub CPU execution

The [RAM-05 workflow](../../.github/workflows/lean-ram05-prospective.yml)
runs independent train-only label replay, then nine CPU fits, then
a separately provisioned original theorem evaluation job. The first
training stage deliberately avoids building the four new heldout
MathCan modules. The evaluation job starts only after all nine
checkpoints are hashed and the train-only artifact is archived.

- Python 3.11, PyTorch 2.6.0 CPU, psutil 7.1.3, Lean 4.35.0.
- Looped32, Looped64 and flat64, seeds 17,29,43, development-only
  checkpoint selection, four recurrent passes.
- Same primary budget: 30 seconds, 48 logical evaluations,
  eight transitions per path. Same secondary: 60 seconds, 128 logical
  evaluations, sixteen transitions per path.
- At least one **new recurrent-model-only certified theorem** versus
  contemporary native search at the primary budget is the registered
  primary criterion. Report recurrence-specific gain only if it also
  exceeds flat64 and train-frequency policies.

The workflow uploads original model weights and full compiler evidence
as artifacts. The successful proof count may be zero; any true negative
result is preserved.

## Local Windows RTX 3050 execution

Reuse the Python 3.13/PyTorch2.6.0 environment from RAM-03, with
`psutil==7.1.3` installed. The local run enforces the preexisting
process-tree RAM cap of 6 GiB, CUDA allocation cap of 2.5 GiB and RAM
and dedicated VRAM headroom. It chooses CPU unless a short identical
CUDA pilot is at least 1.15× faster. Previous local experiments found
CPU faster for these sub-250k-parameter models.

Prepare three immutable local source/input directories:

- Original MathCan checkout at the exact revision above.
- Compactification-Theory pinned gym checkout at
  `e8a780254b5cf41d479095f4f22bcfc4aa76ed72`.
- The original FC-12 train-only artifact obtained from GitHub workflow
  `43547762837`, named `fc12-frozen-typed-lean-models`, containing
  `train_source/actual_goal_states.jsonl` and
  `train_probes/train_candidate_outcomes.jsonl`.

To fetch the original artifact, run from an authenticated GitHub CLI:

`gh run download 43547762837 -R MoralityLabAI/LoopedTransformers --name fc12-frozen-typed-lean-models --dir "D:/Research Engine/RAM-05/old-fc12"`

From a local checkout of this RAM-05 branch:

`powershell -ExecutionPolicy Bypass -File experiments/lean_ram05/run_windows.ps1 -MathCan "D:/Research Engine/MathCan" -Gym "D:/Research Engine/Compactification-Theory" -OriginalFC12 "D:/Research Engine/RAM-05/old-fc12" -Out "D:/Research Engine/RAM-05/runs/first" -Python "D:/Research Engine/ram03-env/Scripts/python.exe" -LeanBin "D:/Research Engine/lean435/bin"`

The supplied paths are examples. Use a new output directory to preserve
all earlier artifacts. The PowerShell script checks the source SHA and
toolchain before collection; it only builds the full original MathCan
project after all nine new models are frozen.

## Direct Linux / WSL2 stage commands

With Lean4.35 on PATH and the original compiled gym on PYTHONPATH,
compile the RAM-03 `ProbeServer.lean` into its directory and add that
directory to LEAN_PATH. Source-only collection and fitting:

`python experiments/lean_ram05/collect.py --data /data/fc12-train --mathcan /src/MathCan --lean lean --config experiments/lean_ram05/config.json --out /out/ram05-data`

`python experiments/lean_ram05/train_local.py --data /out/ram05-data --config experiments/lean_ram05/config.json --out /out/ram05-models`

Only after audit status passed and all checkpoints have been hashed:

`python experiments/lean_ram05/evaluate.py --mathcan /src/MathCan --data /out/ram05-data --training /out/ram05-models --lean lean --out /out/ram05-test`

`python experiments/lean_ram05/audit_results.py --results /out/ram05-test --lean lean --out /out/ram05-audit`

All accepted source proofs receive one additional independent
post-score compiler run, whose result cannot change the frozen scored
coverage. The final receipt records original source SHA, model
checkpoint hashes, per-theorem failures, and native-only and model-only
successes. Future tuning of simplifier actions against these twelve
test theorems belongs in a separately named RAM-06 experiment.
