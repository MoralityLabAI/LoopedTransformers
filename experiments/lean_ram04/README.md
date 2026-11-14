# RAM-04 | prospective family-held-out Lean reasoners

[Preregistration](PREREGISTRATION.md) was committed as
`ae7be99b6367434f0dd724ae232cfa218ed99342` before
any RAM-04 scorer ran. This branch is stacked on the unmerged RAM-03
branch, preserving its train/dev results and original proof certificates.

## Research scope

The new primary suite consists of twelve original MathCan theorems from
OpcodeRefinement, FoundryProbe, TOETwoFormTriad and TOETripletJordan,
with three source-order declarations per file. These source families
were omitted from the previous imported training and evaluation suites.
The original MathCan source is pinned to
`c84e2025334e9a967c1297425a490f98d5b3d189`, Lean 4.35.0.

RAM-04 retains the RAM-03 persistent Lean proof backend, branch goal
tracking, 17 registered tactic families, source renderer and axiom gate.
Every candidate is instantiated from the actual printed compiler
goal and its visible local context. All branches must close before
a complete original theorem is separately compiled and replayed.
The source-reference oracle is counted separately.

The models are loop32 (~85k), loop64 (~207k) and flat64 (~182k).
Fit each from train-only compiler evidence with seeds 17, 29 and 43,
then select the seed on the same five-theorem development split used
in RAM-03. The source-family test selector remains untouched until
the independently audited dataset, all nine fits and their exact
checkpoint hashes have been frozen.

Compare fixed native ordering, training-only family frequency, and the
three development-selected models with 30 seconds, 48 logical Lean
evaluations and eight steps per path. A separately preregistered
secondary comparison uses 60 seconds and 128 evaluations for native
and loop64. Both use the same 17-family tactic grammar.

## GitHub Actions experiment

[RAM-04 training and prospective evaluation workflow](../../.github/workflows/lean-ram04-prospective.yml)

The workflow runs on CPU-only Ubuntu runners with Python 3.11,
PyTorch 2.6.0 CPU, psutil 7.1.3 and pinned Lean 4.35. It:

1. Downloads the SHA-frozen original FC-12 train-only data artifact
   from workflow run `43547762837`.
2. Builds original MathCan and the unchanged RAM-03 persistent server.
3. Collects original compiler-observed descendant goals from the 27
   TRAIN theorem roots and independently replays every candidate label.
4. Trains nine small models on train-only records, calibrating CPU
   throughput and enforcing the RAM-03 resource guard.
5. Uploads the audited training data, weights, optimizer states and
   SHA256 hashes before the held-out evaluation job begins.
6. Checks the fixed new theorem selector, compiles the original proof
   oracle and executes the matched native/frequency/model search.
7. Recompiles every accepted Lean proof in an additional independent
   post-score process and archives every original script, compiler
   transcript, failure and replay.

If the compiler data or action semantics change on Linux, the
independent audit is a hard prerequisite and the mismatch is reported.
The scored evaluation is skipped when training fails. CI artifacts are
retained for 30 days; copy the named artifacts into a durable workspace
if long-term retention is required.

## Reproduce on the Windows RTX 3050 laptop

Use the existing RAM-03 Python/PyTorch environment and unchanged
original MathCan and Compactification-Theory checkouts. The current
local experiment directory should be new; RAM-02 and RAM-03
checkpoints and proof artifacts remain untouched.

The original FC-12 train-only artifact can be obtained using the
GitHub CLI after signing in:

`gh run download 43547762837 -R MoralityLabAI/LoopedTransformers --name fc12-frozen-typed-lean-models --dir D:/ResearchEngine/ram04-input`

From the LoopedTransformers working directory on this branch:

`powershell -ExecutionPolicy Bypass -File experiments/lean_ram04/run_windows.ps1 -MathCan D:/ResearchEngine/MathCan -Gym D:/ResearchEngine/Compactification-Theory -OriginalFC12 D:/ResearchEngine/ram04-input -Out D:/ResearchEngine/RAM-04/runs/first -Python D:/ResearchEngine/ram03-env/Scripts/python.exe -LeanBin D:/ResearchEngine/lean435/bin`

The runner reconstructs and audits the training states, fits nine
models and only afterward starts the prospective comparison. If the
main script fails, its output and compiler logs remain in the new run
directory for forensic inspection. CPU is selected whenever CUDA
fails the predeclared 1.15× throughput threshold. A GPU is optional.

A certificate earns proof credit only after a full, sorry-free Lean
compile and a separately verified byte-identical replay. A developer
regression score, scratch closure or source oracle is not a learned
complete theorem proof. The evaluation report records model-only and
native-only cases, including all failures. No threshold or case
selection may be tuned on these new twelve theorem results.

## Reproduce with explicit commands

Linux/WSL2 (using paths adjusted to your environment, a matching Linux
Lean executable, and an accessible original training artifact):

`python experiments/lean_ram03/collect.py --data /data/fc12-original --mathcan /src/MathCan --lean lean --config experiments/lean_ram04/config.json --out /out/ram04-data`

`python experiments/lean_ram03/train_local.py --data /out/ram04-data --config experiments/lean_ram04/config.json --out /out/ram04-train`

`python experiments/lean_ram04/evaluate.py --data /out/ram04-data --training /out/ram04-train --mathcan /src/MathCan --lean lean --out /out/ram04-test`

`python experiments/lean_ram04/audit_results.py --results /out/ram04-test --lean lean --out /out/ram04-audit`

These commands require LEAN_PATH to include the original MathCan
compiled library and the directory containing ProbeServer.olean.
Run the full published unit tests before training.
