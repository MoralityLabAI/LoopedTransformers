# RAM-08 | naturally diversified exact reasoning and PEF transfer

[Preregistration](PREREGISTRATION.md) was committed at
`b8001e50e89401b44c8acb276f4c15e02c4b5b05` before
collecting RAM-08 training outcomes or inspecting the new
PEF theorem goal states.

## Core experiment

The old original MathCan train source is pinned to
`c84e2025334e9a967c1297425a490f98d5b3d189`.
The new original test source is a separate fixed copy of
MathCan PR #44 at `83f61997948818b68da94429c2fe736d93ea3093`.
Both use Lean4.35.0. The four PEF test modules are
CausalInteraction, OrderedTangent, PairedHistory and
PrecisionEntropyRobust, first three original source-order theorems
each. All reference theorems are compiled separately as a source
oracle and never enter model or baseline search.

The natural TRAIN root list preserves RAM-07's 47 original
MathCan roots, then adds 18 preselected older original
conjunctive/implicational theorem roots. The collector explores
up to 18 genuinely observed states per root and depth six.
It prioritizes the consequences of a legitimate source-derived
`intro ram07_argN` and a true compiler-accepted
`constructor`, without generating assumptions or using
reference proof scripts. All new candidate effects receive
independent Lean replay in fresh sessions before fitting.

A **hard scientific gate** requires at least two distinct
old original TRAIN theorem roots and three unique original
Lean-positive exact candidate observations. If missing, CI
stops before model training and before opening the new test
source. The original failed mining artifact is preserved.

The shared 20-family grammar adds core Lean `decide` to
RAM-07's 19 contextual parameterized tactic families,
including typed exact, rw, rcases, induction, and bounded
single-definition simplification. All native, train-frequency,
goal-shape-frequency, flat and recurrent controllers use
the exact same generated concrete tactic candidates.

Loop32 (four tied passes), loop64 (four tied passes), flat64
(no recurrence), seeds17/29/43, train from original
compiler-verified effects. The training batch reserves 50%
for original multigoal states and 10% for naturally verified
exact positives. Development-only early stopping selects
checkpoints before test access. No synthetic exact curriculum.

Primary original theorem search runs 30s, 48 logical
evaluations and eight transitions per theorem for six
controllers. Secondary uses 60s, 128 evaluations, sixteen
transitions for native/global-frequency/shape-frequency/loop64.
Every accepted proof needs a full sorry-free original PEF theorem
compile and independent byte-identical replay, with permitted
axioms only. An additional separate compiler audit validates
every accepted source and source SHA without changing scores.

The registered learning criterion requires at least one
PEF original theorem certified by a recurrent policy and
missed by all three non-neural controllers. If flat64 misses
the same original, report that separately as recurrence
specific. Negative results remain permanent.

## GitHub CI

[RAM-08 source-only training and original PEF heldout CI](../../.github/workflows/lean-ram08-prospective.yml)

The training job pins the **old** MathCan commit and the
unchanged FC-12 source gym, builds the old import graph,
runs the diverse-natural-exact smoke, collects original
theorem successors and independently replays every candidate
before fitting nine CPU models. Frozen checkpoint artifacts
are written before the dependent evaluation job begins.

The evaluation job separately checks out only the
**new pinned PR #44** source and builds its complete imported
Lean modules. It checks all nine original checkpoint hashes
before the 120-case new-source theorem search.

CPU-only hosted runtime is Python3.11, PyTorch2.6.0,
psutil7.1.3, Lean4.35.0 and two optimizer threads, with
at most 6GiB total process-tree RSS. All error logs, original
complete theorem scripts, source SHAs, optimizer and RNG state
are preserved. No automatic merge.

## Local Windows with RTX 3050 4GiB VRAM, 10GiB free RAM

Use **different folders** for the old and new MathCan commits.
Pin the old TRAIN clone to `c84e2025334e9a967c1297425a490f98d5b3d189`
and the new PEF TEST clone to `83f61997948818b68da94429c2fe736d93ea3093`.
Keep the original Compactification-Theory source gym pinned to
`e8a780254b5cf41d479095f4f22bcfc4aa76ed72`.

Download the original FC-12 train-only artifact:

`gh run download 43547762837 -R MoralityLabAI/LoopedTransformers --name fc12-frozen-typed-lean-models --dir "D:/Research Engine/RAM-08/original-FC12"`

From the RAM-08 LoopedTransformers branch:

`powershell -ExecutionPolicy Bypass -File experiments/lean_ram08/run_windows.ps1 -MathCan "D:/Research Engine/MathCan-old" -TestMathCan "D:/Research Engine/MathCan-PEF" -Gym "D:/Research Engine/Compactification-Theory" -OriginalFC12 "D:/Research Engine/RAM-08/original-FC12" -Out "D:/Research Engine/RAM-08/runs/first" -Python "D:/Research Engine/ram03-env/Scripts/python.exe" -LeanBin "D:/Research Engine/lean435/bin"`

Replace paths with the actual local checkouts. The runner
creates a fresh output directory, calibrates CPU versus CUDA,
and uses CUDA only when >=1.15x faster under the 2.5GiB
allocated GPU cap, 6GiB system process-tree RSS cap and
2GiB minimum system RAM headroom.

Direct Linux stage commands with exact pinned checkouts,
after setting `PYTHONPATH` to the source gym and `LEAN_PATH`
to the old compiled MathCan and ProbeServer dependencies:

`python experiments/lean_ram08/smoke_diverse_natural_exact.py --mathcan /src/MathCan-old --lean lean --out /out/ram08-smoke`

`python experiments/lean_ram08/collect.py --data /data/fc12-train --mathcan /src/MathCan-old --lean lean --config experiments/lean_ram08/config.json --out /out/ram08-data`

`python experiments/lean_ram08/train_local.py --data /out/ram08-data --config experiments/lean_ram08/config.json --out /out/ram08-weights`

Only after training and model SHA verification, change
`LEAN_PATH` to the separately built new PEF source:

`python experiments/lean_ram08/evaluate.py --data /out/ram08-data --training /out/ram08-weights --mathcan /src/MathCan-PEF --lean lean --out /out/ram08-test`

`python experiments/lean_ram08/audit_results.py --results /out/ram08-test --lean lean --out /out/ram08-audit`
