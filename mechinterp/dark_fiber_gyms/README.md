# Local Dark Fibers gyms: ARG-03-N and LOOPSCOPE-04

**Status (November 15, 2026):** Experimental local pilot infrastructure. No new GPU measurements are claimed. Prior ARG-02 receipts and v014 manuscript are unchanged. Mathematical rationale: [Causal Quotient ARG-03](../../papers/shared_weights_dark_fibers/CAUSAL_QUOTIENT_ARG03.md).

## Models, resources, Python

| Gym | Compute | Local hardware |
| --- | --- | --- |
| arg03_roster: fresh algebraic public split and isolated EvalMath grader | CPU | Any |
| arg03_nanbeige: 16 candidate rows, native Nanbeige 3B loop 1/2/4/8 and context interventions | BF16 | RTX 5080 16 GiB |
| sudoku_roster: unique-solution novel pilot puzzles | CPU | Any |
| loopscope04_trm: 7M Sudoku TRM, FP32 sphere-preserving intervention | GPU | RTX 3050 4 GiB or RTX 5080 |
| loopscope04_trm: cellwise history/snapshot cache ridge fits | GPU + CPU | RTX 3050 + RAM |

Run every script as **python -m mechinterp.dark_fiber_gyms.MODULE** from the LoopedTransformers repository root. Required local checkouts:

- LoopedTransformers at research/causal-dark-fibers-arg03-20261115, PR #47.
- Compactification-Theory at research/arg02-recurrent-qwen-20261110, PR #141, retaining the checkpoint lock and original Nanbeige model adapter.
- EvalMath providing spiral_ln/algebraic_reasoning_oracle.py for grade and split preparation.

The existing Windows environments from ARG-02 can be reused: $HOME/.venvs/bvg-v03/Scripts/python.exe (transformers 4.46.3, torch 2.11.0+cu128) for Nanbeige, and $HOME/.venvs/bluebeam/Scripts/python.exe (torch 2.11.0+cu128, numpy 2.5.3) for TRM. Keep the Qwen transformers 5.17 env separate. For new Python 3.13 environments install torch 2.11.0 cu128, numpy 2.5.3, pytest; Nanbeige additionally needs transformers 4.46.3 and safetensors. PyTorch and NumPy are needed for CPU tests.

## Smoke (no model weights)

PowerShell:

~~~powershell
$P = "$HOME/.venvs/bluebeam/Scripts/python.exe"
& $P -m pytest -q mechinterp/dark_fiber_gyms/tests
& $P -m compileall -q mechinterp/dark_fiber_gyms
~~~

## ARG-03-N: new algebraic roster, model-only run, independent grade

~~~powershell
$E = "C:/path/to/EvalMath"
$C = "C:/path/to/Compactification-Theory"
$R = "runs/dark-fibers/arg03"
$N = "$HOME/.venvs/bvg-v03/Scripts/python.exe"
& $P -m mechinterp.dark_fiber_gyms.arg03_roster prepare --evalmath-root $E --compactification-root $C --out-dir "$R/public"
$env:ARG02_SNAPSHOT_NANBEIGE = "C:/path/to/models--Nanbeige--Nanbeige4.2-3B/snapshots/451ed48c3273ecef7ea8faaa43c31ce529763bb1"
& $N -m mechinterp.dark_fiber_gyms.arg03_nanbeige --public-jsonl "$R/public/development.jsonl" --compactification-root $C --out-dir "$R/nanbeige_dev" --device cuda:0 --limit 4
& $P -m mechinterp.dark_fiber_gyms.arg03_roster grade --evalmath-root $E --compactification-root $C --split-dir "$R/public" --split development --tape "$R/nanbeige_dev/answer_tape.jsonl" --out "$R/dev_grades.json"
~~~

The generated development split has 96 new positive examples from seen generator families. The held-out test has 168 positives across all 7 families and holds out the final-R program family. Earlier EvalMath and ARG-02 signatures are forbidden. The runner sees only the 6 ARG-02 public input fields; source graphs and withheld futures stay in a separate oracle-only grading process.

The Nanbeige runner verifies SHA-256 hashes for every pinned checkpoint file and does not download weights. It constructs same-prompt perturbed one-pass donors; replaces only context positions preceding the final three scored positions; and checks every full-vocabulary log probability at those three sites is unchanged *before* running further loops. The output is an answer_tape.jsonl containing 64 joint program scores per case/arm/depth, including one-loop temperature grids and no-feedback open-loop controls, plus parity and resource data.

A full development run uses --limit 0. **Do not run the held-out test until dev settings and acceptance rules are fixed.** Once registered, run the same Nanbeige command adding --permit-heldout-test and using --public-jsonl "$R/public/test.jsonl", --out-dir "$R/nanbeige_test", --limit 0. Grade separately using --split test --full to enforce complete coverage.

The full-vocabulary head at a single answer site may have full column rank. This gym tests an actual *sequence-level* observational fiber of previous context tokens, with exact full-logit preservation, rather than presuming a nonzero kernel at an answer token. Initial ARG-03 roster is positive-only. Negative and collision controls still need their own new registration.

## LOOPSCOPE-04: fresh Sudoku pilots on RTX 3050

For representative, confirmatory Sudoku experiments, supply new external 81-character question,answer CSVs with broader puzzle-generation provenance. The included sudoku_roster generator makes **engineering pilots** by digit/row/column permuting a single valid solved-grid family and removing clues, checking uniqueness with an independent exact solver. The exclusion audit checks exact and digit-relabel duplicates, but does not yet cover all Sudoku automorphisms.

Pass prior LOOPSCOPE-02 benchmark files via repeated --exclude-csv arguments. Include any previously evaluated train/test files.

~~~powershell
$S = "runs/dark-fibers/loopscope04"
$OldBench = "C:/path/to/test_sudoku_bench.csv"
$OldHard = "C:/path/to/test_hard.csv"
$TRM = "C:/path/to/step_21700"
& $P -m mechinterp.dark_fiber_gyms.sudoku_roster --out-dir "$S/public" --exclude-csv $OldBench --exclude-csv $OldHard --n-train 24 --n-eval 12 --min-clues 32 --max-clues 38
& $P -m mechinterp.dark_fiber_gyms.loopscope04_trm --repo-root . --checkpoint-file $TRM --tag gaoxin492 --mode swap --eval-csv "$S/public/eval.csv" --exclude-csv $OldBench --exclude-csv $OldHard --out "$S/swap_pilot.json" --device cuda:0 --limit 4 --patch-step 4 --horizon 16
& $P -m mechinterp.dark_fiber_gyms.loopscope04_trm --repo-root . --checkpoint-file $TRM --tag gaoxin492 --mode cache --train-csv "$S/public/train.csv" --eval-csv "$S/public/eval.csv" --exclude-csv $OldBench --exclude-csv $OldHard --out "$S/cache_pilot.json" --device cuda:0 --train-limit 24 --limit 4 --steps 8
~~~

The TRM runner loads the original checkpoint only after SHA-256 matches the LOOPSCOPE-02 pin. It uses float32 even if other experiments used BF16, to audit exact all-11-class readout parity. Donor states come from a randomized fork on the *same* board; donor 501-D orthogonal components are rescaled to the recipient Post-RMSNorm sphere. Prefix tokens and zL remain recipient-owned. Comparisons: clean, donor swap, equal-radius random fiber, self-control. Reported outcomes: next-iteration logit divergence and final exact board solve.

The sampled delayed-observability probe is also available locally, using
a small number of float32, equal-logit and equal-radius perturbation pairs.
It estimates a finite-horizon Gramian in an 8-direction random subspace
rather than the full 501-dimensional kernel:

~~~powershell
& $P -m mechinterp.dark_fiber_gyms.loopscope04_trm --repo-root . --checkpoint-file $TRM --tag gaoxin492 --mode gramian --eval-csv "$S/public/eval.csv" --exclude-csv $OldBench --exclude-csv $OldHard --out "$S/gramian_pilot.json" --device cuda:0 --limit 1 --patch-step 4 --probes 8 --angle 0.04 --gramian-horizon 4
~~~

The Gramian receipts include pre-intervention full-head parity, norm error,
the sampled eigenvalue spectrum, and a per-lag response curve. This is a
local first-order diagnostic that requires finite causal intervention
validation on subsequent datasets.

The cache baseline uses identical-width ridge regressors on board-cell [zH_orthogonal,zL] with (clues, position, step, V_t, V_t) vs (clues, position, step, V_t, V_{t-1}), scoring held-out MSE/R2. This is a bounded *cellwise* baseline, not the complete 97-token C5 transformer used in LOOPSCOPE-03. For the second TRM checkpoint use --tag yagizdevre with its SHA-pinned model.pt.

## Safety / evidence

Start with --limit 1 on the RTX 3050. Use the 16 GiB RTX 5080 for Nanbeige; the runner fails when available GPU capacity is smaller than 12 GiB unless an override is supplied. No script downloads models or spends money, and all grade identities are checked against the oracle. Keep local datasets, tapes, GPU traces and model weights outside version control. All generated results are pilot evidence until a fresh preregistered confirmatory experiment and replication are completed.
