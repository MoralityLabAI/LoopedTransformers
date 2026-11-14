# RAM-02 local Lean reasoner campaign

Small candidate-outcome policies trained against frozen compiler observations,
with whole-source-module development selection and a bounded search over all
remaining Lean goals. See RESULTS.md for measured outcomes and PROTOCOL.md for
the protocol frozen before fitting/scoring. The implementation is exploratory;
cached action accuracy and scratch closure never count as theorem certificates.

## Environment

Native Windows or Linux/WSL2, Python 3.12/3.13, PyTorch 2.6.0 and psutil 7.1.3.
No torch.compile, large dataset preloading or GPU compilation. This campaign
reused an existing PyTorch 2.6.0+cu124 installation through a new isolated venv;
it did not upgrade or alter the existing environment. JSON-compatible YAML is
read with the standard library. Tests use unittest.

When dependencies are missing, create an isolated environment on a drive with
at least 10–15 GB free. Redirect TEMP/TMP and pip's cache there too. Install
PyTorch from the appropriate official wheel index:

```powershell
python -m venv D:\ram02-env
D:\ram02-env\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
D:\ram02-env\Scripts\python -m pip install psutil==7.1.3
```

For CPU-only environments use `https://download.pytorch.org/whl/cpu`. On WSL2,
use the environment's Linux Python and Lean binaries and the corresponding
Linux wheel; do not run the Windows venv. CUDA is optional. The trainer measures
identical FP32 CPU/CUDA updates before choosing a device. CUDA requires at
least 1.15x the measured CPU throughput. FP16 was not needed for these models.

## Frozen inputs

- RAM-01 base: `ca1982833af7f1adeb55029ef886f56ebf82b5d4`, PR #37.
- FC-12 training artifact: run `43547762837`, artifact `14095156819`.
  ZIP SHA256: `d91a12d725ca0a611b4de308c80365bfd3888d88449b03b2c5f07137a1c2025a`.
- RAM-01 original model artifact: run `43557076648`, artifact `14098014086`.
  ZIP SHA256: `5f5604ceaa84de0edbbb52669be1af578d42a99e7bedb39e9865644923cb6866`.
- Original gym: Compactification-Theory `e8a780254b5cf41d479095f4f22bcfc4aa76ed72`.
- Original MathCan: `c84e2025334e9a967c1297425a490f98d5b3d189`.
- Lean 4.35.0 official Windows ZIP SHA256:
  `5fc033e93e1a7bc59c5ebf311103d2c9cd18d4b3eddda4ffa3414e8adb0ef79e`.

The loader checks the exact 51-state file and the registered 961-row semantic
effect serialization. It checks state/action pairing, original source metadata,
the 39/12 whole-module split and positive exact/rw/rcases/induction support.
Reference proof commands, filenames, theorem IDs, future actions and outcome
labels do not enter network features. No original labels are regenerated.

## Reproduce training

Run from the repository root, using a **new** output directory each time:

```powershell
python -m unittest discover -s experiments/lean_ram02/tests -v
./experiments/lean_ram02/run_windows.ps1 -Data D:/ram02-data/fc12-train -Out D:/ram02-runs/campaign-01 -Python D:/ram02-env/Scripts/python.exe -TempDirectory D:/ram02-temp -IncludeFlat
```

Linux/WSL2 equivalent:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/lean_ram02/train_local.py --data /data/fc12-train --out /data/ram02-runs/campaign-01 --include-flat
```

The required six loop32/loop64 fits finish before the three optional flat64
controls. Seeds are 17, 29 and 43. The microbatch is 16 with accumulation4;
effective batch64; maximum2,000 updates. Every100 updates, save full train/dev
metrics and last/best checkpoints. Early stopping uses development CE only.
The selected checkpoints retain the 39-state training split. No final refit
uses development labels as training data. Every checkpoint includes optimizer,
sampler/RNG state, config and frozen-data hashes. Failed attempt directories
are preserved. Memory failures halve the microbatch and increase accumulation
to retain batch64, provided actual hardware recovers above the safety floors.

`resources.py` samples total process-tree RSS, available RAM/commit, CUDA
allocated/reserved VRAM and nvidia-smi dedicated VRAM. Limits are 6 GiB RSS,
2.5 GiB allocated VRAM, at least2 GiB available RAM/commit and512 MiB dedicated
GPU headroom. The low-memory floors stop work before exhausted memory/commit.
The CUDA allocator is additionally capped. RSS/headroom checks run every0.1s;
actual dedicated VRAM is polled every2s. Peaks are sampled campaign high-water
marks; torch's allocator also reports its exact calibration peak.

## Reproduce original checkpoint audit

```powershell
python experiments/lean_ram02/audit_ram01.py --data D:/ram02-data/fc12-train --weights D:/ram02-data/ram01-models --out D:/ram02-runs/ram01-audit.json
```

This verifies all16 RAM-01 checkpoints against their original receipts and
reproduces all8 original development top1 counts, plus native3/12 and family
frequency6/12. It never writes original inputs. RAM-01's Unix-only `resource`
import is avoided by the derived portable model/data module.

## Real Lean regression

Use the pinned gym checkout on PYTHONPATH, Lean4.35 binaries on PATH, and the
original MathCan `.lake/build/lib/lean` on LEAN_PATH. Set LEAN_NUM_THREADS=2 and
redirect TEMP/TMP to the artifact drive. Build and evaluate:

```powershell
$env:PYTHONPATH='D:/ram02-source/Compactification-Theory'
$env:LEAN_PATH='D:/ram02-source/MathCan/.lake/build/lib/lean'
$env:LEAN_NUM_THREADS='2'
python experiments/lean_ram02/evaluate.py --mathcan D:/ram02-source/MathCan --gym D:/ram02-source/Compactification-Theory --weights D:/ram02-runs/campaign-01 --out D:/ram02-runs/fc13-regression-01 --lean D:/lean435/bin/lean.exe --lake D:/lean435/bin/lake.exe --build
```

Only the already published FC-13 suite is evaluated, explicitly post-hoc.
New theorem families remain unevaluated. The source oracle independently
recompiles all16 reference proofs; its commands never enter candidate ranking.
Each policy receives the same grammar,20s deadline,12 actual Lean calls and
three successful transitions per path. Final certification counts toward both
budgets. Deadline termination may incur process-kill overhead; no result
completed after the deadline receives credit.

Search tracks the entire ordered goal list, count, per-goal hashes and a
canonical state signature. It continues on the first active goal after a
branching tactic, retaining all siblings. Deduplication is syntactic, preserves
goal order/multiplicity, and does not claim mathematical equivalence.

Candidate/probe outcomes are new search receipts, separate from immutable
training labels. All branches must close. Accepted proofs require a separate
complete original-theorem compile, an explicit axiom audit allowing only
propext/Classical.choice/Quot.sound, and fresh compilation of the exact final
source bytes. Sources, failures, stdout logs, selected scripts and SHA256s are
preserved. Synthetic preflight proofs never enter regression proof counts.

## Artifacts

Checkpoints, full logs and candidate sources live under gitignored `runs/`,
not Git. New output directories are required; old results are never replaced.
The local campaign root is `D:\Research Engine\RAM-02`. The new branch is
`feat/lean-ram02-local-3050`, stacked on the still-open RAM-01 feature branch.
Only compact measured receipts and analysis are tracked for review.

`analyze.py` verifies every candidate/log hash and final-source replay, reruns
the cached pass ablations, and exports `metrics.json`, checkpoint locations,
a full artifact hash manifest and deduplicated certificate sources.
`report_results.py --analysis <analysis-dir> --audit <ram01-audit.json> --out <RESULTS.md>`
renders the measured report. Exact, rcases, induction and multi-goal positive
development recall are unavailable because this fixed split has no such
positive outcomes. The neural model still trains on single-goal contexts;
full branch-state text is observed during search, not fabricated for training.
