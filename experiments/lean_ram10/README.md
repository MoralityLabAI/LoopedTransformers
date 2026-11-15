# RAM-10 | Fresh authentic Lean action curriculum and source-family proof transfer

**Status (2026-11-15):** The [original preregistration](PREREGISTRATION.md)
was committed before we inspected the newly chosen MathCan TRAIN/DEV theorem
statements or accessed any TEST source. The pinned [config](config.json) fixes
all source Git blob SHAs, splits, budgets and training gates. Python-only
contract tests pass locally. The actual Lean4.35 compiler collection, nine
model fits and fresh-source theorem results require a working CI or local Lean
runner. No original TEST theorem has been scored yet.

## Scientifically separate from RAM-08 and RAM-09B

Original RAM-08 trained on `MathCan@c84e2025`. The pinned new `MathCan@7515eec`
contains 20 later modules. Six earlier PEF-ancestry modules are excluded as
**target roots**. Of the remaining 14 modules, RAM-10 registers eight for
TRAIN, two for DEV and four as unseen TEST. File contents for the TEST modules
are excluded from the TRAIN checkout, and the TEST checkout happens only
**after** the original nine-model byte-hash freeze.

One registered TRAIN module, `MemoryProjection`, imports
`MathCan.PairedHistory`, a PEF-ancestry dependency; DEV's `GradientStability`
imports `MemoryProjection`. These original imports can expose PEF-related
symbols as context during compiler execution. They provide **no PEF theorem
root, proof script or candidate label** to RAM-10. The family holdout is
module-level and time-ordered, rather than cryptographically secret or
semantic-independence guaranteed. Never use RAM-10 checkpoints to reinterpret
RAM-08's original untouched PEF test.

## Data and trust boundary

The original source files are pinned by Git *blob* SHA as well as commit.
For each genuine theorem or lemma root, the collector starts with the
Lean4.35 original goal, applies only RAM-08's 20 registered source-context
candidate families, and independently replays *every* positive, failed and
progressing compiler action in a fresh Lean session. A family-fair 80-action
cap retains the first four source-visible scripts per family before filling
remaining slots in the original order. Discarded generated candidates are
censored. Compiler crashes and request timeouts are **unknown**, and they
abort the dataset rather than enter failed-action supervision.

Positive `exact`, `rw`, and `rcases`/`induction` evidence from multiple
original TRAIN theorem roots is **required** before fitting. If a family
fails the registered gate, publish the failed corpus and stop the pipeline
with zero prospective scores. There are no manually inserted generated
hypotheses or source teacher prefixes.

After a passing gate, fit loop32, loop64 and flat64, three seeds each, on
TRAIN only. DEV-only selection freezes three best checkpoints and their
SHA256 byte hashes. The fresh 4-module TEST cohort selects up to eight
original top-level declarations and scores all six primary policies under
30s/48-action/8-transition budgets and four secondary policies under
60s/128-action/16-transition budgets. A complete proof must compile twice
inside its clock budget and pass a third independently compiled source/axiom
audit. The original source reference proofs serve solely as a separate oracle.

## GitHub Actions

[`.github/workflows/lean-ram10-fresh-source.yml`](../../.github/workflows/lean-ram10-fresh-source.yml)
contains a separated source-only train job and a dependent full TEST job.
Both pin Python3.11, PyTorch2.6 CPU and Lean4.35, and use the original pinned
FC-12 compiler interpreter. The TRAIN job deliberately removes four untouched
TEST source module files before invoking Lean. Its failure artifact preserves
any genuine incomplete/negative train-only collection, including an unmet
family-positive gate. A runner assignment failure is an infrastructure blocker,
never a proof-search outcome.

## Local reproducibility

From the RAM-10 branch, prepare *two separate* checkouts of
`patrickdugan/MathCan@7515eecf51a60e7f8cfcd0801fddf8a3cf039e89` and
the original `MoralityLabAI/Compactification-Theory@e8a780254...` compiler gym.
The TRAIN checkout must have the four TEST module files removed. Compile
TRAIN/DEV modules, set `LEAN_PATH` to the original `experiments/lean_ram03`
server and the MathCan `lake env printenv LEAN_PATH`, and use a Python3.11
environment with torch2.6 CPU/psutil7.1.3.

```bash
python -m unittest discover -s experiments/lean_ram10 -p 'test_*.py' -v
python experiments/lean_ram10/collect.py --mathcan /src/MathCan-TRAIN --lean lean --out /out/ram10_data
python experiments/lean_ram10/train.py --source /out/ram10_data --out /out/ram10_training
```

Only after a *passing* independently audited original TRAIN tactic gate and a
complete nine-fit freeze, create a separate pinned unmodified TEST checkout,
build its imports, and score:

```bash
python experiments/lean_ram10/evaluate.py --mathcan /src/MathCan-TEST --data /out/ram10_data --training /out/ram10_training --lean lean --out /out/ram10_evaluation
python experiments/lean_ram10/audit_results.py --results /out/ram10_evaluation --lean lean --out /out/ram10_independent_audit
```

Both outputs must be new paths. Even a positive ranked first action is only a
compiler observation. The registered theorem result requires all policies,
matched resource accounting and full Lean kernel-certified original theorem
proofs. All negative outcomes and resource failures remain in the receipts.

## WSL/Linux two-stage runner

The checked-in [`run_local.sh`](run_local.sh) rejects mismatched original
MathCan/compiler-gym commits and the pinned registered configuration blob.
Use two separate MathCan clones at the same exact SHA. In the **TRAIN-only**
clone, remove the four protected TEST source files after checkout; the runner
refuses a checkout that still contains them. Install Lean4.35 and the pinned
CPU Python requirements.

```bash
bash experiments/lean_ram10/run_local.sh train /src/MathCan-TRAIN /src/Compactification-Theory /src/lean-4.35.0-linux/bin /out/ram10-first-train
```

After the collector gate, all nine fits and every original checkpoint SHA
pass, invoke TEST with its own unchanged source checkout:

```bash
bash experiments/lean_ram10/run_local.sh test /src/MathCan-TEST /src/Compactification-Theory /src/lean-4.35.0-linux/bin /out/ram10-first-test /out/ram10-first-train/data /out/ram10-first-train/training
```

This is suitable for WSL on the local Windows machine. Every output directory
is single-use, and TRAIN never opens TEST files.
