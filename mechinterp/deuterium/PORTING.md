# Deuterium relocation: provenance and execution contract

This folder moved the Deuterium review package from `MoralityLabAI/param-decomp` to
`MoralityLabAI/LoopedTransformers/mechinterp/deuterium/`, **without changing the
original experimental programs or their scored records**.

## Immutable historical and prospective sources

From the verified incremental Git bundle, the original authoring commits are:

- parent source `param-decomp` commit: `2df0092d5be4d8e4748b3c66ef8325dedef17549`;
- historical audit commit `da2c3197fdc02693d00360533ee6a4c8b125ecfd`;
- experiment registration source commit `f1e3e9ba663ea906a6468d76426bb7381a6e23ad`;
- original local result/paper commit `2441c4330bb138f3218183eb46cde9c6db9eb903`.

The immutable science files have the original SHA-256 values recorded in
`experiments/deuterium/registration.json`: `steering.py`, `protocol.json`, and
`requirements-steering.txt`. The original `steering.py` is copied unchanged.
Its `registration_check()` validates the **old repository's** commit and worktree.
After relocation it cannot be invoked directly for score mode. The new
`experiments/deuterium/relocated_runner.py` replaces only that pre-execution
registration check at runtime; it authenticates those original source bytes,
requires the exact pinned upstream `param_decomp` commit and clean worktrees,
and records the actual new host commit. The original training, VPD optimization,
selection, intervention, scoring and aggregation functions remain unchanged.
The wrapper is itself reviewed source and should be tested before trusting
new result receipts. The original independent arithmetic verifier contains two
historical `git show` calls that also refer to the original repository.
`relocated_verifier.py` permits only those exact protocol and steering lookups,
after rechecking the frozen source and upstream checkout, then calls the
byte-identical original verifier. Its 265 arithmetic checks remain intact. Tests fail closed on altered original bytes or protocol.

A Git SHA is a source-history pointer, not proof that a result was externally
preregistered. This registration existed as a local Git commit and was recorded
before scoring; it was not hosted externally when the test set was unopened.
The reserved seeds 103, 211 and 307 have already been scored and inspected.
Any GitHub Actions rerun is a reproducibility exercise, **not new independent
prospective evidence**.

## Module paths and dependency boundaries

Run Python modules from this folder, with `PYTHONPATH` containing this folder
and the pinned upstream checkout. The historical repo and its source code remain
external, installed by the Actions job via `actions/checkout` at the exact commit.
A pinned torch CPU environment and supporting dependencies are documented in
`experiments/deuterium/requirements-steering.txt`.

```
cd mechinterp/deuterium
sha256sum -c experiments/deuterium/results/SHA256SUMS
PYTHONPATH=. python -m unittest experiments.deuterium.tests.test_artifacts experiments.deuterium.tests.test_records experiments.deuterium.tests.test_relocated_runner -v

# With original checkpoint archive extracted to $RECORDS and pinned upstream checkout:
DEUTERIUM_UPSTREAM=/path/to/param-decomp-PINNED PYTHONPATH=.:/path/to/param-decomp-PINNED \
  python -m experiments.deuterium.relocated_runner preflight --records "$RECORDS" --out /tmp/deuterium-preflight

DEUTERIUM_UPSTREAM=/path/to/param-decomp-PINNED PYTHONPATH=.:/path/to/param-decomp-PINNED \
  python -m experiments.deuterium.relocated_runner toy --phase score \
  --registration-commit f1e3e9ba663ea906a6468d76426bb7381a6e23ad --out /tmp/deuterium-reproduction
```

The upstream checkout must have no tracked edits and its HEAD must equal
`2df0092d5be4d8e4748b3c66ef8325dedef17549`.

## Original bytes and reproducible derived outputs

`experiments/deuterium/results/SHA256SUMS` authenticates the original audit,
model-replay receipts, pilot failure receipts, per-seed saved logits, VPD
checkpoints, paper and figures. The scientific status is
`NO_REGISTERED_ADVANTAGE` for the synthetic follow-up and
`BLOCKED_ARCHITECTURE_PROVENANCE` for recovered-target task inference.

The 1.33 GB original recovered-record ZIP remains on Hugging Face at the
revision and SHA-256 in `experiments/deuterium/archive.json`.
`deuterium-mechinterp-recovery.yml` fetches, hash-checks and audits the originals.
`deuterium-mechinterp-followup.yml` independently runs and scores the already
published seeds on a CPU runner and preserves every failure receipt.

The old workflow YAML files are preserved under this folder's `.github/workflows/`
for archive-byte verification. The active GitHub workflows reside in the root
repository's `.github/workflows/` and use the relocated paths.

## Historical limitations

- The earlier full-arena's sweep is missing. Its reported 23 accepts remain summary-only.
- Six-seed replication: 16 accepted but all tied selected random on scored task metrics.
- ARC fold overlap leaves only 2–6 genuinely unseen examples per heldout card.
- Wordle's 18/72 extra correct observations are from a constrained rule applied
  to cached candidate scores, with incomplete upstream puzzle lineage.
- Governance cleanup reduces collateral accuracy to 1% in the recovered report.
- Recovered model's forward class, tokenizer source and independent reference
  logits are missing. The 91/91 tensor-lineage match cannot authenticate model
  inference or support multi-step causal steering.
- Three synthetic seeds and a zero-crossing across-seed interval limit the
  evidential scope. Green Actions checks authenticate execution, not the hypothesis.

## Hosted execution chronology

On PR #211, archive verification run
[44448038368](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44448038368)
passed on the initial port. First steering run
[44448038362](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44448038362)
completed training, archived-checkpoint preflight and scoring, then failed the
independent verifier's two original-repository `git show` lookups. The source
check itself and the scientific calculations succeeded. The follow-up commit
adds `relocated_verifier.py` to redirect only those two lookups through the
byte/hash gate, with dedicated negative tests. The failed first run remains a
visible CI receipt. Re-runs continue to use previously exposed seeds.
