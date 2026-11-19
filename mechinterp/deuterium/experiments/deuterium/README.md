# Deuterium: recovered evidence and frozen steering follow-up

This directory contains the retrospective audit, original-artifact manifest,
frozen follow-up protocol, measured local results, and GitHub Actions workflows.
The updated manuscript is at
[`papers/VPD_TRM_Feature_Steering/vpd_trm_feature_steering_paper.md`](../../papers/VPD_TRM_Feature_Steering/vpd_trm_feature_steering_paper.md).

## Execution status

| Work | Observed status |
| --- | --- |
| Archive SHA-256 and inventory | Verified locally; 406 members and 401 embedded source-manifest entries match |
| Historical replication recount | Completed; 16 accepts, 12 preview entries; all 16 tie random on every recorded task delta |
| ARC/Wordle and later claims | Cached records reconciled; provenance and split limits retained |
| Recovered-model follow-up | `BLOCKED_ARCHITECTURE_PROVENANCE`; 91 frozen tensors match, original forward/tokenizer unavailable |
| New synthetic recurrent-gate follow-up | Completed on local CPU; `NO_REGISTERED_ADVANTAGE` |
| GitHub Actions execution | Blocked: GitHub integration rejected branch/tree writes with HTTP 403 |
| Remote branch and PR | Uncreated; review work is committed locally |

The primary local effect, random minus VPD target negative log-likelihood, is
0.000890401851696 nats across three independent seeds. Its 95% Student t interval
is [-0.000159858838392, 0.001940662541784], which fails the frozen positive gate.
Every prerequisite passes. The task-gradient arm has lower target loss in all
three seeds. Full per-seed, per-arm metrics and uncertainty appear in
[`results/followup/scored/steering_summary.json`](results/followup/scored/steering_summary.json).

## Provenance and local freeze

The starting repository snapshot is
`2df0092d5be4d8e4748b3c66ef8325dedef17549` on `rust-accel-scoring`.
The review branch is `research/deuterium-recovered-followup-20261119`.

Protocol `DEUTERIUM-FOLLOWUP-01` and its implementation were frozen in local
commit `f1e3e9ba663ea906a6468d76426bb7381a6e23ad`. That commit and the protocol hash
were announced in the conversation before reserved-seed scoring. Scoring then
ran from a clean detached checkout of the exact commit. Repository write access
prevented external publication of the freeze before execution.

Protocol SHA-256:

```text
c6bd49b7423091f02fd55812a51a24aae8f4f07332b3178e006a64a015f908a9
```

[`registration.json`](registration.json) records the freeze and source hashes.
After scoring, the experimental code, numerical protocol and experiment
dependency pins remain unchanged. Reporting code and workflow integration may
change as results are incorporated. A future Actions execution with these seeds
is a reproduction. Independent confirmation requires untouched seeds and a new
protocol.

## Original archive

The immutable URL, revision, expected sizes and digest are pinned in
[`archive.json`](archive.json). The archive is 1,326,795,233 bytes, expands to
3,716,109,041 bytes, and has SHA-256:

```text
a45db5ef5e603852efb4f247112655c5572eff5aa85f9ab94674a4ae9dc8f003
```

Allow at least 7 GB of free space for archive retrieval, extraction, and the CPU
environment. Original archive members are extracted without changing their
contents and are made read-only. The downloader checks the archive before any
extraction and checks the embedded source manifest. Checkpoints load only after
their expected hashes match, using `torch.load(..., weights_only=True)`.

The earlier full-arena **summary** is recovered. Its sweep is absent from this
package, whose missing-file note says that sweep was not requested. The six-seed
replication sweep is available and independently recounted. Original large
artifacts remain on Hugging Face; Git contains hashes and derived results.

## Reproduce the retrospective audit

The archive and historical audit use Python's standard library. Run from the
repository root:

```bash
sha256sum --check experiments/deuterium/results/SHA256SUMS

python -m experiments.deuterium.artifacts \
  --archive .cache/deuterium/original.zip \
  --extract .cache/deuterium/records \
  --manifest .cache/deuterium/archive_manifest.json

cmp experiments/deuterium/results/archive_manifest.json \
  .cache/deuterium/archive_manifest.json

python -m experiments.deuterium.records \
  --records-root .cache/deuterium/records \
  --out-dir .cache/deuterium/retrospective

diff -ru experiments/deuterium/results/retrospective \
  .cache/deuterium/retrospective
```

Use `--offline` on `artifacts` when the verified ZIP already exists. It still
verifies the archive and all inventoried members. Existing extracted files with
changed bytes cause a failure.

The audit reads every replication row and retains exact source line numbers for
accepted grafts and their random comparators. It also reconstructs cached action
policy scores, measures overlaps, and copies historical roundout tables with
their evidence status. It does not execute the historical arena or original
language-model scorer.

## Reproduce the steering experiment

The hosted workflow specifies Ubuntu 24.04 and Python 3.12.8. The measured local
run used Python 3.12.14, CPU PyTorch 2.5.1, NumPy 2.3.5, and two PyTorch threads on
an AMD EPYC 9V74 host. Environment receipts retain these differences. Dependencies
for the small standalone experiment are isolated from the repository's main
Python 3.13 development environment.

The scored receipt records eleven package versions, all of which match their
frozen pins. A contemporaneous full transitive `pip freeze` was not captured for
that local execution. The hosted workflow captures both the named runtime
receipt and a complete installed package list. Later support-environment
observations have their own receipt and cannot reconstruct missing historical
version evidence.

```bash
python3.12 -m venv .venv-deuterium
. .venv-deuterium/bin/activate
python -m pip install setuptools==84.2.1
python -m pip install torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/deuterium/requirements-steering.txt
python -m pip check

python -m unittest discover -s experiments/deuterium/tests -p 'test_*.py' -v

python -m experiments.deuterium.steering preflight \
  --records .cache/deuterium/records \
  --out .cache/deuterium/recovered-preflight

PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled \
python -m experiments.deuterium.steering toy \
  --phase score \
  --registration-commit f1e3e9ba663ea906a6468d76426bb7381a6e23ad \
  --out .cache/deuterium/scored-reproduction
```

Use a fresh output directory for every execution. The scoring command refuses
dirty tracked source, changed authoritative files, untracked Python that could
shadow imports, an invalid registration commit, or an occupied output directory.
The dataset generator hashes every split. A receipt is written before held-out
examples are generated; exceptions after exposure retain that fact.

The auxiliary pilot command uses seed 997 and never creates held-out data:

```bash
python -m experiments.deuterium.steering toy \
  --phase pilot --out .cache/deuterium/pilot-reproduction
```

The stock optimizer records step indices 0 through 300 inclusive and skips the
optimizer update at the terminal index. It therefore performs 300 main updates
after 200 faithfulness warmup updates; supervised training uses 500 updates.
The frozen prose incorrectly counted 301 main updates. The documentation erratum
in `PROTOCOL.md` corrects that description after scoring; the numerical config,
source, dependency pins and measured results retain their frozen bytes.

The final local scored seeds consumed 4.17, 3.13 and 3.43 seconds inside their
instrumented execution regions. Their sum, 10.73 seconds, excludes dependency
installation, archive download and extraction, process startup, and reporting.
Each seed has a 600-second wall-time limit on CPU. Failed gates remain visible;
an engineering exception returns a nonzero process exit after saving diagnostics.

## Verify saved results and regenerate the paper

`verify_followup.py` independently recomputes the measured results from saved
rows, checkpoints and edit matrices. It does not train a model or evaluate new
examples. Run it in the steering environment:

```bash
python -m experiments.deuterium.verify_followup \
  --results experiments/deuterium/results/followup/scored \
  --out .cache/deuterium/saved-result-verification.json
```

Paper rendering uses a separate environment because its exact figure/PDF
dependency snapshot differs from the experiment's transitive packages:

```bash
python3.12 -m venv .venv-deuterium-paper
. .venv-deuterium-paper/bin/activate
python -m pip install -r experiments/deuterium/requirements-paper.txt

python -m experiments.deuterium.summarize_followup \
  --recovered experiments/deuterium/results/followup/recovered/recovered_preflight.json \
  --pilot experiments/deuterium/results/followup/pilot/steering_summary.json \
  --scored experiments/deuterium/results/followup/scored/steering_summary.json \
  --failure experiments/deuterium/results/followup/pilot_001_config_failure/steering_summary.json \
  --out experiments/deuterium/results/followup/paper_summary.json

python -m experiments.deuterium.render_paper \
  --followup experiments/deuterium/results/followup/paper_summary.json
```

The renderer produces the Markdown tables, source ledger, figures and PDF. Its
manifest hashes inputs, outputs and fonts. Both historical manuscript snapshots
are preserved separately under the paper's `archive/` directory.

## GitHub Actions and review handoff

The prepared workflows are:

- `.github/workflows/deuterium-recovery.yml`: download and verify originals,
  reproduce the retrospective, compare derived records, and rebuild the paper.
- `.github/workflows/deuterium-followup.yml`: install the pinned CPU environment,
  test execution boundaries, audit recovered prerequisites, reproduce the frozen
  synthetic comparison, and upload all results including failed prerequisites.

Actions are pinned to full action commit SHAs. Both workflows have read-only
repository permissions, retain artifacts for 90 days, and leave the review branch
unmerged. Scientific verdicts appear explicitly in the follow-up job summary.

The integration returned `403 Resource not accessible by integration` for both
branch creation and Git-tree creation. The terminal has no authenticated push
credential, and the permitted browser fallback is signed out. The exact receipts
are in [`results/remote_status.json`](results/remote_status.json). No workflow-run
URL or PR URL exists for this work yet.

Once repository and workflow writes are available, push the prepared review
branch. Its push triggers the workflows; then create a draft PR targeting
`rust-accel-scoring`. No merge or paper publication is authorized by this package.
Downloaded Actions outputs should be retained as a separate run directory,
compared with the local records, and committed with their observed run URL and
tested SHA. Preserve the local run as its own evidence record.

## Remaining gaps

Recovered-checkpoint task execution needs the original `HermesHRMClassifier`,
dataset/tokenizer implementation, exact dependencies, and independent reference
logits. Historical governance lacks its exact command, selected rows, dirty
update and projection coefficients. Adapter subset identities and exact commands
remain missing. The original full-arena sweep remains outside the supplied
package. Upstream Wordle puzzle lineage, historical execution commits, and fresh
confirmation data for broad steering or recursive-improvement claims remain
unresolved. The local synthetic result is confined to its stated task and dose.
