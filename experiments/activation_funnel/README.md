# AF-01: joint continuation preservation

AF-01 tests whether a compact recurrent state retains the information needed to
choose a valid proof candidate and recover the provenance of earlier search
previews. The first experiment uses a small newly trained shared GRU. Its
observations come from a finite interface over Horn proofs, and selected proofs
receive an independent Lean check against the original theorem.

The experiment is isolated from LR-01 and the FC-12 tactic-policy branch. The
general mathematical definitions remain in EvalMath. This directory owns the
neural experiment, its exact finite task, and its receipts.

## Primary result

The [14 November 2026 report](results/20261114/RESULTS.md) records the completed
three-seed run. Joint compression reached 100% task and provenance recovery on
the deeper-certificate family with the original preview order. Reversed-order
transfer failed both registered gain gates. The full checkpoint and proof
archive, exact finite quotient, and machine-readable metrics accompany the
report.

The [CI replay](results/20261114/ci_replay.json) passed both jobs and reproduced
all 48 arm/seed/family task, audit, and tainted-attempt scores. Its independent
audit also reproduced both failed registered gates.

## Frozen experiment

`PROTOCOL.json` records the model sizes, training steps, seeds, compression arms,
evaluation split, and decision rules. The first registration was published as
`07dd51055e81006bd50a97ec950ada11a9014db2` before neural scores. Version 2 resolves
the per-split generator seeds and makes the public-depth donor constraint
explicit before training. Hyperparameters and improvement thresholds remain
fixed. Changes after scored evaluation require a new exploratory registration.

## The task and what it measures

Each host theorem has two implication chains ending at one target. A branch can
be valid, dead, or tainted. The host supplies the root premise of a valid branch.
A dead preview remains open. A tainted worker preview claims closure using an
external root that the host never imports.

The controller has a budget of three actions. The scored policy receives two
registered preview observations followed by a final goal marker, then chooses
branch 0, branch 1, or STOP. Its final visible goal omits the earlier receipts.
Remembering which branch closed and where its premise came from determines the
decision. The host rejects a tainted submission and records the attempt.

The corpus balances all nine ordered branch-status combinations. Training uses
144 renamed theorem instances and development uses 36. Each held-out family has
72 instances: one changes certificate-chain depth, and one also reverses the
preview order. The model sees nine numerical public features, including depth;
it never receives the private facts, target labels, proof traces, case IDs, or
status tuple. The observation sequence repeats across renamed certificates.
Reports count unique full inputs and semantic patterns explicitly.

This is an interface-memory experiment. Longer Horn chains exercise the
certifier and the public depth feature. General Lean proof search and native
looped-transformer mechanisms require subsequent experiments with their actual
states and architectures.

## Compression and continuation

The teacher has a width-32 GRUCell invoked twice per observation. A learned
width-4 encoder and decoder intervene after each recurrent call. The resulting
state feeds the next call and the final frozen action and audit heads. The
codec therefore changes subsequent computation.

Learned codec objectives use task targets, activation reconstruction,
one-step commutation, or joint task/audit continuation. Train-only rank-4 PCA
and a seeded random projection provide matched representation controls.
An additional width-4 GRU student learns the transition directly and uses a
decoder into the frozen source heads. All parameter and arithmetic costs are
reported. Codec arms retain the source GRU and add work; the compact-transition
arm is evaluated separately for a compute gain.

The four-coordinate boundary contains 128 bits at float32 precision. Codec
execution also needs its decoded 32-coordinate working state. This accounting
describes an information boundary and does not imply a 128-bit total runtime.

Counterfactual trials replace the state after the first preview with a donor
state from the same first-branch and public-depth stratum. The recipient suffix
then runs normally. Exact targets come from the composed preview history.
The records distinguish those counterfactuals from proof submissions against
the original host theorem.

## Proof and audit boundary

`certify.py` independently reconstructs the original theorem from its hash-bound
fields and replays the selected Horn chain. The model supplies a branch index.
All accepted declarations are compiled by Lean 4.19.0 and must have an empty
`#print axioms` dependency list. Failed decisions and abstentions remain in the
receipt. Missing Lean makes a requested kernel audit fail.

The fixed native baseline uses Lean `first` over the two original candidate
terms in branch order. It receives the original theorem and records its own
compiler cost. Its access and cost differ from the preview interface, so the
report gives its result as a native reference with that scope.

Safety measurements concern attempts to submit an unauthorized proof and the
ability to recover provenance. The host's refusal to add external roots is an
enforced boundary. The experiment does not establish autonomous deception,
general safety, or a learned mechanism in a pretrained transformer.

## Run the registered experiment

Run these commands from the repository root with Python 3.12. Use an empty
output directory for each run; the runners reject reuse that would overwrite
earlier checkpoints or evaluation evidence.

```bash
python3.12 -m venv .venv-af01
source .venv-af01/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
python -m pip install -r experiments/activation_funnel/requirements.txt
python -m unittest discover -s experiments/activation_funnel/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel/tests -p 'test_*.py' -v
python -m experiments.activation_funnel.gym quotient --out artifacts/af01/exact_quotient.json
python -m experiments.activation_funnel.train --out artifacts/af01/checkpoints --threads 2
```

The training command fits only on the registered training corpus and uses
development inputs for eligibility diagnostics. It writes the final checkpoint
hashes and `train_manifest.json` before the evaluation process starts.

Install the official Lean 4.19.0 Linux release for the kernel stage:

```bash
AF01_TOOLCHAIN_DIR="$(mktemp -d)"
curl --fail --location --retry 3 https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst -o "$AF01_TOOLCHAIN_DIR/lean.tar.zst"
echo "6fe3ce97a58f44e2b3567d455b994eacec5bfe9ae7774f2a573444480ba813fe  $AF01_TOOLCHAIN_DIR/lean.tar.zst" | sha256sum --check --strict
tar --zstd -xf "$AF01_TOOLCHAIN_DIR/lean.tar.zst" -C "$AF01_TOOLCHAIN_DIR"
export PATH="$AF01_TOOLCHAIN_DIR/lean-4.19.0-linux/bin:$PATH"
lean --version
```

Evaluation verifies the frozen manifest and source hashes. Complete it from the
same source revision as training:

```bash
python -m experiments.activation_funnel.evaluate --checkpoint-dir artifacts/af01/checkpoints --out artifacts/af01/evaluation --threads 2
python -m experiments.activation_funnel.certify all --episodes artifacts/af01/evaluation/test_episodes.jsonl --outcomes artifacts/af01/evaluation/outcomes.jsonl --out artifacts/af01/kernel --lean lean --timeout 120
python -m experiments.activation_funnel.audit_run --checkpoints artifacts/af01/checkpoints --evaluation artifacts/af01/evaluation --kernel artifacts/af01/kernel --out artifacts/af01/independent_audit.json
```

The separate `--smoke` training option produces an explicitly marked engineering
run. Its checkpoints require `--allow-smoke` at evaluation and cannot satisfy
the primary independent audit.

### Continuous integration

[The AF-01 workflow](../../.github/workflows/activation-funnel-af01.yml) trains
and freezes in one CPU job. A dependent job downloads that artifact, checks out
the same commit, then runs held-out evaluation, real Lean certification, and the
independent audit. Both jobs use Python 3.12 and PyTorch 2.5.1+cpu. The Lean archive
is checked against the SHA-256 above before extraction.

Source, test, protocol, requirement, and workflow changes trigger this pipeline.
Results-only and README-only commits leave the frozen run intact. Manual runs
are available through `workflow_dispatch`. Artifact upload uses `always()` so
partial checkpoints, errors, and compiler logs survive failed jobs; artifacts
have a 30-day retention setting. Scientific gate verdicts stay in the evaluation
report. A failed scientific hypothesis leaves CI successful when all execution,
source-integrity, and proof-audit checks pass.

## Sources

- [EvalMath evaluation calculus](https://github.com/MoralityLabAI/EvalMath/blob/a899141efeb5e7f4bcefa80ef792e855acdb9b28/paper/evaluation_design_calculus.md)
- [Intervention-preserving compactification](https://github.com/MoralityLabAI/EvalMath/blob/a899141efeb5e7f4bcefa80ef792e855acdb9b28/research/interventional_compactification/RESEARCH.md)
- [Interface-relative state equivalence](https://github.com/MoralityLabAI/EvalMath/blob/a899141efeb5e7f4bcefa80ef792e855acdb9b28/paper/interface_relative_compactification.md)
- [Evaluator integrity example](https://github.com/MoralityLabAI/EvalMath/blob/a899141efeb5e7f4bcefa80ef792e855acdb9b28/paper/reward_hacking_stateful_awareness.md)
- [LoopedTransformers semantic echo](../../papers/rmp_looped_mechinterp/semantic_echo/theory.md)

The general compactification statements in those notes have their own evidence
status. AF-01's Lean receipts certify the selected Horn proofs.
