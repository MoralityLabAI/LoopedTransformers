# RMP-INK-PC01: trained recurrent gate, serial-step and state-patch positive control

Status: development-only, 2026-11-21. A *real trained scalar* in an otherwise
**explicitly supplied graph-transition operator**. This is a CPU functional
positive control for the loop instrumentation, not evidence that an LLM
learned to perform graph reasoning. It also does not establish any feature
of Inkling or the Goodfire coordinate-2027 routing hypothesis.

## Why this arm exists

The existing RMP experiments report chance-level results in several
independently trained tiny LoopedDecoderLM cells. Huginn-0125 has a
pretrained positive result on one-hop variable-copy chains but reaches
only 26% on three-hop examples at 16 recurrent passes. The native Inkling
variable-loop research checkpoint requires a supported memory-suitable
runtime and is not used in this CPU instrument.

A small known-working recurrent computation gives us a **required** check:
if a program claims to patch the active state coordinate and still cannot
affect the correct output, its instrumentation is broken.

## Model and data

There are 17 possible graph nodes. Each source root defines a random
single-cycle permutation P and starting node i. A task requests a number
h of graph hops and its exact target is computed by h successive
applications of P, independently of model output.

The operator P is **given explicitly** to the recurrent system. The only
learned parameter is theta, with alpha = sigmoid(theta). Each pass
proposes m = z P and mixes

    z_(t+1) = (1-alpha) z_t + alpha (z_t P).

The model is informed of h and freezes its state after h passes. The
training objective is cross-entropy on the exact target at h, with
analytic gradients through the unrolled recurrence. Training uses
48 independent roots with hop lengths 1–4, 70 epochs, learning rate .75.
The development suite uses 64 new roots, hop lengths 1–8, and inference
budgets 1/2/3/5/7/10. Multiple tasks from one graph share a root and
are correlated; inference depth observations are repeats, not fresh roots.

Training and evaluation are deterministic, with independent source seed
pairs (23 and 20261121). The source generator never opens an external
test split. GitHub Actions runs it on CPU, retaining a checkpoint and
source fingerprints in an artifact rather than committing generated data
to the repository.

## Loop-boundary causal control

At visit 2, for tasks of length 4–8, patch the raw 17D activation at a
fixed L2 dose .85. The *selected* direction subtracts from the current
argmax coordinate and adds to the next coordinate. The selected index is
computed from the running model state, not a named pretrained neuron.
The perturbation tests are clean, zero, selected, seeded Gaussian random
direction, and seeded random coordinate. Each physical displacement is
recomputed and checked against the nominal dose. The same tasks and
model are used for every arm.

The dose was selected in a separate synthetic development pilot, so the
following results and any CI replication are **development descriptive
engineering checks**. A comparison of this selected active coordinate
with random-coordinate shams is intentionally asymmetric: the selected
coordinate is guaranteed to be active. It cannot establish a discovered
causal feature in a native neural network.

## Run and independently review receipts

    python -m pip install numpy
    python -m unittest discover -s research_gym/inkling_positive_control \
      -p 'test_experiment.py' -v
    OPENBLAS_NUM_THREADS=1 python -m research_gym.inkling_positive_control.experiment \
      --out runs/rmp-ink-pc01

The command writes:

- checkpoint.json: theta, alpha, training recipe and learning-curve snapshots.
- depth_matrix.jsonl: source-root paired verified correctness, with fields
  compatible with EvalMath's development-only variable-loop depth auditor.
- interventions.jsonl: every arm, source root, case ID, nominal and realized
  L2 dose, intervention visit and final externally recomputed target.
- summary.json: SHA-256 source and output hashes, root-cluster effect
  summaries, per-depth accuracy, positive-control gates and limitations.

The exact per-case correctness verifier walks the supplied source
permutation; it is separate from the differentiable model forward pass.

The local engineering result from an initial fresh-root run was 64/512
at one pass, 128/512 at two, 192/512 at three, 320/512 at five,
448/512 at seven and 512/512 at ten. The learned alpha was 0.9924545.
On 320 cases whose requested hop length exceeded the patch visit,
clean and zero arms scored 320/320, selected 0/320, seeded Gaussian
sham 319/320 and random-coordinate sham 309/320. Those numbers are
development observations, not a prospectively confirmed result.

## Native successor gate

Before claiming a general recurrent mechanism, move to a model with
learned graph interpretation and fixed checkpoint identity. Score
exactly the same independent task roots for all pass counts. Measure
both model correctness and first-valid passes. Then extend the adapter
to give a named block-boundary state, a real cross-loop reader, and a
tied control with no intervention. Random-direction shams need matched
actual state displacement; judge only independently verified final
answers after the recomputed native forward pass.

The 2027 coordinate observed by Goodfire belongs to the fixed-pass
Inkling checkpoint and cannot be assumed in a different model. The
BPD source-only reader protocol is in its independent
research/inkling-cross-loop-causal-20261121 branch; its existing
NC09 and AMM sealed cohorts remain untouched. EvalMath FSI-01
root-level power, multiple-comparison and prospective-source gates
apply before any scientific confirmation.
