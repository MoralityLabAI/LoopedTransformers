# PEF-03F | Posthoc future reactivation of frozen head-nullspace memory

## Frozen provenance

This experiment was selected after the PEF-03 original registered
evaluation and the posthoc PEF-03P/PEF-03N analyses. It constitutes a
separate exploratory **internal hidden-state intervention**, rather
than a new prespecified model-evaluation criterion.

The successfully audited original checkpoint source is hosted PEF-03
run 43680455308. The independent
[pinned PEF-03F CI run 43681766062](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43681766062)
loaded its exact checkpoints, verified both SHA256 digests, performed
posthoc tests, and archived artifact
pef03f-pinned-original-future-reactivation-receipts.

- Seed 17 frozen checkpoint SHA256:
  c174fbfd8bef4fa026806e62e66ad48a38827935c1121a487514de9948f61432
- Seed 29 frozen checkpoint SHA256:
  76dfcf7c401a8d60ae4570cf281b0aa788362bb76bde7f0c040438333a5a6e01

This workflow performs no training or model update.

## Mathematical intervention

Let W be the frozen final four-class head applied to the 16D
recurrent memory, with orthogonal row projection PR and kernel
projection PN. At each visit j, compare the memory h_j from an
original eight-bit history with h'_j from the same history with
the oldest bit flipped. Both trajectories receive the same
newer input bits.

Construct the hybrid memory

    h_mix_j = PR h_j + PN h'_j.

By construction it retains the original head-visible row component
and replaces only the exactly head-invisible 12D component.

When j is the final visit (j=7), the last-head logit value
W h_mix_7 + bias equals the original W h_7 + bias, up to
floating-point rounding. For j<7, pass h_mix_j through the
remaining frozen recurrent updates with the same later bits and
measure the *final* task logits.

A hybrid state may be off the model's natural history manifold.
This is an intervention on the recurrence, not an observation of
ordinary input-history behavior.

## Host-verified finite measurements

The experiment uses the original 52 heldout histories and their
oldest-bit-only counterfactuals.

| Patch location | Seed 17 mean final logit L2 change | Seed 17 task-class changes | Seed 29 mean final logit L2 change | Seed 29 task-class changes |
| --- | ---: | ---: | ---: | ---: |
| After visit 0 | 9.068 | 14/52 | 1.485 | 0/52 |
| After visit 3 | 1.135 | 1/52 | 2.088 | 0/52 |
| After visit 6 | 0.488 | 2/52 | 1.040 | 0/52 |
| After final visit 7 | below 0.00001 | 0/52 | below 0.00001 | 0/52 |

The seed-17 first-visit hybrid can change the final predicted task
class. Its effect is different from an ordinary oldest-bit flip,
which changed the original task-head predicted class on only 2/52
histories in the same checkpoint.

The seed-29 hybrids still measurably change final logits after
earlier visits. A zero class-change rate means only that the
winning discrete class remains the same.

## Direct local differential readout test

For the lexicographically first heldout history, calculate the
Jacobian of the *remaining recurrent computation followed by
the final head* with respect to memory immediately after visit j.
For the frozen final-head null basis N, record

    G_j = || D_h(final head after visits j+1..7) N ||_F.

These are exact PyTorch autograd derivatives evaluated on the
original unpatched trajectory, and separate from the finite
hybrid-state transplant.

| Visit j | Seed 17 G_j | Seed 29 G_j |
| --- | ---: | ---: |
| 0 | 1.8469 | 4.4933 |
| 3 | 1.7803 | 3.8859 |
| 6 | 1.1096 | 2.1320 |
| 7 | below 1e-12 | below 1e-12 |

The values establish that current-head-null directions can
become future-head-visible through recurrent mixing. There
is no additional update after j=7, and the Jacobian returns
the original head W, whose restriction to N is zero.

## Pure mathematical companion

The finite Lean module MathCan/OrderedTangent.lean contains
a simple exact example:

    head(x,y) = x,
    U(x,y) = (x+y,y).

The direction (0,1) lies in the head kernel, while
head(U(0,1)) = 1. The theorem proves a dormant head-null
direction can become visible after one update. The same
module proves the generic additive affine-head kernel
invariance and the nonfactorization of a varying hidden probe
through a constant task output.

These are formal algebraic statements about exact maps.
The trained-model measurements are an independently
verified finite experiment and do not constitute a
universally formalized theorem about learned transformers.

## Reproduction

Run with the original SHA-pinned checkpoint directory:

    python -m pytest research/pef03/test_pef03_future.py -q
    python -m research.pef03.pef03_future --dir /path/to/original/artifact

The workflow .github/workflows/pef03f-pinned-future.yml retrieves
run 43680455308, validates exact SHA256s and original receipt
headers before any analysis, tests checkpoint-tamper guards,
and archives JSON results. Artifacts are subject to GitHub
retention limits.

Claims remain specific to this 16D trained two-token synthetic
recurrent model, two frozen checkpoints and their 52
heldout source histories. Future work should measure
intermediate-state manifold distances and downstream
task effects under more natural interventions.
