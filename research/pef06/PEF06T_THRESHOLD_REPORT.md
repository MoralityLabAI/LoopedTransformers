# PEF-06T | Threshold-only pair effects versus nonadditive frozen logits

**Status:** posthoc mechanism control, chosen after the initial
PEF-06 frozen-model audit. The original PEF-05 registered
negative verdict is preserved.

## What the additional control measures

Pair-only predicted-class changes occur when flipping either
older input bit alone preserves the predicted class, while
flipping both changes it. This may be caused by a simple
threshold on an additive logit score. On a complete natural
intervention quartet, define the logit surrogate

    z_add(x^ij) = z(x^i) + z(x^j) - z(x).

It has zero finite four-corner logit difference by
construction. Applying argmax to this surrogate can still
produce pair-only class changes.

For every actual pair-only class change, the audit
records whether the surrogate predicts the *same* actual
joint-intervention class. A mismatch certifies a nonzero
four-corner logit difference on that source quartet and a
discrete class effect that this additive surrogate
cannot reproduce. Matching predictions remain compatible
with additive-score thresholding; neither outcome is a
complete circuit attribution.

Also record

    D_ij z(x) = z(x^ij)-z(x^i)-z(x^j)+z(x)
    D_ij h(x) = h(x^ij)-h(x^i)-h(x^j)+h(x).

The full 256-word source cube has all last-two-bit task
labels fixed under each of the 15 oldest-six-bit pairs.
Counts below sum 15*256 intervention-source cases per
model. They must not be interpreted as independent source
histories or statistical effect sizes.

## Independent compatible-model CPU replay of the ORIGINAL source

A local read-only CPU audit loaded all twelve exact
original PEF-05 checkpoint files from registered
[run 43687506414](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43687506414),
artifact 14150782670. It verified each full checkpoint
SHA256, strictly loaded the 16-dimensional tied-attention
model state, and scored every bitword at both h8 and h10(01).
The local environment was Python 3.13.5 with
PyTorch 2.10.0+cpu.

**Primary descriptive arm: eight-visit original task head.**

| Seed | Arm | Original task correct /256 | Pair-only class events | Compatible with additive surrogate | Class differs from additive surrogate | Pair-only events from correctly predicted original task |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 101 | baseline | 244 | 90 | 15 | 75 | 81 |
| 101 | protect | 248 | 52 | 11 | 41 | 45 |
| 101 | sham | 247 | 57 | 17 | 40 | 49 |
| 109 | baseline | 248 | 24 | 0 | 24 | 24 |
| 109 | protect | 252 | 24 | 0 | 24 | 24 |
| 109 | sham | 252 | 24 | 0 | 24 | 24 |
| 127 | baseline | 250 | 41 | 3 | 38 | 39 |
| 127 | protect | 249 | 40 | 2 | 38 | 38 |
| 127 | sham | 249 | 40 | 1 | 39 | 38 |
| 157 | baseline | 256 | 0 | 0 | 0 | 0 |
| 157 | protect | 256 | 0 | 0 | 0 | 0 |
| 157 | sham | 256 | 0 | 0 | 0 | 0 |

Across these models the h8 head has nonzero mixed
continuous hidden-state/logit finite differences,
including seed157 where the *predicted task class*
never changes under oldest-bit intervention.
Class-only indicators are an incomplete proxy for
latent nonlinear interactions.

**Secondary diagnostic: ten visits with two extra
bits (0,1).** The original head received training
supervision at h8. Its h10 predictions are an
out-of-training-horizon continuation, and their
pair-only class event counts are:

| Seed | Baseline | Protect | Sham |
| --- | ---: | ---: | ---: |
| 101 | 213 | 183 | 174 |
| 109 | 42 | 49 | 48 |
| 127 | 261 | 210 | 239 |
| 157 | 0 | 0 | 0 |

These h10 scores must not be labeled successful or
failed supervised task predictions at the original
training horizon.

The complete local independent 12-model JSON receipt
is available as conversation artifact
PEF06T-sandbox-threshold-surrogate, generated
2026-11-15, and a separate pinned hosted
GitHub workflow is provided in this branch.

## Exact finite negative controls

- Additive score z=x0+x1, class threshold 2:
  at all zero, neither singleton changes the class,
  but the pair does. Mixed logit and hidden second
  differences are exactly zero.
- Nonlinear score z=3*x0*x1-1, threshold class at zero:
  the same pair-only class effect arises and its
  finite mixed logit response is nonzero.
- For the original PEF-06 XOR-of-three-AND latch,
  the complete pairs (0,1),(2,3),(4,5) are active
  at all zero, while singleton bits are inactive.

The first two are independently certified by
[MathCan CoalitionThreshold.lean](https://github.com/patrickdugan/MathCan/pull/52).
The audit also counts source cases by supervised
target class and by whether the original model
prediction was correct, constraining a decision
boundary explanation.

## Interpretation and limitations

These findings establish finite nonadditive
logit responses on a synthetic source cube for
particular frozen checkpoints. The additive
surrogate is a negative control, not an
independently trained baseline. A disagreement
to that surrogate may occur even without a
specialized AND/XOR circuit and cannot
alone identify a unique mechanism.

All measurements are posthoc on reused source
histories. The original PEF-05 negative
gradient-rescue verdict remains intact.
No capability improvement or behavior of a
pretrained language model is inferred.

Source:
https://github.com/MoralityLabAI/LoopedTransformers/pull/64
