# PEF-06 | Frozen model joint-intervention audit

**Prior source:** original PEF-05 registered negative-result
[GitHub Actions run 43687506414](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43687506414).
This read-only analysis was selected posthoc. Its detailed scope
was committed independently in PRE_AUDIT.md *before running*
the PEF-06 model measurements.

Three distinct September 2026 OpenAI math preprints motivate
the independent finite null. The source references and
mathematical proof are in [Compactification-Theory PEF-06](https://github.com/MoralityLabAI/Compactification-Theory/pull/146).
The finite Lean certificates are in
[MathCan PEF-06](https://github.com/patrickdugan/MathCan/pull/52).

## The calibrated Boolean phenomenon

For earliest six bits, define a mathematical target F3
as parity of three disjoint AND pairs. At the all-zero
input every singleton flip is inactive, while each of
the three complete two-bit pairs changes the output.
This provides a controlled example where singleton
causal sensitivity is zero but coalitional intervention
is nonzero.

Under independent BSC(p) noise from all zero,
P(F3(noise)=1)=3p²-6p⁴+4p⁶. At p=1/4
the exact false activation is 169/1024. Over
uniform sources, expected sensitivity is 3,
and the noisy output-disagreement probability is
[1-(1-p)^6]/2. These are independent finite nulls,
not assumed learned-model laws.

## Frozen recurrent-model measurement

The original PEF-05 run has twelve final trained
recurrent checkpoint states: four seeds × three
arms (baseline, protect, sham). The full checkpoint
SHA256 and model parameter hash are checked before
and after frozen analysis. For each checkpoint,
the exact 256-point eight-bit input cube is
evaluated once through the original tied attention
recurrent model and its fixed two-visit natural
continuation (0,1).

For each oldest input bit i in 0..5, report the
total and average hidden-state/logit difference
and predicted task-class changes under a bit flip.
For each of the 15 unordered pairs i,j, compute

    D_ij h(x)=h(x^ij)-h(x^i)-h(x^j)+h(x)

and the analogous task-logit difference, together
with pair-only class changes where neither of the
two singleton flips changes the original class.

The last two supervised target bits 6,7 are
held fixed in every intervention. All perturbed
states are naturally generated model trajectories;
there is no off-manifold activation splicing.

Separately, fit an oldest-six-bit coalition-latch
ridge readout on the original PEF-05 probe
partition and score its heldout histories,
with a seed-matched untrained random trunk control,
a heldout majority-label baseline, and the
optimal source-only recent-seven-bit decoder ceiling
computed from the exact paired histories.

This probe is **posthoc** and reuses the original
finite eight-bit source universe. A good readout
cannot be cited as an independently preregistered
capability gain or as proof that the recurrent
task head itself calculates the coalition latch.

## Original registered result preservation

The source archive must still report PEF-05's
registered criterion_not_met conclusion. No
weights are retrained, no optimizers instantiated
for updating the model, and no original trial
threshold is reconsidered. The source-file
and checkpoint IDs remain attached to every
reported model/arm.

The 256-point intervention sweep is a mechanistic
diagnostic, with no claims about production
language models, safety modes, or improved
Lean reasoning.

## Reproduce

    python -m pytest research/pef06/test_frozen_coalitions.py -q
    python -m research.pef06.frozen_coalitions \
      --archive /tmp/pef05_original \
      --out /tmp/pef06_frozen

The Github Actions workflow downloads the original
immutable PEF-05 source artifacts; no GPU or paid
compute is required.
