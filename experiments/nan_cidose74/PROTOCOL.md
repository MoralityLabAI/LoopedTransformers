# NAN-CIDOSE-74 equal-dose BPD attention gate replay

CIGATE73 found a large contribution from sphere rescaling in the
two-pass BF16 controller and unequal dynamic and static physical
doses. CIDOSE74 compares edit directions from the same frozen T2
bank on the same layer-21 native attention writes. Use a 2% L2
displacement relative to each world's unedited write, common
across arms within that world. The learned gate and the original
static CAL mask remain fixed. All fits and native captures retain
their source seals.

Replay both masks with and without the registered sphere rescale,
for four readout comparisons. Extract those directions before
the final BF16 round. Retain three controls: the learned
sphere-on edit with its native-write radial component removed,
the signed radial component alone, and a root/phase-seeded
random tangent to the native write. The dynamic radial sign
is defined by projection onto the unedited native write.
Anchor erased twins to the same seed, and require identical
outputs on identical native states. No trial direction uses
answer labels. The four principal directions hold the chosen
mask fixed and scale its output displacement. This condition
does not imply that any alternate mask would produce the same
operation within the original BPD framework.

Match physical BF16 doses within 2% of nominal and require at
least 0.98 direction cosine. Also require all arm doses to fall
within a 2% joint band per world. A root qualifies within an
arm family only if both worlds in both passes satisfy every
gate. Retain registered and unknown denominators, plus all
failure reasons. Record realized radial share and norm drift.
A continuous tangent can still have nonzero BF16 radial
leakage and finite norm drift. Record these quantities instead
of treating tangency as a norm-preserving edit.

Use the fixed task numerator direction and the observed
mean inverse residual RMS at each pass. Subtract pass 1 from
pass 2 after computing signed A-minus-B effects. Primary
descriptive outputs on focused DEV are dynamic-minus-static
with sphere on and sphere off, their difference, and dynamic
sphere-on versus the matched signed radial control. Retain all
three source arms on 8 CAL and 16 DEV roots. Report signed
root means, median absolute root contrasts, and coverage. As a secondary sensitivity check,
scale the realized write difference to the exact nominal dose
in the linear readout. Keep that arithmetic check separate
from the delivered BF16 edit.
This is an exposed-development numerical replay. It measures
the answer-aligned local write on fixed clean inputs. It
does not run the downstream MLP, the remaining model, or a
native rescue. There are zero formal comparisons and no
calibrated causal sham or meaningful behavioral effect floor.
