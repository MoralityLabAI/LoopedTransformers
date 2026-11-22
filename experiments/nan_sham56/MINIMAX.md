# SHAM56 peak-logit design on fine BF16 support

The initial fine-coordinate sham families cleared physical dose and leakage
but failed the full-head peak limit on CAL. Fix a second method using only
the supported ambient space and the stricter BF16 spacing cap 0.25. Use the
same frozen active edits, signed target, 0.75% residual dose and admission
criteria. Construct a 24-dimensional deterministic physical basis with the
previous seeded ambient sham as its starting direction. Remove the original
contrast, margin-gradient and reached-active directions on that support.

Stream the analytical RMSNorm directional derivative into coordinates of the
frozen vocabulary head. Search only in the small coefficient space. For at
most eight iterations at each of four fixed step sizes, take a sphere-tangent
step using a soft weighting of the 64 largest absolute predicted logits.
Accept a step only if it lowers the maximum absolute continuous logit shift.
No observed rounded head is used to select a direction. Freeze the recipe
after the initial sham results and score every world in both earlier CAL
panels, with no further tuning.

Prepare the chosen direction in both signs through the same BF16 lattice and
evaluate the actual full first-pass head on every physically qualified
four-arm group. Compare actual rounded peak and RMS drift to the initially
registered ambient fit sham. Report every world, not only successes. This is
a CAL engineering atlas and uses zero pretrained decoder calls or formal
tests. The deterministic finite-head search is a design choice for the sham.
Future-only responses and the current task's broader causal claim still
require a separately registered pretrained pilot.
