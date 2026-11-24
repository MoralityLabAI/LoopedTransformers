# NAN-ROUTEMATCH-86 matched phase-1 writer geometry

Use the RECIPE85 learned and ROUTE50 static gate preferences at the
same soft mass under the T2 pass-mean and sphere recipe. The comparison
uses the actual captured layer-21 pass-1 down_proj writes. At the site
of intervention keep the physical L2 dose at 2% of the observed write,
with 2% per-arm and joint dose tolerance, a signed radial gap up to 1%
of nominal dose, and a reached norm gap up to 0.0002 of the clean norm.
The original directional cosine floor is 0.98. These tolerances come
from CIDOSE74 and RADIAL76.

Before touching the BF16 lattice, check whether the natural learned
and static directions can attain the same normalized radial share. An
angle constraint around an intended unit direction gives a latitude
interval on the unit sphere around the clean write axis. Multiplying
this interval by the allowed L2 interval gives a relaxed bound on
attainable signed radial displacement. If the resulting two intervals
are farther apart than the permitted radial gap, no direction meeting
the registered cosine and dose can pass, even with continuous writes.

For each root use a seed-shared random tangent orthogonal to both
planned tangents. Compare independent natural rays, radial-matched
three-arm writes preserving the original angles, and a separate
zero-radial target applied to the two projected tangents and seeded
sham. Test the finite BF16 reached writes and qualify each complete
three-arm set jointly. The tangent control is a new intervention
direction, and its alignment with the original recipe is recorded.
Record byte-identical learned and static writes and qualify both
worlds per root for a distinct comparison.

The scoped endpoint is instrument feasibility before continuation.
Use eight exposed COMP14 CAL roots and sixteen exposed DEV roots.
Worlds and precisions repeat graph roots. The rules and small candidate
grid are fixed for both splits. No task-outcome comparison, decoder
forward, inferential test or sealed ARG-03 access enters this audit.
A later experiment must measure the actual first-pass and second-pass
heads and the independent task oracle from the same native prefill.

A supplemental branch fixes the largest-magnitude BF16 write coordinate
and projects the two tangents and seeded control in the remaining
coordinates. It retains the 0.98 cosine gate to the unconstrained
tangent, and records alignment with the original natural direction.
Run the existing LATTICE53 exact-rational angular infeasibility proof
on each original learned and static direction. A proof covers arbitrary
finite BF16 writes at the same dose and angle. A missing proof leaves the
direction unresolved. The supplemental branches never change the
original-angle search or the registered pairwise tolerances.

A second supplemental branch anchors the comparison to the actual
reached learned tangent. For a chosen BF16 change at the largest
coordinate, solve for the radial coefficient and tangent length in the
remaining coordinates using the observed learned L2 and signed radial
targets. Enumerate BF16 values of that coordinate within the permitted
dose and keep the small correction grid on the remainder. The reached
static comparison must still meet 0.98 cosine to its unconstrained
tangent. This matches finite radial leakage instead of requiring each
rounded arm to land at zero radial. Retain the same joint geometry gates.

The anchored sham is prescribed in the same fixed-coordinate shell as
the two reached routes. Its cosine gate compares the rounded vector to
its pre-rounded shell vector. The learned and static cosine gates still
bind their unconstrained tangents. The sham's actual tangent in the
remaining coordinates must have absolute cosine at most 0.05 with each
route's actual remainder tangent.
