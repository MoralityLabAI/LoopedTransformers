# NAN-CENTERSHAM108 redistributes the chord on an exact BF16 midpoint

HEADSWAP107 tries different signs on the attained BF16 chord coordinates.
It lowered differential first-pass RMS gaps on all 100 signed pairs, and
zero cleared 0.005 logits. The coordinatewise magnitude restriction
limits how much weight can move between directions that couple differently
to the head. This stage tests a different control family. It keeps the
same exact midpoint and matches the reached chord length and each endpoint
radius to within 3%. The maximum absolute chord cosine remains 0.10.

Use all HEADSWAP107 signed pairs. Let m be their attained midpoint, h
their attained half chord and g the difference between m and the clean
state. Both endpoint distances are determined by the squared half-chord
length and the inner product g dot h. Build a tangent plane orthogonal to
h. When feasible, also search an affine slice in that tangent that fixes
g dot h. At each site use the 512 token rows from the HEADSWAP107 design
grid, the checkpoint output weights and the continuous RMSNorm Jacobian
at m. Use 16 projected power iterations from a deterministic Gaussian
seed to find a high-response axis. Combine it with each of two projected
random perpendicular axes. Sweep 24 equally spaced angles and nine fixed
multipliers from 0.84 to 1.16. Scan the tangent before the optional
radius-split slice, with a separate seeded stream per backend,
root, world and sign.

For each target near m plus or minus the proposed half chord, try the
nearest BF16 value and both neighbors for either endpoint. Compute the
other endpoint by subtracting from the exact original sum. Keep only
exact BF16 endpoints with the original sum. Retain the original
coordinate pair in either order as a fallback in the local stencil.
Choose by minimum coordinatewise half-chord error, then check all reached
geometric gates on the assembled candidate.

Evaluate every geometry-qualified candidate with the actual rounded CPU
head on the design rows. Rank by the indicator for sampled peak drift
above 0.125, maximum of the absolute common and differential RMS gaps,
differential RMS gap, then sample peak drift and first candidate. Lock
the candidate digests and design results before streaming the full
head. Score the established per-endpoint greedy, full peak and paired
answer-margin gates. Require both RMS gaps to be at most 0.005 on the
full vocabulary and on the 165,632 excluded vocabulary rows. Report all
refusals, signed pairs and both-world, both-sign root qualification.
Coordinatewise dose is allowed to change, so any eventual causal
contrast concerns direction at a matched overall dose.

This is archive calibration on exposed development roots. The root is
the unit. The RMS tolerance is an instrument-matching criterion. The
stage uses zero formal tests, zero decoder forwards and zero ARG-03 test
cases. A native comparison of first-pass and delayed responses still
requires fresh roots, a causal effect floor and targeted reader rescue.
