# ROUND54 BF16-aware route contrasts on CAL

DOSE51 and LATTICE53 found that most individual route-write directions
cannot be realized faithfully in BF16 at 0.25% residual L2. Evaluate a
different physical target, the learned write minus the equal-soft-mass
CAL-static write from the same native input. This removes their common
native-write reference. The contrast is defined in physical coordinates
using the pinned controller and factors. It does not identify a unique
factorization or a unique VPD component.

For each world, divide the signed physical contrast by its L2 norm. Retain
coordinates where the larger of the up and down adjacent BF16 steps is at
most `t` times the coordinate's absolute expected displacement at the fixed
nominal dose. Compare the dense comparator and four fixed values of `t`.
Project the supported vector off the current bare-label margin gradient
on the same support. Construct one seeded sham on that support by signed
permutation within equal BF16 spacing classes. Remove its margin and route
contrast projections. Keep the same sham seed for all support choices of a
world, with no retries. Retain mask coverage, the projected cosine to the
original contrast, and the BF16-rounded arm diagnostics. Coordinate
screening and projection can leave the fitted BPD write span.

Reuse the DOSE51 physical radius, 3% dose and pair tolerances, 0.98
per-arm direction cosine, and 0.125 current-head and margin thresholds.
In addition, both active signs must retain cosine at least 0.50 with
the signed original write contrast. Each rounded sham must have absolute
cosine at most 0.10 to that original target, with same-sign arm separation
at least one nominal radius. Score all four full heads only when the
group passes the physical and contrast checks. Require all four to
preserve the original greedy token and pass the full-head gates.

The DARK44 four-root CAL panel, with both worlds and both precisions, fits
one shared support rule. Rank the five rules by number of roots with both
worlds passing in both precisions, followed by passing world-by-precision
count. Break ties in stricter-to-looser order, with dense last. If the
winner admits zero worlds, retain it for the disjoint engineering check
and report calibration failure. The remaining four CAL roots check only
the winner and fixed dense comparator. Select before opening their case
states. Both panels were previously used elsewhere in the research gym.
No DEV or sealed ARG-03 state is evaluated. The independent unit is the
source root and the engineering comparison uses no formal inference or
downstream decoder calls. The sham is a physical contrast control and
does not yet supply a measured future-behavior noise floor.
