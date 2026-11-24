# DOSE38 post-rounding CAL feasibility

READPATH37 exposed phase-specific intervention dose failures in the fixed
IO18 writes. This CAL-only instrument test asks whether scaling each
intervention before BF16 rounding recovers the requested physical L2.
Use the same three COMP14 CAL arms, both worlds, both passes, both
precisions, the eight target-matched IO18 directions, the same eight
orthonormal directions, and residual coordinates 2278, 2279 and 2280.
Inspect only pinned CAL states. No new head or decoder calculation,
task-score reads, model load, causal selector fit, or DEV analysis.

For each nonzero unit direction and each sign, the realized displacement
norm is monotone in the nonnegative input scale when the reference state
is exactly BF16. Bracket the requested L2 between input scales zero and
four times its straight-line value, then take 36 binary steps. Compare
the realized norms just below and above the target in the final bracket.
Choose the smaller absolute error, breaking ties toward the lower scale.
Compare this calibration against the existing one-shot scale separately
for the original target fractions 1% and 2%. Keep the original 3%
physical-dose tolerance and also audit cosine to the intended signed
direction. A usable instance requires both signs within tolerance and a
cosine of at least 0.98 for each. Report direct one-shot counts and
calibrated counts under that same cosine criterion. Retain lower/upper
gaps, sign asymmetry and per-direction coverage.

Use 8 independent exposed CAL roots, with repeated prompts and
directions per root. This is an instrument feasibility test, with zero
formal comparisons and no observed behavioral effect. Hold the resource
lease at 5 GiB start, 4 GiB reserve, 0.75 GiB private, one numerical
thread, Below Normal, and 600 seconds. Keep individual dose records local.
The original 168-case test stays sealed.
