# DUALHEAD61 joint peak and RMS feasibility in the NF4 sham atlas

Use the same 24-dimensional low-spectrum atlas and eight exposed CAL
roots as HEADNULL60. The full-vocabulary analytic derivative matrix is
A and its squared RMS operator is G = AᵀA / vocabulary size. Set the
continuous peak target to 0.125 and the RMS interval to the mean signed
active rounded RMS ±0.005. This continuous target is a design diagnostic.
Evaluate rounded BF16 shams with the earlier full-head and signed RMS gates.

The upper endpoint R of the RMS interval is impossible together with
the peak target P if, for any nonnegative multipliers mu_i, the matrix
G + sum_i mu_i a_i a_iᵀ - (R² + P² sum_i mu_i) I is positive definite.
This follows by testing the quadratic form against a unit coefficient
vector and using each absolute logit bound. Use the prior worst rows
and a fixed coordinate-ascent schedule to search for a witness, with
up to six additional maximizing derivative rows. The numeric search
chooses multipliers. The final check uses exact rational LDL and an
outward error allowance for the float64 Gram: each entry's error is at
most M² times 2^-32, where M is the maximum stored absolute atlas entry.
The atlas entries are float32 and the reduction uses fewer than 2^18 rows.
Products of converted float32 entries fit in float64, and the rounding
bound allows float64 accumulation and division. Shift the matrix
downward by dimension times that entry allowance before LDL. The claim
is about the frozen stored derivative atlas and the stated continuous
limits. It does not certify BF16-quantized or whole-residual infeasibility.

Independently search for a unit witness with small maximum constraint
ratio, combining peak and both RMS endpoints. Use the minimum-RMS,
dual-eigenvector, original worst-row and ordinary peak-aware starts.
The fixed tangent line search selects from those starts and never sees
rounded logits for candidate choice. Replay the selected direction
with both signs. Require exact active replay and the original finite
physical, peak, margin, and native-winner gates. Report full-world and
both-world-root coverage, additional 0.005 RMS matches, and unresolved
continuous worlds. Do not infer feasibility from the lack of a dual
certificate.

The root is the independent unit. The study is descriptive on already
exposed CAL, uses the archived first-pass boundary, and does not read
DEV or sealed ARG-03 vectors. New pretrained decoder forwards and
formal comparisons are zero.
