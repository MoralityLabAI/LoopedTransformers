# LATTICE53 an upper bound on BF16 route alignment

DOSE51 found that most raw learned and CAL-static route directions lost
angular fidelity at 0.25% residual L2. This follow-up uses the same exposed
roots, route directions, both signs, and 3% radius band. It bounds any finite
BF16 displacement at that radius, including edits that the DOSE51 ray search
would never try. The native archive and DOSE51 receipt identify every row.

Let `h` be the clean BF16 residual and `d` the signed unnormalized route
direction. For positive scale `s`, let `m` be the squared distance from
`h + s*d` to the nearest finite BF16 vector. Coordinatewise rounding
minimizes squared Euclidean distance on this product lattice. For any
reachable difference `x` of norm `r`,

```text
dot(x,d) <= (r² + k) / (2s)
k = s² ||d||² - m >= 0
```

The nonnegative `k` follows because `x = 0` is reachable. Dividing by
`r ||d||` gives an upper bound on cosine. The maximum of the bound on the
allowed interval occurs at an endpoint because `r + k/r` is convex.
A fixed grid of scales screens for a useful bound. The selected scale and
floats are interpreted as exact dyadic inputs for a rational recheck. This
brackets each probe coordinate by adjacent BF16 values, verifies the nearest
squared distance, and compares squared expressions at both radius endpoints
using the registered decimal thresholds as rationals. Only a strict
rational inequality below 0.98 counts as a certificate.

A certificate excludes every finite BF16 edit in the radius band for that
signed direction. An unresolved row may be feasible or require a tighter
bound. The exercise does zero model calls and tests no future behavior.
Counts are descriptive on the 16 exposed DEV roots. No ARG-03 seal access.
