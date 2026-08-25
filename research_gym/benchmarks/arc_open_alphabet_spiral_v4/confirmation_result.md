# Confirmed synthetic mechanism result

The unchanged v3 policy passed its sealed eight-root v4 confirmation.

## Task result

- `srt_expand_collapse`: 47/64 exact solves (73.44%).
- strongest matched control, `residual_beam`: 35/64 (54.69%).
- absolute margin: +18.75 points (registered minimum +10).
- root-block deltas: `[+0.125, +0.625, +0.125, 0.000, +0.250, +0.250, +0.250, -0.125]`.
- 95% root-block bootstrap interval: `[+0.046875, +0.34375]`.
- family deltas: +25.0 mask/fill/symmetry, +25.0 conditional component
  motion, +12.5 crop/repeat/recolor, and +12.5 select/translate/recolor.
- generated simple-null exact: 68.75% SRT versus 50.00% residual beam;
  novelty did not hurt.

Structural target reachability and oracle-at-budget exact solve were both 100%.
The exact same candidate-set shuffle tied the primary at 47/64, proving that the
gain is candidate-set coverage, not candidate order. Direct TRM top-1/top-k solved
0%, graph search 34.38%, edit 0%, random 1.56%, and Sobol 1.56%; the exhaustive
target-program ceiling was 100%.

## Measurement and operation

The evaluator audit returned `ACCEPT`. Across 64 independent items there were 15
SRT-only wins, 3 residual-only wins, and 46 ties. Conditional on the 18 decisive
items, the SRT win rate was 83.33% with Wilson interval `[60.78%, 94.16%]`.
Repeat flip, position sensitivity, and hard-fail selection were all 0%; deterministic
coverage was 100%; measurement grade was B and met all declared limits.

The run completed in 21m07s under 2 GiB RAM, 50% of one logical CPU, and a
30-minute timeout. PID-scoped cleanup passed with no lingering owned process or GPU
compute application.

## Claim boundary

This confirms the bounded synthetic claim: on held-out operator pairs, motifs, and
families, a train-only learned function-space D2C3 SRT preserves useful rewrite
hypotheses better than the strongest matched residual search control. It does not
show that candidate ordering matters, that SRT is a general transformer
replacement, or that the current DSL solves public ARC.
