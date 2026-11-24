# GAUGE42 BPD transport and PercInterp sensitivity

The full BPD write and fitted cross-pass prediction stayed numerically
constant through 19 coordinate systems per precision. The pruned
PercInterp path sum changed under orthogonal rotations for all 16 focused
DEV roots at each of the three partial cutoffs. Equality of the full
operator therefore leaves component-level path attribution ambiguous
under mixing of the factors.

The experiment reused audited COMP14 twins at physical layer 21 and the
independently fitted IO18 banks. A zero-intercept ridge map fit on eight
focused CAL roots predicted the pass-2 BPD write difference from pass 1.
The map uses an orthonormal basis for the physical write span. The
evaluation covered 16 exposed DEV roots per precision, separately in
focused, copy and graph-erased arms.

## How much did the CAL fit predict?

| Focused DEV endpoint | NF4 | BF16 |
|---|---:|---:|
| Physical transfer squared error / zero baseline | 1.371 | 0.995 |
| Physical error of the CAL root-shift control / zero baseline | 0.993 | 1.015 |
| Direct head tangent transfer error / zero baseline | 1.152 | 1.041 |
| Native bare-label margin swing, mean absolute, pass 1 → pass 2 | 0.012 → 10.578 | 0.015 → 13.195 |
| BPD write tangent, mean absolute, pass 1 → pass 2 | 0.002 → 0.044 | 0.002 → 0.085 |

An error ratio below one beats the zero prediction of the measured
quantity. BF16 was close to baseline and NF4 was worse on the physical
write, and both exceeded baseline on the direct-head tangent. The
counterfactual margin rose sharply in the actual late state. The tangent
measures only the fitted write difference at a common state through the
continuous RMS-normalized head. It has different scope from the finite
native margin swing. This observational fit provides no signed cross-pass
edge measurement. Erased twins had zero BPD differences and zero native
margin swing, and their ratio denominators were zero.

## What changed after pruning?

| Focused DEV rank cutoff | NF4 roots with rotation-dependent signed sum | BF16 roots with rotation-dependent signed sum | NF4 / BF16 with both signs across rotations |
|---|---:|---:|---:|
| 0.25 | 16/16 | 16/16 | 13/16 · 12/16 |
| 0.50 | 16/16 | 16/16 | 12/16 · 9/16 |
| 0.75 | 16/16 | 16/16 | 7/16 · 3/16 |
| 1.00 | 0/16 | 0/16 | 0/16 · 0/16 |

Reciprocal scaling, sign changes and matching permutations preserved
pruned path summaries. General orthogonal mixing preserved the full
prediction and changed the component decomposition, with a mean range
of 0.061 logit in NF4 and 0.018 in BF16 among rotated sums at the 0.50
cutoff. Native path and local in/out selectors had identical membership
on 9/16 NF4 and 6/16 BF16 focused roots at that cutoff. The comparisons
use one fixed bank and an observational transport map. Causal effects and selector performance across refits require separate tests.

Keep the factor bank and physical intervention directions fixed when
testing DELAY39. Compare PercInterp to local degree at equal intervention
budgets, and use physical-subspace agreement to check whether a proposed
path survives a refit. The surrogate error rules out using this CAL map
as the measured input to that causal comparison. DELAY39's matched
native edits supply the first direct test of those proposed edges.

The pinned archived layer-21 logits replayed exactly in both precisions.
The continuous head tangents matched two-sided differences to within
1.2e-12 absolute per unit L2. Whole-map and reciprocal graph closure had
maximum absolute numerical discrepancy below 2.0e-12. A second calculation
recounted all cutoff sensitivity and sign-reversal cells, and an independent
two-component path enumeration matched the implementation. The CPU run
peaked at 118.4 MiB sampled private commitment and 100.1 MiB working set.
It used zero new pretrained forwards and zero formal tests. The probe
used exposed COMP14 data, with the original ARG-03 test sealed.

[Protocol](../../experiments/nan_gauge42/PROTOCOL.md) ·
[Aggregate results](nan_gauge42_results.json)
