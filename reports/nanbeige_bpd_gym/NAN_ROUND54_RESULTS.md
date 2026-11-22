# ROUND54 BF16-reachable adaptive-route contrast on CAL

The intervention starts from the learned write minus the equal-mass CAL-static
write. It screens coordinates by local BF16 spacing and projects the retained
contrast off the current bare-label margin. Its sham uses the same support and
dose with a seeded signed permutation within spacing classes. Both signs
of both arms must pass the finite checks. The original route contrast is held
fixed within each world. The screened edit can extend outside the fitted BPD
write span.

The predeclared fit selected the **0.8 spacing rule**. It tied the 1.6 rule
on the fit criterion and won the stricter-support tiebreak. The original dose
of 0.25% residual L2, the 3% dose and pair tolerance, and the 0.125 full-head
and bare-label margin thresholds were retained. Each active sign had to retain
cosine at least 0.50 with its signed original contrast.

| Panel | Precision | Selected rule with all four doses matched | Selected rule with all gates passed | Dense comparator with all gates passed | Selected rule, both worlds passed |
| --- | --- | ---: | ---: | ---: | ---: |
| Fit | NF4 | 8/8 worlds | 0/8 | 0/8 | 0/4 roots |
| Fit | BF16 | 8/8 worlds | 5/8 | 0/8 | 1/4 roots |
| Disjoint CAL check | NF4 | 8/8 worlds | 0/8 | 0/8 | 0/4 roots |
| Disjoint CAL check | BF16 | 8/8 worlds | 3/8 | 1/8 | 1/4 roots |

Every checked NF4 world passed the four-arm dose gate under the selected rule
and failed the original-contrast alignment gate. Its supported contrast retained
a median of 0.178 of the original unit vector norm on the check panel,
compared with 0.488 in BF16. The subsequent continuous margin projection
changed those medians by less than 0.001. The fit panel showed the same
pattern. All three physically qualified BF16 check worlds also preserved the
full head, with maximum absolute vocabulary-logit shift at most 0.0703125.

## Exact unrestricted feasibility

We removed the mask and current-head constraints from a separate mathematical
audit, allowing any finite BF16 displacement in the registered dose band.
The original cosine requirement was mathematically unreachable for **32/32
NF4 signed targets** and **8/32 BF16 targets** over both CAL panels. Both the
one-sided support bound and the LATTICE53 nearest-point certificate returned
the same counts. NF4 remained unreachable in both worlds and both signs of
all eight roots. Uncertified targets may still fail physical or head gates.

For the measured NF4 route contrast, using more candidate masks at this dose
cannot recover alignment. A higher-dose comparison would need its own CAL
feasibility and full-head check. A comparison of the reachable projection
would test a narrower physical contrast. The current BF16 check supplies one
both-world root for an engineering fixture, with zero full-coverage roots
across the two precisions.

Both CAL panels were previously used elsewhere in the gym. Counts are
descriptive over eight distinct roots with repeated worlds and precisions.
The full 166,144-row head was replayed for 112 clean or qualified states.
Clean top and selected logits matched exactly, and the largest normalizer
error was 4.05 × 10⁻⁷. The main audit used about 142.2 MiB peak sampled
private memory and made zero new pretrained decoder calls. The original
ARG-03 seal was preserved. [Protocol](../../experiments/nan_round54/PROTOCOL.md)
· [Calibration aggregate](nan_round54_results.json)
· [Feasibility proof](../../experiments/nan_round54/FEASIBILITY.md)
· [Feasibility aggregate](nan_round54_feasibility.json)
