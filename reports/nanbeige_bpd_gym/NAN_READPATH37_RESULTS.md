# READPATH37 native margin, BPD write dose and PercInterp control

The archived layer-21 residual states reproduced every checked native head
logit exactly, including the original operator-alias logits and each
archived winning token, across **576 phase states** from the paired COMP14
subset. This uses the frozen snapshot’s RMSNorm and verified head rows.
It allowed a direct output-readout test while the NF4 model remained busy.

## Matched focused pointer roots

All 16 DEV pairs per precision had matching semantic lengths and native B1
execution. CAL has eight distinct exposed roots. The direct margin uses the
bare single-token correct label minus the opposite world’s bare label.
The native unrestricted correct-answer endpoint is shown separately.

| Precision | Pass | Mean root bare-label margin, logits | Both worlds with positive bare-label margin | Native unrestricted both worlds correct |
|---|---:|---:|---:|---:|
| NF4 | 1 | 0.003 | 0/16 | 0/16 |
| NF4 | 2 | 5.289 | 15/16 | 13/16 |
| BF16 | 1 | 0.004 | 0/16 | 0/16 |
| BF16 | 2 | 6.598 | 16/16 | 16/16 |

Focused erasure had zero mean signed margin and zero both-positive
roots at either pass or precision. The same-canvas comparison clears
the batching confound on this pointer subset. It establishes a
phase-specific observed readout, with the internal path still open.

## Post-rounding dose and direct sensitivity

On DEV phase-1 states, component 0 of both IO18 fits missed the 1% target
in all 96 registered worlds per precision across the three arms. The other
seven writes qualified in all 96 per precision. In the DEV focused arm,
neither precision had complete phase-1 matched data for all eight
directions at either dose. Channel 2279 qualified in 0/32 focused worlds
per precision at 1%; at 2% it qualified in 1/32 NF4 and 0/32 BF16.
These are explicit dose-alignment failures under the 3% L2 tolerance.

For DEV focused phase 2 at 2%, the mean absolute symmetric half-shift
in the bare-label margin was **0.0645 vs 0.0977 logits** for IO18 writes
versus orthonormal controls on the two complete NF4 roots. On the eight
complete BF16 roots, it was **0.0659 vs 0.0737 logits**. Both sides of
each selected direction passed the requested post-rounding dose. The
incomplete roots remain in the published denominators. The calculation
changes the archived residual and replays norm plus head, with no
continuation or future-correctness counterfactual.

## What PercInterp adds here

A three-stage graph using real fitted current-write magnitudes and signed
output response had **exactly the same backbone and local-degree membership
at every one of 1,890 cutoffs across 193 fully qualified graphs**. There were
990 nonempty cutoffs. With one interior stage, this agreement is a property
of the topology. A computation through successive loop occurrences requires
measured signed edges between those occurrences. Use matched-budget
selector comparisons after those edges exist.
[PercInterp rank and bottleneck definition](https://github.com/MoralityLabAI/percolation-interp/blob/282184519f7bbd05a81ae76c3faa5d655945ee04/src/percinterp/measure.py) ·
[BPD and PercInterp bridge](NAN_PERCINTERP_BPD_BRIDGE.md)

The exposed retrospective roster contains 24 roots shared across precision,
worlds and passes. The audit used **101.1 MiB** maximum sampled private RAM
and retained at least **7.32 GiB** system available. Zero new neural
forwards and zero formal tests. A delayed behavioral effect and sham floor
remain to be measured. The original ARG-03 test stays sealed.
[Sanitized aggregate](nan_readpath37_results.json) ·
[Frozen protocol](../../experiments/nan_readpath37/PROTOCOL.md)
