# NAN-ATTAIN101 CAL-selected BF16 carrier dose

Use the archived COMP14 focused CAL and exposed DEV graphs and the
same native donor input and audited IO18 factors as BANKGAP100.
Select one carrier rule per precision from CAL geometry. The rule
uses native states and physical directions with no answer or head
inputs.

Round the positive and negative normalized SVD writes along each
fixed seed radius from 0.5% through 4%. Use the mean of the two
reached lengths as the shared target within that world. Preserve
both signs, require each reached anchor to clear 0.98 signed cosine
and require their norm spread to fit within 3% of the shared target.
The reached target must lie between 0.4% and 4.2% of the original
residual. Keep effective doses and both signs in the record.

Normalize the full T1 and T2 directions to that shared radius. The
gain-one pair retains the original directions. For higher gains,
amplify the T2-minus-T1 chord while keeping the pair on the same
sphere. The common midpoint direction is fixed and its radius
shrinks as the chord grows. Keep the gain ladder 1, 2, 4, 8, 16 and
32. A degenerate pair or chord that reaches a diameter is refused.
Record the midpoint displacement from amplification.

Use BANKGAP100 joint BF16 rounding centered on the shared SVD
carrier plus the fit offsets. Compare to independent nearest
rounding as a geometry diagnostic. Preserve the individual
0.98 signed cosine and 3% absolute norm gates, the 3% spread across
the four edits of a world, and the 0.98 cosine and 3% norm gates
on the actual reached T2-minus-T1 vector for each sign. Retain
exact aliases and differential-stencil refusals.

On the eight CAL roots, require at least six faithful both-world
roots. Choose the smallest eligible gain, then the smallest median
reached radius among the eligible rules at that gain, then the
lowest seed radius. Record the best observed CAL unamplified
coverage even if it falls below the gate. Freeze this choice on
disk before replaying only the locked rule on all 16 exposed DEV
roots of its precision. A missing rule closes with no DEV scores.
Both signs and both worlds jointly determine a root vote.

This archive calibration measures attainable geometry. It opens
no new confirmation roster and uses zero formal comparisons and
no pretrained forwards. An amplified chord supplies a separate
intervention from the original full-fit contrast. Fresh-root
native full-head and passed sham controls are required to
interpret a delayed effect. Preserve the sealed ARG-03 test.
