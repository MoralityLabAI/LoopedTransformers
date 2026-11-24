# NAN-HEADSWAP107 calibrates first-pass response of exact swap shams

The Inkling Mini routing pilot pairs realized intervention dose with
reader response, rescue and final-token measurements. This stage
upgrades the Nanbeige contrast before a comparable delayed-reader
test. CHORDSHAM106 matched physical geometry on 100 exposed signed
pairs, but its sham differential RMS was lower in every case.
Goodfire's [Inkling Mini analysis](https://www.goodfire.com/research/interpreting-inkling-2-mini/)
and the [native routing pilot](https://github.com/MoralityLabAI/LoopedTransformers/pull/311)
motivate measuring the physical edit, the current output and the reader
as distinct quantities.

Use every signed CHORDSHAM106 parent pair. Keep the same exact BF16
midpoint and coordinatewise chord magnitudes, 0.10 absolute chord
cosine cap and 3% per-endpoint radius tolerance. Start from the same
256 seeded swaps. Compute a continuous RMS-normalized first-pass
head Jacobian at the midpoint using token IDs floor(i * vocab / 512)
+ 17 for i from 0 to 511. Starting from the three feasible swaps with
highest linearized head RMS, make at most 48 single-coordinate
flips per start. Each flip maximizes the increase in linearized
squared response among flips that preserve both geometry gates.
Rank distinct proposals, including the feasible seeds, by absolute
linearized differential RMS gap from the attained active pair on the
design rows. Recompute the best 16 with the rounded CPU head on
those same rows. Select by exact design differential RMS gap, then
common RMS gap, then shortlist order.

Write the selected rows and digests before streaming the full head.
On the remaining vocabulary, measure both RMS gaps. Use the full
vocabulary and paired answer-label margin for the original per-endpoint
gate. Report full-vocabulary RMS too, and require both RMS gaps to be
at most 0.005 logits on each vocabulary partition used for validation
(the full vocabulary and its non-design subset). The full-head gate
also requires the clean greedy token, peak drift and paired margin
drift within the existing 0.125-logit limits. No validation outcome
feeds selection. Compare against CHORDSHAM106 on the same roots and
report coverage in both worlds and both signs per root. A finite search
refusal does not prove infeasibility.

The root is the unit. This is calibration on already exposed model
states, with a vocabulary partition for engineering verification.
The 0.005 RMS tolerance is instrument matching, with zero formal
tests, zero pretrained decoder forwards and no calibrated native
causal floor or power claim. A delayed-reader run still needs
fresh development roots, matched post-rounding interventions and
its own reader response and rescue gates. Keep ARG-03 sealed.
