# NAN-CHORDSHAM106 exact-midpoint BF16 swap sham

Use the frozen COMMONHEAD102 clean-centered endpoints on the roots
qualified by ATTAIN101, including every parent signed pair and its
head refusals. The parent centers the intended T1/T2 chord on the
clean residual, then rounds both endpoints. This experiment uses
those attained asymmetric endpoints and their actual midpoint.
The estimand is a change in orientation of the attained chord.
The previous per-endpoint symmetric-target gate remains in its
original experiment.

For each coordinate leave the two BF16 endpoints in place or
exchange them. This preserves their unordered coordinate pair,
the midpoint and the absolute coordinatewise chord exactly.
It also preserves the sum of squared endpoint displacements
from the clean residual. The individual endpoint radii can
change. Require both radii to match their active counterparts
within 3% and the absolute active/sham chord cosine to be at
most 0.10. Search 256 fixed seeded sign assignments on nonzero
chord coordinates. Select by smallest worst endpoint relative
radius error, then smallest absolute cosine, then first trial.
Selection sees physical states and source identity only.

If a coordinate contains fraction p of the total squared chord
length, the absolute cosine of every coordinate-swap sham is at
least max(0, 2p-1). Use the largest p to certify dominant-coordinate
refusals. A failed seed search with no such certificate remains
a search refusal.

For the selected candidate evaluate the exact CPU vocabulary head
on both original and both sham endpoints. Require both pairs to
preserve the clean unrestricted greedy token, 0.125-logit maximum
absolute full-vocabulary drift and 0.125-logit paired-label margin
drift at each endpoint. Match the vocabulary RMS of the signed
half difference and also the vocabulary RMS of the shared average
shift, with an absolute tolerance of 0.005 logits for each. Report
the signed paired-label half-difference gap descriptively. Equal
RMS values do not imply equal tokenwise effects.

Report signed-pair counts and both-world, both-sign root coverage.
The frozen destination is an exposed development replay, with no
head-based sham selection, decoder forwards, formal tests or
confirmation access. A native follow-up would apply the same
paired interventions with a separately calibrated future-effect
floor. Keep the original ARG-03 test sealed.
