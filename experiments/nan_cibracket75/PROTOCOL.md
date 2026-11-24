# NAN-CIBRACKET-75 CAL-only BF16 dose compensation

CIDOSE74 required all seven attention edits to match a 2% relative
dose and a 0.98 cosine after BF16 rounding. The original scalar ray
search qualified zero focused CAL and DEV roots. Its focused failures
missed the dose despite passing the angle gate. CIBRACKET75 tests a
separate, fixed geometric repair on CAL only.

Keep every direction, relative dose, angular threshold and joint
dose tolerance from CIDOSE74. Preserve any eligible original ray.
For a failed ray, locate the two neighboring scalar ray endpoints
around the desired L2. The coordinates whose rounded values differ
are the dose-jump coordinates. For each endpoint, hold precisely
those rounded coordinates fixed. Independently re-scale the original
direction on the complement to fill the remaining squared L2.
Reject candidates that exceed the dose on the fixed coordinates,
violate the full-dose band, or lose the full-vector alignment gate.
Choose the feasible candidate with least dose error, breaking ties
by alignment. The compensating direction is allowed to leave the
original affine ray. Angular error and the number of fixed
coordinates are retained. The budget and controls still apply
per world, independently of answer labels and task readouts.

Both worlds and both physical passes must qualify in all seven
arms for a root to enter the local readout summary. Keep pass
and root denominators, and retain the signed local projections
and norm drift. Use the same archived attention readouts and
mean inverse residual RMS as CIDOSE74. A feasible bounded
construction disproves infeasibility for that specific state,
direction and dose band. A failed search makes no such claim.

Read the eight COMP14 CAL roots, both worlds and all three arms.
No DEV capture is used to select or check the repair. The public
CIDOSE74 aggregate records the original failure. This numerical
control calibration has zero formal tests or new pretrained
forwards. A native trial needs a separately commissioned
physical write injection and clean-reader/behavioral scoring
on fresh roots.
