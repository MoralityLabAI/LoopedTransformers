# HEADNULL60 worst-row nulling for the signed NF4 sham

LOWHEAD59 found that the lowest-RMS directions lie close to the active
RMS target but give isolated vocabulary-logit shifts above the full-head
peak gate. Reuse its pinned spectral proposal, physical support, 24-column
local basis, 0.75% residual dose, and all signed gates. Use the same two
exposed CAL panels and the same independently verified active geometry.

At each step, minimize continuous vocabulary RMS over unit coefficient
vectors orthogonal to the previously selected vocabulary gradient rows.
If the predicted maximum absolute logit shift exceeds 0.12, add the
current maximizing row to the forbidden span and repeat. Allow at most
ten added rows per world. Upon reaching the predicted peak gate, compare
17 normalized chord points back toward the peak-aware solution, retaining
the one with RMS closest to the previously measured active signed RMS
average. When no point is feasible, use the candidate with the smallest
predicted peak across the search and peak-aware baseline. Ties follow the
fixed iteration order. Direction choice uses the continuous derivative.
The actual rounded logits are only used in the independent readout.

Report every world and all failures at the original physical and head gates.
Report absolute RMS gap at most 0.005 in both signs as a separate criterion
among full-head passes. Keep private the selected vocabulary row identities.
Independent units are the eight earlier CAL roots. The result is a
descriptive first-pass engineering audit, with zero new decoder calls,
formal inference, DEV state access, or sealed ARG-03 access.
