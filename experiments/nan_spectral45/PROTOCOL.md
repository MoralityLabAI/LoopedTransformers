# SPECTRAL45 phase-1 full-head preservation in the BPD span

DARK44 kept Nanbeige's two-label margin at 1% residual L2 while every
candidate failed the full-vocabulary preservation gate. Here the entire
continuous head derivative informs the choice of direction. Read the
other four COMP14 focused CAL roots in the same eight-root pool,
both worlds and separately fitted NF4 and BF16 IO18 T2 banks. This
complement is disjoint from DARK44's four roots. Use the same physical
boundary, seven eligible writes, checkpoint and donor construction.

For recipient residual h and physical orthonormal write basis B, let
J be the Jacobian of the continuously normalized, frozen vocabulary
head at h. Stream the checkpoint weights to form the 7-by-7 Gram
matrix B^T J^T J B / vocabulary size. Include the radial term of
RMSNorm in the derivative. Use a six-dimensional orthonormal basis N
of the write coordinates orthogonal to the donor-minus-recipient
margin gradient. The resulting Gram matrix in this space controls
continuous uniform RMS logit sensitivity. Report its eigenvalue floor
and full rank. This is a local derivative characterization.

Compare three unit directions. The reference is the margin-null donor
from DARK44. The least-sensitive direction is the smallest-eigenvalue
unit direction of the constrained Gram. The regularized donor solves
(G + λ I) v = u before normalizing, with u the normalized reference
donor and λ the median eigenvalue. Orient the smallest-eigenvalue
direction toward the reference donor. The registered alignment gate is
cosine at least 0.8 to the original full-span donor, before sign and dose.
Report its loss separately from the current-readout preservation gates.

Run both signs at 1%, 0.5%, and 0.25% of the phase-1 residual L2. Measure
the actually reached BF16 edits with DARK44's post-rounding physical L2,
cosine and matched-dose gates. Stream the exact rounded full vocabulary
readout for clean and edited states. Score margin preservation at an
absolute difference of 0.125 logits, and full-head preservation at a
maximum difference of 0.125 logits plus unrestricted greedy agreement.
Also report finite RMS, maximum logit change and clean-to-edited KL.
For both comparison arms keep matched and missing contrasts at each
radius. The 1% dose is primary. The two smaller doses describe
engineering tradeoffs with a fixed common radius within each comparison.

The calibration establishes which measured interventions preserve the
current head at their tested dose. It uses four previously exposed
source roots, two matched precisions and zero inferential tests or
decoder forwards. A native continuation is required for a future
dark-to-visible effect. No sealed ARG-03 data are used. The lightweight
CPU lease is the same as DARK44.
