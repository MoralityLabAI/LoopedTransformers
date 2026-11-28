# MOLT-13: exact carry-cut quotient for the typed operation clock

Development continuation of MOLT-12. Both addressed teachers passed competence.
The [corrected native BPD workflow](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45920276560)
passed under unchanged limits: 16 fitted components, 8 behavioral steps,
clean teacher parity zero before and after fit, relative physical-weight
residual 0.447411537, final reconstruction KL 0.0002685641, mean active
components 15.947917. Null calibration was disabled. This is an engineering
factor fit, not a validated sparse mechanism.

## Exact suffix quotient

At the post-block cut of stage r let the full native answer residual be h,
unscaled physical MLP write displacement be d, residual multiplier beta,
and q the actual final-normalized unembedding. With legal logits selected
by L, the carried register is

    p(d) = softmax(L q(h+beta d)) in Delta^16.

Given the frozen parameters and public future operands, future stages read
only p(d). Define F to stack all centered, unrestricted-vocabulary logits
strictly after the cut. Then

    F(d) = G(p(d)),
    D F = D G E E^T D p,

where E is an orthonormal 17-by-16 chart for the zero-sum tangent space.
Since 1^T p(d)=1, 1^T Dp=0, hence E E^T Dp=Dp. This gives exact rank <=16
for the future observer. The quotient is exact for fixed future operands
and checkpoint: equal carries produce equal full future trajectories by
induction over the typed stage map. It does not include the current
unrestricted vocabulary readout, which can distinguish directions that
the legal carry discards. No general native TRM or attention quotient
follows without its own state-boundary proof.

The first 16 carry entries plus the constraint reconstruct the 17th in real
arithmetic. The program checks floating-point carry, complete future-logit
and argmax parity against the ordinary patched trajectory. Exact carry
replay trades stored targets for frozen suffix computation. Store the
teacher, operands, frame and binding as required; per-record byte counts
do not establish reduced total RAM or speedup. The cut's own readout, if
required by a response loss, needs separate storage.

## Development panel

Use only the predeclared addressed seed 157 and fitted native factors from
the corrected MOLT-12 contract. The actual native dense shared block
runs the whole clock, including the residual beta. For eight fresh worlds
each at affine depths 4 and 8, edit the public addition at primitive
visit four by +1 modulo 17, leaving prefix and suffix operands identical.
The independent host computes the donor and anchor traces. Use an additive
write at the anchor's answer position from the fitted V,U factors. This is
a source-related edit, but a single-site swap need not reproduce the whole
operand edit. Report anchor and donor correctness, patch terminal and
suffix correctness, and every replay error. This code does not rerun
native swap_acts API parity; it evaluates the additive fitted-factor write
in float64. No donor-specific causal superiority is claimed.

Compute independent forward-mode Jacobians for the full centered future
observer, the legal carry and the suffix function in the simplex chart.
Check chain parity and tangency. Retain singular spectra in the ambient
write chart, native U span, and cut-included observer at a fixed absolute
1e-8 numerical threshold. That threshold can suppress saturated directions;
the mathematical bound follows from the factorization, not measured rank.
Unit tests use an untrained model and synthetic factors only to check the
instrument. No protected readout or promotion. Fixed checkpoint development
worlds are the evaluation units, not individual logits or repeated probes.

The CI recreates the pinned teacher and fit, then evaluates this small
panel in a new child-free process using unchanged MOLT-12 resource caps.
The source preflight is read-only outside the worker. A hard address-space
limit is distinct from the sampled RSS guard. No hard process-tree physical
memory cap is established.
