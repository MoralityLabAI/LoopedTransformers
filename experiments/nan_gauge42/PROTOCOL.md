# GAUGE42 counterfactual transport and component-coordinate sensitivity

Use the audited COMP14 A/B twins at the last prompt token on physical
layer 21, before and after the loop boundary. Cover all 24 pointer roots,
both precisions, and focused, literal-copy and graph-erased arms. For the
frozen IO18 rank-eight bank write the component approximation as
`x V U`, where the rows of U are the physical writes. Take differences
within every A/B root-arm pair on each pass. The sign is always world A
minus world B. Read all eight fitted components. DELAY39's admissible
physical component doses remain a separate experiment.

Map the two pass-specific differences into an orthonormal basis for the
shared physical write span using the Gram matrix `G = U U^T`. Fit a
linear pass-1-to-pass-2 map on the eight focused CAL roots only. The
objective is squared error in physical output coordinates with ridge
penalty `0.1 tr(C1^T C1)/8` and zero intercept. Controls predict no
change, carry the pass-1 write forward unchanged, or fit with a fixed
cyclic shift of the CAL target roots. The held-out prediction target is
the actual pass-2 fitted BPD write difference. Score physical error and
a separate direct head tangent. Erased twins must have zero source and
target differences. Copy is a separate evaluation arm.

Compare the unaltered native bare-label margin swing between the A/B
counterfactuals at each pass against the BPD direct tangent readout.
The margin always uses that root-arm's A label minus B label.
Differentiate the continuous precision RMSNorm and frozen head margin
at the A/B midpoint at each physical pass. Check the derivative against
a two-sided direct finite difference on the same continuous head.
Separately reproduce native archived head logits at both boundaries.
The derivative omits rounding and measures a local head response.
The fitted relation is an observational surrogate for a cross-pass
Jacobian. Causal cross-pass effects require perturbed native trajectories
with the same residual, KV and position state.

Build source → pass-1 component → pass-2 component → margin graphs using
the signed pass-1 physical coefficient differences, the calibrated
transport map in unit-write coordinates and the pass-2 head derivative
along the corresponding unit write. Reuse the PercInterp per-stage
rank cutoff and signed path product summary. Compare the registered
four cutoffs in the original bank, an equivalent reciprocal
scale/sign/permutation, its orthonormal reparameterization and 16
seeded orthogonal rotations of the orthonormal representation.
Assert the full operator action, fitted physical prediction and full
signed graph sum agree numerically across gauges. The reciprocal gauge
must also preserve the pruned graph counts and signed sums. Rotations mix
component identities, so compare physical projected predictions and
truncated scalar path sums, rather than matching component numbers.
Report range and count of changed sums over the registered bases and
separate path versus local in/out selector coverage.

This stresses what can be inferred from equality of the whole factor
operator. Reparameterization does not preserve a factorization's routing
gates, origin, regularizer or its trained component identity. Use a frozen
factorization, matched physical interventions and evidence for stability
across refits when claiming component-specific pathways. Gauge closure is
an algebraic check. Predictor errors and graph variation on reused roots
are descriptive development evidence. There are zero modeled interventions,
zero formal tests and zero new model forwards. The resource lease uses
5 GiB start, 4 GiB reserve, 0.75 GiB private, one thread and Below Normal.
The original ARG-03 test is outside this audit.
