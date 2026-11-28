# MOLT-16: finite-dose legal-decision feasibility

Development follow-up to the measured MOLT-15 result. The learned observer
CE arm recovered 30/32 changed suffixes, native CE 27/32, and raw native
swap 21/32. All preserved the same-target controls. We now ask whether
a missed legal cut decision is compatible with the declared physical budget.
The cut winner and future suffix correctness are separate endpoints.

## Bias-aware LayerNorm geometry

Fix a recipient residual h, orthonormal support Q, radius R from the
raw native swap, and the donor teacher's legal winner t. Let C be the
centering projector and s(c)=sqrt(||C(h+Qc)||²/n+eps). For each rival j,
a_j is the unembedding contrast (W_t-W_j) multiplied by LayerNorm gamma,
and b_j=(W_t-W_j) eta, with eta the learned LayerNorm bias. Legal winner
requires a_j^T C(h+Qc) >= -b_j s(c). This is an exact real-arithmetic
inequality, including the learned bias.

Write z=Ch and B=CQ, and split z into its projections parallel and
orthogonal to range B. The centered displacement has norm at most
R||B||_2 and belongs to range B. Thus

    l² = (||z_perp||² + max(0,||z_parallel||-R||B||)²)/n + eps
    u² = (||z_perp||² + (||z_parallel||+R||B||)²)/n + eps

give l <= s(c) <= u on the ball. Use the smaller of (-b_j)l and
(-b_j)u for a necessary halfspace threshold and the larger for a
sufficient threshold; subtract a_j^T z and use rows a_j^T B.
The sign of the bias matters. A sufficient linear system inside the
ball implies legal winner feasibility. The necessary system relaxes it.

For A c >= b and ||c||<=R, any nonnegative lambda must satisfy

    lambda^T b <= lambda^T A c <= R ||A^T lambda||.

A positive separation gives a dose-budget obstruction for the necessary
system, hence the legal cut winner. The inequality is valid for any
lambda>=0; convergence of the numerical dual solver is unnecessary.
If no positive separation is found, impossibility is unestablished.
If ||A^T lambda||>0, the quotient lambda^T b/||A^T lambda|| lower
bounds the required radius for this *fixed* necessary system. The system
itself depends on the original tested R, so do not extrapolate that
number as a global minimum physical dose at another radius.

The implementation uses ordinary fp64, numerical SVD, normalized dual
weights, and disclosed comparison padding. A positive numerical witness
is conditional on real-arithmetic bounds. No directed rounding or
formal certificate. Use the actual nonlinear margin to verify every
primal witness. A contradictory witness and obstruction aborts.

## Fixed comparison

Use the same predeclared teacher and native fit, replay MOLT-15 training
once, and bind the learned controller artifacts to the exact checkpoint
and factors. Preserve its three fixed charts. Draw 16 new roots at each
affine depth 4 and 8, excluding prior training and development rosters
and their changed/same-target donors. Include both source-related edits.
No gold labels or future outcomes select a repair.

In each chart compare the projected raw swap, learned CE controller,
warm analytical CE solver, and the necessary/sufficient halfspace
projection candidates. For each linear system use 64 fixed sweeps of
dual coordinate projection, clip the candidate to the original radius,
and assess the actual nonlinear margin. Select best_cut_gap by largest
actual legal winner margin, then smaller dose and fixed name. This is
a per-case ensemble with additional work, not a matched-FLOP policy.
Record all individual candidate outcomes, solver calls, selected
candidate, dual vector/separation, denominator range and physical dose.
Report witnessed strict winner, conditional obstruction, or unresolved;
the three statuses exhaust the roster. Negative margin with no witness
does not establish optimization failure or infeasibility.

Evaluate actual unmodified downstream execution after each additive
physical write. Keep donor suffix accuracy and same-target preservation
separate from cut feasibility. A cut obstruction cannot establish suffix
unreachability. The evaluation is frozen teacher/controller development
only, not a new learning campaign, population claim, or native-specific
mechanism. Source roots are the units; both edits remain paired. No
protected split, paid GPU or promotion.

    python -m pytest -q tests/test_molt_carry_feasibility.py
    python -m research_gym.interp.molt_carry_feasibility \
      --clock /tmp/molt16/clock --factors /tmp/molt16/bpd \
      --bpd-checkout .native-bpd --bpd-source-receipt /tmp/molt16/bpd-source.json \
      --controllers /tmp/molt16/repair --out /tmp/molt16/feasibility

All phases use the existing child-free CPU worker caps. The RSS guard is
sampled and the worker kernel high water is recorded. The hard address
space limit is not a physical process-tree RAM certificate.
