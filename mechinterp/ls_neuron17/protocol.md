# LS-NEURON-17: independent restarts as the beam control

## Development question

LS16's width-three beam repaired 11/24 fresh initializations, versus 9/24 for its single-path greedy control. Every certified beam ancestry first crossed the numerical threshold through improving steps. This motivates isolating the contribution of retaining alternate bootstrap codebooks from later exchange among beam slots.

LS17 compares the archived dynamic beam with three independent greedy continuations from the same top-three bootstrap states. It uses the already inspected LS16 seeds 493..516 at the inherited 15-percent perturbation dose. The entire cohort, both previous arms and their trajectories were exposed before this design. Evidence class is **EXPOSED_DEVELOPMENT_ONLY**. The comparison creates no fresh confirmation cohort and cannot provide a prospective population-effect claim.

Parent commit: `57b19f8d38a624f6a47b91386e95663d324d0878`, LS16 draft PR389. Initial states, common bootstrap receipts, beam search receipts and the complete parent certificate file are copied byte-for-byte. `parent_manifest.json` pins all 73 input files and unchanged inherited source files. The experiment does not generate initializations.

## Search allocation

The inherited initial least-squares fit and bootstrap expansion supply three distinct codebooks in ascending numerical-bound order. Each codebook starts a separate branch. Across 12 rounds, every branch expands once and greedily selects the best state among its own parent and six fitted children. Branches never exchange candidates or reassign work. If two branches converge to the same codebook, both continue and consume their remaining calls.

After the final round, choose the best of the three terminal states by raw numerical bound, with stable branch order for exact ties. This uses the unchanged LS16 selection function. Its candidate fitter first applies the LS13 nonlinear least-squares step, then the LS15 conditional minimax LP; the LP state wins only for a decoded numerical-bound improvement exceeding 1e-12. The proposal family, two scale anchors, shortlist and all solver settings remain unchanged.

Each arm's opportunity accounting includes the shared bootstrap: 37 expansions, 13,098 profile slots, 223 LS calls and 222 LP calls per seed. The new arm executes 36 expansions and 216 LS/LP calls per seed. Across 24 seeds, new physical execution is 864 expansions, 305,856 profile slots, 5,184 LS calls and 5,184 LP calls. The archived beam and bootstrap consume no new fitting calls. Historical beam wall time and new multistart wall time are labeled by provenance and cannot support a concurrent timing claim.

Each LS call has a 40-function-evaluation cap, and each LP has a 1,000-iteration cap. Search uses at most three CPU workers with one numerical thread per worker. Every branch gets its full budget. Exact certificates and rollout scoring are evaluated after all 24 new searches complete.

## Outcome, analysis and exposure

The unit is the original independently seeded perturbation. The primary binary endpoint is `initial == POINT_VIOLATION and final == CERTIFIED`. Every assigned seed stays in the denominator. Initially unresolved models receive zero proven-repair outcome, with any later certificate acquisition reported separately. All exact verdicts are retained.

The sole descriptive contrast is dynamic beam minus independent multistart. The official FSI-01 bridge receives `sealed_readout=false` and `hypothesis_preregistered=false`. It reports nominal exact paired McNemar calculations and uncertainty summaries, with exposure explicitly blocking confirmation. Its per-arm intervals are marginal model-based intervals on a cohort selected for this follow-up. The known 32 transition cases also guide fitting.

The inherited scientific floor of 0.25 is a planning reference. LS16's operational threshold of at least12 beam repairs and at least6 net gains is recorded as historical context: the archived beam count11 was already known to fail the count condition before LS17. It is not a new acceptance gate. No direct beam-versus-multistart effect or matched-sham calibration exists before this run. The strict official preflight remains `REDESIGN_OR_PILOT_ONLY`; its historical LS16 contrast and universal binary support bound are explicitly scoped. Neither a null result nor matching counts establishes equivalence.

The source and analysis procedure are locked locally and committed remotely before running the new multistart cohort. This chronology prevents further implementation changes after new outcomes; it does not restore an unexposed readout. The old seed469 supplies a separate engineering smoke test. All other new-code outcome inspection waits until the source lock. No protected ARG or BPD cohort is accessed.

## Exact scope and audit

The eight-GELU scalar mod4 recurrence uses 52 integer codes in [-7,7], four positive scales and two decoded FP32 offsets. The inherited exact verifier treats those stored coefficients as rational constants. It checks the full radius-1/8 state intervals for all32 state/input cases and requires transition error at most3/32. Additive disturbance bounded by1/32 preserves the state regions, supporting induction over every finite input sequence from a valid initial region. Hardware rounding remains outside the exact guarantee.

The independent auditor reuses LS16's separately implemented decoder, local-curvature interval metric and LP primal/dual checks. It reconstructs every shortlist, verifies that each branch can select only its own candidates, checks parent lineage and per-branch monotonicity, and confirms final selection among all three terminal states. Archived beam search receipts undergo the unchanged independent LS16 selection audit. The verifier reruns the positive construction, all24 initial models, all24 archived beam endpoints and all24 new endpoints:73 exact calls. Recomputed initial and beam certificates must match the archived rational certificates exactly.

Inherited numerical evaluation uses256 sequences at T8 and T128 for each initial and final endpoint, totaling144 horizon conditions. These are repeated measurements within each seed. Descriptive outputs include per-seed certificate pairs, both new-seed lists, final numerical bounds, chosen bootstrap branch, every branch's final bound, distinct expanded codebooks, repeated expansions, path depth, all candidate-level LP outcomes, and the prior LS16 discordant seeds501 and516. These diagnostics add no hypothesis tests.

Retain full raw receipts, every exact outcome, both source locks and all hashes. A subsequent fresh study requires its own design and planning decision based on the development result.
