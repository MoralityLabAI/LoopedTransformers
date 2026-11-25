# COL-OBS-02: prospective neural discrimination gate

Status: design for a later run. COL-OBS-01's complete finite census is public
exploratory evidence. This document registers the questions and required
controls; a model-specific sealed manifest follows development calibration.

## Question

Can a bounded recurrent state support basin decisions on previously unseen
parameter families while retaining information beyond the conserved quotient?
The exact prefix decoder in COL-OBS-01 makes seed memorization a known competing
explanation. The next task must expose that explanation in its controls.

Use pairs `(d,n,m)` and ask whether n and m reach the same cycle. Basin minima
are parameter-specific identifiers, so a binary relation gives a common
output across parameters. Construct an equal mix of positive and negative
pairs within one shared quotient. Keep trivial identity pairs separate.
Report scores by the earliest exact coalescence depth and by cycle length.

## Information budget

The model receives d and two initial seeds in fixed-width binary form. An
external transition oracle, when used, charges each emitted `(u,a)` pair.
Report this cost separately from recurrent block visits and from training
compute. Freeze any allowed quotient feature before fitting.

Short-prefix tasks use at most four observations. Require
`2^(2r+1) <= d-1` for the chosen horizon r, so the universal finite-domain
injectivity guarantee does not make every task a seed code. Audit the actual
prefix partition for each parameter; parameters whose observations already
identify all states belong in a separate capacity-control stratum.

## Controls and mechanism endpoint

Fit a tied recurrent model and report both a parameter-matched untied model
and a compute-matched feedforward baseline. An untrained recurrent trunk with
the same fitted readout measures random-feature capacity. Keep the exact
quotient baseline, the affine seed-recovery algorithm, and full graph lookup
as separately costed structural references.

A fitted probe may recover q without causing correct basin decisions. For a
mechanistic claim, first identify a candidate subspace using development
parameters only. Then swap or ablate it at a fixed loop depth and fixed norm.
Match donors on q and input scale, and include within-basin donors,
between-basin donors, a shuffled donor assignment, and random subspaces with
the same rank and intervention norm. Record every paired task outcome.

The primary causal effect is the change in correct same-basin decisions under
the specified intervention relative to its matched sham. The scientific
minimum is five percentage points at the parameter-macro level. The operational
floor is the larger of that minimum, instrument resolution, and the calibrated
matched-sham envelope.

## Source separation and stopping gates

All `d <= 1025` parameters and `d=2021` are exposed development sources.
Future confirmation selects entirely new parameters and records their prime
factorization families as clusters where appropriate. All seeds, pairs,
loop depths, candidate interventions, and checkpoints sharing one parameter
remain correlated observations on that parameter. Give each root equal
weight in the primary estimate.

Obtain at least 59 independent development sham roots before using their
maximum as a 95th-percentile envelope at 95% confidence. Run the repository's
mechanistic-power preflight on parameter-level pilot differences. Freeze the
resulting root count, complete comparison family, model/checkpoint hashes,
input projection, scientific floor, and sealed readout manifest before
confirmation. Calibration may require redesign. This stage has no approved
GPU allocation or trained checkpoint.

Count all screened features and intervention directions in the comparison
family. Apply FSI-01 to the registered paired endpoint with its actual cluster
structure. Preserve failures and raw root outcomes. The confirmation gate
requires a causal effect beyond the operational floor on fresh roots; probe
accuracy alone supplies an observational result.

## Minimum handoff

A run must publish the frozen manifest hash, unit and cluster definitions,
parameter roster, every model seed, allowed inputs, all candidate/control
outputs, exact solver cost, sham calibration, planned and actual comparisons,
power calculation, confidence procedure, selection history, and verdict.
The existing exact-domain receipt remains a separate evidence class.
