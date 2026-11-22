# BPD and PercInterp on Nanbeige’s tied loop layers

The next causal comparison should use a source family where Nanbeige
already solves the clean task at two passes on both precisions. COMP14
pointer development is the current starting point. The storyworld
development floor makes story trajectories useful for checking the
readout instrument and phase geometry. READOUT34 measures where that
floor changes across shorter contexts, output formats and physical depth.
Use fresh pointer roots for CAL and any later confirmation.
[Pointer competence](NAN_COMP14_RESULTS.md) · [Story audit](NAN_STORYAUDIT30_SUMMARY.md) ·
[READOUT34 protocol](../../experiments/nan_readout34/RUNTIME.md)

## What to represent

Represent the tied physical site as one operator
`W = R + sum_i w_i r_i^T`. Use distinct nodes for occurrences of each
component, keyed by physical layer, module, component identifier, loop
pass and declared token position. Store the shared factor and target
digest separately from the occurrence key. Record the actual reached
coefficient `r_i^T h`, gate, and physical write at each occurrence.
Factor sign, scale and assignment can change across independent fits.
Compare the physical operator and read/write projectors across fits,
and bind component identities within each fit. Qualify whole-target
identity, all-one native closure and parameter residual before looking
at subsets. [Learned VPD/BPD comparison](NAN_LEARNED_VPD_BPD_COMPARISON.md) ·
[Goodfire VPD methods](https://arxiv.org/abs/2610.20748)

The historical T2 measurements gave the static CAL mask as much exact
DEV correctness as the adaptive alternatives. The next fit comparison
needs shared train/CAL roots, physical site and effective rank, the
same optimizer and budget, and explicit constant/static and
input-conditioned routing. Report one- versus two-pass readouts as
a separate experimental axis. Their current recipe difference includes
fit depth and cause of ablation. Use matched reconstructed output,
full-vocabulary drift, future verified correctness and control doses.
[IO18 learned routing](NAN_FIT16_IO18_RESULTS.md) · [Factor and SAE geometry](NAN_XREF25_RESULTS.md)

## How to measure edges

A phase marker can affect the geometry of the residual without carrying
the world distinction. For a clean, fixed prefix, intervene on a phase-1
component write at a registered physical site. Measure a signed downstream
change in a named phase-2 read or verified donor-versus-recipient logit
margin, with positive and negative finite steps and a checking point
for local linearity. Freeze the sites, horizons, dose and endpoints on
CAL. Keep direct readout, rolling observed state, and the delayed
response separate. Require current full-readout preservation for claims
about readout-dark changes and measure preservation on the actual
finite, rounded intervention. A local Jacobian null gives first-order
information. The true continuation needs the relevant residual, KV
state, positions and attention mask. Use finite changes and matrix-free
JVP/VJP over the registered horizon where memory admits it.
[TRM observer calculus and state controls](../../mechinterp/trm_roc01/PROTOCOL.md) ·
[Inkling cross-loop read paths and lenses](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

Compare a task-matched donor, self, zero, main static, and randomized
intervention with the same post-rounding physical L2 displacement,
at the same site, pass and token support. Retain both donor-minus-clean
and donor-minus-sham, then count repairs and regressions under the
independent exact oracle. A destructive sham can make donor-minus-sham
positive even when donor harms correctness. Include held-out CAL audit
of phase/position-matched coefficient permutations to separate pathway
selection from activity and uniform bias. An unchanged readout after a
small perturbation certifies the observed check only.
[BPD TRM dose-matched pilot](https://github.com/MoralityLabAI/BPD/blob/8f658b789138d73f497b1a2a05f5783745147d40/mechinterp/FSI04_NATIVE_PILOT.md)

## Where PercInterp enters

Export a directed acyclic graph on the **unrolled occurrences** using the
verified earlier-to-later edge measurements. Preserve signs and units.
Rank edge magnitudes within registered comparable module/phase groups.
PercInterp’s forward/backward bottleneck entries identify nodes that
lie on a retained source-to-readout path at each cutoff. Retain signed
effect data alongside the graph, since equal magnitudes can hide
cancellation. Apply the phase/module-preserving degree and strength
nulls, and report the separate coverage and input/output alignment
terms. Select on CAL, freeze identical alive-component budgets per
module and phase, then compare the sieve against local degree, weighted
degree, mean CI, absolute attribution and activity. Check exact
membership equality before launching paired GPU runs. Equal
memberships give the same intervention and can be verified without
finishing both. The calibration and confirmation roots, prospective
effect floor, source custody, and complete paired error counts determine
the eventual causal claim. [PercInterp E09 utility audit](https://github.com/MoralityLabAI/percolation-interp/blob/ae871eac4a95d787b980dad0abd5a60d9140b4e4/docs/e09_utility_decision.md) ·
[Equal-budget E10 comparison](https://github.com/MoralityLabAI/percolation-interp/blob/ae871eac4a95d787b980dad0abd5a60d9140b4e4/docs/e10_decisive_gate.md)

Use the Inkling same-checkpoint depth comparison and the TRM
current-versus-future observer design together. Separately compare
phase-conditioned lens fidelity to the final output and oracle-scored
correctness. The first native Nanbeige lens comparison can be done on
pinned phase boundary states. A tuned lens must fit CAL only, and the
intermediate and final readouts require explicit closure checks. Deep
repairs, regressions and compute are separate endpoints. No selected
causal graph, dose-matched donor outcome or confirmation result is
available at this milestone. The original 168-case test remains sealed.
