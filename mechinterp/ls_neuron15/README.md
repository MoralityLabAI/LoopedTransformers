# LS-NEURON-15: conditional minimax fitting

**Both development gates FAILED.** A conditional minimax LP improves many local scores but yields no net additional exact certificates over minimax-selection least squares. Parent LS14 [PR #387](https://github.com/MoralityLabAI/LoopedTransformers/pull/387). Sources and protocol were locked locally before fresh draws; this remote publication follows execution.

## Results

24 fresh paired seeds 469..492, 15% tensor-RMS perturbation, eight-GELU scalar mod4 cell, exact-real disturbance radius 1/32.

| Arm | New certificates | T8 accuracy | T128 accuracy | T128 all-prefix | Median numerical bound |
|---|---:|---:|---:|---:|---:|
| mean_ls |9/24|97.53%|92.14%|89.84%|.15422179|
| max_ls |11/24|91.81%|79.62%|73.42%|.15782561|
| max_lp |11/24|88.20%|73.16%|64.97%|.12071171|
| bound_lp |11/24|88.18%|72.18%|64.24%|.12070849|

All 24 initial models have exact point violations. All 96 final endpoints resolve: 42 certificates, 54 point violations, no unresolved cases. 38 distinct certified decoded weight sets; 12 seeds certify in at least one arm (pooled, not a method success rate). Numerical proxy threshold agrees with every final exact verdict. Primary max_lp vs max_ls: candidate-only seed473, control-only seed491, 22 ties. Secondary bound_lp vs max_lp: same 11 successful seeds, 24 ties. Both exact paired McNemar p-values and Bonferroni-adjusted p-values are 1 under the paired-null assumptions. Neither meets the locked >=12 new and >=6 net thresholds. FSI-01 PILOT_ONLY, no calibrated sham or direct prospective population power. Equal counts do not establish equivalence.

## Conditional program and controls

See protocol.md. All arms receive 12 rounds, two nonlinear scale anchors, 177 profile slots/anchor and three shortlisted codebooks/anchor (including no-op). The common LS13 nonlinear fit is followed by a second least-squares fit or a conditional LP. Fixed up/bias tensors and integer codebook; LP minimizes t subject to sigma*(r+aA+qD+o)+q*kappa<=t for both signs at each endpoint, with positive skip and down scales a,q, bounded offset o. There are 1088 inequalities in four variables. Point LP sets kappa=0; interval LP uses local GELU curvature interpolation padding. The continuous ideal family differs from the FP32 decoder. All proposals are decoded and rescored; the first fit is retained when the second fit gives no improvement. Rational verification is terminal. Equal calls are not equal FLOPs; adaptive paths can differ. This pilot has a smaller search cap than LS14.

| Arm | Profiles | First fits | Secondary calls | LS residual evaluations | LP successes | Improved secondary scores |
|---|---:|---:|---:|---:|---:|---:|
| mean_ls|101952|1752|1728|36203|0|480|
| max_ls|101952|1752|1728|35495|0|451|
| max_lp|101952|1752|1728|28456|1728|1609|
| bound_lp|101952|1752|1728|28187|1728|1653|

3456/3456 LP solves successful. Logged max primal residual 9.676e-9, absolute dual gap 3.081e-15, stationarity residual 3.095e-11. Replay checks primal constraints to 1e-6, not independently the full dual certificate or solver optimization trajectory. Each LP arm has one candidate-level numerical bound threshold crossing after LP. Local score improvement alone does not imply a new endpoint certificate.

Post-run diagnostic: among failures, zero visited candidates passed the numerical threshold in every arm. Last accepted update by round six: 14/15 mean_ls, 12/13 max_ls, 10/13 max_lp, 10/13 bound_lp. Median final update round among failures is four in each arm. This motivates testing candidate screening and nonmonotone/beam exploration rather than another local score adjustment. No adaptive repairs performed on this cohort.

## Audit and reproduction

121 exact verifier calls including positive control, 240 numerical horizon conditions, 407808 profile slots, 13920 fitted states including common first fits, 379 accepted code-edit rounds. Deterministic replay verifies proposal arrays, shortlists, candidate scores, common initial fits, decoded weights, LP primal feasibility and selected states. It reuses the same numerical score routine rather than an independent implementation; optimizer trajectories and dual vectors are not replayed.

Run in this directory with NumPy 2.3.5, SciPy 1.17.0, CPU Torch 2.10.0:
```sh
sha256sum -c lock.sha256
python search15.py
python audit15.py
```
Search requires initial/ and results/ absent. Full companion archive LS-NEURON-15-minimax-LP.zip (separately supplied, SHA256 22ecf568dd145d2b9fc1d953c2c2be08d4ce60d17ba2659c8ce3a779b543b5dd) contains all 120 model/search receipts, certificates, numerical outcome vectors, diagnostics, environment and manifest. Certificate JSON SHA256 bb07bf181d6d3005536f5db91bce6b74e9e943245e1a0e7c68679dd5bacc46cc; audit_results.json SHA256 7a192b48ebd599d9d616421a1103ded5c30dd0316e914a1153089fb0c87fd806. Remote source blobs match the executed source for search15.py, audit15.py, protocol.md, lock.sha256. No paid GPU, protected cohorts, Lean proof or hardware rounding guarantee. Exact certificates concern stored dyadic coefficients and exact GELU with beta=1/5. Point violations concern the prescribed regions and need not be reachable from initialization.
