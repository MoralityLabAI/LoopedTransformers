# LGA-07 / LGA-07B: EvalMath resolution check before new native data

**Date: 18 November 2026. Evidence class: pre-data planning using already exposed LGA-06 results.** Neither reserved source seed 31059 nor 31060 was generated to prepare this assessment.

The frozen 44K TRM teachers' original four-pass IID token accuracy averaged **98.62%** in the completed LGA-06 test (source 31058). LGA-07's registered 64-new-IID-base × 8-goal design has 512 goal rows/teacher. Under representative IID sampling, its *expected* number of originally incorrect goal tokens is 512×(1−0.9862)=**7.07 per teacher**. Every set of eight goals shares a base sequence, and the models are only **three** previously inspected frozen checkpoints. Hence the original wrong-to-right improvement rate has a low finite denominator, its root-level uncertainty cannot be resolved by reporting 1,536 nominal goal rows, and original training-horizon correction is marked **REDESIGN_OR_PILOT_ONLY**. This is a design/power inference, not a new model result.

LGA-06's four-pass Block9 accuracy averaged 56.60%, leaving greater correction headroom but weak native solver competence. Its 12-pass IID accuracy averaged 10.21%, making the extrapolated endpoint primarily a breakdown stress. Both remain descriptive diagnostics without native mechanism certification.

**LGA-07B solution being tested:** source-legal donor counterpart changes a single predecessor digit, which changes exactly one donor target at the selected goal. Recipient/donor clean four-pass behavior should be competent on most IID cases, while the intervention now has a substantive alternative target to transfer. This uses fixed eight goal sites and preserves 64 independent base IDs as the cluster unit. Donor-target transfer is an independently defined behavioral endpoint, unlike maximizing the original model's own chosen logit. The proposed label-blind edit projects the physically reachable donor-state difference into the *recipient's first-order energy/head-nullspace*, with eight equal-L2 random controls drawn within precisely the same K. It is still an **off-trajectory hybrid intervention** and does not prove causal semantics for arbitrary pretrained language models.

Strict EvalMath status **DEVELOPMENT_ONLY** for both 3-teacher protocols. The new LGA-07B primary pilot must be calibrated on separate model-root shams before any claim of independent confirmation. The science minimum of **+15 percentage points** donor-target recovery versus matched shams is a preregistered engineering floor. Model-root bootstrap and the future 59 sham/32 confirmation requirements remain separate; Stage-B frozen original teachers can support no prospective root-level significance claim.

Required handoff:
- **unit:** original 64 independent IID base strings per frozen teacher for descriptive clustering; three exposed teacher checkpoints
- **n_units:** 3 exposed teachers; 64 new source clusters/teacher when stage B eventually runs
- **n_cases:** 512 donor goal pairs/teacher, correlated eight per source
- **primary contrast:** correct donor-target rate, projected donor vs median eight equal-dose tangent/null shams
- **physical dose:** 0.25× current recipient native single-goal update L2 (full donor-token control larger, separate)
- **scientific variation floor:** +0.15 absolute transfer rate for engineering candidate, scientific floor requiring independent-model calibration PENDING
- **matched-sham floor with confidence/source:** PENDING, requiring new independent calibration roots; none estimated from the already exposed teacher examples
- **planned comparisons:** one k2 original-horizon primary; k6→12, inverse-sign, unprojected donor and full donor controls exploratory
- **resolution/power:** root-level confirmation BLOCKED, only three reused models and no 59 independent calibration worlds
- **readout selection exposure:** old trained teachers/potentials exposed by LGA-05/06; new legal donor source seed 31060 reserved
- **causal controls:** source-reachable donor, same-K sham, same dose and actual finite head/E change, negative sign and full-donor positive reference
- **verdict:** REDESIGN_OR_PILOT_ONLY, LGA-07B separately preregistered for new development
- **provenance:** [LOOPSCOPE-09 RESULTS](../loopscope09/RESULTS.md); [DONOR_PROTOCOL](DONOR_PROTOCOL.md); original frozen [LOOPSCOPE-08 RESULTS](../loopscope08/RESULTS.md)
