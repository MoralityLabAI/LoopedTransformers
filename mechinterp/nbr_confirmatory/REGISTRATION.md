# NBR-01/01B confirmatory registration

25 November 2026. This registration precedes any checkpoint access or
model inference on the 144 candidate boards. It is a separate confirmatory
campaign, not a reclassification of the six-board pilot. Engineering pilot
and fixed-basis reanalysis results remain in PRs #351 and #354.

## Frozen source

Source-only PR #352, repaired commit 3e80bcb8f13ce5a7c3be94ec78d32a965615730b,
Actions 45518646045 artifact 14979392586, name nbr-candidate-source.
roster.json SHA256 763ea63c8565245928a7dd59f7b56b7916fad6e2f8d3126075a801c56a36f224.
48 cal, 48 eval, 48 rep, each exactly 24 singles-solvable 52-hole
and 24 singles-stalled 56-hole unique puzzles. Fixed median blank site.
Source generation admitted 144 after 29 failed digs and 89 difficulty
rejections, no collisions. Two source solvers agreed. Separately rechecked
all 144 with an independently written bitset exact-cover search,
sequential naked/hidden singles, validity and symmetry invariants.
All passed; stalled remaining-blank ranges cal 13..53, eval 13..49,
rep 16..53. The named historical comparison covers 573 records, including
the known failed LS-SAE07 53-hole attempted puzzle and all six prior NBR
pilot boards. Additional independent comparison of all 20 LS-SAE08
admitted source records, including four pilot boards, found no exact
input/solution or invariant collision (Actions 43851165980 source-only
artifact 14218016060, source_only.json SHA256
1e90b2bb541291f34fb760cd0e02a74f6fd522cdb74bc0436454dc57334a3cc9).
Reviewed LS-SAE09/10/13 provenance: prior rosters reused. COE-07 reuses
LoopScope02; LCD-06 native and ROC-TRM-01 were still pending in their
published status; the LOOPSCOPE04 44K experiment uses cyclic-copy
sequences, not Sudoku. Disjointness claim is limited to the 577 named
historical records; other unpublished or unenumerated experiments and
training-data disjointness are not certified. This explicit source-scope
qualification is fixed before model access.

## Model and inference

Use the existing unchanged native.py and nbr_fixed_basis/control.py from
parent fa3ab6fd27b5ee1d1324996f92ecb7f09da934e4. Gaoxin492 checkpoint step
21700 SHA256 932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3.
Python 3.11, torch 2.7.1 CPU FP32, NumPy 2.1.3, four threads, math SDPA,
batch one. No tuning. ACT3/6/12 native local transported Jacobians at
the final L/H updates of the final H cycle. Freeze all 144 records; no
post-inference exclusions or replacements. Abort and record INCOMPLETE
on a parity, checkpoint, nonfinite or receipt failure. All model calls
are within the fixed native runner. Fixed-direction reanalysis loads
checkpoint weights but uses saved Jacobians, no further board inference.

## Primary gates

NBR-01 generic stability: ACT3->6 final L role, physical layer 0, raw
same-ID cosine minus norm-stratified shuffled-ID mean. For each of eval
and rep require >=40 valid boards of 48, mean advantage >=0.10 and
strictly positive lower 95% board bootstrap bound (2000 resamples, NumPy
default_rng seed 20261125). Both families must pass. This tests direction
reuse relative to different neuron IDs, not neuron-specific semantics.

NBR-01B specificity: same ACT, role and layer, residualized cosine
difference native minus fixed covariance-matched Gaussian mixtures.
Eight draws x 256 fixed mixtures, seed 202611251031; same coefficients
across boards and ACTs. Remove each direction's own static head projection.
For each of eval and rep require >=40 valid boards of 48, mean advantage
>=0.05 and strictly positive lower 95% board bootstrap bound, same bootstrap
rule. Both families must pass. This gate tests native-basis specificity
beyond generic shared write geometry; it does not prove a causal primitive.
Hierarchical scientific interpretation: only claim both generic stability
and basis specificity if BOTH gates pass; a raw pass and specificity fail
supports stability but not privileged neuron basis. Report all signed
contrasts and both gates, including failure and missingness. No threshold
change after first inference.

Preassigned singles/stalled strata are separately summarized without
additional gates. ACT6->12, H-role, later layer, static alignment, clean
accuracy, native detector AUROCs and coverage are diagnostics only.
The six-board pilot had large shuffled-ID advantages but fixed-mixture
advantages slightly below zero. It motivated the control; it is not
part of the registered denominator. NBR-02 through 08 remain exploratory
until separately registered, with calibration sealing, appropriate
active-state coverage, multiple comparisons and hardware replay as
specified in their continuation ledger.

## Execution boundary

Run the frozen source artifact hash check and independent admission check
before checkpoint loading. One confirmatory workflow on this branch; no
automatic re-run for scientific failures or changes of threshold. Any
engineering rerun requires documented reason, preserves prior receipts,
and is identified by run ID/attempt. Publish original NPZ/report receipts
and gate summary as artifact. No main merge.
