# LS-SAE08 prospective source-admitted hard-Sudoku causal test

**Registered November 16, 2026, before LS-SAE08 source or model results.** LS-SAE07 PR #131 registered 8 fixed 54-hole grids and encountered one source grid that could reach only 53 holes without losing unique solvability. LS-SAE07's incomplete source admission and original protocol remain untouched. LS-SAE08 uses a new precommitted model-blind deterministic first-admissible grid algorithm for hard 52/54-hole Sudoku and reuses the already registered LS-SAE07 numerical test and fixed gate.

## Frozen pretrained model and source-fit boundary

Keep original gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att step_21700 source file SHA256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3** and original LoopScope02 PureSudokuTRM code at CPU float32 PyTorch 2.7.1, NumPy2.1.3, four CPU threads, math attention. The immutable original fit-only LS-SAE04 512→48 PCA and 48→96 top-four SAE NPZ is loaded from Actions run **43818817318** source-calibration-only artifact **14203533086**; require original file SHA256 **073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25**, manifest SHA256 **cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a**, fit-array SHA256 **83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec**, and physical-dictionary SHA256 **14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb**. No checkpoint or SAE/PCA updates.

## Registered model-blind source admission

Generate exact 9×9 complete solutions via the original fixed-seed randomized MRV+diagonal-box builder, retaining the positive alternating rectangle invariant. For each required puzzle slot i, use up to **32 candidates**, in ascending order j=0..31, with full-grid seed **base + 100*i + j**. Each candidate uses the original deterministic 81-cell random permutation with dig seed full-grid seed+1000000 and the original exact bounded two-solution uniqueness checker. **Accept the first candidate** whose resulting puzzle has exactly the registered 52 or 54 holes and exactly one solution, with the correct noncanonical invariant; record all skipped source grid seeds and the reason unique digging fell short. If none of the 32 candidates qualifies, fail the original slot and scientific cohort. Never choose on the model's accuracy, margins, hidden-state, SAE activation or source-direction gain. Keep the one-puzzle-per-slot denominator.

- Separate engineering-only pilot: **2** 52-hole slots at base **202628020000** and **2** 54-hole slots at base **202628021000**, excluded from full scientific gates.
- Prospective full: **8** 52-hole slots at base **202628030000** and **8** 54-hole slots at base **202628040000**.

All 20 selected complete targets and visible puzzles must be mutually distinct and source-disjoint from LS-SAE02 through LS-SAE07 fixed/pilot/full seed rosters and every calibration source set. Preserve every selected and attempted grid seed, exact uniqueness, hole count, original solution, visible puzzle and canonical JSON SHA256. Visible-only difficulty statistics use the original synchronous naked+hidden single propagation rules (number of rounds, initially naked singles, number of deduced cells, zero-round stall), and are archived before model evaluation. Source intervention site is the **median sorted visible missing-cell index**.

## Native intervention and fixed comparators

Reuse the exact LS-SAE07 native source-direction algorithm and full81 scoring with new LS-SAE08 source rosters. At ACT1 inject exactly one unit-normalized 512D z_H board token direction with every other source token, z_L, input embedding and network weights unchanged. Freeze four directions before reading true answer digits:
- STATE_POWER: four matrix-free JVP/VJP power updates on the full 97×512 hidden-state Jacobian from ACT1→ACT3;
- LOGIT_POWER: four equal-budget matrix-free JVP/VJP updates on centered all-81×9 predicted digit logits over the same ACT1→ACT3 continuation;
- SAE: frozen strongest positive top-four source SAE feature decoder direction lifted through original PCA into 512D unit input, with positive strict-top-one gap >1e−8 and source-blind abstention;
- RANDOM: independent SHA-seeded unit isotropic source perturbation.

Use the identical original LS-SAE06 algorithmic seed contracts (family strings test_52 and test_54) for direction selection and original LS-SAE07 score/geometry budgets. Do not reselect based on correctness or edit effects.

All eligible arms receive physical L2 input radii **0.20, 2.00 and 8.00**, BOTH signs ±, with source token norm and each physical dose/norm ratio recorded for OOD interpretation. Score genuine native ACT3, ACT6 and ACT12 complete **81×9 digit logits** for clean and every arm/sign/radius. Reuse the LS-SAE06 exact integer Sudoku 27-unit/row/col/box verifier, original givens consistency, unique correct solution, blank-cell correct count, changed prediction count, correct→wrong, wrong→correct, centered logit Frobenius effect, and frozen clean true-digit-versus-strongest-other-digit margin per originally blank cell. Both original four-step JtJ transcripts are saved and independently auditable.

## Frozen original scientific gate and accuracy interpretation

**Primary:** total ACT6, physical input radius **8.00**, symmetric missing-cell predicted digit flips F=½(flips under +8 + flips under −8) relative to original clean, paired D=F(STATE_POWER)−F(RANDOM). A source-blind ineligible power vector has F=0 and puzzle remains in denominator. For each 8-puzzle independent full family, resample puzzle IDs 2000 times with replacement using **random.Random(202627130+len(family))**, sorted CI indexes 50 and 1949. Family success requires at least **6 of 8 puzzles with D≥1.0** and a strictly positive lower 95% bootstrap bound; overall success requires both full 52/54-hole families. Pilot does not count. No changes to original LS-SAE07 thresholds, doses, horizons, full-test size or direction budgets after outcomes.

**Separate correctness:** report all signed blank-correct gains and incorrect→correct vs correct→incorrect transitions, exact board solves, original clue consistency, and delayed clean ACT12 recovery. The retrospective best-sign count on originally clean-inexact ACT6 puzzles is explicitly descriptive, not a preregistered success criterion. Changed predicted digits are a causal-output effect, while verified accuracy improvement must be separately demonstrated, including the possibility of net harm. Compare against equal-budget LOGIT_POWER and frozen SAE directions at identical physical doses.

**Independent stdlib-only numeric auditor:** reconstruct exact solution uniqueness, model-blind first-success candidate selection and source-disjointness, original source file/fit SHA, complete original 512D JtJ transcripts, source unit vectors, physical dose ratios, all clean and finite 81×9 Sudoku logits/ACT3/6/12, exact Sudoku validity/target digits, paired signed flips and accuracy counts, and original 2000 puzzle-cluster bootstrap and both fixed scientific gates. Replay checks saved numerical statements and does not rerun the pretrained neural model or certify interval FP32.

This new prospective source-admitted experiment never revises LS-SAE07's original failed source-admission record. It extends the previously specified hard-Sudoku hypothesis to a cohort generated by an explicitly locked model-blind first-success rule.
