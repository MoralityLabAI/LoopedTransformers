# LS-SAE07 prospective harder-Sudoku causal task transfer

**Registered 2026-11-16 before any LS-SAE07 hidden-state/test measurements.** The source-frozen LS-SAE06 experiment measured continuous all-81 digit-logit effects but no digit flips on 16 fully solved 30/34-hole puzzles, failing both registered family gates. LS-SAE07 fixes harder independently generated unique-solution Sudoku cohorts, stronger finite source edit doses and ACT12 delayed-recovery scoring. Original LS-SAE06 outcomes remain unchanged.

## Source, solution constraints and admission

Frozen pretrained model: original gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att step_21700 file SHA256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**, original LoopScope02 PureSudokuTRM, FP32 PyTorch 2.7.1, NumPy 2.1.3, four CPU threads, math SDPA, no model finetuning. Retain the original preheldout LS-SAE04 fit-only PCA512→48/SAE48→96 source artifact, Actions #43818817318 artifact #14203533086, NPZ byte SHA256 **073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25**, manifest SHA256 **cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a**, fit-array SHA256 **83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec**, physical dictionary SHA256 **14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb**. Any source SHA mismatch aborts.

New complete Sudoku solutions use the source randomized MRV+independent diagonal boxes and positive alternating 2x2 intercalate invariant. Dig exactly 52 or 54 holes using source fixed random permutation under random.Random(seed+1000000), and the original exact bounded two-solution uniqueness checker. If a registered full solution cannot reach its exact requested number of holes while preserving uniqueness, fail that source cohort and do not substitute puzzle seeds or change the denominator.

- Engineering pilot only, excluded from scientific gate: **2** 52-hole puzzles with full-grid seeds **202627020000..202627020001**; **2** 54-hole puzzles with seeds **202627021000..202627021001**.
- Prospective full heldout: **8** 52-hole solutions at **202627030000..202627030007** and **8** 54-hole at **202627040000..202627040007**.

All 20 registered complete solutions and visible inputs must be mutually distinct and source-disjoint from LS-SAE02,03,04,05,06 fit, pilot and test rosters. Archive canonical JSON SHA fingerprints. Record exact visible-only structural difficulty via existing Sudoku propagation trace (rounds, numbers solved by singles, whether stalled), but never select on model correctness, logits, clean margins, SAE feature scores or hindsight. One source intervention site per puzzle is the **median missing-cell index in sorted visible blank mask**, fixed before neural inference.

## Label-blind four-arm physical source controls

Capture frozen z_H and z_L after ACT1. Intervene on exactly one physical 512-dimensional board token in z_H, preserving the other 96 tokens, full z_L, original input embedding and original checkpoint weights. Source direction selection uses real native F2 continuation to ACT3 (two further ACT steps; 42 original L-level calls).

Reuse the exact existing LS-SAE06 frozen source-direction algorithm, with both family strings changed to test_52 or test_54:
1. STATE_POWER: four 512D forward-JVP/reverse-VJP JtJ iterations targeting the full 97x512 native hidden-output norm, original fixed LS-SAE05 source-seed contract.
2. LOGIT_POWER: four *equal-budget* JVP/VJP JtJ updates targeting the centered predicted all-81x9 native digit logits, original LS-SAE06 separate seed contract.
3. SAE: original source-fitted strongest strictly positive (gap >1e-8) top-four SAE decoder direction, projected through sealed 512x48 PCA and unit-normalized. No post-edit SAE reencoding gate.
4. RANDOM: independent SHA-seeded isotropic 512D unit source direction, original LS-SAE06 random-final contract.

All directions are frozen before reading the correct Sudoku solution; preserve full state/logit JVP norm and four-step JtJ numerical transcripts and the cosine between power directions.

## Fixed finite interventions and complete Sudoku evidence

For every eligible arm, run physical L2 source edit radii **0.20, 2.00 and 8.00**, for BOTH signs plus and minus, applied to the single selected source token after ACT1. Record physical radius divided by the source token's original L2 norm to expose possible OOD edits. Do not clip radius adaptively, tune on outputs or select a favorable sign. Recompute the clean and all intervention trajectories to total ACT horizons **ACT3, ACT6, ACT12** (2,5,11 additional ACT steps). Archive complete clean/edited **81x9 digit logits** for every arm, sign, dose and horizon.

Use the existing LS-SAE06 exact all-81 Sudoku scorer for original clean and edited full-board digit argmax, originally blank correct count, correct-to-wrong and wrong-to-correct transitions, changed missing cell and total all81 predictions, given-clue violations, 27-unit Sudoku repeated-digit conflict counts, valid complete solution, frozen clean strongest nontruth rival task margins and centered-logit Frobenius response. The registered original uniquely solved Sudoku target defines correct answer, not any model-generated solution. Report original clean inexact counts, correction on ACT12 and the exact worst margin gap.

## Frozen primary endpoint and secondary correctness

At **ACT6, physical radius 8.00**, define symmetric missing-cell digit prediction flips F(u)=[flips(plus r)+flips(minus r)]/2 against *the original clean* digit predictions. The paired per-puzzle effect is D=F(STATE_POWER)-F(RANDOM), with ineligible state power counted as zero. Each fresh 8-puzzle family uses exactly 2,000 bootstrap resamples over puzzle IDs, RNG random.Random(202627130+len(family)), 95% percentile indices 50 and 1949. Per-family PASS requires **at least 6 of 8 puzzles with D>=1.0** and strictly positive lower bootstrap CI for mean D. Both independent families required for overall PASS. No changes after test to thresholds, seeds, horizons or physical radius.

Separately record symmetric blank-cell correctness gain C(u)=[correct(plus)+correct(minus)]/2-correct(clean) and verified full-board exact solves. Prespecify a **secondary diagnostic count**, over puzzles with at least one clean incorrect blank cell at ACT6, of cases where *at least one* signed STATE_POWER edit increases blank-correct count and remains completely clue-consistent; it is a descriptive best-sign observation and cannot alter the primary success gate. Publish negative cases, wrong-to-correct and correct-to-wrong components, and ACT12 delayed recovery. LOGIT_POWER is the equal-compute direction comparator; SAE and RANDOM are independent source baselines. Original ±0.20 bridge dose and new 2.00/8.00 radii must all be archived.

A fully separate **stdlib-only independent numerical replay** must check original checkpoint and sealed calibration SHA, exact solution uniqueness and cross-source disjointness, median visible blank site, frozen 512D four-arm unit directions, both full four-step JtJ update transcripts, all ACT3/6/12 81x9 native logits and original physical doses, exact Sudoku clue and 27-unit validity, clean true-digit margins, signed task effects and fixed 2,000-draw clustered bootstrap and original decisions. It audits recorded neural outputs but does not reexecute the trained checkpoint or certify FP32 interval bounds.

Large clean winning digit gaps can still block prediction flips. Use the exact centered-logit finite winner-gap bound from LS-SAE06 to interpret such outcomes. A positive flip result would be causal prediction sensitivity relative to the equal-norm random source edit; Sudoku correctness improvement requires separate audited task accuracy evidence.
