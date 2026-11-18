# LS-SAE12: causal interventions during active reasoning, scored by verified future deductions

**Prospective registration, 2026-11-17.** The commit that first adds this file is the registration. It fixes the source pool, the observable-error site rule, the intervention budget, the controls, the seeds, the outcome and the scientific gate before any LS-SAE12 puzzle content is fetched, before any checkpoint is run on a pool puzzle, and before any LS-SAE12 intervention exists. Pool file names and Git blob SHAs were read from the pinned public tree listing; no pool puzzle digits, solver output or model output were examined before registration.

LS-SAE08 (negative), LS-SAE09 (incomplete) and LS-SAE11 (registered gate failed in both families, see `../results/LS_SAE11_RESULTS.md`) stay as they are. Nothing here re-scores them.

## What LS-SAE11 showed that this design answers

- Edits on unsolved boards were amplified by the recurrence regardless of direction: a ±0.20 ACT1 edit reached median hidden-state gain ≈ 80 by ACT6, and the displacement saturated near the same norm for every arm and dose. Hidden-state amplification therefore cannot discriminate directions; only the outputs can.
- Every edit arm, including random, turned some stuck boards into exact ACT12 solutions (44 signed edits) and broke some boards the clean model was about to solve (11 signed edits). The outcome that matters is the *schedule of verified deductions*, not the size of the hidden displacement.
- The donor-transfer directions came from other puzzles. LS-SAE12 computes every non-random direction on the very state it edits, at the moment it edits it.
- A same-platform local re-execution of the LS-SAE11 hard stage changed clean argmax digits on unsolved boards relative to the GitHub runner. LS-SAE12 outcomes are therefore defined inside one same-forward batch, and all registered model stages run in one process on one machine.

## Frozen model and numerics

Original pretrained `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att` `step_21700`, SHA256 `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`, standalone PyTorch implementation `mechinterp/loopscope02/scripts/trm_sudoku.py`, CPU float32, PyTorch 2.7.1, NumPy 2.1.3, 4 threads, MATH attention, no fine-tuning. An ACT step is 3 H-cycles × 6 L-cycles. Predictions are the first-index argmax over the 9 digit logits after each ACT step, steps 1..16. Every receipt records Python, PyTorch and NumPy versions, OS and CPU model.

## Source pool (source-only, before any model access)

Public Champagne collection, `Snow-Hoe/sudoku-database` commit `72f77030afd36ac3a31bc0708ec9179f5d1b8fc4`, directory `sudoku-champagne-collection/` (tree `02ec0254a90be989b4253571341245aed3ed48a4`, 1000 files), sorted by exact codepoint order of file names. Positions 0..19 are LS-SAE11's pilot and hard cohorts. **The LS-SAE12 pool is positions 20..67, 48 files**, listed with their Git blob SHAs in `families.py` (`POOL_FILES`): `104087.txt` … `11340.txt`.

Admission of every slot, all required, before any model stage:
1. exactly 81 bytes of `0..9`, Git blob SHA equal to the pinned SHA;
2. exactly one solution under **two independent exact solvers** that must return the same grid: LS-SAE11's 324-column Algorithm X two-solution enumeration (`native_withinrun_v1.families.solve_unique`) and a separately written bitmask/MRV depth-first two-solution search (`families.solve_dfs`); the grid must be a valid completion of every given;
3. deterministic visible naked+hidden singles stall with blanks remaining (`native_hard_task_v1.families.visible_structure`);
4. no duplicate input or solution within the pool, and none shared with the 20 LS-SAE11 Champagne puzzles or the 16 LS-SAE08 reference puzzles.

A failed slot aborts the whole pool. There is no replacement, no exclusion and no pool extension. The source receipt records raw SHA256, solution, both solvers' node counts, singles statistics and a canonical pool fingerprint. The public puzzles may have been seen in pretraining; the pool is source-disjoint from earlier LS-SAE cohorts, not certified disjoint from training data or inequivalent under Sudoku symmetries.

## Screening and the observable-error site rule (clean model only)

Each admitted pool puzzle is run clean, batch size 1, ACT1..ACT16. For each blank cell `c`, the **verified deduction step** `T(c)` is the first step `s` such that the argmax digit equals the exact-solver digit at every step from `s` through ACT16; `T(c) = 17` if the ACT16 digit is wrong.

- Intervention time is fixed: **`t* = 4`** (the state after ACT4, before ACT5).
- A **recovering error** is a blank cell that is wrong after ACT4 and has `6 ≤ T(c) ≤ 16`: an incorrect intermediate prediction that the clean model later corrects and keeps correct.
- A puzzle is **eligible** if it has at least one recovering error.
- **Selected site** (observable-error rule): the recovering error with the smallest `T(c)`, ties by smallest cell index.
- **Control site** (model-blind rule, secondary): the median sorted blank cell, the LS-SAE11 rule, whatever its status.
- **Families:** in pool order, the first 12 eligible puzzles form `active_a`, the next 12 form `active_b`. Ineligible puzzles stay in the screening receipt with their trajectories. If fewer than 24 puzzles are eligible, each family with fewer than 12 states is registered **INCOMPLETE**: it is run and reported, and it receives no gate verdict.

Only clean predictions and exact-solver digits enter screening. No intervention is run before families are fixed.

## Interventions

For every family state and each site rule, start from the clean batch-1 `(z_H, z_L, x_emb)` after ACT4 and edit only the `z_H` row of the site token (`16 + cell`). Nothing else changes: no other token, no `z_L`, no embedding, no weight.

Directions, all unit L2 in the 512-d token space, computed on that state without labels:
- **STRUCTURAL:** top right-singular vector of `J_H = ∂ z_H(after ACT5, all 97×512) / ∂ z_H(after ACT4)[site]`, by 24 power iterations on `JᵀJ` (J·u by double backward through `w ↦ Jᵀw`), initial vector `gaussian_unit(2026111900000 + 1000·pool_position + 100·rule + 0)`.
- **LOGIT-SENSITIVE:** top right-singular vector of `J_Y = ∂ (per-cell centered 81×9 digit logits after ACT5) / ∂ z_H(after ACT4)[site]`, same procedure, initial seed `… + 1`.
- **ISOTROPIC:** 8 draws `gaussian_unit(2026111800000 + 1000·pool_position + 100·rule + k)`, k = 0..7.

`gaussian_unit(seed)` is 512 draws of Python `random.Random(seed).gauss(0, 1)` divided by their L2 norm, so the auditor regenerates it bit-for-bit. `rule` is 0 for the observable-error site and 1 for the control site. Power-iteration sign is fixed so the largest-magnitude component is positive; signs are symmetrized in scoring anyway.

**Equal budget:** every edit adds `dose × sign × unit` with the same physical L2 dose for every arm. Doses **0.02 (primary), 0.25, 2.00**; signs ±. Token norm after ACT4 is √512 ≈ 22.6 (RMS-normalized state), so the doses are about 0.09 %, 1.1 % and 8.8 % of the token norm. The primary dose is the smallest because LS-SAE11 showed larger edits saturate to a direction-independent displacement.

Selected site: clean + STRUCTURAL and LOGIT at 3 doses × 2 signs + ISOTROPIC 8 draws × 3 doses × 2 signs = **61 rows in one native batch**. Control site: clean + all arms at the primary dose only = **21 rows**. Each batch continues natively from ACT5 to ACT16. Argmax digits are saved for every row and step; full 81×9 logits for every row at ACT5 and ACT16; hidden-state transport (gain `‖Δz_H‖_F/dose`, `‖Δz_L‖_F/dose`, site share, centered 9-digit readout rowspace fraction, sufficient margin certificate `m_min > 2ε`, flips) at ACT5, ACT8 and ACT16.

The same-forward clean row (row 0) is the reference for every outcome. Steps 1..4 are the screened clean steps for every row. The batch-1 rerun of the clean trajectory must reproduce the screening digits exactly or the run stops as an engineering failure. Agreement of row 0 with the batch-1 screening digits at steps 5..16 is recorded as a diagnostic.

## Outcomes

For every row, the verified deduction schedule `T'(c)` over all blank cells, and relative to the same-forward clean schedule `T(c)`:
- **VDD** (verified deduction displacement) `= Σ_c |T'(c) − T(c)|`, with 17 for never;
- blocked recoveries (`T ≤ 16`, `T' = 17`), new recoveries (`T = 17`, `T' ≤ 16`), accelerated (`T' < T`), delayed (`T' > T`);
- selected-site shift `T'(site) − T(site)` and whether its recovery is blocked;
- ACT16 blank-cell correctness and exact board.

Per state and dose, `V_arm` is the mean VDD over the arm's rows (2 rows for STRUCTURAL and LOGIT, 16 for ISOTROPIC). Contrasts: `D_SI = V_STRUCTURAL − V_ISOTROPIC`, `D_LI = V_LOGIT − V_ISOTROPIC`, `D_SL = V_STRUCTURAL − V_LOGIT`.

## Registered scientific gate

Primary endpoint: `D_SI` at the observable-error site and dose 0.02. A family passes when **at least 9 of its 12 states have `D_SI > 0`** and the **lower bound of the 95 % percentile bootstrap of the mean `D_SI` is > 0**: 2000 resamples of 12 states with replacement, `random.Random(2026111813)` for `active_a` and `random.Random(2026111814)` for `active_b`, sorted draws, index 50 lower and 1949 upper. **Both families must pass** for the claim "directions taken from the model's own one-step dynamics change verified future deductions more than equal-norm isotropic directions during active reasoning". If either family fails, the claim fails. An INCOMPLETE family gives no verdict.

Secondary, reported in full with the same rule and no gate: `D_LI` and `D_SL` at every dose; `D_SI` at 0.25 and 2.00; the control-site contrasts at 0.02; blocked/new/accelerated/delayed counts; ACT16 correctness changes; hidden-state gains. An outcome where STRUCTURAL has a larger hidden-state gain than ISOTROPIC while `D_SI` fails is reported as amplification without deduction specificity.

No threshold, dose, seed, site rule, family size or pool changes after any LS-SAE12 model output exists.

## Engineering pilot

Before the registered model stage, the same code runs screening and interventions on the four LS-SAE11 engineering-pilot puzzles (`100153.txt`, `100371.txt`, `100587.txt`, `10080.txt`), which are not in the pool. The pilot only shows the code runs on the real checkpoint and measures run time. It has no gate, and nothing in this protocol changes because of it.

## Execution order and audit

1. Unit tests (CPU, toy models): both solvers, schedule definitions, site rules, gate and bootstrap, power iteration against an explicit Jacobian SVD, toy end-to-end run through the independent auditor.
2. Source-only admission of the 48 pool files, then an independent re-check of the receipt (both solvers again, blob SHAs, singles, disjointness).
3. Engineering pilot.
4. Registered model stage, one process on one machine: screening of all 48, then `active_a`, then `active_b`.
5. Independent standard-library auditor (`audit.py`) on the saved receipts: recomputes source admission, argmax digits from saved logits, every schedule, eligibility, both site rules, family assignment, isotropic vectors from their seeds, every per-row outcome, every per-state contrast, both family gates and bootstraps. It cannot rerun hidden states, Jacobians, power iterations or unsaved logits; every report states this.

Failed stages, failed source admissions and incomplete families stay in the public record.
