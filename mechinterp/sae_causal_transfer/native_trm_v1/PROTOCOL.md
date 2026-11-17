# LS-SAE02 prospective native TRM/SAE depth-gradient experiment

Registered **2026-11-16 before LS-SAE02 heldout measurements**. Stacked on LS-SAE01 PR #112. The finished SMC-17 result (1/4 positive pair gates) is a historical source result. This study uses a genuinely different frozen native LoopScope-02 TRM.

## Source and state
Use gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att checkpoint step_21700, exact original file SHA256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**. Load with the pre-existing LoopedTransformers PureSudokuTRM implementation, FP32, CPU, PyTorch 2.7.1, eval(), torch.set_num_threads(4). Preserve original weights and record newly measured FP32 clean accuracy separately from BF16 results. Freeze the 512D board-token z_H after **one complete ACT step**, keep all other 96 tokens, corresponding z_L and input embedding fixed at injection. Continue **0,1,2** additional full ACT steps (each 3 H cycles with six L cycles and an H update). At each depth the correct digit is compared with the strongest *clean* non-target digit, frozen for all arms.

## Frozen Sudoku split
New exact-unique noncanonical Sudoku full solutions use the source SMC-13 seeded randomized MRV generator, positive 2x2 intercalate count, and unique-hole removal (dig seed = full-grid seed + 1000000). Mutually disjoint complete solutions and visible inputs, and seed ranges disjoint from SMC13–17 and LoopScope02:
- SAE fit: 32 independently generated 30-hole puzzles, grid seeds **202619000000..202619000031**.
- Calibration-only feature support diagnostics: 16 30-hole puzzles, seeds **202619010000..202619010015**. Do not retune on this set.
- Test30: 16 30-hole puzzles, seeds **202619030000..202619030015**.
- Test34: 16 34-hole puzzles, seeds **202619040000..202619040015**.
Pilot is first 4 fit + 4 diagnostics + 2 of each test, used for engineering checks only, with its own separate SAE fit. Generate and hash canonical roster JSON before accessing either test family. Select exactly two goal positions per test puzzle from the visible blank mask at floor(n_blank/3), floor(2*n_blank/3), without target digits.

## Fit-only sparse autoencoder
Collect all 81 board-token z_H step-one vectors from each SAE-fit puzzle. PCA from **only** that training set: 512D mean, fixed-sign 48-column float64 SVD basis (largest absolute coordinate positive), fit-projection means and per-coordinate positive scales clipped at 1e−6. Fit one 48→96 top-four ReLU SAE: 200 AdamW updates, batch256, lr0.003, wd0.0001, weight seed71, data sampling generator seed20261971, decoder MSE + 0.001 mean code-activation penalty, FP32, encoder bias initially +0.5. Freeze all PCA, normalizer and SAE weights and SHA/fingerprints **before heldout-state access**.

For each test goal choose its own strict positive SAE top-one feature (top-one gap >1e−8). Use LS-SAE01 physical-coordinate adapter and causal observer to kill the matched feature to preactivation −0.02, preserving the other three active features (full SAE reencoding drift ≤1e−4), physical dose ≤6× native decoder contribution. Same-dose controls: first valid active-preserving sham within 4 fixed-seed tries, and best of baseline plus two signed least-squares tangent shams plus 128 seeded random tangent shams, chosen solely by minimum inactive preactivation mismatch. Ineligible goals abstain. Lift each 48D edit to the native 512D z_H token using the *same* PCA basis, preserving other tokens.

## Registered readouts
For each eligible test goal at continuation depth k∈{0,1,2}, compute clean, kill, baseline-sham and best-inactive-matched-sham true-digit-minus-fixed-clean-rival margins. Primary paired effect d_k = margin(best) − margin(kill). Compute genuine native Torch autograd g_k of that margin w.r.t. the clean step-one goal hidden state, project gradient into the frozen 48D PCA basis, and compare ell_k = g_k·(kill−best), residual d_k−ell_k, and |d_k−ell_k|/(1+|d_k|). Freeze labels and rivals only *after* geometric selection. No detached ACT approximation for k>0. Record clean-correct→wrong flips.

Rank-one comparisons: the clean **predicted winner vs runner-up** lm_head readout projected into the same 48D subspace as task-only baseline; and **128 independent seeded random rank-one observer directions per goal/depth**, distinct from sham candidates. Score absolute cosines to g_k and rank p=(1+# random ≥ SAE)/(129); use source-independent seed SHA256 of (LS-SAE02, family, grid seed, goal slot, depth, draw index).

## Statistical gate and audit
Primary depth=2 per family: ≥12 eligible goals over ≥8 independent puzzles, 2000 *puzzle-ID-clustered* bootstrap resamples using random.Random(202619130+len(family)), positive lower percentile bound (indexes 50 and 1949), and kill clean-correct→wrong count strictly greater than best sham. Both test families required for overall pass. Include all failures, coverage, numerical Jacobians, random rank p and per-depth margins. An independent standard-library receipt replay checks original checkpoint SHA, registered rosters and goal positions, source-lock metadata, finite active/inactive geometry, observed margin arithmetic, gradient dot products and puzzle bootstrap. Its replay does not reexecute original network forward passes. This is within-checkpoint native LoopedTRM evidence, *not* cross-teacher equivalence or certified unique feature causality.
