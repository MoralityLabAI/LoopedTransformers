# LS-SAE04: prospective full-512D native TRM JVP with a sealed calibration artifact

**Registered November 16, 2026, before any LS-SAE04 pilot or test hidden state was generated.** This succeeds LS-SAE03's engineering source-replay failure while retaining LS-SAE02's original two scientific failures and SMC-17's 1/4 source comparison. LS-SAE03 pilot successfully reproduced the historical LS-SAE02 fit once, but independent hosted CPU machines later failed its exact historical fit SHA check *before* the first full-family model evaluation. This study starts a new **source-fit-only calibration** and seals the resulting weights as a single binary artifact before any LS-SAE04 pilot or heldout inference.

## Frozen pretrained network, serial stage gates and calibration

Use the original pretrained gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att step_21700 with exact source file SHA-256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**, unchanged 7M native LoopScope02 PureSudokuTRM code, FP32 CPU, PyTorch 2.7.1, NumPy 2.1.3, four CPU threads.

Native step-one source state is a 512D z_H board token, keeping its paired z_L, all other 96 tokens and original input embedding intact when injecting edits. An outer ACT step contains 3 H cycles, each with 6 L-cycle and 1 H-update L_level call. Readout after **0, 1, 2** further complete ACT steps.

**Three strictly ordered CI phases:**

1. Calibration-only job generates 32 independent complete noncanonical Sudoku solutions at fixed **202624000000..202624000031**, with exactly 30 uniquely solvable holes per puzzle and dig seed = grid seed+1000000. Refit the source SAE only on these 32 puzzles with 81 board-token z_H samples each, training **512→48 PCA and 48→96 top-four SAE** using existing LS-SAE02 algorithms: 200 AdamW updates, batch256, learning rate 0.003, weight decay 0.0001, SAE seed71, minibatch seed20261971, reconstruction MSE plus 0.001 mean code activation penalty. Save fit arrays as a versioned NPZ with float64 PCA/normalizer and source FP32 encoder/decoder weights. Separately seal a canonical JSON manifest with original checkpoint SHA, fit puzzle roster SHA, schema, fit/PCA fingerprint SHA, physical dictionary fingerprint SHA, **NPZ file-byte SHA-256** and algorithm tag. Upload both files as a single immutable Actions artifact. Calibration job must produce no LS-SAE04 pilot/test full-grid or hidden-state observations.
2. Pilot job downloads that **same calibration artifact**, verifies all source hashes and original file-byte NPZ SHA, loads the arrays without fitting or modifying them, and evaluates only two 30-hole puzzles at **202624020000..202624020001** and two 34-hole puzzles at **202624021000..202624021001**. These pilot puzzle solutions are unique and disjoint from calibration and full test seeds. Pilot is an engineering-only study and cannot count toward the registered scientific gate. Original pilot numerical effects cannot tune thresholds or choose new test seeds.
3. Full job downloads the **identical calibration artifact from phase 1**, verifies the manifest and NPZ bytes again and evaluates **12 fresh 30-hole puzzles at 202624030000..202624030011**, and **12 fresh 34-hole puzzles at 202624040000..202624040011**. None were used for calibration, pilot or LS-SAE02/03 and SMC13–17. Retain complete solution, puzzle mask and SHA-256 rosters. Evaluate exactly one target per puzzle: the median missing-cell position among visible blanks (floor(n_blanks/2)). This selector uses no true digit or model output.

Calibration source seal is **created before first test hidden state is computed**. Later CPU variations cannot silently change the PCA/SAE; byte/hash mismatches abort the run. A source-fit fingerprint is measured and published as an outcome of calibration itself, never chosen to maximize pilot/test behavior.

## Full 512D JVP and matched controls

Follow the mathematical/numerical LS-SAE03 protocol for all intervention and scoring definitions, with the **same fixed control random-seed algorithm tag LS-SAE03** to reuse its tested implementation. In particular:

- From the frozen clean 512D step-one goal state, select the naturally largest positive SAE feature in the precommitted 48D PCA plane, map its physical decoder column to 512D and normalize. Strict positive top-one gap >1e−8 required, with explicit admission failure counting.
- Compare with the clean-model predicted winner-minus-runner native 512D digit-head direction, the projection of the SAE direction onto the nullspace of the nine digit-head rows, and one equal-norm isotropic random direction. All selected before reading target labels or task gradients.
- Insert a unit 512D tangent into *one* specified board z_H token. Compute genuine torch.func.jvp with math SDPA through 0/1/2 additional ACT steps to obtain full 97-token by 512D tangent states. Keep every token's tangent norm, local goal 512D tangent, global/goal/off-goal gain, goal-token energy fraction, task-head slope and readout cosine.
- Independently compute the native full-unrolled reverse-mode gradient of the true-digit minus strongest clean non-target digit margin; verify its directional dot-product against forward-JVP slope at depth2 to numerical tolerance max(0.02,0.05|slope|). Compute paired actual finite interventions at radius ±0.2 to record symmetric task-margin secants and nonlinear errors.
- Separately score **128 rank-one random observer directions** per puzzle/depth against the source task-gradient (the 128 nulls are separate from the one random-JVP direction). Report empirical rank p=(1+#random cos>=SAE cos)/129. Frozen calibration method and clean predictions determine source directions; heldout target labels only determine evaluation readout after source direction freeze.

All four direction classes are evaluated on each original 12-puzzle test family regardless of SAE status; the source feature abstention is recorded as an event failure for the primary gate.

## Predeclared scientific gate and audit

The inherited LS-SAE03 per-goal event is

    E(u) := [abs(S_0(u))>=0.05]
           AND [abs(S_2(u))/(abs(S_0(u))+0.000001)<=0.10]
           AND [||J_2 u|| >= 0.25]

where S is the exact first-order true-digit-versus-clean-rival margin slope after k further ACT steps, and the full-state global tangent norm is computed over all 97×512 coordinates. The paired per-puzzle primary outcome is **E(SAE)−E(random physical)**. For each family, bootstrap the 12 puzzle IDs with replacement **2000 draws**, random.Random(202622130+len(family)), report percentile indices 50 and 1949. Success per family requires >=10 SAE-admitted goals, >=8 positive SAE events and a strictly positive lower 95% bootstrap bound for the paired event difference. Both fresh families must pass overall. Preserve every failed gate and coverage reason. Head-only and head-null controls, depth-specific readout alignments, and 128 random observer ranks are secondary measurements.

The previous LS-SAE03 method implementation supplies the native forward-mode JVP and independent stdlib replay. A new independent auditor for LS-SAE04 must additionally check that the full raw receipt names the exact **pre-heldout sealed calibration manifest and NPZ file-byte SHA**, that fit and test/pilot source roster seeds and SHA match, that every task-label/clean rival is fixed after direction selection, all 512D readout projections and state energies reconstruct from saved vectors, and all rank-p and paired bootstrap gates reproduce. It does not independently execute the pretrained network and is not interval certification.

**Interpretation:** A positive gate supports preserved full-state influence coupled to lost target task readout sensitivity on these new independent puzzles relative to a norm-matched random direction. It does not establish hidden-feature uniqueness, a transferable cross-checkpoint mechanism or general task competence. The exact originally registered LS-SAE02/03 observations and failures remain separate.
