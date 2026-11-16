# LS-SAE05: prospective native recurrent Jacobian singular-direction experiment

**Registered 2026-11-16 before any LS-SAE05 heldout hidden states.** This follows the completed LS-SAE04 audited negative gate: along measured SAE, head and random unit directions the native recurrent full-state tangent contracted after two ACT steps. LS-SAE05 tests whether a narrow high-gain singular direction exists.

## Frozen source and independent Sudoku cohorts

Pretrained source: unchanged gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att file step_21700 SHA-256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**, PureSudokuTRM CPU FP32 PyTorch 2.7.1, NumPy 2.1.3, math attention, four threads.

**Reuse the LS-SAE04 preheldout sealed source-fit calibration artifact** from GitHub Actions run #43818817318, artifact ID #14203533086, name ls-sae04-original-preheldout-calibration-only. Require exact prior NPZ *file bytes* SHA-256 **073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25**, calibration fit transcript SHA-256 **83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec**, physical SAE dictionary SHA-256 **14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb**, calibration-roster SHA-256 **a49925eb3ec76b4cdfd1a412720695b5295047ac7134ae64c4794ef30b77220a** and source manifest SHA-256 **cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a**. No refitting or selecting among calibrations.

Generate fresh unique-solution noncanonical Sudoku complete grids using the source seeded randomized-MRV solver, positive alternating-rectangle invariant and fixed unique-hole digger(seed+1000000). Engineering pilot: two 30-hole puzzles with seeds **202625020000..202625020001** and two 34-hole with seeds **202625021000..202625021001**. Full registered evaluation: **8** independently generated 30-hole puzzles with seeds **202625030000..202625030007** and **8** 34-hole puzzles with seeds **202625040000..202625040007**. All target full solutions and visible puzzles must differ from LS-SAE02, LS-SAE03 and LS-SAE04 fit, pilot and test cohorts, and across LS-SAE05 splits. Exactly one goal per puzzle: median missing-cell position selected by input mask alone. Archive canonical JSON roster SHA before reading task outcomes.

## Native Jacobian and matrix-free power iteration

Inject one physical 512D vector into the target board-token **z_H after one ACT step**, retaining all other native z_H tokens, every z_L token, the input embeddings and all model weights. Let F₂ map the injected token vector to the entire 97×512 z_H state after exactly two additional ACT steps (42 L_level calls). Its Jacobian at the clean anchor is J₂.

Choose power initial u₀ by deterministic isotropic Gaussian seed from the first 63 bits of SHA256 of literal text LS-SAE05|family|puzzle_seed|power-start, normalized to unit L2. Perform **exactly four** power updates, for t=0,1,2,3: compute v_t=J₂u_t using full native torch.func.jvp with math SDPA; calculate w_t=J₂ᵀv_t by full reverse-mode VJP of the same F₂ on the same clean state and detached v_t; normalize u_(t+1)=w_t/||w_t||. Abstain on nonfinite or norms <=1e-12; record reason and retain puzzle in denominator. Finally score u₄ with a fifth JVP and archive all intermediate v_t and w_t norms and normalized directions. No task label, digit loss, target-output gradient or clean rival may influence this singular direction search.

Frozen source-direction controls chosen before task labels: highest strictly positive SAE feature's decoder direction through the source-frozen 512×48 PCA basis; clean step-one predicted winner-minus-runner full 512D digit-head direction; **eight** independent isotropic normal 512D JVP controls using literal SHA seeds LS-SAE05|family|puzzle_seed|random-gain|i for i=0..7. All unit L2. The eight one-JVP controls provide a best-of-eight random gain comparator; compute use is 4 JVP+4 VJP+1 final power JVP versus 8 random JVPs per puzzle, and is reported separately. The power search is geometry-adapted to each test puzzle and has a different budget from the task-only/SAE source observers.

Archive, for each scored vector, the full state JVP gain ||J₂u||, goal-token gain, off-token gain and all 97 token norms, goal token 512D tangent and energy fraction. Only then read Sudoku correct digit and select strongest *clean depth-2 non-target* logit rival; compute exact task-head JVP slope and full native reverse-mode goal gradient to check JVP/VJP agreement at tolerance max(0.02,0.05|slope|). For power, SAE, head and first random JVP direction record ±0.05 actual finite-margin secants and nonlinear errors after exactly two native ACT steps.

## Frozen scientific gate and uncertainty

The prospective per-puzzle event A is true exactly when BOTH

    ||J₂ u_power|| >= 0.25
    ||J₂ u_power|| >= 1.20 × max_i ||J₂ u_random_i||, i=0..7

hold. Missing or numerically ineligible power directions count as A=false without dropping a puzzle. The paired continuous effect is gain(power) − gain(best-of-eight random). Report every single per-puzzle value and distribution.

Bootstrap 8 independent puzzle IDs with replacement for **2,000** draws per family, with deterministic random.Random(202625130+len(family)), CI indexes 50 and 1949 after sorting. Registered family success requires **at least six of eight positive A events**, all 2,000 valid draws, and positive 95% lower CI bound on the continuous paired gain difference. Overall success requires both fresh 8-puzzle families. Retain failed outcomes without modifying thresholds, start seeds, power steps or test roster.

Secondary observations: power-vs-SAE and power-vs-head gain ratios; positive-event coverage; source task-margin slope and readout alignment; transient off-token energy; progress of each of four matrix-free power updates; eight-random empirical rank; frozen source calibration and checkpoint digests.

## Audit and limits

A separate standard-library receipt auditor must verify original source/NPZ SHA, new exact Sudoku manifests and median blank masks, all unit physical inputs, JVP norm decomposition, task readout dot products from saved full 512D goal tangents, clean-target/rival digit margins, finite secants, full power-iteration progress receipts, fixed eight random controls, puzzle-cluster bootstrap and both scientific gates. The auditor replays recorded numerical outputs; it does not independently execute model forward passes or certify intervals.

An observed high-gain power direction would demonstrate a narrow state-amplifying channel at the frozen model, intervention site and tested horizon. A negative result would constrain this exact four-step search and seeded start, not prove a full Jacobian operator-norm bound. LS-SAE04 original failures remain unchanged.
