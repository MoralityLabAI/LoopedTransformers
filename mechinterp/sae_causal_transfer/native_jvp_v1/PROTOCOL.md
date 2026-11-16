# LS-SAE03: Prospective full-512D native LoopedTRM Jacobian-vector transport

**Registered November 16, 2026 before creating any LS-SAE03 heldout hidden states.** This is a new experiment following source-locked LS-SAE02; its two negative registered gates, all earlier puzzle IDs and numerical results remain unchanged. LS-SAE03 tests why a direction with a large immediate task-margin effect can become nearly invisible after repeated native ACT steps. Its scientific hypothesis is that a surviving hidden-state perturbation becomes poorly aligned with the task readout, with an explicit alternative of generic contraction.

## Source lock and access order

The frozen network is gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att at original step_21700 file SHA256 **932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3**, loaded through the existing native LoopScope02 PureSudokuTRM code, FP32 CPU, PyTorch 2.7.1. All weights are eval/frozen and no training is applied to the TRM.

Reuse the exact source-fitted LS-SAE02 **32-puzzle fit30 calibration** with seeds 202619000000..202619000031, source-normalized 512→48 PCA and 48→96 top-four SAE (200 optimizer updates, batch256, source seed71). Verify the deterministic original training fingerprint **f217453dc17e078b055e99f3cebc3cf736af7e7cfae2ab120353c91bf2554c9d** and physical dictionary fingerprint **decf90fd73137bd232ba40104cf387e894c4a3d1f4f007036c591d173f3e238c** before creating new test hidden states. Calibration is a replay of pre-existing registered fit, not a refit chosen on LS-SAE03 outcomes. Failure to reproduce these SHA-256 hashes aborts the experiment.

New complete-grid seed bases are **202622030000** for 12 uniquely solvable 30-hole puzzles and **202622040000** for 12 uniquely solvable 34-hole puzzles. Use the source deterministic randomized MRV Sudoku generator, positive 2x2-intercalate invariant and unique-hole dig(seed+1000000) with exact two-solution cutoff. New full solutions and puzzle inputs must be mutually distinct and must not appear in any LS-SAE02 or SMC13–17 fit/calibration/test cohort. Preserve canonical roster JSON and SHA-256 manifest. Each puzzle contributes **one visibly selected median blank**: the blank at floor(n_blank/2) in ascending board-position order. The selected cell is fixed before looking at the answer or any network output. The engineering pilot uses only the first two puzzles in each family, does not count toward the full scientific gate, and must not retune registered parameters. All 24 new test examples enter the denominators even when their SAE code is zero or an intervention direction is numerically unusable.

## Intervention site and frozen controls

Capture clean z_H and z_L immediately **after native ACT step 1** on the unchanged question, and inject a perturbation in **one of the 81 board-token z_H vectors** (within a 97-token total sequence). Preserve every other z_H token, all z_L tokens, original embeddings and network weights at injection. Continue 0, 1 and 2 more full ACT steps (each step has 3 H cycles and 7 L-level calls per H cycle).

Choose directions from pre-outcome clean state and frozen calibration dictionary:
- **SAE:** the positive strictly largest top-four feature's physical 48D decoder column mapped to 512D by frozen PCA basis; normalize to unit L2. If highest feature is nonpositive/tied within 1e-8 or physical decoder column has norm ≤1e-8, label outcome-blind abstention, retaining that puzzle in coverage statistics.
- **Task-only head:** the highest and second-highest predicted digits at the **clean step-1** readout, with their native 512D lm_head row difference normalized to unit L2. It is based on predicted logits, not ground-truth digits.
- **Head-null SAE:** project the SAE direction into the orthogonal nullspace of the nine actual digit head rows (lm_head indices 2..10), normalized to unit length; abstain if its norm ≤1e-8. This direction's immediate clean digit logits are unchanged in exact arithmetic.
- **Random physical:** one separately seeded isotropic normal vector in 512D, unit-normalized, with seed = first 63 bits of SHA256 of literal LS-SAE03|family|grid_seed|random-jvp. This is the equal-rank, equal-L2 random intervention control.

Additionally score **128 independent rank-one random observer directions per puzzle** against the true-digit margin gradient at depth 2, using distinct SHA seeds LS-SAE03|family|grid_seed|rank-null|draw_index, with empirical rank p=(1+#random cosine>=SAE cosine)/129. These 128 directions are observer nulls, not 128 unrolled JVPs. Source checkpoint, SAE and direction selection are frozen before any target label is read. Their normalized physical dose is r=0.2; if an arm is ineligible, retain its pre-outcome abstention reason. A clean-model predicted winner-minus-runner head direction remains the task-only comparator.

## Full 512D Jacobian-vector measurement

For direction u in the full 512D hidden state, construct the tangent tensor V with u at the single selected board token and exactly zero at the other 96 tokens. Use **torch.func.jvp** on the actual native ACT continuation to obtain J_k V for k=0,1,2. With the CPU PyTorch 2.7.1 implementation, place the operation in torch.nn.attention.sdpa_kernel(backends=[SDPBackend.MATH]) so the selected attention implementation has forward-mode autodiff. Do not replace this with a finite-difference-only proxy. Validate the native JVP at the pilot using one exact small-model autograd-VJP cross-check and a central finite secant at radius 0.01, with nonfinite or unsupported-forward-AD as execution failure.

At each horizon record full 97×512 tangent-state norm R_global(k)=||J_k V||2, target token 512D norm R_goal(k), off-goal norm sqrt(R_global²−R_goal²), and fraction on the target token R_goal²/R_global². Freeze the correct Sudoku digit and strongest non-target clean competitor separately per depth using *clean logits* after geometry selection, and record the scalar margin derivative

    S_k(u) = [lm_head(y) - lm_head(clean_rival_k)] · [J_k V]_goal .

This exact first-order readout projection distinguishes retained state influence from task-margin sensitivity. Record the absolute task/readout alignment cosine |S_k|/(||lm_head(y)-lm_head(clean_rival_k)||*R_goal), with explicit zero-norm abstention. Also compute reverse-mode full-BPTT gradient g_k with respect to the original injected 512D goal state at k=2 and independently check S_2= g_2·u to absolute tolerance max(0.02,0.05|S_2|). These tolerances are for numerical validation, not scientific acceptance.

For each direction compute actual finite causal margins after injecting +0.2u and -0.2u and continuing all 0,1,2 steps, holding all other tokens identical at injection; compare symmetric secant [m(+r)-m(-r)]/(2r) to the JVP slope. Preserve original task labels and clean competitors after direction freezing. Record per-goal per-direction effects, numerical nonlinearity, clean task correctness and all 97×512 JVP norm diagnostics. These are numerical PyTorch measurements, not interval certificates or trained cross-checkpoint feature identity.

## Registered prospective scientific gate and uncertainty

Define the **SAE task-blind sensitivity attenuation with preserved hidden signal** at horizon k=2 as a Boolean per goal:

    E(u) := [abs(S_0(u)) >= 0.05]
         AND [abs(S_2(u))/(abs(S_0(u))+0.000001) <= 0.10]
         AND [R_global(2,u) >= 0.25].

Use the same predicate for the independently seeded random physical direction. An ineligible SAE is counted as E=false with an explicit admission-abstention record, never silently removed. The primary score is the puzzle-clustered proportion difference E_SAE−E_RANDOM within each 12-puzzle fresh family. For each family, resample its 12 puzzle IDs with replacement for **2000** bootstrap draws, random.Random(202622130+len(family)); report percentile indexes 50 and 1949.

**Registered success** on each family requires at least 10 SAE-numerically-eligible goals, at least 8 positive SAE events, and a strictly positive lower bootstrap bound of mean(E_SAE−E_RANDOM). Both families must pass for the whole prospective scientific gate. All families, admission abstentions, direction-specific effects, relative slopes and source hashes must be reported even if this is a negative result. The separate task-only and head-null directions are mechanistic controls, not alternative gates. The 128-random rank test is secondary and reported as rank-based p, not a z-score.

An independent **stdlib-only receipt auditor** must validate roster hashes, puzzle/goal masks, source SAE provenance, 0/1/2 depth grid, 512D geometry norms, one-hot input tangent placement metadata, JVP norm decomposition, readout-gradient arithmetic from saved goal tangent and nine digit head rows, true-digit/clean-rival margins and secants, random rank null p, frozen event predicates, 2000 cluster bootstrap and both registered decisions. The auditor checks saved numerical observations and *does not reexecute PyTorch neural passes*.

## Interpretation

A positive event can identify preserved global state influence with diminished task-margin readout sensitivity, but its cause might include redistribution over other tokens, rotation within the target token, or source-head alignment. Readout cosine and off-token norm measure these alternatives. If state gain itself collapses, the generic contraction interpretation is favored. Large finite secant errors weaken the linearized interpretation even if the JVP checks pass. No claim from LS-SAE02 is revised, and all LS-SAE03 outcomes must remain independent of its earlier inspected test puzzles.


## Dated source-replay engineering amendment, November 16, 2026, before LS-SAE03 full heldout states

The first source-locked pilot on [Actions #43816815113](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43816815113) rebuilt the **exact original LS-SAE02 calibration fingerprints**, successfully completed its 2+2 fresh-puzzle JVP pilot and independent standard-library audit. The first **full-job attempt aborted during the source-calibration hash check**, before constructing any new full-cohort hidden test states: separate GitHub Actions CPU runners did not reproduce the calibration arrays bit for bit. The original pre-registration and all numeric hypotheses, labels, seeds, interventions and acceptance thresholds remain frozen.

A source-only handoff now serializes the pilot's SHA-verified full 32-puzzle PCA+SAE arrays (mean, basis, normalization, encoder/decoder weights) into a versioned NPZ stored as a pilot workflow artifact. The same workflow's full job **downloads those exact frozen arrays**, validates the original expected fit/PCA and physical-dictionary canonical float64-array fingerprints, and uses them for the already preregistered 12+12 heldout evaluation. A missing, malformed or hash-mismatched frozen fit aborts; no refit or choice among calibration candidates is permitted in the full job. The pilot independently trained all 200 original SAE updates on the full 32-puzzle calibration roster and used no LS-SAE03 heldout data for the fit.

This is an engineering byte-identity repair. It cannot change a scientific gate, per-goal feature selection or heldout puzzle roster. The raw original model file SHA and both original LS-SAE02 calibration-array digests remain mandatory.
