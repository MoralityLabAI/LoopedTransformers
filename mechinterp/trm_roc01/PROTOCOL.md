# ROC-TRM-01 | Source-locked Sudoku TRM observer sprint

**Registration date:** 2026-11-17. **Status:** source and prospective measurement choices frozen before any new ROC-TRM-01 puzzle generation, fitting, Jacobian measurement or pretrained-checkpoint rollout. `CONFIG.json` is the machine-readable specification. Changes to seeds, hypotheses, endpoints, or gates require a dated amendment committed before their affected measurements.

**Target:** `MoralityLabAI/LoopedTransformers/mechinterp/trm_roc01/`. This is a distinct experiment from ROC-02 prospective delayed-rule binding/release and retrospective ROC-02A rank-eight ToyLoop.

## Scientific target and existing evidence

Explain how a small tied recurrent Sudoku model turns successive updates into durable solutions. Distinguish a stable solve basin, a solved-then-lost trajectory and a stall by finite-time dynamical response, head-visible state, latent-to-visible coupling, and work beyond hand-coded propagation.

Use the two original 7M checkpoints only as *frozen diagnostic targets*. LOOPSCOPE-02 reports 226/256 exact solves for `gaoxin492` and 184/256 for `yagizdevre` on `items256`; LOOPSCOPE-03 gives separate visible/cache/residual counterfactuals and path dependence. These are preexisting results, not ROC-TRM-01 outcomes. Large LLM zero-shot comparisons are descriptive; the fair learning baselines are supervised tied/untied toy models and transparent symbolic solvers.

The attached handoff from November 17 and the predecessor MathCan/EvalMath work supply the research rationale. This registration does not import unmeasured BPD-v2 gate claims as evidence of decomposition.

### Source pins

- MathCan ROC-01 observer math / amendment C: `patrickdugan/MathCan@37bbac210c175ccb98e0ff0d4aa8bdd2be0764e6`, [Part 2](https://github.com/patrickdugan/MathCan/blob/research/roc01-unified-observer-card-20261116/research/observer_calculus/PART2_THEORY.md) and [Amendment C](https://github.com/patrickdugan/MathCan/blob/research/roc01-unified-observer-card-20261116/research/observer_calculus/ROC01_AMENDMENT_2026-11-16C.md).
- EvalMath evaluator / BPD-01 intervention scorer: `MoralityLabAI/EvalMath@4351c15966c45ac09a0613da3531d4914344715a`, [PR #36](https://github.com/MoralityLabAI/EvalMath/pull/36).
- Existing [LOOPSCOPE-02 runner](../loopscope02/scripts/trm_sudoku.py), [LOOPSCOPE-03 report](../loopscope03/docs/report.md), and [ROC-02A retrospective report](../../experiments/observer_calculus_roc02a_v1/ROC02A_FROZEN_NRF_RANK_CAUSAL.md).
- Original model downloads via existing `load_trm_checkpoint`, SHA256 verified *before inference*: primary `gaoxin492/step_21700` = `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`; replication `yagizdevre/model.pt` = `d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39`.

## Architecture and state definition

The pinned pretrained runner has 81 Sudoku cells, 16 prefix tokens, width 512, and *two* carried buffers `z_H,z_L`, each `97×512`. Hence the complete state is `S=(z_H,z_L)∈R^99328`. One **outer ACT step** is 3 H cycles, each with six L updates and one H update, totaling 21 tied core calls. All state-space measurements must use the complete paired transition `F_x(S)` with fixed board input `x`, including intercell and interbuffer effects. A state-local output projection is not a complete recurrent state.

The model has an 11×512 `lm_head`, while `digit_logits` selects vocabulary rows 2:11 (nine digits) on 81 board cells of `z_H`. The *native Sudoku readout* therefore has rank **at most** 81×9=729 over the full state. The complete 11-row vocabulary observation has rank at most 81×11=891. Both ranks are measured from actual checkpoint matrices before defining the exact readout-containing rank floors. Prefix and L buffer coordinates may influence future digits despite having no direct digit-logit output. Treat `q_head`/halt output as a separately named exploratory readout.

Define three explicitly different observers:

1. **Global readout observer:** the direct-sum rowspace of the 81 digit heads embedded in the full paired state. This is the primary native observable subspace. Its constrained rank sweep uses measured global readout rank plus `{0,1,2,3,4,7,8,12,16,32}` added dimensions. The 11-row vocabulary version is a separate secondary sweep.
2. **Local-cell diagnostic:** the head rowspace of one 512D `z_H` cell and local ranks `{1,2,3,4,7,8,9,10,11,12,13,16,32}`. Here all other state coordinates are conditioned upon / held fixed for the partial derivative. Results quantify a local conditional sensitivity, **not** a full-state closed quotient. Ranks below actual digit readout rank are unconstrained diagnostics only.
3. **Learned task observer:** low-rank train-only subspaces with task-only, leakage-regularized, and PCA comparators, using identical optimization budgets and split discipline. Report a basis rank `k` and its containment constraint explicitly.

## Mathematical operators and reporting

At a declared captured state `S_t`, outer horizon `n∈{1,2,4}`, compute matrix-free `T=D(F_x^n)(S_t)` and `D=T-I`. For orthonormal observer basis `U`, `P=UU^T`, `Q=I-P` interpreted as a projector or orthonormal complement:

- `A=U^T D U` (observed → observed), `B=U^T D Q` (dark → observed), `C=Q D U` (observed → dark).
- `a,b,c=||A||F,||B||F,||C||F` divided by `||D||F`; also retain raw norms per `sqrt(k)`.
- `G_0=Dg(S_t)` is the *current-state* declared readout Jacobian. An exact readout-containing closed observer must satisfy `G_0P=G_0`.
- `G_f=D(g∘F_x^n)(S_t)` is the *future-output* Jacobian. Its Gram is `H=G_f^T G_f`, yielding `s(U)=tr(U^T H U)/tr(H)`, with `s` undefined when `tr(H)=0`.
- `E(U)` is a train-only fitted observer's held-out task MSE. Exact puzzle solve, native output margins, first lock step, later flip count and `NeverLock` are distinct endpoints.

Use JVP/VJP with randomized trace/Frobenius estimators and report probe variance and convergence. For nonlinear / nonnormal dynamics, top singular growth of `T`, spectral radius, and finite-kick contraction are distinct quantities. A single eigenvalue estimate does not establish contraction. Compare local behavior at early, pre-lock, locked, lost-lock and stalled states. Report board and checkpoint variance.

### Exact closure and limits

With `G_0P=G_0`, define `L*(k)=min_U ||U^T(T-I)Q||F` over exact rank-k readout-containing subspaces. The smallest closed readout-containing space is the `T^T`-invariant span of `G_0^T,(T^T)G_0^T,...`. Exact-rank closure may be non-monotone: `T=1⊕R90`, `G_0=e1^T` gives `L*(1)=0,L*(2)=1,L*(3)=0`; the `1⊕R90⊕R90` witness produces interior even-rank obstructions.

Use the MathCan finite-horizon **lower bound**, with `O_N=[G_0;G_0T;...;G_0T^(N-1)]`:

`L*(k) >= sqrt(sum_{i>k}sigma_i(O_N)^2) / (||G_0||F sqrt(sum_{j=1}^{N-1}j²||T||op^(2(j-1))))`.

The denominator must be positive; the numerator can be zero. A sampled readout-containing completion yields a feasible **upper** bound. A partial numerical SVD cannot certify the entire tail without an explicit residual/error certificate. Label neural bounds as estimated if matrix-free numerical error has not been certified. Keep full-model `L*(k)` separate from local-cell and empirical task-error knees.

## ROC-01 null calibration and interventions

Use **128 SHA256-keyed rank-matched Gaussian/QR subspaces**, seed IDs 941001..941128, conditional on exact checkpoint, state, rank and horizon. Require 100 distinct projectors with fingerprints and per-state numeric rows. For raw `a,b,c,s` compute same-state rank-null medians and predeclare six tests `a↓,b↓,c↓,b↑,s↑,s↓`; tie-inclusive add-one empirical p and six-direction Holm-adjusted q≤0.05. A null-relative near-zero A/B/C needs ratio ≤0.20, large B ≥1.20, high s excess above median ≥0.80 of remaining range, low s ≤0.20 of null median. Undefined denominators, too few distinct bases, or estimated norm errors too large to resolve thresholds yield **unclassified**. Tags `closed quotient`, `invariant subspace`, `latched memory`, `output register`, `irrelevant` may overlap. Retain continuous scores in every case.

Train leakage-aware U under task MSE + `λ·mean_i[(b_i/m_{b,i}(k))²]`, primary λ=1 and exploratory `{0,.05,.2,1,2}`. Use the same null medians for every candidate with the same state/rank/clock; preserve task-only equal-budget comparator.

The active physical intervention arms are `dark→observed`, `observed→dark` and `observed→observed`, injected at declared steps on clean captured states. All need a matched physical-L2 isotropic and rank-matched control. Primary physical radius is 0.2; exploratory 0.05 and 0.5. `ablate_readout` is a mandatory separately reported control. Freeze-U acts on A and B together and is exploratory only. Record native logits, exact solve, time to lock, flips, and outcome at steps `{6,12,16}` for intervention steps `{2,4,8}`; only applicable forward horizon pairs are evaluated. Evaluate stable versus stalled strata separately.

### Separate BPD mechanism gate

The failed BPD-v2 Haar *component read-rotation* null is **not** repaired by applying more of those rotations. ROC's rank-matched random **subspace observer** null serves a different test; keep the two procedures and their claims distinct.

Before *either pretrained 7M checkpoint* is instrumented, rerun a cheap `StationaryRoutedLooper` planted 2-of-8 routing toy with **activation-permutation nulls**: permute each component's own activation coefficients across batch/token to preserve its exact empirical distribution. Record matched mean ablation, as an important uniform bias can otherwise look non-specific. Freeze support-density settings `{1/8,1/16,1/32}`, scale `{0.5,1,2,4}`, and 199 permutation draws per condition, seed IDs 280001..280199. Require planted active-gate precision and recall ≥0.80 and one-sided planted-effect p≤0.05 in at least 8/12 density×scale conditions. A failed gate stops pretrained promotion; report the full 12-condition table and planted-factor maximum-versus-selected cosine. These thresholds are new prospective ROC-TRM-01 gates, **not** claims about the historical BPD-v2 results.

## Experimental sequence

### S0: registration, static tests and memory preflight

- Commit this protocol and `CONFIG.json` before drawing any new puzzles, training weights or building new intervention data. Pin original hashes and preserve source manifest. Test registration consistency and tiny exact linear-algebra witnesses, requiring zero model compute.
- Check host available physical memory and Windows committed bytes before starting any later process. Earlier B4 v2 PID 275016 had ~21.7 GB committed at the time of its report; its current status is **unknown here**. Do not assume suspension releases committed memory.
- After this registration, implement and test a 6×6 exact uniqueness oracle, hidden/naked singles, pairs, triples, bounded-guess tiers, and exact solver reference.

### S1: train miniature TRMs on disjoint 6×6 families

- Use 2×3 boxes on a 6×6 board; 36 tokens, two recurrent 96D buffers, tied attention-based updates, 2H cycles × (2L+1H) calls per outer step; model seeds 173/179/181, generator seed 2026111701, split seed 2026111702. Deduplicate by canonical completed-board symmetry orbit across train/calibration/heldout before assigning folds. Cap at 4096 training, 512 calibration and 512 held-out boards; zero overlapping canonical solution orbits.
- Cap 2000 optimizer updates per model, batch≤8 and one process. Implement truncated gradient with detached previous outer state, using a single declared update clock. Freeze evaluation at ACT steps `{1,2,4,8,16,32}`; include an equal-parameter/compute untied task model, identical task-only observer and exact symbolic oracle.
- Hold at least 64 held-out puzzles unresolved by singles/pairs/triples propagation; if the generator cannot supply them under the frozen family definition, **fail the data gate** and publish the generator audit. Separate hard-to-solve-from-propagation and truly guess-dependent boards with the exact solver.
- Advance if at least 2/3 seeds solve ≥10% of the nontrivial held-out subset above deterministic propagation and improve exact solve by ≥5 percentage points from ACT step 4 to 16 on that subset. Report untied model performance as a comparator whether it wins or loses. Never substitute easy-board reconstruction for the gate.

### S2: matched Jacobian / observer studies on frozen toy checkpoints

- Freeze checkpoints at their preregistered update ceiling regardless of evaluation outcome. Choose states early, prior to first stable lock, durably locked, lost-lock, and stalled; use the same board IDs across observers and seeds. Compute T and finite-kick growth at outer horizons 1/2/4.
- Measure the native global and per-cell observer cards, plus learned task-only/leakage/PCA bases. Run 128 matched basis nulls and the physical interventions. Report per-seed paired errors, rank-normalized B and future-output capture with raw rows, plus exact symbolic-tier baseline.
- Only for small toy states with adequate RAM, use dense Jacobians as a cross-check against JVP/VJP for selected states; this is never the default route.

### S3: frozen pretrained pilot, conditional on S1/S2 and BPD gates

- Use original SHA-locked loaders, *batch one*, no training, no full 99,328² matrix. Select eight WB-Prop and eight WB-Stall items from `items256` per checkpoint by ascending SHA256 of `board_id|2026111703` within each stratum; publish actual IDs and hashes before measurement.
- Start with streaming JVP/VJP estimates of the global digit-head and 11-row head block metrics and finite-output changes. Run state/rank-matched nulls sequentially. Use realistic error bars: if 128-null/full-card completion would break the memory or time budget, explicitly label the inference incomplete.
- Link ROC outcomes to prior LOOPSCOPE-03 cache/residual and path-independence observations, conditioned on exact board IDs. Full `items256` can serve as later confirmation only under a separately dated registration.

## Stopping, system memory and evidence

Single-process CPU by default; optional RTX 3050 4GB toy acceleration. Abort safely if **process RSS reaches 6 GiB** or **committed/private bytes reach 8 GiB**; review host-wide headroom before any launch. Batch 1 on pretrained checkpoints; stream input batches, nulls and JVPs; do not retain long autograd trajectories or perform a full-rank `C=512` BPD fit. Logs need peak RSS, peak committed bytes, process and source hash, exact training/observer seed, board ID, step, clock, floating-point dtype and whether gradient checkpointing is used.

Publish failed gates and abstentions. No claim of real pretrained mechanism, global optimum `L*(k)`, or statistical causal classification is authorized by this registration alone. The experiment's principal falsifier is a model that solves well while displaying no systematic native dark-to-visible effect, no stable finite-time contraction signature, or no nontrivial gain beyond symbolic propagation.
