# LOOPSCOPE-01 interim: fragility vs specificity in looped models (Q1–Q3)

2026-11-09. Registration `docs/loopscope01_registration.md` (`6d5f4f6`; amendments 1 `47636c3`, 2 `59c9a11`,
3 `b758714`, 4 `4e70f12`, 5 `8e027b4`). Q1–Q4 are done. Q5, a prospective test of a post-hoc finding from Q4,
is running.

## The instrument

**Transported component attribution (TCA).** For a weight matrix W inside a looped block, at iteration t:
- X is W's input and A is the gradient of the readout (log p of the answer) with respect to W's output,
  taken through all later iterations.
- A rank-one component (v, u) has first-order removal effect `Δ = −vᵀ Xᵀ A u`.
- `F = ‖Xᵀ A‖_F` measures **fragility**: how much any push of that size at t moves the answer.
- Closed-form nulls for a matched random read rotation (sd `‖v‖‖Xᵀ A u‖/√d_in`) or write rotation
  (sd `‖u‖‖Aᵀ X v‖/√d_out`) turn Δ into a **specificity** score z.

One forward and one backward pass give every component, layer and iteration.

**Closest prior work.** Kaissis et al., *Step-Resolved Data Attribution for Looped Transformers* (ICML 2026),
decompose TracIn into per-step terms `φ_t = Σ_j δ_{t,j} ⊗ a_{t,j}`. That is the same per-step object as Xᵀ A
here, used to attribute *training data*. TCA uses it to attribute *weight components*, and adds matched-rotation
nulls that separate fragility from specificity.

## Q1: TCA reproduces measured ablations (Huginn-0125)

The setting is the 2026-11-03 heat map: two-hop variable chains, three-shot, r = 16, the same 64 items, batches
and noise seeds, and SVD components of the core `mlp.proj`.
- Batched noise is reproduced per item (max |Δ log p| 0.048, mean 0.0007).
- Over all 480 measured ablation conditions, the predicted window sums correlate with the measured changes at
  **Spearman 0.75**. The sign agrees on **93%** of the 29 conditions with |measured| ≥ 0.05.
- **Verdict:** valid by the registered rule.
- **Limit:** first order underestimates the largest effects. The regression slope is 1.9. Core layer 3's top
  direction was measured at −0.77 nats in iterations 13–16 and predicted at −0.06. TCA ranks components; it
  does not size the large nonlinear ones.

## Q2: Huginn's late-iteration effects are fragility, not read specificity

- **Fragility:** late iterations (9–16) are 2.3× more fragile than early ones.
- **Read specificity:** |z_read| > 3 in 8.6% of all cells, and in 19.5% of late cells for the 16 most-used
  components per layer. The registered falsification threshold of 20% was not reached, and the predicted
  "< 5% overall" was missed. Read specificity sits almost entirely in core layer 3 (25% early and late), with
  0–7% elsewhere.
- **Write specificity:** |z_write| > 3 in 25% of cells, about equally early and late (13–27% early, 18–34%
  late), with half of those effects positive.
- **Largest late effects:** core layer 3 components c1, c49, c50, c59, c27 and c0 have mean late |z| of 1.8 or
  less.
- **Conclusion:** the components that do damage late are not specially aligned with what the answer reads.
  They hit an iteration where any push of their size does damage. This matches the registered matched-control
  confirmation (Silico experiment #3; HybridTRMLDT `Huginn/NOTE_negative_result.md`): raw late effect
  −0.37 nats, matched control −0.37.

## Q3: the Lean-CGRT single-clock loop (capability from test-time loops)

Checkpoints are from the Lean-CGRT gym (read-only), run on 128 gold-path `test_mirror` states. In the gym's
proof search, the single-clock S-loop's certified success on mirrors rises with test-time loops (48.6% at T = 8,
65.0% at T = 32); the two-clock HRM-S does not.

| model | T | own head agrees with final, t = 1 | t = T/2 | final top choice is optimal |
|---|---|---|---|---|
| S-loop | 8 | 0.62 | 0.94 | 0.914 |
| S-loop | 32 | 0.60 | 0.99 | 0.953 |
| HRM-S | 8 | 0.63 | 0.98 | 0.891 |
| HRM-S | 32 | 0.63 | 0.99 | 0.898 |

- **Readable:** the model's own head, applied at every iteration, already gives the final choice for 60% of
  states after one iteration and 94–99% by mid-loop. The S-loop's gain from more loops (0.914 → 0.953 optimal)
  comes with a decision that stays readable from the state. For this model, more test-time compute does not mean
  less monitorability.
- **Early small pushes are corrected:** at T = 32, the S-loop's first-order fragility at iteration 1 is 0.007
  against 1.07 at the last iteration, a drop of about 150×. HRM-S shows the same pattern. That is the same
  asymmetry as Huginn's, there 2.3×. *Qualified after Q4:* the toy loopers show this signature through discrete
  error correction, while removing whole components early still does large damage. This number is about small
  pushes only, not about whether early iterations matter.
- **Specificity:** |z_read| > 3 for 5–11% of cells for the S-loop's top components, against 0.3% expected under
  the null. The share is flat across early and late iterations.

## Q4: per-loop supervision and the monitorability frontier (toy loopers)

The toy is a tied 2-layer looped transformer (d = 256, 1.6M parameters) that chases pointers over 16 nodes.
- **Training:** depth 8 only, 8 loops, 30,000 steps, with λ × CE(hop j) after loop j < 8 (amendment 2).
- **Capability at depth k:** run k loops and read the answer. The model never sees k; depths 9–16 are held out.
- **Runs:** 12, all valid (depth-8 accuracy 1.00).

| λ | held-out acc (9–16) | probe hop j at loop j | own head at loop j |
|---|---|---|---|
| 0 | 0.46 ± 0.38 (seeds: 0.21, 1.00, 0.17) | 0.76 ± 0.26 (0.92, 0.97, 0.40) | 0.47 |
| 0.01 | 0.79 ± 0.30 (1.00, 1.00, 0.36) | 0.92 | 0.89 |
| 0.1 | 1.00 | 1.00 | 1.00 |
| 1 | 1.00 | 1.00 | 1.00 |

**Decision (registered rule): no trade-off.** Every λ > 0 is at least as capable and at least as readable as
λ = 0. For λ ≥ 0.1, every seed extrapolates to depth 16 and is readable at every loop by a probe and by the
model's own head.

**Answer-only training finds different solutions per seed:**
- **seed 2:** a hop-per-loop staircase that extrapolates;
- **seed 1:** a staircase that is probe-readable (0.92) but neither readable by the model's own head nor
  extrapolating;
- **seed 3:** an **illegible** solution (probe 0.40) that answers depth 8 perfectly and fails elsewhere.

In this toy, the illegible solution is also the least general one. Here, losing monitorability does not buy
capability.

## Q4 TCA: what first-order fragility measures, and a positive control (amendments 3–4)

**Readout.** The answer's log-odds margin; log p saturates in trained arms.

**First order fails on large removals.**
- 5 of 12 arms pass the registered validity gate. ρ is 0.62–0.76 where the gate is met; sign agreement often
  falls just short of 0.80.
- The reason is discrete error correction:
  - First-order fragility at loop 1 is 0.01–1% of loop 8 in every extrapolating arm. Small pushes early are
    rounded off by the next pointer lookup.
  - Removing a whole top component at loop 1 costs −0.2 to −2.1 nats of margin, where first order predicts
    about 0.
- **P1 (early fragility within 3× of the last loop in λ > 0 arms): falsified,** 0/3 valid arms.
- **P2 (flat read-specificity share across loops): held,** 3/3.

**Consequence for Q3.** "Low early fragility" means early small pushes get corrected, by convergence or by
discretization. It does not mean early iterations don't matter. The Lean S-loop statement in Q3 is a statement
about small pushes only.

**Measured matched controls are a positive control for the Huginn null.**
- In every toy arm, single-loop removal of a top component does far more damage than matched read rotations of
  the same size. Mean |raw| is 0.12–1.98 nats; mean |control| is 0.006–0.08 in 11 arms and 0.39 in λ=1 seed 1.
- The same procedure gave raw = control on Huginn's late window (−0.37 vs −0.37).
- So matched controls do detect direction-specific computation when a loop has it. Huginn's null is a property
  of Huginn, not a blind instrument.

**Post hoc: contraction separates extrapolating from non-extrapolating arms.**
- F1/F8 is computed on depth-8 training-distribution items, from the answer's gradient alone.
- It is ≥ 0.11 in the 3 arms that fail to extrapolate and ≤ 0.018 in the 9 that do.
- Tie-aware Spearman: −0.70 with held-out accuracy, −0.86 with probe decodability.
- Final training loss does not separate them (lam0_s1 fails with loss 7e-9).
- Because the threshold was chosen after seeing these arms, **Q5** (amendment 5) tests it prospectively. The
  rule is F1/F8 < 0.05 ⇒ extrapolates, on 12 fresh runs (λ ∈ {0, 0.01} × seeds 4–9).

## Q5: does early-loop contraction predict length extrapolation? (amendment 5, prospective)

**Setting.** 12 fresh toy loopers (λ ∈ {0, 0.01} × seeds 4–9), trained as in Q4. The rule was registered before
training: predict extrapolation iff F1/F8 < 0.05 on depth-8 items; extrapolation means held-out accuracy (depths
9–16) ≥ 0.90. Every arm reached 1.000 at depth 8, so none was excluded as `training_failed`.

**Result: weak** (registered: ≥ 90% correct is supported, ≥ 75% weak). 9/12 classified correctly; 8 arms
extrapolate, 4 do not.

- **No false alarms.** All 5 arms predicted to extrapolate did. All 4 failures had F1/F8 ≥ 0.053.
- **Three misses, all extrapolators above the threshold:** λ0.01_s4 (0.221), λ0.01_s8 (0.060), λ0_s9 (0.0503).
- **The Q4 gap did not replicate.** Q4's arms split cleanly (≤ 0.018 vs ≥ 0.11). The fresh arms fill the region
  between: 0.044 (extrapolates), 0.0503 (extrapolates), 0.053 (fails), 0.060 (extrapolates).
- **The rank signal replicated.** Spearman(F1/F8, held-out accuracy) = −0.72, against −0.70 post hoc in Q4.
  Final training loss gives −0.38. A linear probe gives +0.93, but it needs hop labels; F1/F8 does not.
- **Exploratory (not registered):**
  - AUC of F1/F8 for failure is 27/32 = 0.84.
  - Within λ = 0, the two extrapolating seeds (s8, s9) have the two lowest F1/F8 of the six (0.044, 0.050 vs
    0.053–0.181).
  - All six λ = 0.01 arms extrapolate whatever their F1/F8, as every λ > 0 arm did in Q4.
- **First-order validity.** Only 3/12 arms pass the amendment-3 gate (ρ ≥ 0.6, sign ≥ 0.8). F1/F8 is used here
  as a label-free statistic of the gradient profile, not as a validated attribution.

**Reading.** Contraction is a one-sided signal in this toy. Low F1/F8 has so far always meant the computation
length-generalizes; high F1/F8 does not reliably mean it fails. A fixed 0.05 cut sits inside the overlap.

## Q6: does the contraction signature transfer to Lean-CGRT loopers? (amendment 6)

**Setting.** The 18 Gate-0 checkpoints: 10 single-clock S-loop seeds and 8 two-clock S-rd seeds, trained by the
Lean-CGRT session. F1/F8 was computed from the margin of each model's own top tactic on 69 `lcgrt-mirror/val`
gold-path states, before any eval value was read.

**Result: uninformative.** The registered primary outcome has no variation. Every checkpoint solves 100% of
`test_mirror_deep` under best-first search at budget b4, at both 8 and 32 fast steps.

**Descriptive.**
- **No early contraction anywhere:** F1/F8 = 1.00–2.41. Fragility spikes at step 1, dips at step 2, then rises.
- **No checkpoint benefits from more loops:**
  - best-first gains are ≤ +0.005 at every budget;
  - greedy changes on the deep split range from −0.38 to +0.005;
  - b1 changes on the deep split range from −0.34 to +0.002.
- **Exploratory (not registered):** at the tighter budgets, ρ(F1/F8, gain) runs from −0.07 to +0.34, all
  two-sided p ≥ 0.17.
- **The c1 checkpoints from Q3, measured the same way (exploratory):**
  - S-loop s17b, which gains from extra loops: F1/F8 = 0.64;
  - HRM-S s17b, which does not: F1/F8 = 0.78.

  Neither is near the toy's contraction regime (≤ 0.018). Q3's 150× figure used a log-p readout at T = 32 on
  `lcgrt-v1/test_mirror`, so the size of "early washout" depends on the readout and the budget.

**Reading.** The Lean loopers do not show the toy's discrete, error-correcting hop dynamics, and the Gate-0 seeds
gain nothing from extra loops. The contraction monitor has nothing to discriminate here. Its test stays with Q5
(toy, prospective).

## Q7: contraction restricted to the computation-carrying subspace (amendment 7)

**Setting.**
- The 24 frozen fixed-depth toy checkpoints (Q4 and Q5): 17 extrapolate, 7 do not.
- Transition Jacobians J_t were taken on each model's own trajectory over 256 depth-8 items.
- E_t is the top-16 principal subspace of the cursor state.
- γ_t = σ_max(Q_{t+1}ᵀ J_t^{cc} Q_t), relative to the cursor-norm change. κ_late is the median over items of the
  mean γ over transitions 5–7.
- This was registered with every outcome already known. The bar was F1/F8's AUC of 0.950 over the same 24 arms.

**Result: the hypothesis holds everywhere, so it does not discriminate.**
- **P7a (primary): weaker.** AUC(κ_late) = 0.824, against 0.950 for F1/F8.
- **P7b: not supported.** Two of the three Q5 misses (λ0.01_s4 at 0.384, λ0.01_s8 at 0.404) sit above three
  non-extrapolators (lowest 0.259).
- **P7c: not supported.** Every arm contracts inside E_t at every late transition: 17/17 extrapolators and 7/7
  failures, with κ_late from 0.056 to 0.743.
- **P7d: not supported.** The restricted statistic beats the best global one (cursor block, AUC 0.731) by 0.093;
  the bar was 0.10.
- **P7e (onset): uninformative.** 22 of 24 arms contract from the first transition.

**Descriptive.**
- **Contraction inside, expansion outside.** In every arm the full-state Jacobian expands (relative σ_max 1.5–6.9
  per loop) while the computation subspace contracts (0.06–0.74). Small finite pushes inside E_5 are erased by
  loop 8 (median relative deviation ≤ 0.054, no answer flips at ε = 0.1).
  - The exception is λ1_s1. It extrapolates perfectly but has κ_late = 0.71 and flips 21% of answers at ε = 0.3.
- **The secondaries agree:**
  - label-defined subspace (needs hop labels): AUC 0.765;
  - finite-perturbation deviation: 0.78 at ε = 0.1, 0.76 at ε = 0.3;
  - flip rate: 0.44.
- **Most failing λ = 0 arms start less contracted** (γ_1–γ_3 of 0.4–1.5, against ≤ 0.2 for λ > 0). They settle to
  0.16–0.35 by transition 7, into the same range as the λ = 0 seeds that do extrapolate.
  - The exception among failures is λ0.01_s3, which stays at 0.65 or above throughout.

**Reading.**
- Error contraction in the computation subspace at training depth is something every trained toy looper has,
  whether or not it extrapolates. It is a property of fitting depth 8, not of generalizing beyond it.
- F1/F8 still ranks extrapolation better. It measures how the readout's sensitivity is distributed across loops,
  which is not the same thing as the local gain.
- If failure is a property of loops 9–16, the unmeasured candidate is the trajectory leaving the region where it
  contracts. Measuring that needs depth > 8 items and a new amendment.

## Q8: drift and contraction beyond the training depth (amendment 8)

**Setting.**
- The same 24 frozen arms, on 256 depth-16 items run for 16 loops.
- **Drift (Δr):** the growth, from loops 1–8 to loops 9–16, in the share of the normalized cursor state lying
  outside the depth-8 reference principal subspace.
- **κ_deep:** the Q7 gain over transitions 8–15.
- **Precision:** the explicit transition matches `forward` to ≤ 3.8e-5 relative in float32, and to 1e-14 in
  float64 on the worst arm. The original elementwise 1e-4 check failed at loop 16 from float32 accumulation, so
  the run uses a per-item relative-error check of 1e-3.

**Result.**
- **P8a (drift, primary): weak.** AUC(Δr) = 0.882.
- **P8b (contraction lost): weak.** AUC(κ_deep) = 0.840. κ_deep ≥ 1 in 2 of 7 failures and 0 of 17
  extrapolators.
- **Neither beats F1/F8** (0.950).
- **The drift is real but small.** Every failing arm has Δr between +0.019 and +0.084, and 16 of 17 extrapolators
  have Δr ≤ 0. The exception, λ0_s8 (+0.020), is the extrapolator whose accuracy is already slipping by loop 16
  (0.90).
- **No failing arm crosses the registered 2× drift threshold.** As registered, P8c therefore reads "no drift" for
  all seven.

**Descriptive.** Per-loop readout accuracy (uses hop labels) shows two distinct failure modes.

1. **Contraction lost after loop 8:** λ0.01_s3, and λ0_s3.
   - λ0.01_s3 is phase-locked (one hop per loop, accuracy ≥ 0.98) at every loop 1–8, yet γ jumps from 0.63
     (transition 7) to 0.96 and then 1.22. Accuracy falls below 0.90 at loop 10, the same transition where γ
     first reaches 1.
   - λ0_s3 crosses γ = 1 at transition 9.
   - These are the only two arms with κ_deep ≥ 1.
2. **Schedule failure, with contraction intact:** the other five λ = 0 failures. They contract after loop 8
   (κ_deep 0.24–0.47) and drift only modestly.
   - Trained only at depth 8, they reach hop t at loop t only from some loop L onward.
   - Their held-out accuracy tracks how early that lock-in happens: L is 7 for s1 and s4 and 6 for s5 and s7, all
     failing; s6 (borderline, 0.876) locks at 5.
   - The extrapolating λ = 0 seeds lock earlier: s8 at 6 (accuracy 0.84 by loop 4), s9 and s2 at 3.
   - They lose accuracy within 1–3 loops past the training depth: at loops 10–11, and 13 for s6.

   Exploratory and post hoc: mean readout accuracy over loops 1–7 separates failures with AUC 0.966. This is the
   label-based lens, consistent with the Q4 probe result (ρ = +0.93), so it is not a new label-free signal.

**Reading.**
- The extrapolation failures in these toys are mostly not dynamical instabilities. In five of seven, the model
  stays contracting and near its reference manifold, but runs a hop schedule fitted to the depth-8 deadline, and
  that schedule desynchronizes once the loop count passes it.
- Only two failures lose contraction past the training depth.
- F1/F8 (AUC 0.95) captures both modes better than any dynamical statistic tested here (κ_late 0.82, Δr 0.88,
  κ_deep 0.84). It is label-free, but these results do not say why it works.

## Q9: why does F1/F8 predict extrapolation? (amendment 9)

**Setting.** The same 24 frozen arms. Three explanations were registered, plus one causal test:
- **M (state memory):** the relative state-gradient ratio (‖∂m/∂h_1‖‖h_1‖) / (‖∂m/∂h_8‖‖h_8‖), for the model's
  own top-answer margin.
- **C (loop clock):** ridge-probe accuracy for the loop index t = 2..8 from the normalized cursor state (chance
  0.143). Label-free.
- **Two-factor fit:** log F1/F8 on the lock-in loop L and log κ_late.
- **Causal test:** mean-ablate the top-6 loop-index directions of the cursor after every loop, on depth-16 items,
  against 4 random 6-dimensional subspaces.

**Result.**
- **P9a (state memory): supported.** Spearman(M, F1/F8) = +0.914, and AUC(M) = 0.908.
  - F1/F8 is a weight-space proxy for how strongly the loop-8 answer still depends on the loop-1 state.
- **P9b (clock): supported.** AUC(C) = 0.958, the best label-free statistic so far (F1/F8: 0.950), and
  Spearman(C, F1/F8) = +0.866.
  - The loop index is decodable at 0.79–0.93 in the late-locking λ = 0 failures and at 0.63–0.68 in the two
    contraction-loss failures.
  - It is near chance (0.17–0.26) in the λ ≥ 0.1 arms and at 0.31–0.46 in most λ = 0.01 extrapolators.
- **P9c (decomposition): not supported.** R² = 0.449, so lock-in plus contraction does not account for F1/F8.
- **P9d (clock causal): not supported.**
  - Removing the clock directions from the cursor changes deep accuracy by at most 0.02 in every failing arm
    (0/7 gain > 0.10). acc8 drops in only two failures: λ0_s3 0.88, λ0.01_s3 0.95.
  - In two λ = 1 extrapolators the same ablation is destructive: deep accuracy falls to 0.15 (λ1_s1) and 0.34
    (λ1_s3). There, the loop-mean directions carry the computation itself.

**Reading.**
- F1/F8 works because it measures **non-stationarity of the loop**: whether the loop-8 answer still depends on
  where the trajectory started, and whether the cursor state still carries which loop it is on.
- Models that extrapolate run a time-homogeneous map: the cursor distribution looks the same at every loop, and
  loop 1 is forgotten by loop 8.
- Models that fail are still in a transient at loop 8. Mostly that is a late hop schedule fitted to the deadline;
  in two cases contraction is weak.
- The linearly decodable clock is a symptom, not the mechanism. Mean-ablating it on the cursor does not repair
  extrapolation.
- Not tested:
  - a clock spread over the edge positions or encoded nonlinearly;
  - the full-state version of the ablation.

## Q11: the hop code as an intervention-preserving compactification (amendment 11)

**Setting.**
- **Origin:** proposed by a Fable subagent from a curated brief of these results and the Compactification-Theory
  B1 math (commutation error, near-fiber obstruction, simulation bound). Registered before computation, with
  outcomes known.
- **Abstraction:** π = the model's own belief z_t = softmax(readout(h_t^c)).
- **Abstract operation:** G_a = advance one hop along item a's permutation σ_a, decoded from the prompt. It is
  time-homogeneous and isometric (L = 1).
- **Bound statistic:** ρ_t = ‖z_{t+1} − G_a z_t‖ and ε̂ = max ρ_t over loops 4–7. Ê is the share of items with
  8ε̂ ≥ m(z_8)/√2, the simulation bound under time-homogeneity.
- **Phase obstruction:** Φ is the rate of near-fiber pairs (the same hop code at loops t and t + c on c-cycles)
  whose successors differ.
- **Data:** 4,096 fresh depth-16 items. No hop labels are used in Ê or Φ.

**Result: weak** by the registered rule. Exactly one sub-condition misses, by one arm.
- **(a) AUC(Ê) = 0.992** (item-bootstrap CI 0.992–1.000). This is the best predictor of extrapolation in the
  series (F1/F8 0.950, clock C 0.958).
- **(b) The bound holds in all 17 extrapolators** (Vio ≤ 0.05). λ0_s8 is flagged (Ê 0.148) and stays within the
  bound (E_obs 0.103).
- **(c) All 7 failures are flagged** (Ê ≥ 0.10), but only 5/7 match the Q8 mode labels (the rule needs 6).
  - The two contraction-loss arms were predicted to *violate* the bound but are flagged inside the horizon.
  - λ0.01_s3 is flagged and also violates the bound by a wide margin: Ê 0.135 against E_obs 0.801, so Vio +0.67.
    It is the only arm whose depth-16 error is invisible to the loops 4–7 commutation error.
- **(d) Φ separates all 5 desync arms (0.07–0.15) from every extrapolator** (max 0.032). Φ is exactly 0 in the 14
  phase-locked extrapolators. The same-phase control Φ_late is 0 in every arm.
- **(e) Shuffled-operation control: AUC 0.071**, so Ê depends on the abstract operation.

**Caveats.**
- **The margin clause is inactive:** every arm's loop-8 margin is 1.000. Ê therefore reduces to the share of
  items whose loops 4–7 commutation error exceeds 0.088, and ε̂ alone has the same AUC (0.992).
- **The registered E_obs-vs-on-disk check (within 0.02) misses in three arms:** λ0_s4 0.021, λ0_s5 0.036, λ0_s6
  0.038. The on-disk accuracies come from 256 items (standard error about 0.025); E_obs uses 4,096.

**Reading.**
- **Extrapolators:** the hop code is an intervention-preserving compactification with a time-homogeneous
  successor map, and the B1 bound computed from loops ≤ 8 holds at loop 16.
- **Desync failures:** a phase obstruction that is already visible inside the training horizon. The same hop code
  is followed by different successors depending on the loop.
- **Contraction loss:** shows up as a violation of the bound's hypotheses past loop 8 in λ0.01_s3.
- **Comparison with the earlier signals:** Ê is label-free (model beliefs plus the prompt's permutation) and
  beats F1/F8 and C. It needs the task's transition structure, which the weight-space and probe statistics do not.

## What this adds up to so far

1. **Use matched controls.** Per-iteration ablation of a looped model measures mostly how close to convergence
   the loop is at that iteration. A component's identity adds little. TCA gives the separation analytically;
   matched controls give it experimentally. Both agree on Huginn.
2. **Where loops buy capability** (Lean S-loop), the extra computation refines a decision that the model's own
   head can read from mid-loop. In this case capability and monitorability are not in tension.
3. **No trade-off in the toy (Q4).** Per-loop supervision gave loops that are more capable out of distribution
   *and* fully readable. Answer-only training sometimes found an illegible solution, and that solution was the
   one that failed to generalize.
4. **Instrument lessons.**
   - First-order fragility profiles mislead in loops with discrete states. Use measured matched controls for
     claims about early iterations.
   - Matched controls separate specific from non-specific damage in the toy, so Huginn's null stands.
5. **Label-free monitor, prospective result (Q5): weak.** Early-loop contraction (F1/F8 on training-depth
   items) classified 9/12 fresh toy loopers. It gave no false alarms, and its rank correlation with held-out
   accuracy (−0.72) replicated Q4. The registered 0.05 cut falls in an overlap region. On the Lean-CGRT Gate-0
   loopers (Q6), no checkpoint contracts and none gains from extra loops, so the check there was uninformative.
6. **Subspace-restricted contraction (Q7) is universal, not predictive.** All 24 toy loopers contract inside the
   cursor's principal subspace (κ 0.06–0.74) while their full transitions expand (1.5–6.9×). The statistic ranks
   extrapolation worse than F1/F8 (AUC 0.82 vs 0.95) and does not explain the Q5 misses.
7. **Two failure modes beyond the training depth (Q8).**
   - Five of seven failures keep contracting and drift only slightly. They fail because their hop schedule was
     fitted to the depth-8 deadline: they reach one hop per loop late, then desynchronize.
   - Two lose contraction (γ > 1) right after loop 8.
   - Drift (AUC 0.88) and deep contraction (0.84) are weaker predictors than F1/F8 (0.95).
8. **Why F1/F8 works (Q9): it measures non-stationarity.**
   - F1/F8 tracks how much the loop-8 answer still depends on loop 1 (ρ = 0.91), and how well the loop index can
     be decoded from the cursor (ρ = 0.87).
   - That clock decodability is itself the best label-free predictor so far: AUC 0.958.
   - Extrapolators run a time-homogeneous loop. Removing the decodable clock from the cursor does not fix
     failures, so it is a symptom of the transient, not its cause.
9. **A compactification test is the best predictor so far (Q11, from Compactification-Theory B1).**
   - Treating the model's own hop belief as an abstraction, with a time-homogeneous successor, the simulation
     bound computed from loops 4–7 predicts depth-16 failure with AUC 0.992 and holds in all 17 extrapolators.
   - The phase obstruction Φ (the same hop code with different successors at different loops) isolates the
     desync failures.

## Related work (Sept–Oct 2026)

- Kaissis et al., ICML 2026: step-decomposed influence (above).
- Sesodia, Sock, Torr, Schroeder de Witt, arXiv 2610.24135 (NeurIPS 2026): weight-tied models advance in a
  "staircase" adapted to the iteration budget; untied models need intermediate supervision.
- Wu, Hu, Zhang, arXiv 2609.39892: loops do not follow one-loop-one-step; attention routing selects transitions,
  and intermediate supervision shapes which are available.
- Olszowiec et al., arXiv 2610.18127: a five-lens monitoring protocol over eight looped LMs, including Huginn.

## Receipts

- `experiments/recurrence_monitorability_v0/loopscope_toy_q11/`: per-arm JSON (ρ curve, Ê and E_obs with CIs, Φ,
  controls) and `q11_report.md` and `.json`. Script: `research_gym/scripts/loopscope_toy_q11.py`; tests in
  `tests/test_loopscope_toy_q11.py` (4 pass).
- `experiments/recurrence_monitorability_v0/loopscope_toy_q9/`: per-arm JSON (M, C, lock-in, clock and random
  ablations, per-loop accuracy) and `q9_report.md` and `.json`. Script: `research_gym/scripts/loopscope_toy_q9.py`;
  tests in `tests/test_loopscope_toy_q9.py` (4 pass).
- `experiments/recurrence_monitorability_v0/loopscope_toy_q8/`: per-arm JSON (r_t, γ_t, per-loop accuracy, norms,
  forward check) and `q8_report.md` and `.json`. Script: `research_gym/scripts/loopscope_toy_q8.py`; tests in
  `tests/test_loopscope_toy_q8.py` (3 pass).
- `experiments/recurrence_monitorability_v0/loopscope_toy_q7/`: per-arm JSON (γ profiles, global σ, label
  subspace, finite perturbation) and `q7_report.md` and `.json`. Script: `research_gym/scripts/loopscope_toy_q7.py`.
  Tests in `tests/test_loopscope_toy_q7.py` (5 pass): JVP vs finite differences, power iteration vs full-Jacobian
  SVD, restricted gain with Q = I vs the cursor block.
- `experiments/recurrence_monitorability_v0/loopscope_toy_q5/runs/` (Q5 receipts and metrics) and
  `loopscope_toy_q5_tca/` (per-arm TCA JSON and NPZ, `q5_report.md` and `.json`). Driver:
  `research_gym/scripts/loopscope_q5.sh`. Checkpoints are not in Git.
- `experiments/recurrence_monitorability_v0/loopscope_toy/`: Q4 receipts (`runs/*/receipt.json`,
  `metrics.json`), `report.md` and `report.json`. Checkpoints are not in Git. Scripts:
  `research_gym/scripts/loopscope_toy.py` and `loopscope_toy_all.sh`; tests in `tests/test_loopscope_toy.py`.
- `experiments/recurrence_monitorability_v0/loopscope_lean_q6/`: `predictor.json` (written before outcomes were
  read), `q6_report.md` and `.json`, and `exploratory_c1.json`. Script: `research_gym/scripts/loopscope_lean_q6.py`.
- `experiments/recurrence_monitorability_v0/loopscope_toy_tca/`: per-arm TCA JSON and NPZ, measured-control
  JSON, `report.md` and `tca_profiles.png`. Script: `research_gym/scripts/loopscope_toy_tca.py`; tests in
  `tests/test_loopscope_toy_tca.py`. Those tests check TCA Δ against finite differences and the rotation-null sd
  against sampling.

- `experiments/recurrence_monitorability_v0/loopscope_huginn/tca.json` and `tca.npz`: all per-item arrays and
  the Q1 condition table. Script: `research_gym/scripts/rmp_huginn_tca.py`.
- `experiments/recurrence_monitorability_v0/loopscope_lean/`: per-checkpoint JSON, `report.md` and the plot.
  Script: `research_gym/scripts/loopscope_lean.py`, tests in `tests/test_loopscope_lean.py` (7 pass).
