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
5. **Candidate label-free monitor (Q5, running):** early-loop contraction (F1/F8), measured on training-depth
   items only, as a predictor of whether a looper's computation length-generalizes.

## Related work (Sept–Oct 2026)

- Kaissis et al., ICML 2026: step-decomposed influence (above).
- Sesodia, Sock, Torr, Schroeder de Witt, arXiv 2610.24135 (NeurIPS 2026): weight-tied models advance in a
  "staircase" adapted to the iteration budget; untied models need intermediate supervision.
- Wu, Hu, Zhang, arXiv 2609.39892: loops do not follow one-loop-one-step; attention routing selects transitions,
  and intermediate supervision shapes which are available.
- Olszowiec et al., arXiv 2610.18127: a five-lens monitoring protocol over eight looped LMs, including Huginn.

## Receipts

- `experiments/recurrence_monitorability_v0/loopscope_toy/`: Q4 receipts (`runs/*/receipt.json`,
  `metrics.json`), `report.md` and `report.json`. Checkpoints are not in Git. Scripts:
  `research_gym/scripts/loopscope_toy.py` and `loopscope_toy_all.sh`; tests in `tests/test_loopscope_toy.py`.
- `experiments/recurrence_monitorability_v0/loopscope_toy_tca/`: per-arm TCA JSON and NPZ, measured-control
  JSON, `report.md` and `tca_profiles.png`. Script: `research_gym/scripts/loopscope_toy_tca.py`; tests in
  `tests/test_loopscope_toy_tca.py`. Those tests check TCA Δ against finite differences and the rotation-null sd
  against sampling.

- `experiments/recurrence_monitorability_v0/loopscope_huginn/tca.json` and `tca.npz`: all per-item arrays and
  the Q1 condition table. Script: `research_gym/scripts/rmp_huginn_tca.py`.
- `experiments/recurrence_monitorability_v0/loopscope_lean/`: per-checkpoint JSON, `report.md` and the plot.
  Script: `research_gym/scripts/loopscope_lean.py`, tests in `tests/test_loopscope_lean.py` (7 pass).
