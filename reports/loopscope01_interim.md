# LOOPSCOPE-01 interim: fragility vs specificity in looped models (Q1–Q3)

2026-11-09. Registration `docs/loopscope01_registration.md` (`6d5f4f6`, amendment 1 `47636c3`). Q4 (toy λ
sweep) is in progress.

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
- **Convergence explains fragility:** at T = 32, the S-loop's fragility at iteration 1 is 0.007 against 1.07 at
  the last iteration, a drop of about 150×. HRM-S shows the same pattern. Early pushes are washed out by the
  remaining iterations, and late ones are not. That is the same asymmetry as Huginn's, there 2.3×.
- **Specificity:** |z_read| > 3 for 5–11% of cells for the S-loop's top components, against 0.3% expected under
  the null. The share is flat across early and late iterations.

## What this adds up to so far

1. **Use matched controls.** Per-iteration ablation of a looped model measures mostly how close to convergence
   the loop is at that iteration. A component's identity adds little. TCA gives the separation analytically;
   matched controls give it experimentally. Both agree on Huginn.
2. **Where loops buy capability** (Lean S-loop), the extra computation refines a decision that the model's own
   head can read from mid-loop. In this case capability and monitorability are not in tension.
3. **Open:** Q4 asks whether keeping the loop readable costs capability when the schedule is otherwise free.

## Related work (Sept–Oct 2026)

- Kaissis et al., ICML 2026: step-decomposed influence (above).
- Sesodia, Sock, Torr, Schroeder de Witt, arXiv 2610.24135 (NeurIPS 2026): weight-tied models advance in a
  "staircase" adapted to the iteration budget; untied models need intermediate supervision.
- Wu, Hu, Zhang, arXiv 2609.39892: loops do not follow one-loop-one-step; attention routing selects transitions,
  and intermediate supervision shapes which are available.
- Olszowiec et al., arXiv 2610.18127: a five-lens monitoring protocol over eight looped LMs, including Huginn.

## Receipts

- `experiments/recurrence_monitorability_v0/loopscope_huginn/tca.json` and `tca.npz`: all per-item arrays and
  the Q1 condition table. Script: `research_gym/scripts/rmp_huginn_tca.py`.
- `experiments/recurrence_monitorability_v0/loopscope_lean/`: per-checkpoint JSON, `report.md` and the plot.
  Script: `research_gym/scripts/loopscope_lean.py`, tests in `tests/test_loopscope_lean.py` (7 pass).
