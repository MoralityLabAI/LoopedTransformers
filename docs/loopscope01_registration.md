# LOOPSCOPE-01: separating iteration fragility from component specificity in looped models

Registered 2026-11-09, before any LOOPSCOPE-01 run. Exploratory program with decision rules stated in advance.
Branch `rmp-v0-phase2`.

## Motivation

On Huginn-0125, removing a core-block factor in iterations 9–16 costs up to 0.37 nats and in iterations 1–8
almost nothing. But an activation-matched random rotation costs the same (Silico experiment #3; HybridTRMLDT
`Huginn/NOTE_negative_result.md`). Per-iteration ablation mixes two quantities:

- **fragility:** how much any push of that size at iteration t moves the output, a property of the dynamics;
- **specificity:** how much more this component's direction matters than a random one of the same size.

Matched controls separate them only at the cost of many control forwards per component. This program
introduces an instrument that measures both from one backward pass, validates it, and applies it to three
kinds of looped model.

## Instrument: transported component attribution (TCA)

At core layer l and iteration t, let `X` [positions × d_in] be the input to the matrix W and `A`
[positions × d_out] the gradient of the readout `y` (log-probability of the correct answer) with respect to
W's output at that call. Backpropagating through the remaining iterations carries the readout back to t, so `A`
already includes the loop's later repair. For a rank-one component `(v, u)` writing `(X v) u`:

- first-order effect of removing it: `Δ = −Σ_pos (X v)(A u) = −vᵀ (Xᵀ A) u`;
- **fragility** `F_{l,t} = ‖Xᵀ A‖_F`;
- closed-form matched nulls. For a random read rotation (`v` → random direction, same norm, `u` kept), `Δ` has
  mean 0 and sd `‖v‖ ‖Xᵀ A u‖ / √d_in`. For a write rotation, sd `‖u‖ ‖Aᵀ X v‖ / √d_out`. For both rotated,
  sd `‖v‖ ‖u‖ F / √(d_in d_out)`;
- **specificity** `z = Δ / sd`, per null.

One forward and one backward per item give `Δ` and `z` for every component, layer and iteration.

## Questions and decision rules

**Q1. Validity (Huginn).** Use the 2026-11-03 heat-map setting: two-hop variable chains, three-shot, r = 16, the
same 64 items, batches and seeds. Sum predicted `Δ` over each ablation window and compare with the measured
change in log p for all 480 measured conditions (16 top components + 8 random directions × 4 layers × 5
windows).
- **TCA valid at first order** if Spearman ρ ≥ 0.6 over the 480 conditions and the sign agrees on ≥ 80% of
  conditions with |measured| ≥ 0.05 nats.
- **Otherwise** report "first order fails" and do not use TCA for the claims in Q2–Q4.

**Q2. Huginn specificity.** Over all 64 components × 4 core layers × 16 iterations, compute `z_read`.
- **Prediction (from experiment #3):** the late-window effect is mostly fragility. The share of cells with
  |z_read| > 3 is below 5%, and `F` rises with t.
- **Falsified** if more than 20% of late (t ≥ 9) cells for the top-16 components have |z_read| > 3.

**Q3. Lean-CGRT S-loop (capability from test-time loops; exploratory, descriptive).** Use checkpoints from
`D:\lean_cgrt\runs\c1` (read-only): S-loop seed 17b, and HRM-S `S_t8_s17b` for contrast. On `test_mirror`
proof states at T = 8, 16 and 32, report per iteration:
- the decodability of the final chosen tactic from the state (the model's own head applied per iteration);
- TCA fragility;
- the specificity of components at iterations beyond 8.

The question: is the extra test-time computation readable from the state, or spread thin?

**Q4. Monitorability frontier (from-scratch toy loopers).** Train tied looped transformers on pointer or
variable chains, depths 1–8, with a per-loop intermediate-supervision weight λ ∈ {0, 0.01, 0.1, 1}. Use 3 seeds
per λ, with matched steps, data and parameters. Report:
- **capability:** accuracy at depths 1–8 and at held-out depths 9–16 with extra loops;
- **monitorability:** linear-probe decodability of the hop-k node at loop k, with the probe trained on a
  separate split, plus TCA specificity of the slot-reading components.

Rule:
- A **trade-off** is reported if some λ > 0 is below λ = 0 on held-out depth accuracy by more than 0.05
  (seed-bootstrap 95% interval excluding 0) while above it on probe decodability.
- **No trade-off** is reported if every λ > 0 is at least as capable and at least as readable.
- Otherwise the result is **mixed**.

## Limits stated in advance

- TCA is first order. Saturated or nonlinear repair can make it wrong, which is why Q1 gates the claims.
- Q3 uses one seed per configuration, and Q4 uses toy models. Nothing here is a claim about frontier models.
