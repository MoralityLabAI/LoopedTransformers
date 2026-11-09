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

## Amendment 1 (2026-11-09, before any Q4 training run)

**Why:** the first Q4 build read the answer at loop k for depth-k items without stating k in the prompt.
That supervises every loop implicitly in every arm (the Algernon-02D D1 design), so λ = 0 would not be
answer-only.

**Q4 now:**
- The prompt states the hop count k as a token.
- Every item runs a fixed budget of T = 16 loops, and the answer is read after loop 16.
- λ supervises loop j < 16 with hop min(j, k): one hop per loop, then hold the answer. λ = 0 leaves the
  schedule free.
- **Capability:** accuracy after T loops at depths 1–8 and 9–16, primary; also after 2T loops.
- **Monitorability:** ridge-probe accuracy for hop j from the state after loop j (items with depth ≥ j),
  primary. Also reported: the best loop per hop (`probe_anywhere`) and the model's own head at loop j.
- The decision rule is unchanged: held-out accuracy means depths 9–16 after T loops, and probe decodability
  means the aligned mean over hops 1–16.
- TCA specificity for Q4 is deferred to a follow-up, because the Q4 code does not include it yet.

## Amendment 2 (2026-11-09, before the Q4 sweep; supersedes the Q4 design in amendment 1)

**Pilots** (seed 101, outside the sweep, receipts not used for the decision):
- With the hop count as a prompt token, held-out depths 9–16 use depth tokens never seen in training, so no
  arm can extrapolate. λ = 1 scored 1.00 on depths 1–8 and 0.18 on 9–16.
- With a 16-loop budget and lr 1e-3, λ = 0 did not train (0.22).
- The toy model also gives each edge one position (source + destination embeddings), makes the start node the
  read position, and re-injects the prompt each loop. These were found in debugging; without them, depth 1
  did not train in 600 steps.

**Q4 design for the sweep:**
- No depth token. Every training item has depth 8 and runs 8 loops.
- λ = 0 is answer-only (loss after loop 8); λ > 0 adds λ × CE(hop j) after loop j < 8.
- lr 3e-4, 30,000 steps, batch 128, no curriculum.
- **Capability:** at depth k, run k loops and read after loop k, for k = 1..16. The model never sees k.
  Primary: depths 9–16 (held out). Training-depth accuracy is depth 8.
- **Monitorability:** probe for hop j from the state after loop j, primary; `probe_anywhere` as before.
- **Validity:** an arm is valid if its depth-8 accuracy is at least 0.90 in a seed. Invalid arms are reported
  as `training_failed` and excluded from the decision. If λ = 0 is invalid in all seeds, the Q4 decision is
  "λ = 0 not trainable at this budget", with descriptive comparisons.
- **Pilot observation** (λ = 0, 10k steps, lr 3e-4): depth-8 accuracy 0.60; hop j decodable after loop j at
  0.20–0.35; reading after k ≠ 8 loops is near chance.

## Amendment 3 (2026-11-09, before any Q4 TCA run; Q4 TCA follow-up, exploratory)

Seven of the twelve Q4 runs had finished, and their capability and probe metrics had been read. No TCA quantity
had been computed on any toy checkpoint.

**Readout.** Trained arms are saturated (final loss about 1e-8), so the gradient of log p is about 1e-8 and
first-order TCA on log p is degenerate. The Q4 TCA readout is therefore the **log-odds margin** of the answer
after the read loop: `y = z_c − logsumexp_{j≠c} z_j`. Log p is also reported.

**Setting.**
- Every finished Q4 checkpoint, on CPU in float32.
- 256 depth-8 items read after loop 8, plus 256 depth-12 items read after loop 12. Depth 12 is interpreted only
  for arms with cap@12 ≥ 0.9.
- Components are the SVD of each core matrix: q, k and v (slices of qkv), attention proj, fc1 and fc2, per layer.
- Item seeds: 920008 for depth 8 and 920012 for depth 12.

**Validity (per arm, depth 8).** Measured single-component ablations cover, for each of the 12 matrices, the
top 4 components by summed |mean Δ| plus 2 fixed random components. Each is removed at one loop t = 1..8, or at
all loops, for 648 conditions. The arm's TCA is valid if both hold:
- Spearman ρ(predicted window-sum Δ, measured mean Δ margin) ≥ 0.6;
- the sign agrees on ≥ 80% of conditions with |measured| ≥ 0.05.

Claims below are made only for valid arms.

**Predictions.**
- **P1.** In λ > 0 arms (own head reads hop j at loop j, lens 1.0), the loop executes a sequential algorithm in
  which a wrong pointer at loop t is not repaired. Total fragility at loop 1 is therefore within 3× of the final
  loop: `F_1 / F_8 ≥ 1/3`. Compare a late/early ratio of 2.3 on Huginn and about 150× on the Lean S-loop at T = 32.
- **P2.** In λ > 0 arms the share of top-16 per-item cells with |z_read| > 3 is about constant across loops: the
  max/min ratio over loops 1–8 is ≤ 2.
- **λ = 0 arms:** descriptive only. Same quantities, compared with λ > 0.

## Amendment 4 (2026-11-09, after the first two toy TCA arms; measured matched controls, exploratory)

**What was seen.** On lam0_s1 and lam0.01_s1, first-order TCA puts fragility at loop 1 at 1–11% of loop 8. Yet
removing a top component at loop 1 measured −0.40 nats of margin on average (up to −12) against +0.001
predicted. The validity slope is negative (−0.46 and −0.80). In a pointer chase, small pushes at an early loop
are rounded off by the next discrete lookup, while a full component removal can flip the pointer. First-order
fragility therefore does not measure finite-push fragility here.

**Added measurement (toy, depth 8, every finished arm).** For each of the 12 core matrices:
- take the top 4 components (the same choice as the amendment 3 validation);
- for each single loop t = 1..8, measure the mean margin change for removing the component (raw);
- against it, measure 4 matched read rotations: subtract `s (X r) u` with r a fixed random unit vector in place
  of v, using generator seed 0 per (matrix, component, rotation);
- report per loop the mean raw change, the mean control change, and specificity = raw − control;
- also report the share of (matrix, component, loop) cells whose raw change lies outside the range of its 4
  controls by more than 0.05.

**Descriptive; no decision rule.** The question is whether early-loop damage is direction-specific (raw ≪ control)
or a property of the loop (raw ≈ control), as found for Huginn's late window.

## Amendment 5 (2026-11-09, after Q4 and its TCA; before any Q5 training run): Q5, a prospective test

**Post hoc observation (Q4 TCA, 12 arms).** First-order fragility at loop 1 relative to loop 8 (F1/F8) is
computed on depth-8 items only, using the answer's gradient and no hop labels.
- **Separation:** it is ≥ 0.11 in the 3 arms that fail to extrapolate (held-out 0.17–0.36) and ≤ 0.018 in the 9
  that extrapolate (held-out 1.00).
- **Correlation:** tie-aware Spearman with held-out accuracy is −0.70.
- **Baseline:** final training loss does not separate the arms. lam0_s1, which fails, has a final loss of 7e-9.

Because the threshold was chosen after seeing these arms, this is not evidence yet.

**Q5 design.**
- **Arms:** 12 new runs, λ ∈ {0, 0.01} × seeds 4–9. These are the λ values where outcomes varied. The Q4
  configuration is otherwise unchanged (amendment 2: depth 8, 8 loops, lr 3e-4, 30,000 steps), output to
  `experiments/recurrence_monitorability_v0/loopscope_toy_q5`.
- **Predictor:** F1/F8 from `loopscope_toy_tca.py run`, on depth-8 items with seed 920008.
- **Validity:** depth-8 accuracy ≥ 0.90; invalid arms are excluded and listed.
- **Rule** (fixed now; `loopscope_toy_tca.py q5`):
  - predict "extrapolates" iff F1/F8 < 0.05, where extrapolation means held-out accuracy (depths 9–16) ≥ 0.90;
  - **supported** if at least 90% of valid arms are classified correctly with at least 2 arms in each class;
  - **weak** if at least 75%;
  - **not supported** otherwise;
  - **uninformative** if either class has fewer than 2 arms.
- **Also reported:** Spearman of F1/F8, final training loss, and probe decodability (which needs hop labels)
  with held-out accuracy.
- **Note:** the TCA validity gate (amendment 3) does not apply to Q5. F1/F8 is used as a measured statistic, not
  as an attribution claim.
