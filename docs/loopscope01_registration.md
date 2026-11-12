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

## Amendment 6 (2026-11-09, before any Q6 computation): Q6, the contraction check on Lean-CGRT Gate-0 loopers

**What has and hasn't been seen.** The Gate-0 checkpoints are `D:\lean_cgrt\runs\c1\g0_Sloop_s{17,29,43,53,61,71,83,97,101,113}`
(single clock, width 592) and `g0_Srd_s{43,53,61,71,83,97,101,113}` (two clock, width 384). They were trained
by the Lean-CGRT session on `lcgrt-mirror` at 8 fast steps. Their evaluations at 8 fast steps (`eval_g0_*`) and
at 32 (`evalT32_g0_*`) exist. Only their configs and the key names of the eval JSON have been read. No eval
value has been read.

**Predictor.** F1/F8 is the first-order fragility (summed over the fast-clock matrices att.qkv, att.o, ff.0,
ff.2 and cross.*) at fast step 1, divided by that at step 8, at T = 8.
- **Items:** 128 gold-path states from 64 theorems of `lcgrt-mirror/val` (reservoir seed 0).
- **Readout:** the log-odds margin of the model's own final top tactic. States with fewer than 2 legal tactics
  are dropped.
- **Labels:** none are used.
- **Compute:** CPU, read-only on checkpoints and data. Script: `research_gym/scripts/loopscope_lean_q6.py`.

**Outcomes** (read only after every predictor value is written). The success rate is the `b4` budget of
`summary.all` (divided by `n` if it is a count).
- **Primary:** the test-time gain on `test_mirror_deep` with policy `best`: success at T = 32 minus success at
  T = 8.
- **Secondary:**
  - `test_mirror_deep/best` success at T = 8 (depth extrapolation at the training budget);
  - the same gain on `test_mirror_id`;
  - the greedy policy.

**Rule (primary, pooled n = 18).** ρ is the Spearman correlation of F1/F8 with the primary gain. One-sided
permutation p values use 10,000 permutations with seed 0.
- **Supported** (stronger early contraction goes with more gain from extra loops) if ρ ≤ −0.4 and the one-sided
  p < 0.05.
- **Reversed** if ρ ≥ +0.4 and the one-sided p < 0.05 in that direction.
- **Null** otherwise.
- Within-family ρ (S-loop n = 10, S-rd n = 8) is reported. A pooled result that holds in neither family is
  flagged as a family confound.

## Amendment 7 (2026-11-10, before any Q7 computation): Q7, contraction restricted to the computation-carrying subspace

**Hypothesis.** A looper length-generalizes when its error dynamics contract inside the subspace that carries the
computation, ‖P_{E_{t+1}} J_t P_{E_t}‖ < 1, even if the full transition is not contracting. If so, the Q5 misses
(extrapolators with F1/F8 ≥ 0.05) should look like extrapolators on this statistic.

**What has and hasn't been seen.** The 24 frozen fixed-depth toy checkpoints:
- Q4: λ ∈ {0, 0.01, 0.1, 1} × seeds 1–3, in `loopscope_toy/ckpt`.
- Q5: λ ∈ {0, 0.01} × seeds 4–9, in `loopscope_toy_q5/ckpt`.

All their held-out accuracies and F1/F8 values are known:
- 17 arms extrapolate (held-out ≥ 0.90) and 7 do not.
- Over these 24 arms, the AUC of F1/F8 for non-extrapolation is 113/119 = 0.950. This is the bar.

No Jacobian of any checkpoint has been computed. This registration fixes the statistic, subspace, loops and
rules before that. It is not blind to outcomes.

**Setup.**
- **Transition:** h_{t+1} = core(inject(h_t) + e) on the full state (17 positions × 256).
- **Jacobian:** J_t = ∂h_{t+1}/∂h_t on the model's own trajectory.
- **Items:** 256 depth-8 items (item seed 930008). There is no depth token: fixed-depth arms answer depth k after k loops.
- **Precision:** float32 with TF32 off.
- **Labels:** none in the primary.

**Primary statistic: κ_late.**
- **Subspace:** E_t is the top-16 principal subspace of the cursor (last-position) state h_t^c across the 256
  items, embedded at the cursor position.
- **Per-item gain:** γ_t = σ_max(Q_{t+1}ᵀ J_t^{cc} Q_t) · ‖h_t^c‖ / ‖h_{t+1}^c‖. This is the relative error gain:
  every path out of h goes through a LayerNorm, so scale is not meaningful.
- **κ_late** = median over items of the mean of γ_t over the transitions t = 5, 6, 7 (into loops 6, 7, 8).

**Decision rules.** AUC is computed over the 24 arms, with a higher statistic predicting non-extrapolation and
ties counted half.
- **P7a (primary):**
  - **supported** if AUC(κ_late) ≥ 0.90 and > 0.950;
  - **matches F1/F8** if ≥ 0.90 but ≤ 0.950;
  - **weaker** if 0.75–0.90;
  - **not supported** below 0.75.
- **P7b (the Q5 misses):** supported if all three of λ0.01_s4, λ0.01_s8 and λ0_s9 have κ_late below the minimum
  κ_late of the 7 non-extrapolators.
- **P7c (absolute contraction):** supported if the per-arm median profile is below 1 at every late transition
  (t = 5, 6, 7) in all 17 extrapolators and in at most 3 of the 7 non-extrapolators.
- **P7d (global vs restricted):**
  - Global statistics: σ_max of the full J_t (power iteration, 30 steps, relative to ‖h_t‖_F / ‖h_{t+1}‖_F) and
    of the cursor block J_t^{cc}.
  - Supported if AUC(κ_late) exceeds the AUC of both global late statistics by ≥ 0.10.
- **P7e (late entry, descriptive):**
  - Onset t* is the first transition after which the median profile stays below 1 (8 if it never does).
  - Reported: the onset of the three Q5 misses against the other extrapolators.

**Secondary statistics.** Reported with their AUCs. They are not decision-bearing.
- **Label-based subspace:** E_t spanned by the 15 class-mean directions of h_t^c, grouped by the hop-t node.
  This needs hop labels.
- **Finite perturbation:** first order was blind to discrete flips in Q4.
  - Add ε‖h_5^c‖·Q_5w to the cursor after loop 5, for 4 random unit w per item and ε ∈ {0.1, 0.3}.
  - Rerun loops 6–8.
  - Report the relative deviation ‖Q_8ᵀ(h′_8^c − h_8^c)‖ / (ε‖h_8^c‖) and the readout flip rate.

**Checks.** Unit tests on a tiny float64 model:
- JVP vs finite differences;
- power-iteration σ vs the SVD of the full Jacobian;
- the restricted gain with Q = I equals the cursor-block σ.

The script asserts that its explicit transition reproduces `LoopedChase.forward`.

**Compute.** GPU under a vramorama lease, one job. Script: `research_gym/scripts/loopscope_toy_q7.py`. Output:
`experiments/recurrence_monitorability_v0/loopscope_toy_q7/`.

## Amendment 8 (2026-11-10, before any Q8 computation): Q8, do failing loopers leave the contracting region beyond the training depth?

**Motivation.** Q7 found that every toy looper contracts inside its computation subspace at loops 1–8,
extrapolator or not. Any difference must therefore appear at loops 9–16.

**What has and hasn't been seen.** The same 24 frozen checkpoints and known outcomes as amendment 7, plus
`capability_by_depth` for depths 9–16. No state or Jacobian at any loop beyond 8 has been computed. Q8 is a
mechanism question, not a label-free monitor: it uses the deep trajectory itself.

**Items.**
- Reference: 256 depth-8 items (seed 930008, 8 loops).
- Deep: 256 depth-16 items (seed 930016, 16 loops). Fixed-depth arms answer depth k after k loops.
- Precision: float32, TF32 off.

**Statistics.** None of them uses labels.
- **Drift.**
  - Let u = h^c / ‖h^c‖ be the normalized cursor state.
  - P_ref is the top-16 principal subspace of the reference u over loops 1–8 (2048 states), and μ_ref their mean.
  - r_t is the fraction of ‖u_t − μ_ref‖² outside P_ref, for deep items at loops t = 1..16.
  - **Δr** = median over items of [mean r_t over t = 9..16 − mean r_t over t = 1..8].
- **Deep contraction.**
  - γ_t is the Q7 statistic: per-loop top-16 principal subspaces of the deep items' cursor states, relative gain.
  - **κ_deep** = median over items of the mean γ_t over transitions t = 8..15 (into loops 9–16).

**Decision rules.** AUC is computed over the 24 arms, with a higher statistic predicting non-extrapolation (7
fail, held-out < 0.90) and ties counted half.
- **P8a (drift, primary):** supported if AUC(Δr) ≥ 0.90; weak if 0.75–0.90; not supported below 0.75.
- **P8b (contraction lost):**
  - **supported** if AUC(κ_deep) ≥ 0.90, at least 5 of 7 failures have κ_deep ≥ 1, and at most 3 of 17
    extrapolators do;
  - **rank only** if AUC ≥ 0.90 but the absolute condition fails;
  - **weak** if 0.75–0.90;
  - **not supported** below 0.75.
- **P8c (order, descriptive).** This uses hop labels.
  - Per-loop readout accuracy acc_t on deep items, t = 1..16.
  - For each failing arm, compare:
    - the first loop where acc_t < 0.90;
    - the first loop where r_t exceeds 2× the arm's max over t ≤ 8 (median over items);
    - the first transition at or after 8 where median γ_t ≥ 1.
  - Report whether drift precedes, coincides with, or follows the accuracy loss.

**Also reported.** The Spearman of Δr and κ_deep with held-out accuracy and with F1/F8. The AUC of F1/F8 (0.950)
remains the reference.

**Compute.** GPU under a vramorama lease, one job. Script: `research_gym/scripts/loopscope_toy_q8.py`, which
reuses the tested Q7 Jacobian code. Output: `experiments/recurrence_monitorability_v0/loopscope_toy_q8/`.

## Amendment 9 (2026-11-10, before any Q9 computation): Q9, why does F1/F8 predict extrapolation?

**What is known.**
- On the 24 frozen toy arms, F1/F8 (first-order weight fragility at loop 1 over loop 8, margin readout, depth-8
  items) predicts non-extrapolation with AUC 0.950.
- Q8 gave two failure modes past depth 8:
  - **schedule desync:** five λ = 0 arms that lock onto one hop per loop only at loop 6–7;
  - **contraction loss:** two arms with γ > 1 after loop 8.
- The lock-in loop L (from Q8's per-loop accuracy) and κ_late (Q7) are known per arm. The quantities M, C and the
  ablation below have not been computed for any arm.

**Candidate explanations.**
1. **State memory.** F1/F8 is a weight-space proxy for how strongly the final answer still depends on the loop-1
   state.
2. **Loop clock.** Failing arms carry a loop-count signal in the cursor state, used to land on the depth-8
   deadline, which breaks past 8.
3. **Schedule + contraction.** F1/F8 is a mixture of the lock-in loop and weak contraction.

**Items.** Probe training uses 512 depth-8 items (seed 931008); everything else uses the 256 Q7 depth-8 items
(seed 930008). The ablation uses the Q8 depth-16 items (seed 930016). Float32 with TF32 off.

**Statistics.**
- **M (state memory, label-free).**
  - m is the log-odds margin of the model's own top answer at loop 8.
  - g_t = ∂m/∂h_t on the full state.
  - M = median over items of (‖g_1‖‖h_1‖) / (‖g_8‖‖h_8‖).
- **C (clock, label-free).**
  - Target: the loop index t ∈ {2..8} (7 classes, chance 1/7).
  - Features: the normalized cursor state u_t = h_t^c / ‖h_t^c‖.
  - Probe: `ridge_probe`, trained on the 512 items and tested on the 256. C is the test accuracy.
- **Clock ablation (causal).**
  - P is the top-6 class-mean directions of u_t over t = 2..8 (training items), and μ the pooled mean.
  - After every loop, the cursor state is replaced by ‖h^c‖ · (u − PPᵀ(u − μ)).
  - Run on depth-16 items for 16 loops. Readout accuracy against hop labels is measured at loop 8 (acc8) and
    averaged over loops 9–16 (deep).
  - Control: 4 random orthonormal 6-dimensional subspaces (seeds 0–3), same formula, averaged.

**Decision rules.** AUC is computed over the 24 arms, with a higher statistic predicting non-extrapolation (7
fail) and ties counted half.
- **P9a (state memory):** supported if Spearman(M, F1/F8) ≥ 0.8. AUC(M) is reported.
- **P9b (clock):**
  - **supported** if AUC(C) ≥ 0.85 and Spearman(C, F1/F8) ≥ 0.6;
  - **weak** if AUC(C) ≥ 0.75;
  - **not supported** otherwise.
- **P9c (two-factor decomposition, known inputs):** an OLS fit of log F1/F8 on L and log κ_late over the 24 arms.
  Supported if R² ≥ 0.7. Both inputs are already known, but this fit has not been run.
- **P9d (the clock is causal for failure):**
  - The effect is deep accuracy under the clock ablation minus deep accuracy under the random control.
  - **supported** if at least 4 of the 7 failures gain more than +0.10;
  - **partial** if 2–3 do;
  - **not supported** otherwise.
  - Also reported: the change in acc8 (a drop of more than 0.2 means the clock serves the deadline) and the effect
    in extrapolators.

**Compute.** GPU under a vramorama lease, one job. Script: `research_gym/scripts/loopscope_toy_q9.py`. Output:
`experiments/recurrence_monitorability_v0/loopscope_toy_q9/`.

## Amendment 10 (2026-11-12, before any Q10 training or computation): Q10, prospective depth-32 extrapolation, edge-clock causality, and label-free semantic echo

**What is known.**
- On the 24 frozen Q4+Q5 checkpoints (`rmp-v0-phase2` at `4a924a2`), Q9 found that training-depth cursor loop-index
  decodability `C` (`t ∈ {2..8}`, chance `1/7 = 0.143`) separates Q9 extrapolation failures with `AUC(C) = 0.958`
  (vs `F1/F8` `0.950`), while mean-ablating the top-6 cursor clock directions did not repair deep accuracy (`0/7`
  gained `> +0.10`).
- `SEM-ECHO-01` (`27838d9`) subsequently revealed two structural facts on the same 24 checkpoints:
  1. Two Q9 passes (`lam0_s8`, `lam0.01_s4`) hold through depth 16 (`0.981`, `0.990`) and collapse over loops
     `17–32` (`0.476`, `0.556`), showing that Q9's depth-16 evaluation window missed late schedule/contraction loss.
     Against depth-32 accuracy (`acc_{9..32} ≥ 0.90`), all 9 failing Q4+Q5 arms have `C ≥ 0.633` (`0.633–0.934`) and
     14 of 15 passing arms have `C ≤ 0.621` (`0.166–0.621`, with only `lam0.01_s8` at `0.789`).
  2. In capped interventions, edge-token swaps carry the continuation difference while cursor swaps do not, directly
     pointing to Q9's untested candidate: a loop clock spread across the edge positions (full-state clock ablation).
- All Q9 and `SEM-ECHO-01` findings on Q4+Q5 are retrospective. No prospective test on fresh seeds exists yet for
  `C`, `SEM-ECHO`, or `RMP-MG-01`.

**Prospective cohort (untrained at registration).**
- 12 fresh fixed-depth `LoopedChase` runs: `λ ∈ {0, 0.01} × seeds 10, 11, 12, 13, 14, 15`, trained identically to
  Q4/Q5 (`n_nodes = 16`, `d = 256`, `layers = 2`, `heads = 4`, `loops = 8`, `fixed_depth = True`, `curriculum_frac = 0`,
  `steps = 30,000`, `batch = 128`, `lr = 3e-4`, AdamW).
- Validity gate (unchanged from Amendment 5 / Q5, `loopscope_toy_tca.py q5`): any arm with depth-8 accuracy
  `acc_8 < 0.90` is marked `training_failed` and excluded from primary classification counts (because `λ = 0` arms
  receive zero supervision on loops `1..7`, intermediate `acc_{1..7}` is not gated).

**Primary outcome (`extrap32`).**
- Measured on `N = 256` held-out depth-32 items (`seed = 930032`, 32 loops, float32, TF32 off).
- `acc_{9..32}` is the mean own-head oracle accuracy over loops `t = 9..32`. An arm **extrapolates (`extrap32 = True`)**
  iff `acc_{9..32} ≥ 0.90` (and **fails** iff `acc_{9..32} < 0.90`).
- `acc_{9..16}` and `acc_{17..32}` are reported alongside for backward comparison with Q5/Q9.

**Frozen label-free predictors and causal tests.**
1. **`C_cursor` (training-depth cursor clock, label-free, loops `2..8` only):**
   - Exact Q9 `C` definition: 7-way `ridge_probe` accuracy (`t ∈ {2..8}`) on normalized cursor state `u_t^c`, trained on
     512 depth-8 items (`seed = 931008`) and tested on 256 depth-8 items (`seed = 930008`).
   - **Frozen prospective rule (`P10a`):** predict `extrap32` iff **`C_cursor < 0.63`**.
2. **`D_echo_own` (label-free 32-loop own-head continuation disagreement):**
   - Computed over 64 unlabeled items (`seed = 940117`, `max_loops = 32`, `horizon = 4`) with **no oracle gate**: a pair
     of visits `1 ≤ t_1 < t_2 ≤ 28` on the same item is gated solely on the model's own predictions agreeing
     (`pred(t_1) == pred(t_2)`), and step disagreement is the fraction of steps `r ∈ {1..4}` where
     `pred(t_1 + r) != pred(t_2 + r)`. `D_echo_own` is the unweighted mean over items with at least one own-gated pair
     (items with zero own-gated pairs score `1.0`).
   - **Frozen prospective rule (`P10b`):** predict `extrap32` iff **`D_echo_own < 0.05`**.
   - Companion oracle-gated `SEM-ECHO-01` (`loopscope_semantic_echo`) and `RMP-MG-01` (`loopscope_mg_cohort`) receipts
     are also recorded on all 12 fresh Q10 checkpoints.
3. **`C_edge` and full-state clock ablation (`P10c`, tested on both the 24 frozen Q4+Q5 arms and the 12 fresh Q10 arms):**
   - `C_edge`: 7-way `ridge_probe` accuracy (`t ∈ {2..8}`) from the normalized mean edge-token state
     `unit(mean_{i < 16} h_t[:, i])` (`512` train `seed = 931008`, `256` test `seed = 930008`), plus pooled per-position
     concatenated edge probe `C_edge_full`.
   - **Full-state clock ablation:** For each token position `p ∈ {0..16}` (all 16 edge tokens plus the cursor token),
     fit the top-6 loop-index (`t ∈ {2..8}`) class-mean subspace `P_p` and pooled mean `μ_p` on normalized token states
     from the 512 training items. After every loop `t = 1..32` on the 256 depth-32 items (`seed = 930032`), mean-ablate
     every token position (`u_p ← u_p − P_p P_pᵀ (u_p − μ_p)`, restoring each token's norm). Compare `acc_{9..32}` against
     4 matched random 6-dimensional subspaces per token position (`seeds 0..3`).

**Decision rules.**
- **`P10a` (prospective training-depth cursor clock on Q10):**
  - **supported** if `C_cursor < 0.63` classifies `≥ 11/12` (`≥ 90%`) valid Q10 arms correctly on `extrap32` AND
    `AUC(C_cursor, ~extrap32) ≥ 0.90` (if both classes have `≥ 1` arm);
  - **weak** if classification accuracy is `≥ 9/12` (`≥ 75%`) or `AUC ≥ 0.80`;
  - **not supported** otherwise.
- **`P10b` (prospective label-free 32-loop semantic echo on Q10):**
  - **supported** if `D_echo_own < 0.05` classifies `≥ 11/12` (`≥ 90%`) valid Q10 arms correctly on `extrap32` AND
    `AUC(D_echo_own, ~extrap32) ≥ 0.90`;
  - **weak** if classification accuracy is `≥ 9/12` (`≥ 75%`) or `AUC ≥ 0.80`;
  - **not supported** otherwise.
  - Secondary comparison: report paired difference in prospective AUC and accuracy against Q5's `F1/F8 < 0.05` rule.
- **`P10c` (edge/full-state clock causality):**
  - Evaluated first on the 9 depth-32 failing Q4+Q5 arms (and separately on failing Q10 arms):
  - **supported** if full-state clock ablation raises `acc_{9..32}` by `> +0.10` over the matched random control in at
    least `5/9` failing Q4+Q5 arms;
  - **partial** if `2–4/9` gain `> +0.10`;
  - **not supported** otherwise (establishing that non-stationary failure is not a removable additive per-position
    clock even across edge tokens).

**Compute.** GPU under vramorama leases (`rmp-q10-*`). Script: `research_gym/scripts/loopscope_toy_q10.py`; tests in
`tests/test_loopscope_toy_q10.py`. Outputs: `experiments/recurrence_monitorability_v0/loopscope_toy_q10/` (receipts and
metrics; checkpoints gitignored) and `experiments/recurrence_monitorability_v0/loopscope_toy_q10_eval/`.
