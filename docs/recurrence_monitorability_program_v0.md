# Recurrence Monitorability Program v0

Status: design draft. Nothing here is registered, and no outcome exists. Protocol ID once frozen:
`recurrence_monitorability_v0` (RMP v0).

## Question

When a model moves serial computation out of emitted tokens and untied depth into tied latent iterations,
does its computation become harder to monitor at matched accuracy, compute, width, and monitor budget? And
is the per-iteration trajectory a stable, causally faithful object, or does it only look that way through a
sparse autoencoder?

The working question the whole program is built to answer for any example and any iteration is:

```text
What changed internally between loop t and loop t+1, and how do we know the answer is the model's
own account rather than the dictionary's?
```

## Starting Point In This Repository

| Need | What exists | Gap |
|---|---|---|
| Looped model | `LoopedDecoderLM` runs any `module_word`: tied `[0]*R`, untied `[0..R-1]`, periodic. It returns the residual stream after every visit (`visit_outputs`), runs other loop counts at inference (`depth_visits`), and already computes per-visit logits through `final_norm` and the tied unembedding (`auxiliary_logits`). | No within-block capture, attention weights are discarded (`need_weights=False`), and there is no way to patch, splice, skip, or resume from a visit. |
| Loop analysis interface | `lsa.kappa_probe.ResidualLoop` (`states()`, `suffix()`) with JVP/VJP suffix sensitivity. | `LoopedDecoderLM` does not implement it. |
| Tasks with known depth | `architecture_discovery/tasks.py`: pointer chase, modular recurrence, rewrite, 4x4 Sudoku, each with an explicit step count. | No per-step ground truth is stored, and the splits hold out depth (train 1-4, calibration 5-6, evaluation 7-8), so they measure extrapolation only. |
| Probes | `neural/probes.py`: group-disjoint binary probe, within-group shuffled-label floor, matched-norm random and orthogonal control directions. | Binary only. No multiclass, control-task selectivity, description length, or cross-visit transfer. |
| Authorization semantics | `core/hybrid.py` membrane and `SoundnessType`. | Not expressed as a token task a looped decoder can learn. |
| Monitor-gaming precedent | `gaming_vs_improvement` v1 and v2: an exposed latent probe is evaded in two of three seeds. | GRU proposer on public features, not a transformer loop. |
| Registration discipline | Hash-registered configs, `valid_null_result`, hierarchical bootstrap, Holm correction, capped job runners. | Reused as is. |
| Dictionaries, patching, similarity | None. | All new. |

### Blocking fact: no model in the repository has learned an iterative algorithm

All 27 LSAD v1 cells sit at chance on the calibration split they were scored on (depths 5-6). The nine A2
cells end at training loss 2.26 to 2.39; the entropy of the training targets is 2.2325. The eighteen A1 cells
stopped earlier, at training loss 3.7 to 9.1. On the macro score every cell falls below a predictor that
emits one fixed token per family:

| Family | Uniform chance | Best constant token | A2 range over nine cells |
|---|---:|---:|---:|
| Pointer chase (8 nodes) | 0.125 | 0.160 | 0.117 to 0.164 |
| Modular recurrence (mod 17) | 0.059 | 0.090 | 0.027 to 0.090 |
| Rewrite normalization (6 symbols) | 0.167 | 0.234 | 0.168 to 0.195 |

A2 ran 128 optimizer steps at batch 32, or 4,096 examples. The models learned which answer range belongs to
which family and nothing else. The audit is `reports/checkpoint_lsad_architecture_discovery_v1_addendum_1.md`.
Two consequences:

1. The LSAD v1 closeout bound (`|mean delta| <= 0.005209`) compares chance-level models. It does not bound
   the effect of dispatch order on a model that computes.
2. Phase 0 here must produce a model that solves the task. The run checkpoints are also git-ignored and
   `run_training_cell` overwrites a single `latest.pt`, so no checkpoint series exists to analyze.

## Arms

Every arm is the same pre-norm decoder block. Only where the serial computation lives changes.

| Arm | Construction | Serial steps happen in |
|---|---|---|
| `LOOP-T` | `module_word = [0]*8`, one physical block | tied latent iterations |
| `FF-U` | `module_word = [0..7]`, eight physical blocks | untied depth |
| `COT` | two untied blocks, trained to emit the trace as tokens before the answer | emitted tokens |
| `LOOP-T-wide` | `LOOP-T` with width solved to the `FF-U` parameter count | tied latent iterations |
| `LOOP-DS` | `LOOP-T` with loss at every visit at or after the required depth | tied iterations, early commit |
| `LOOP-COT(R)` | tied block, R visits per emitted trace token, R in {1, 2, 4, 8} | both |
| `TRM2` (successor, not v0) | two-state recursion: answer state `y`, latent `z` | `z` updates between `y` updates |

`LOOP-T` against `FF-U` is the iso-compute pair: same width, same eight block applications, same FLOPs.
`LOOP-T-wide` against `FF-U` is the iso-parameter pair. `COT` is the reference in which the trace is
externally visible by construction. `TRM2` maps onto the repository thesis directly (`y` is `a_t`, `z` is
`h_t`). It is out of scope for v0 and belongs to a successor, once the single-state arms are understood.

### Smallest useful model

`LOOP-T` at rung `M0`, below the existing `S0`:

| Property | Value |
|---|---|
| Hidden width | 128 |
| Heads | 4 (head width 32) |
| MLP | 4x, GELU, pre-norm, `alpha=1.0`, `beta=0.25` |
| Visits | 8 at training, 1 to 32 at inference |
| Vocabulary | the existing 2,048-token scheme, sinusoidal positions |
| Unique parameters | 459,776 (embedding 262,144; block 197,120; norms 512) |
| `FF-U` at the same width | 1,839,616 |

Three deliberate choices:

- Four heads instead of the ladder's `architecture_head_dim=8` (which would give sixteen). Fewer, wider heads
  keep per-head attribution tractable. This is a decided deviation from the v1 shape policy and will be
  recorded as such in the registration.
- No visit embedding. A tied block with no clock input is an autonomous map, so anything visit-dependent
  must be carried in the state. That makes the program-counter question well posed. `LOOP-T-CLK`, with a
  supplied visit embedding, is the contrast arm.
- No input re-injection. Input tokens stay in the residual stream at their own positions. Re-injection is
  held back as a fallback if Phase 0 fails to train.

## Tasks And Data

A new trace-annotated bundle, `rmp_task_bundle_v0`, generated by a new module. The v1 generator and its
hashes are left untouched.

| Family | Source | Trace variables stored per algorithmic step `s` |
|---|---|---|
| `pointer_chase` | existing generator, 16 nodes | `cur_node(s)`, `remaining(s)` |
| `gated_pointer_chase` | new | `cur_node(s)`, `latch(s)`, `first_violation_step`, held grants |
| `masked_pointer_chase` | new | `cur_node(s)`, `unknown_touched(s)` |
| `reachability` | new, 12 nodes, out-degree at most 2 | BFS `frontier(s)`, `visited(s)`, `goal_found(s)` |
| `sudoku4` | existing generator | per-cell candidate set after `s` propagation rounds, `stalled(s)` |
| `modular_recurrence`, `rewrite_normalization` | existing | accumulator or string state after `s` steps (replication only) |

- `gated_pointer_chase`: edges carry a lock class and the prompt lists the grants held. Traversing a locked
  edge without its grant makes the correct output `REFUSE`. This is the membrane's rule as a token task. The
  refusal rate is 50%, the violation step is uniform over the path, and every instance has a sibling with one
  grant flipped, so lock counts alone do not predict the label.
- `masked_pointer_chase`: one mapping entry is replaced by `UNK`. If the path touches it the correct output
  is `ABSTAIN`. This is the `SoundnessType.UNKNOWN` analogue.
- Trace steps are algorithmic steps, not visits. How visits align to steps is a finding, not an assumption.

Splits are by instance-structure hash, so siblings stay together:

| Split | Use | Size per family |
|---|---|---:|
| model-train | generated on the fly, depths 1-8 | at least 1M seen |
| monitor-train | dictionaries, probes, alignment searches | 20,000 |
| monitor-calibration | site, threshold, and hyperparameter selection | 5,000 |
| locked | reported numbers only | 5,000 |
| OOD stress | depths 9-12, never used for fitting | 2,000 |

Each locked example carries minimal-pair siblings that change exactly one trace variable at one step: one
edge on the path rewired, the step count changed by one, the start node changed, one grant flipped, or one
Sudoku clue changed. Each sibling records the changed variable and the step at which the traces diverge.

## Training And Checkpoints

Gate `G0`: at least three seeds per arm reach locked exact match of 0.98 or more at depths 1-8 on every core
family. No interpretability claim is made about a model below 0.90. OOD accuracy is reported, not gated.

Retained, hash-attested checkpoints per seed:

| Label | Definition | Why |
|---|---|---|
| `C0` | step 0 | random-initialization control for every dictionary and probe |
| `C1` | first step with depth-1 accuracy at 0.5 or more | onset |
| `C2[j]` | first step with depth-`j` accuracy at 0.99 or more, each `j` | whether depths are acquired in order |
| `C3` | first step meeting `G0` | primary analysis checkpoint |
| `C4` | four times the `C3` step count | monitor aging after the behavior is stable |
| log grid | steps `2^k` | trajectory |

One additional control model per arm is trained on targets permuted within family. It can only memorize.

## Instrumentation

Sites, for `t = 0..R`:

```text
resid[t]     residual stream entering visit t; resid[0] is the embedding, resid[R] feeds final_norm
attn_out[t]  per-head attention output at visit t
mlp_out[t]   MLP output at visit t
delta[t]     resid[t+1] - resid[t]
lens[t]      embedding.weight @ final_norm(resid[t])
```

For the pre-norm block the update is exact and needs no dictionary:

```text
delta[t] = (alpha^2 - 1) * resid[t] + alpha * beta * attn_out[t] + beta * mlp_out[t]
```

Code changes, in order of need:

1. `ScheduleExecutor`: an optional `visit_hook(visit, module_index, state) -> state` called after each visit.
   Capture, patch, splice, skip, repeat, and early exit all go through this one seam.
2. A block with explicit attention (the form already in `lsa/mini_transformer_probe.py`) so per-head outputs
   and attention patterns are available.
3. `LoopedDecoderLM.states()` and `suffix()` so the existing JVP/VJP tools apply unchanged.
4. Named checkpoint retention in the training cell, and loss over `auxiliary_logits` for `LOOP-DS`.
5. Greedy token-by-token generation for `COT` and `LOOP-COT`.
6. An activation store: fp16 shards keyed by example, visit, and position, with a hashed manifest. It keeps
   the answer position, every structural position, and four random positions, plus full sequences for a
   2,000-example atlas.

Residual norm drifts with `t` under `alpha=1`. Dictionaries and similarity measures therefore operate on
RMS-normalized states, which is the quantity the block's own LayerNorm reads. The raw norm is kept
separately as a convergence signal.

## Dictionaries

"Middle and late layers" in a tied model means middle and late visits. Five dictionaries, each with one job:

| Name | Trained on | Job |
|---|---|---|
| `D-tok` | `resid[0]` | Token-level dictionary. Defines what is given by the input rather than computed. |
| `D-shared` | `resid[t]` pooled over `t = 1..R` | One basis for all visits, so a change between visits is a sparse difference in fixed coordinates. |
| `D-visit[t]` | `resid[t]` alone, every `t`; `t = 4` and `t = 8` are the reported middle and late dictionaries | Measures what sharing costs. |
| `D-cross` | all visits jointly, one latent with a decoder vector per visit | Feature lifetime (decoder norm by visit) and feature rotation (decoder cosine between visits). |
| `D-trans` | reads `resid[t]`, predicts `delta[t]`, with a linear skip term | A sparse approximation of the transition function. One transcoder covers every visit. |

A sixth, `D-delta`, trained directly on `delta[t]`, is a cross-check on differences computed in `D-shared`.

All are TopK autoencoders with unit-norm decoder columns, a pre-encoder bias, and an auxiliary dead-latent
loss. Grid: width in {1024, 2048, 4096, 8192}, `k` in {8, 16, 32}. The selection rule is frozen before
outcomes: the smallest width and then smallest `k` with at least 0.95 of cross-entropy recovered on single-visit
splice in monitor-calibration. Every claim is repeated at the adjacent width. `FF-U` gets the same set, with
layers in place of visits; it needs eight transcoders where `LOOP-T` needs one.

The gap between `D-shared` and `D-visit[t]` reconstruction is itself a result. A small gap means the code is
stationary across visits. A large gap means the representation rotates between visits, and `D-cross` is then
the right instrument.

## Feature Extraction Pipeline

| Stage | Output |
|---|---|
| 1. Capture | activation shards and manifest |
| 2. Trace join | trace variables per example and step, joined by `example_id` |
| 3. Fit | dictionaries with training receipts |
| 4. Encode | sparse codes indexed by example, visit, position, feature |
| 5. Feature cards | one record per feature |
| 6. Transition tables | Loop Delta Reports, per example and aggregated per visit |
| 7. Causal validation | intervention scores written back to the cards |
| 8. Report | figures and a checkpoint report |

A feature card holds the activation profile by visit and by position, top examples, association with every
trace variable under both raw-visit and step-aligned indexing, direct logit effect through `lens`, maximum
cosine to `D-tok` (given versus computed), cross-seed match score, and, after stage 7, its causal scores.

Features are typed by trajectory shape, with definitions fixed before labeling:

| Type | Definition | Candidate for |
|---|---|---|
| clock | activation depends on `t` and not on the input | loop clock |
| latch | off, then on for all later visits, input-conditional | authorization, unknown-touched |
| transient | active for at most two consecutive visits | current subgoal |
| ramp | monotone magnitude over visits | confidence, convergence |
| persistent | active from birth to `R` at stable magnitude | stored result |

## Loop Delta Report

The report for one example and one transition `t -> t+1` has six lenses. The first is exact; the others are
approximations that must agree with it and with each other.

1. Component split. `delta[t]` by head and MLP, from the identity above. For each head, the source positions
   it read from.
2. Dictionary difference. Codes in `D-shared` at `t` and `t+1`: features born, features that died, and
   features whose magnitude changed, ranked by change times decoder norm.
3. Transition rules. `D-trans` latents active at `t`: for each, what it read from `resid[t]` and what it
   wrote into `delta[t]`.
4. Variable difference. Decoded trace variables at `t` and `t+1`.
5. Output difference. `KL(lens[t] || lens[t+1])`, change in the correct answer's rank, change in entropy.
6. Geometry. Relative update size, cosine between successive updates, effective rank across the batch.

Each update is split into the dataset-mean update at that visit and the example-specific remainder. The mean
part is the same for every input; it can carry a clock but says nothing about the example.

For minimal pairs, the difference between the clean and sibling updates isolates the part of the update
attributable to one trace variable.

### Fidelity: what makes a report an explanation

A report nominates an explanation `E` of `m` components (features, rules, heads, or variable subspaces). Its
fidelity is measured by rebuilding the next state from only those components and running the remaining
visits:

```text
resampled[t+1]   = resid[t] + delta'[t]                 delta' taken from another example, same family and visit
explained[t+1]   = resampled[t+1] + P_E (delta[t] - delta'[t])
DEF(E)           = (score(explained) - score(resampled)) / (score(clean) - score(resampled))
```

`P_E` projects onto the span of the chosen components. `score` is the log-probability of the clean answer at
the final visit; a local variant scores decoded variables at `t+1`. `S90` is the smallest `m` with `DEF` at
0.9 or more.

A report is accepted when `S90 <= 16` and its `DEF` at that size exceeds each control at the same size and
norm: random features, top principal components of the update distribution, top MLP neurons, and matched-norm
random and orthogonal directions from `matched_control_directions`. The complementary necessity test removes
`P_E` from the true update and measures the drop.

The same report is produced for `FF-U` between adjacent layers, so the comparison is like for like.

## Recurrence-Aligned Visualizations

| Figure | Axes | Reads off |
|---|---|---|
| Loop raster | features (sorted by birth visit) by visit | which features exist when; drawn twice, indexed by raw visit and by algorithmic step |
| Lens strip | position by visit, colored by correct-answer rank | when and where the answer becomes available |
| Staircase | decoded variable value by visit, over the ground-truth step trace | steps per visit: slope 1 is sequential, exponential is doubling |
| Arrival matrix | trace step by visit, colored by patching recovery | the schedule the model actually uses |
| Transfer matrix | probe trained at visit `i` by tested at visit `j` | stationary code (block) or rotating code (diagonal) |
| Similarity matrix | visit by visit CKA, and `LOOP-T` visit by `FF-U` layer | which visits correspond to which layers |
| State trajectories | first principal components at the answer position, colored by depth | fixed points, fan-out by required depth |
| Convergence fan | relative update size by visit, grouped by required depth | whether updates shut off when the work is done |
| Delta waterfall | one example, each transition stacked by head, MLP, and top features | a single trace end to end |
| Transition graph | `D-trans` rules as edges between `D-shared` features | one time-homogeneous diagram for the whole loop |

## Causal Interventions And Activation Patching

Patching uses minimal pairs. Denoising (clean activation into the sibling run) is primary; noising is
secondary. The score is normalized logit difference between the clean and sibling answers.

| Intervention | Unit | Question |
|---|---|---|
| Residual patch | visit and position | Arrival matrix: by which visit has step `s` been processed, and where does it live? |
| Head and MLP patch | visit and component | Does the same head do the same job at every visit (role stability)? |
| Subspace interchange | learned low-rank subspace for one variable | Does swapping `cur_node`, `remaining`, or `latch` produce the counterfactual output that variable predicts? Scored as interchange intervention accuracy (IIA). |
| Cross-time patch | `resid[t]` moved into visit `t'` | Time-translation equivariance. A clock-free loop should continue from the moved state. |
| Skip, repeat, early exit | whole visit | Does a missing visit only delay the result? Does running 2R or 4R visits change committed answers? |
| Feature pulse and clamp | one `D-shared` feature, one visit or all | Does the dynamics restore a perturbed feature (attractor) or propagate it? |
| Rule ablation | one `D-trans` latent | Does removing a transition rule stop exactly the step it is labeled with? |

Guards against the subspace-patching illusion: the patched subspace must also carry probe-decodable
information, a full residual patch at the same site sets the upper bound, and patched states must stay in
distribution on norm and `D-shared` reconstruction error.

## Probes

Multiclass logistic probes per variable, visit, and position, extending `fit_grouped_probe`:

- Group-disjoint fit, selection on monitor-calibration, reporting on locked.
- Within-group shuffled-label floor (existing) and control-task selectivity.
- Online description length over eight data portions, so capacity is charged.
- A one-hidden-layer probe alongside each linear probe. A large gap means the variable is present but not
  linearly accessible.
- Baselines at the same site: the `C0` model, `resid[0]`, and the permuted-target model.

A variable counts as computed at visit `t` only if it clears the shuffled floor, the control task, and all
three baselines. Hop count, start node, and grants are input tokens and will be decodable at `resid[0]`; that
is why the baseline is part of the definition.

Cross-visit transfer is the matrix of accuracy for a probe trained at visit `i` and tested at visit `j`. Its
summary is mean off-diagonal over mean diagonal normalized accuracy.

## Representation Similarity

- Debiased linear CKA, visit by visit within a model, layer by layer within `FF-U`, across seeds, and
  `LOOP-T` visit against `FF-U` layer with a monotone alignment path. At least 4,096 examples per estimate.
- Orthogonal Procrustes between `resid[t]` and `resid[t+1]`: how much of the change is rotation of existing
  content and how much is new content.
- Representational similarity against model matrices built from each trace variable at the aligned step, with
  variance partitioning: which variable organizes the geometry at each visit.
- Dictionary-level: `D-cross` decoder cosine across visits, and the `D-shared` to `D-visit` gap.
- Jacobian spectrum of the block at late visits, from the existing power-iteration code.
- Nulls: example permutation, the `C0` model, and both visit-mean-centered and uncentered states.

## Concept Hunts

| Concept | Ground truth | Decodability evidence | Causal test | What failure looks like |
|---|---|---|---|---|
| Program counter | `remaining(s)`; visit index `t` | counter probe that transfers across visits; clock-type features | interchange `remaining` from a 2-left example into a 4-left example: output must be the node two hops earlier. Cross-time patch tests whether an absolute clock is used. Running 2R visits tests overshoot. | counter decodable only at `resid[0]`; interchange has no effect; the model stops by clock, not by count |
| Subgoal state | `cur_node(s)` | arrival matrix ridge; transient features | interchange `cur_node`: output follows the source node under the base map | arrival matrix has no ridge, or one consistent with doubling instead |
| Uncertainty | lens entropy; will-be-wrong label; Sudoku candidate count; `unknown_touched(s)` | error-prediction probe above lens-entropy baseline; ramp features | steer the feature: `ABSTAIN` rate must move, and move more on instances that are closer to ambiguous | probe adds nothing over output entropy |
| Authorization | `latch(s)`, grant set | latch features aligned to the violation step | flip the latch subspace at visit `t`: `REFUSE` must toggle. Compare the monitor's direction with the causal direction. | latch decodable but flipping it does nothing; or refusal is decided in one visit from a lock-count heuristic |
| Search state | BFS `frontier(s)`, `visited(s)`; Sudoku `stalled(s)` | multi-label probes; set overlap between decoded and true frontier at the best-aligned step | add a node to the frontier on crafted graphs where that node reaches the goal: answer must flip | answer correct with no frontier-like state at any visit |
| Convergence | commit visit (lens top-1 equals final and never changes); required depth | relative update size, lens divergence, lens entropy as predictors of committed | in `LOOP-DS`, clamp the done feature early: the answer should freeze at the intermediate node of that visit | no signal separates committed from uncommitted; answers drift when run past `R` |

For authorization, the latch may be decodable before the pointer reaches the violation. That would mean the
model checks locks globally rather than along the path. It is a different algorithm and it gives a monitor
positive lead time, so it is reported as a finding and not as an error.

## Controls Against Spurious Dictionary Interpretations

1. Random initialization. The full pipeline runs on `C0`. Every interpretability score is reported as the
   margin over `C0`.
2. Structure-free input. Dictionaries trained on Gaussian data with matched covariance and on
   position-shuffled activations. Features that survive there are covariance artifacts.
3. Seeds. Three dictionary seeds and three model seeds. Only features matched across dictionary seeds
   (decoder cosine and activation correlation) are labeled.
4. Size. Each labeled feature is tracked across widths. Splitting and absorption are recorded, and claims are
   made about the spanned subspace.
5. Supervised ceiling. The tasks have known variables, so a supervised dictionary of conditional-mean vectors
   is built for each. The learned dictionary is scored against it on sufficiency, necessity, and sparse
   controllability.
6. Causal requirement. A label needs decodability out of group, an intervention effect the variable predicts,
   and a margin over matched-norm random, orthogonal, best-neuron, and top-principal-component baselines. The
   feature's IIA is reported as a fraction of the supervised subspace's IIA.
7. Planted positive. One model is trained with an auxiliary loss that writes a chosen variable into a chosen
   subspace. The pipeline must find it.
8. Absent negative. A label unrelated to the task (a hash bit of the example ID) is run through the same
   pipeline. It must find nothing.
9. Given versus computed. Features with high cosine to `D-tok`, or explained by position alone, are excluded
   from computed-feature counts.
10. Visit identity. In `D-shared`, features that only encode which visit this is are detected by comparing
    visit-mean-centered and uncentered training.
11. Closed-loop faithfulness. Experiment N1 below.
12. Multiplicity. Feature-by-variable-by-visit tests use hierarchical bootstrap and Holm correction. Primary
    feature-variable hypotheses are named before the locked split is read.

## Monitorability Metrics

A monitor is defined by its access tier, its class, and its labeled-data budget `n`.

| Tier | Access |
|---|---|
| 0 | final output only |
| 1 | emitted tokens |
| 2 | per-visit or per-layer lens decodes |
| 3 | linear probes on activations |
| 4 | dictionary features |
| 5 | interventions |

| Metric | Definition |
|---|---|
| Trace recoverability, TR(tier, n) | Share of registered (variable, step) targets a tier's monitor recovers on locked, as chance-normalized accuracy, after subtracting control-task accuracy. |
| Causally verified trace recoverability, CVTR | Share of targets with normalized accuracy at 0.9 or more, all probe baselines below 0.5, and IIA at 0.8 or more on the monitor's own readout. For tier 1 the intervention is a token edit. |
| Description length ratio | Online code length of trace labels given the site, over label entropy. |
| Detection latency | On `gated_pointer_chase`, the smallest compute fraction at which AUROC for "will output `REFUSE`" reaches 0.99, minus the fraction at which the true latch flips. |
| Transfer | Cross-visit (or cross-layer) transfer; depth transfer to the OOD split and to extra visits; aging from `C3` to `C4`. |
| Explanation size | Median `S90` per transition, and total dictionary description length per algorithmic step. |
| Closed-loop fidelity | Exact match with every state replaced by its reconstruction. |
| Evasion cost | Experiment N3: task accuracy lost per unit of fresh-monitor AUROC removed. |

The primary endpoint is CVTR at tier 3 with `n = 1,000`. It asks what share of the computation a cheap
white-box monitor can read and have confirmed by intervention. The headline figure is CVTR by tier for each
arm: `COT` reads the trace at tier 1, `LOOP-T` cannot by construction, and the question is how much of that
gap tiers 2 to 4 close.

Fairness conditions:

- Accuracy: every arm at 0.98 or more. If one arm cannot reach it, the stronger arm is compared at an earlier
  checkpoint with matched accuracy.
- Compute and width: the iso-compute pair is primary, the iso-parameter pair secondary.
- Sites: `LOOP-T` and `FF-U` both offer nine residual sites. Site choice happens on monitor-calibration.
- Monitor: same class, same `n` (swept over 100, 1,000, 10,000), same optimizer budget.
- Examples: the same locked examples for every arm, analyzed as pairs.
- Seeds: five model seeds per arm for the confirmatory comparison. LSAD v1 showed a single seed selects noise.
- Statistics: hierarchical bootstrap over seeds and instance groups, Holm across the primary family, and an
  equivalence margin of 0.05 CVTR declared in advance so that "no difference" is a licensed conclusion.

## Experiments

| ID | Experiment | Needs dictionaries | Can return a strong negative |
|---|---|---|---|
| E0 | Train to `G0`; add instrumentation; verify the literature registry | no | |
| E1 | Arrival matrices from minimal-pair patching | no | |
| E2 | Trace probes, transfer, description length | no | |
| E3 | Similarity and Jacobian analyses | no | |
| E4 | Dictionaries, controls 1-10, closed-loop replacement | yes | N1 |
| E5 | Loop Delta Reports and fidelity | yes | |
| E6 | Concept hunts with causal tests | partly | N2 |
| E7 | CoT displacement sweep | no | yes |
| E8 | Monitorability comparison, confirmatory | partly | |
| E9 | Monitor evasion | no | N3 |

### N1: closed-loop sparse replacement (primary negative-capable experiment)

A dictionary that reconstructs each visit well can still describe states the loop cannot run on, because a
tied block re-reads its own output and compounds any structured error.

- Open-loop: replace `resid[t]` by its `D-shared` reconstruction at one visit and run the true block onward.
- Closed-loop: replace after every visit.
- Replacement model: `resid[t+1] = resid[t] + D-trans(resid[t])` for all `t`; the block is never called.
- Controls: isotropic noise of the same norm as the reconstruction error at every visit; principal-component
  truncation at the same mean squared error; `FF-U` under the same procedure.

Outcome labels, frozen before the run:

| Label | Rule |
|---|---|
| `faithful_state` | Some dictionary with `k <= 32` keeps closed-loop exact match at 0.90 or more of clean on depths 5-8 and beats the matched-error truncation. |
| `open_loop_only` | Every dictionary with open-loop cross-entropy recovered at 0.95 or more at every visit has closed-loop exact match at 0.5 or less of clean, while the matched-norm noise control stays at 0.9 or more. |
| `fragile_dynamics` | The noise control also fails. The loop is sensitive to any perturbation of that size; the dictionary is not specifically at fault. |
| `unresolved` | Anything else. |

`open_loop_only` is the strong negative. It means per-visit feature trajectories in this model are not a
faithful account of the recurrent computation, and tier 4 is then dropped from E8 instead of being tuned. It
is a `valid_null_result`. E4 runs before E5 and E6 for this reason.

### N2: abstraction mismatch

Three candidate step-wise abstractions of pointer chasing are registered: one hop per visit, doubling, and
parallel lookahead. Each is tested by interchange on its own variables. If accuracy is 0.98 or more and the
best IIA across the three is below 0.6 on locked, no registered legible trace is causally realized. Trace
variables may still be decodable, but a monitor reading them is not reading the model's computation.

### E7: CoT displacement

`LOOP-COT(R)` emits the trace as tokens and has R latent visits per token. A trace token is edited to a
different valid node during generation. The displacement index is one minus the share of edits the final
answer follows. A second measure is whether the final answer is decodable from the latent state before the
trace is emitted.

Registered prediction: displacement rises with R. At R = 1 the answer must be read from the trace; at R = 8
the latent loop can compute it alone and the trace may become decorative. Teacher forcing on consistent
traces makes both solutions loss-optimal, so the outcome reflects inductive bias, which is the object of
study. A flat index within 0.05 across R is the negative: latent capacity alone does not displace the trace
at this scale.

### N3: monitor evasion

Converged `LOOP-T` and `FF-U` models are fine-tuned for a fixed budget on task loss plus a penalty on a
frozen, exposed latch monitor. Measured afterwards: the frozen monitor's AUROC, a fresh monitor of the same
tier, a fresh monitor one tier up, task accuracy, and latch IIA. Fresh same-tier AUROC at 0.6 or less with
task accuracy at 0.95 or more and latch IIA preserved is the strong negative: the model still uses the latch
and linear monitoring no longer sees it. The arm difference says whether tied recurrence makes this cheaper.

## Hypotheses

| ID | Statement | Decided by |
|---|---|---|
| H1 | Tied recurrence gives a more stationary code: cross-visit transfer in `LOOP-T` exceeds cross-layer transfer in `FF-U`. | E2, E3 |
| H2 | A clock-free tied loop builds a counter it causally uses. | E6 |
| H3 | One transcoder explains `LOOP-T` transitions at smaller total description length than eight explain `FF-U`. | E5 |
| H4 | At matched accuracy, tier-3 CVTR in `LOOP-T` is within 0.05 of `FF-U`. | E8 |
| H5 | Tier-2 TR in `LOOP-T` is far below tier-1 TR in `COT`; tiers 3-4 close most of the gap. | E8 |
| H6 | Monitors fitted at `R = 8` lose validity when the loop runs to 2R even where the model stays correct. | E2, E8 |
| H7 | Displacement rises with latent visits per token. | E7 |

H1 to H3 say recurrence produces stable, compact trajectories. H5 to H7 say it produces opaque ones. They are
not exclusive: a loop can be easier to explain to someone with full access and harder to watch for someone
without it.

## Phases And Gates

| Phase | Work | Gate to proceed |
|---|---|---|
| 0 | E0 | `G0` |
| 1 | E1, E2, E3 on `C3` and `C0` | at least one family shows an arrival-matrix ridge above the `C0` control |
| 2 | E4, then E5, E6 | N1 label is recorded; controls 7 and 8 pass |
| 3 | E7, E8 on fresh seeds and the locked split | registration of H4, H5, H7 before the locked split is read |
| 4 | E9, transfer to `S0` | |

Phases 0 to 2 are exploratory and labeled so. Phase 3 is confirmatory: its config, hypotheses, thresholds,
and dictionary selections are hash-registered after Phase 2, and it uses model seeds that no earlier phase
touched. Every phase ends with a `reports/checkpoint_rmp_*.md`.

Proposed layout: `research_gym/interp/` (`tasks`, `capture`, `dictionaries`, `delta`, `patching`,
`similarity`, `monitorability`), scripts under `research_gym/scripts/bench_rmp_*`, configs under
`configs/rmp_v0*`, and an `interp` optional extra for plotting.

## Claim Boundary

Results will concern models under two million parameters on synthetic tasks with known traces. A positive
result means these monitors recover and causally confirm these variables in these models. It does not show
that large looped language models are monitorable, that the variables found are the only ones computed, or
that a monitor robust here is robust under deployment pressure. The task set is chosen so that ground truth
exists; real reasoning has none, and that gap is the main external-validity limit.

## Literature To Register

`architecture_literature_registry_v1.json` already covers the looped-architecture sources. An RMP registry
should add the interpretability and monitoring sources below. They are listed from memory and have not been
checked against primary sources; verification is part of E0.

- Sparse dictionaries: TopK autoencoders, transcoders, crosscoders, multi-layer residual autoencoders.
- Dictionary failure modes: autoencoders on randomly initialized transformers, feature absorption,
  pathological reconstruction error, supervised-dictionary evaluation.
- Probing: control tasks and selectivity, minimum description length probing.
- Causal methods: interchange interventions and distributed alignment search, activation patching practice,
  the subspace-patching illusion, causal scrubbing.
- Lenses and similarity: logit lens, tuned lens, CKA.
- Latent reasoning: continuous-thought training, recurrent-depth language models and attempts to decode them,
  logical extrapolation with recurrent networks, looped transformers and length generalization.
- Monitoring: chain-of-thought monitorability, chain-of-thought faithfulness tests, obfuscated activations.

## Decisions

Recorded 2026-10-01.

1. Head width is four heads of 32.
2. `TRM2` is deferred to a successor program.
3. The literature list stays unverified until E0, where each source is checked against its primary record
   before the registry is frozen.
4. The LSAD v1 chance-level observation is filed as
   `reports/checkpoint_lsad_architecture_discovery_v1_addendum_1.md`.
