# RMP v0 Phase 2 plan (E4 first)

Status: written 2026-10-12, before any Phase 2 result. Branch `rmp-v0-phase2`. Phase 2 is exploratory.
The program orders Phase 2 as E4, then E5 and E6, and gates it on two things: the N1 label is recorded, and
controls 7 (planted positive) and 8 (absent negative) pass. This plan covers E4. E5 and E6 get their own
plans once N1 is labeled, because an `open_loop_only` outcome changes what they are for.

## Models

The gate-passing models: `LOOP-T` seeds 1, 2, 3 (`C3` of `LOOP-T-core4-ds-s{n}`) and `FF-U` seed 3
(`C3` of `FF-U-core4-ds-inj-s3`, run with its input re-injection). The untied arm has one passing seed of
three; every arm comparison here is one untied model against three tied ones. Random-init controls: the
`C0` of each seed's plain cell, run with the same variant. One planted model for control 7 (below).

## Activations

Generated on the fly from each model during dictionary training, never stored in bulk: states are cheap to
recompute and a store of every position and visit would be tens of gigabytes. Inputs are the base rows of
`monitor_train` for the four core families (20,000 each), all 64 positions. Sites follow the program:
`resid[0]` is the state entering visit 0, `resid[t]` for `t = 1..8` the state after visit `t - 1` (after the
input re-injection, for `FF-U`), and `delta[t] = resid[t+1] - resid[t]` for `t = 0..7`. Every dictionary
reads RMS-normalized states (`x / rms(x)`, the quantity the block's LayerNorm reads); reconstructions are
scaled back by the same norm before they are spliced into the model. Selection and N1 use
`monitor_calibration` and `validation`; neither is ever used for fitting. `locked` stays unread.

## Dictionaries in E4

| Name | Reads | Predicts | Use in E4 |
| --- | --- | --- | --- |
| `D-tok` | `resid[0]` | itself | control 9 (given versus computed) |
| `D-shared` | `resid[1..8]` pooled | itself | N1, controls 1-4, 7, 8, 10 |
| `D-visit[t]` | `resid[t]` | itself | the shared-versus-visit gap, at the selected size only |
| `D-trans` | `resid[t]`, `t = 0..7` | `delta[t]` | N1 replacement model; one for `LOOP-T`, eight (one per layer) for `FF-U` |

`D-cross` and `D-delta` are left to E5, where feature lifetimes and difference cross-checks are used.

Architecture, as the program specifies: TopK sparse autoencoders with unit-norm decoder columns
(renormalized after every step), a pre-encoder bias subtracted before the encoder and added after the
decoder, and an auxiliary dead-latent loss (a latent is dead if it has not fired in the last 2,000 steps;
the auxiliary term reconstructs the residual error from the top `min(256, width / 2)` dead latents, weight
1/32). The transcoder adds a linear skip term from input to output. Training: Adam, learning rate 3e-4,
constant for 80% of steps then linearly to zero, 10,000 steps, 32 examples per step (all positions and
sites, so 16,384 vectors per step for `D-shared`), dictionary seed 0 unless stated.

Grid: width in {1024, 2048, 4096, 8192}, `k` in {8, 16, 32}. Full grid for `D-shared` on all four models,
and for `D-trans` on `LOOP-T` seed 1 and `FF-U` seed 3; `D-trans` on `LOOP-T` seeds 2 and 3 at the size
selected on seed 1 and the adjacent width.

Selection rule (the program's, made concrete): splice a dictionary's reconstruction in at one visit, at all
positions, and run the true block onward. Cross-entropy recovered at that visit is
`(CE_mean - CE_splice) / (CE_mean - CE_clean)`, where `CE_mean` replaces the state with its per-position
mean over `monitor_train` at that visit and `CE` is the answer-position cross-entropy on the selection half
of `monitor_calibration`. The selected configuration is the smallest width, then the smallest `k`, whose
recovery is at least 0.95 at every visit `t = 1..8`. For `D-trans` the splice replaces the update at one
visit by the transcoder's prediction. If no configuration qualifies, the one with the highest minimum over
visits is used and that is stated. Every reported claim is repeated at the adjacent width.

## N1: closed-loop sparse replacement

As the program specifies, per model, on the `validation` split (4,000 examples per family), scored as exact
match on depths 5 to 8 pooled over `pointer_chase`, `gated_pointer_chase` and `reachability`
(`sudoku4` has no depth beyond 3), each relative to the model's own clean exact match:

- open-loop: replace `resid[t]` at all positions by its `D-shared` reconstruction at one visit `t`;
- closed-loop: replace after every visit;
- replacement model: `resid[t+1] = resid[t] + D-trans(resid[t])` at every visit; the block's output is
  discarded;
- noise control: closed-loop, adding to each state isotropic Gaussian noise whose norm equals that state's
  `D-shared` reconstruction error;
- truncation control: closed-loop with the projection onto the top principal components of the `D-shared`
  training states, keeping the number of components whose mean squared error on `monitor_calibration`
  matches the dictionary's.

Labels, as frozen in the program, applied to each model, over every `D-shared` dictionary in the grid
(all have `k <= 32`):

| Label | Rule |
| --- | --- |
| `faithful_state` | Some dictionary keeps closed-loop exact match at 0.90 or more of clean and beats its matched-error truncation control. |
| `open_loop_only` | Every dictionary with open-loop cross-entropy recovered at 0.95 or more at every visit has closed-loop exact match at 0.5 or less of clean, while its noise control stays at 0.9 or more. |
| `fragile_dynamics` | As `open_loop_only`, except that the noise control also falls below 0.9 of clean. |
| `unresolved` | Anything else. |

The Phase 2 gate needs the label recorded for `LOOP-T`; it is reported for every model. A label on a tied
seed with no dictionary reaching the open-loop bar is `unresolved` by these rules, and that is stated.

## Feature-variable association (used by controls 1, 2, 7, 8)

For a binary variable, or each class of a categorical one against the rest: encode the answer-position
state at every site `t = 1..8` with a model's selected `D-shared`; compute the AUROC of each feature's
activation for the variable. Select-then-confirm, to avoid testing every feature on the same data: on the
selection half of `monitor_calibration` pick the single (feature, site) with the largest `|AUROC - 0.5|`;
on the reporting half, the pipeline "finds" the variable if that same feature and site reach
`|AUROC - 0.5| >= 0.35` with a permutation p-value below 0.001 (1,000 label permutations).

## Controls in E4

| # | Control | Procedure | Pass |
| ---: | --- | --- | --- |
| 1 | Random initialization | The full selection and association pipeline on each `C0` model; trace-variable association reported as the margin over `C0`. | reported |
| 2 | Structure-free input | `D-shared` at the selected size trained on Gaussian data with the training states' covariance; the association pipeline run with it on the real states. | reported |
| 3 | Dictionary seeds | `D-shared` at the selected size with seeds 0, 1, 2; a feature is matched across seeds when its best decoder cosine in every other seed is 0.9 or more. | reported |
| 4 | Size | N1 and association results repeated at the adjacent width. | reported |
| 7 | Planted positive | See below. | must pass |
| 8 | Absent negative | The variable is bit 0 of the SHA-256 of `example_id`, which no model can compute. The association pipeline is run for it on every model. | passes if no model's pipeline finds it |
| 9 | Given versus computed | Each `D-shared` feature's maximum decoder cosine to `D-tok`; features at 0.9 or more are counted as given and left out of computed-feature counts. | reported |
| 10 | Visit identity | `D-shared` retrained on visit-mean-centered states; a feature of the uncentered dictionary counts as a visit-identity feature if visit index alone explains 90% or more of its activation variance. | reported |

Controls 5 (supervised ceiling), 6 (causal requirement), 11 (closed-loop faithfulness, which is N1 itself)
and 12 (multiplicity, handled by select-then-confirm above) belong to the labeling and causal work of E5 and
E6.

Planted positive (control 7). A copy of `LOOP-T` seed 1 is fine-tuned from its `C3` weights for 3,000 steps
with the stage-2 recipe (deep supervision, batch 256 per family, learning rate 0.001) plus an auxiliary loss
that writes a chosen variable into a chosen direction: `BCE(sigmoid(4 * <x_t, u>), v)` on the
RMS-normalized answer-position state `x_t` after every visit, where `u` is a fixed random unit vector
(seed 7) and `v` is the parity of the input token at position 4. (The first draft said position 3; a check
of the data, before any run, showed position 3 holds the same token in every example of the three graph
families, so its parity is constant. Position 4 varies in all four families: 49 to 68% odd in the graph
families, 24% in `sudoku4`.) The variable is given by the input and
irrelevant to every task, so the parent model has no reason to align anything with `u`. The planted model's
task accuracy on `validation` is reported. Control 7 passes if, on the planted model, the association
pipeline finds `v` and the found feature's decoder has cosine 0.5 or more with `u`, while on the parent
model no feature selected for `v` has decoder cosine 0.5 or more with `u`. Cosines are taken in absolute
value (settled before any run): TopK features are non-negative, so the variable can be carried by a feature
along `u` that fires for odd tokens or one along `-u` that fires for even ones, and either has found the
planted direction. The planted model's `D-shared` is trained at the size selected for its parent.

## Order

1. Dictionary and replacement code with tests.
2. `D-shared` grid and selection on the four models; `D-trans` grid; `D-tok`; `D-visit` at the selected
   size.
3. N1 on all four models, with both controls and the adjacent width. Record the label.
4. Planted model; controls 7 and 8; then 1, 2, 3, 9, 10.
5. `reports/checkpoint_rmp_phase2_e4.md`.

GPU jobs run one at a time through `vramorama run`; heavy artifacts stay under
`C:\Users\patri\Documents\Codex\rmp_v0\phase2`; commits at each milestone, never pushed.

## Details settled on 2026-10-25, before any run of controls 1, 2, 3, 9, 10 or the trace association

The plan above leaves these open; they were fixed before any of these analyses ran.

- **Trace variables.** The association pipeline is run for Phase 1's E2 variables, each at every algorithmic
  step it has, on the examples deep enough to have that step:
  - `cur_node(s)` in both pointer families;
  - `latch(s)` in `gated_pointer_chase`;
  - `goal_found(s)` in `reachability`;
  - the queried cell's candidates after `s` rounds in `sudoku4`.

  A binary variable is one test. A categorical one is one test per class against the rest, for classes with
  at least 20 examples on each side in both halves of `monitor_calibration`.
- **Control 1.** Reported per variable as the share of tests found, trained model minus its `C0`, each with
  its own selected dictionary.
- **Control 2.** The Gaussian data has the mean and covariance of the RMS-normalized `resid[1..8]` states of
  400 `monitor_train` examples, at all positions. It is drawn fresh at each step, 16,384 vectors per step as
  for `D-shared`. Each dictionary is trained at its model's selected size. The association pipeline is then
  run with it on the real states.
- **Controls 3 and 9.** Decoder cosines are signed, since TopK features are non-negative and a feature
  pointing the opposite way is a different feature. Counts are given over the features that fire at least
  once on calibration states, and over the features the association pipeline found. Features that never
  fire stay at their random initialization.
- **Control 10.**
  - Centering subtracts each visit's mean RMS-normalized state, taken from 400 `monitor_train` examples at
    all positions.
  - The share of activation variance explained by visit index is computed over 300 examples from the
    selection half of `monitor_calibration`, at all positions and visits 1 to 8.
  - The centered dictionary is reported by its FVU and its count of visit-identity features.
- **Shared versus visit gap.** Cross-entropy recovered at visit `t`, comparing `D-visit[t]` with `D-shared`
  spliced at that same visit.
- **Adjacent width.** The next larger width in the grid; for width 8192, the next smaller.
- **`D-trans` on `LOOP-T` seeds 2 and 3.** Trained at the size selected for seed 1's `D-trans` and at its
  adjacent width.
