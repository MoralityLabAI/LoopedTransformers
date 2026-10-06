# RMP v0 Phase 1 plan

Status: written 2026-10-06, before any Phase 1 result was computed. Branch `rmp-v0-phase1`.
Phase 1 is exploratory (program document, "Phases And Gates").

## Decision this plan follows

Gate `G0` was not met: `LOOP-T` passed with three seeds, `FF-U` failed on seed 1 at 0.960 on
`reachability` (`reports/checkpoint_rmp_phase0.md`). On 2026-10-06 the program owner chose to run Phase 1
on the three `LOOP-T` models and to keep `FF-U` seed 1 as a comparator marked below the gate. Its 0.960 is
above the program's 0.90 floor for interpretability claims.

The program's fairness rule applies to every arm comparison: "If one arm cannot reach it, the stronger arm
is compared at an earlier checkpoint with matched accuracy." Phase 1 therefore adds one accuracy-matched
`LOOP-T` checkpoint, chosen by the rule below before any Phase 1 measurement.

## Models

| Role | Model | Checkpoint |
| --- | --- | --- |
| primary | `LOOP-T` seeds 1, 2, 3 | `C3.pt` of `LOOP-T-core4-ds-s{1,2,3}` |
| comparator, below the gate | `FF-U` seed 1 | `latest.pt` of `FF-U-core4-ds-inj-s1` (step 40,000), run with its input re-injection |
| accuracy-matched tied comparator | `LOOP-T`, one checkpoint | selection rule below |
| random-initialization control | one per model above | `C0.pt` of the seed's plain cell (`LOOP-T-core4-s{n}`, `FF-U-core4-s1`) |
| permuted-target control | one per arm | new; see E2 |

Matched-checkpoint rule. Candidates are every retained checkpoint of the `LOOP-T` gate cells (named, log
grid, and final) of seeds 1 to 3. Each is evaluated on the full validation split. A candidate qualifies if
its pooled `reachability` is within 0.010 of 0.960 and every other core family is at 0.98 or more. Among
qualifying candidates, one from a deep-supervision stage is preferred (its training is closest to the
`FF-U` model's), then the smallest `reachability` distance, then the lowest seed. If none qualifies, the
comparison is reported without a matched checkpoint and that is stated.

## Data

The monitor splits are regenerated from the bundle config into `C:\Users\patri\Documents\Codex\rmp_v0\bundle`
(outside OneDrive). The rebuild reproduced the Phase 0 bundle hash
(`f1d81a3a...`) and every shard hash. `monitor_train` fits probes and similarity estimates;
`monitor_calibration` supplies minimal pairs and is split by `group_id` into a selection half and a
reporting half. The `locked` split is not written and not read: the program reserves it for Phase 3, after
registration. This departs from "reporting on locked" in the program's probe section.

## E1: arrival matrices

Pairs: `edge_rewire` siblings in `monitor_calibration` whose answer differs from their base. The sibling
records the hop `s` at which the rewired edge is first used. Families: `pointer_chase` (4,912 pairs),
`gated_pointer_chase` (2,797), `reachability` (1,501). `sudoku4` has no step-indexed sibling and gets no
arrival matrix.

Measurement, per model and family, on the pairs the model answers correctly in both members:

- Run the sibling (corrupt) and replace the residual state at the answer position after visit `t` with the
  base (clean) run's state at the same visit and position (denoising). `t` runs over visits 0 to 6; after
  visit 7 the replacement fixes the logits, so that column is 1 by construction and is excluded.
- Score: `r = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)`, with `LD` the logit of the base answer
  minus the logit of the sibling answer. Pairs with `|LD_clean - LD_corrupt| < 0.05` are dropped and counted.
- Arrival matrix: mean `r` by divergence hop `s` (rows) and visit `t` (columns), also split by depth.
- Arrival visit `a` of a pair: the first `t` with `r >= 0.5`, or 7 if none.

Ridge, declared now: `rho` is the mean over depths of the within-depth Spearman correlation between `s` and
`a` (depths with at least 20 pairs and three distinct `s`), weighted by pair count. A family has a ridge in
a model when `rho > 0`, a permutation test that shuffles `s` within depth (1,000 permutations) gives
`p < 0.01`, and the bootstrap 95% interval (1,000 resamples of pairs) of `rho` minus the matched `C0`
model's `rho` excludes zero. The `C0` model is scored on the same pairs.

Phase 1 gate (program: "at least one family shows an arrival-matrix ridge above the `C0` control"): met if
one family has a ridge in all three `LOOP-T` seeds.

Changes after the first E1 run (2026-10-06). The first run had covered the three `LOOP-T` seeds when it
stopped on a degenerate case, so these changes were made after seeing those partial results:

- The `C0` control of a model that runs with input re-injection now runs with it too (affects `FF-U` only).
  The first run was stopped and restarted for this before any `FF-U` result existed.
- As first written, the model's `rho` used only pairs on which the `C0` model also had a logit gap of at
  least 0.05. Random-init gaps are mostly smaller, so this discarded most pairs and made the model's own
  statistic depend on its control. The model's `rho`, its permutation test, and the paired interval now use
  every pair the model keeps; the `C0` model is scored on those same pairs with no gap filter, and the
  number of its gaps under 0.05 is reported.
- A depth whose arrival visit is the same for every pair counts as `rho = 0` with its pair weight, rather
  than being skipped; skipping could leave no depth at all. This can only lower a trained model's `rho`.

Secondary, on 200 pairs per family: the same patch at every position and visit, to show where step `s` is
held before it reaches the answer position.

## E2: trace probes, transfer, description length

Multiclass logistic probes of each trace variable at its algorithmic step (`cur_node(s)`, `remaining(s)`,
`latch(s)`, reachability `goal_found(s)` and frontier membership, Sudoku candidate sets), read from
`resid[t]` at the answer position for every visit `t`. Fit on `monitor_train`, regularization chosen on the
selection half of `monitor_calibration`, reported on the reporting half. Group-disjoint throughout.
Baselines at the same site: `resid[0]`, the `C0` model, the permuted-target model, a within-group
shuffled-label floor, and a control task. A variable counts as computed at visit `t` only if it clears all
of them. Cross-visit transfer matrix and online description length over eight portions. A one-hidden-layer
probe beside each linear one.

Permuted-target control: one model per arm, seed 1, the stage-1 recipe (beta 1.0, batch 256 per family)
without the depth curriculum or the zero-hop pool, trained on the stage-1 training pool with each family's
targets permuted across examples, for 20,000 steps. The curriculum is dropped because it unlocks depths on
validation accuracy, which a model trained on shuffled labels never reaches. It can only memorize. The
budget is a choice, recorded as a departure; the program gives none.

E2 definitions, fixed on 2026-10-06 after the E1 runs had started and before any probe was fit:

- Variables, each probed at every algorithmic step `s` it has, on examples deep enough to have it:
  `cur_node(s)` for `s >= 1` in both pointer families (16 classes), `latch(s)` in `gated_pointer_chase`,
  `goal_found(s)` in `reachability`, and the candidate set of the queried cell after `s` rounds in
  `sudoku4`. The program counter `remaining(s)` is left to Phase 2: a probe for it needs a visit-to-step
  alignment, which E1 is meant to supply.
- Sites: the answer position's state entering visit 0 and after each visit, RMS-normalized.
- Probe: multinomial logistic regression, full batch, weight decay chosen from {1e-4, 1e-3, 1e-2} per
  probe on the selection half; a one-hidden-layer probe (256 units) beside it.
- Baselines at the same site: the `C0` model, the permuted-target model, and labels shuffled within depth
  (the floor). The program's `resid[0]` baseline is taken as the mean over all positions of the state
  entering visit 0, a bag of input tokens: at the answer position alone it holds no input at all.
- Control task: the same probe on a fixed random relabeling of the family's given variable (the start node
  in the graph families, the queried cell in `sudoku4`), with as many classes as the probed variable.
- A variable counts as computed at visit `t` if its accuracy on the reporting half exceeds every baseline
  and the control task at that site by at least 0.05.
- Transfer: the probe fit at site `i`, scored at site `j`. Description length: online code over eight
  portions (1/128 to all of `monitor_train`), divided by the label entropy.

## E3: similarity and Jacobian analyses

Debiased linear CKA, visit by visit within each model, across `LOOP-T` seeds, and `LOOP-T` visit against
`FF-U` layer with a monotone alignment path; at least 4,096 examples per estimate, from `monitor_train`.
Orthogonal Procrustes between `resid[t]` and `resid[t+1]`. Jacobian spectrum of the tied block at late
visits by power iteration. States are RMS-normalized; raw norms are kept as a convergence signal. Nulls:
example permutation, the `C0` model, and visit-mean-centered against uncentered states.

## Order and constraints

1. Matched-checkpoint selection.
2. E1 and its gate check, then commit.
3. Permuted-target controls trained on the GPU while E1 and E2 code is written.
4. E2, then E3, each committed.
5. `reports/checkpoint_rmp_phase1.md`: what was run, what was only written, every departure.

Train only on mains power; never stop another session's process; install nothing; read-only use of the
`bluebeam` interpreter; heavy artifacts outside OneDrive; commit at milestones, never push. Out of scope:
dictionaries and everything from E4 on, the `COT` arms, and the `locked` split.
