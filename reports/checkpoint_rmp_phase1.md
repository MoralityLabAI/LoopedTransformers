# Checkpoint: RMP v0 Phase 1 (2026-10-07; untied seed 3 added 2026-10-12)

Phase 1 is exploratory. The plan, with every rule fixed before the measurement it governs and every later
change recorded with its reason, is `docs/rmp_phase1_plan.md`. Results are under
`experiments/recurrence_monitorability_v0/phase1/`.

## Outcome

The Phase 1 gate is met. The program asks for "at least one family [that] shows an arrival-matrix ridge
above the `C0` control"; every one of the three `LOOP-T` seeds shows such a ridge in all three families
that have step-indexed minimal pairs (`pointer_chase`, `gated_pointer_chase`, `reachability`).

In the tied loop, the effect of an edit at hop `s` reaches the answer position later the larger `s` is:
about one hop per visit for the first two or three hops, then two to three hops per visit, with hops 5 to
8 arriving within about one visit of each other.

The untied arm was first represented by seed 1, which never reached the gate (0.960 on `reachability`).
It ordered the hops in the two pointer families and not in `reachability`. Untied seed 3, the first untied
model to pass the gate (added 2026-10-12, after training finished), has a `reachability` ridge as well
(`rho` 0.37) and a pointer schedule close to tied seed 1's. The missing `reachability` ridge was therefore
a property of the weaker seed, not of the untied arm. What still separates the arms is trainability: every
tied seed passed the gate in its second stage, while two of three untied seeds failed after the third and
the one that passed did so by a single example.

The probes (E2) give the same schedule from the other side. In tied seed 1, the BFS flag `goal_found(s)`
and the gate latch `latch(s)` each reach their plateau right after visit `s`, and in seeds 2 and 3 within
a visit of that. In untied seed 1 every `goal_found(s)` first clears its baselines at the same layer; in
untied seed 3 they do so in a loose order (layers 4 to 6). The intermediate node of a pointer chase is
transient in the tied loop and only partly linear. Similarity measures (E3) separate little:
answer-position CKA is about as high between arms as between tied seeds, and random networks reach similar
values. What does separate is how much each visit changes the state: trained models rewrite it mostly in
early visits, random ones barely at all.

Every result here is exploratory. The untied arm has one gate-passing seed and one below-gate seed in
these analyses. Nothing here is a registered claim, and no number was read from the `locked` split.

## Models

| Role | Model | Checkpoint | Full-validation accuracy (pointer, gated, reach, sudoku) |
| --- | --- | --- | --- |
| primary | `LOOP-T` seed 1 | `LOOP-T-core4-ds-s1` `C3`, step 8,500 | 0.995, 0.997, 0.987, 0.999 |
| primary | `LOOP-T` seed 2 | `LOOP-T-core4-ds-s2` `C3`, step 8,750 | 0.999, 0.998, 0.980, 1.000 |
| primary | `LOOP-T` seed 3 | `LOOP-T-core4-ds-s3` `C3`, step 23,250 | 0.999, 0.999, 0.982, 0.999 |
| comparator, below the gate | `FF-U` seed 1 | `FF-U-core4-ds-inj-s1` final, step 40,000, with input re-injection | 0.998, 0.997, 0.960, 0.999 |
| comparator, at the gate (added 2026-10-12) | `FF-U` seed 3 | `FF-U-core4-ds-inj-s3` `C3`, step 23,250, with input re-injection | 1.000, 0.996, 0.980, 1.000 |
| accuracy-matched tied | `LOOP-T` seed 1 | `LOOP-T-core4-ds-s1` `C2_7`, step 5,000 | 0.987, 0.989, 0.960, 1.000 |
| random-init controls | each seed | `C0` of the seed's plain cell | chance |

The matched checkpoint was chosen by the rule in the plan, from 123 retained `LOOP-T` checkpoints, of which
24 qualified (`phase1/matched_checkpoint.json`). Its `reachability`, 0.9595, is within 0.001 of the
untied model's 0.9603. Its training differs from the untied model's in one respect that matters: the untied
model's last stage added input re-injection.

## E1: arrival matrices

Pairs are `edge_rewire` siblings in `monitor_calibration` whose answer differs from their base, kept when
the model answers both members correctly. The clean state at the answer position after visit `t` replaces
the corrupt one; the score is normalized logit-difference recovery. `rho` is the pair-weighted mean over
depths of the within-depth Spearman correlation between the edited hop `s` and the first visit at which
recovery reaches 0.5.

| Model | `pointer_chase` | `gated_pointer_chase` | `reachability` |
| --- | --- | --- | --- |
| `LOOP-T` s1 | 0.43 (C0 0.02) | 0.20 (C0 0.00) | 0.53 (C0 -0.06) |
| `LOOP-T` s2 | 0.47 (C0 -0.00) | 0.45 (C0 0.02) | 0.38 (C0 0.02) |
| `LOOP-T` s3 | 0.90 (C0 0.02) | 0.64 (C0 -0.00) | 0.45 (C0 -0.02) |
| `FF-U` s1, below gate | 0.52 (C0 0.06) | 0.35 (C0 0.04) | -0.12 (C0 0.06), no ridge |
| `FF-U` s3, at the gate | 0.88 (C0 -0.02) | 0.39 (C0 0.01) | 0.37 (C0 -0.03) |
| `LOOP-T` s1, matched | 0.32 (C0 0.02) | 0.11 (C0 0.00) | 0.45 (C0 -0.07) |

Every entry marked with a ridge has a permutation p-value of 0.001 (the smallest 1,000 permutations can
give) and a bootstrap 95% interval for `rho` minus the control's that excludes zero
(`phase1/e1_arrival.json`). Pairs kept: 4,793 to 4,906 (`pointer_chase`), 2,695 to 2,785 (gated), 1,199 to
1,330 (`reachability`).

Mean recovery for `LOOP-T` seed 1 on `pointer_chase`, by edited hop (rows) and visit (columns 0 to 7; the
last column is 1 by construction):

| Hop | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 0.62 | 0.75 | 0.81 | 0.89 | 0.95 | 0.99 | 1.00 |
| 2 | 0.00 | 0.00 | 0.54 | 0.70 | 0.82 | 0.93 | 0.98 | 1.00 |
| 3 | 0.00 | 0.00 | 0.06 | 0.56 | 0.75 | 0.90 | 0.97 | 1.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.26 | 0.65 | 0.87 | 0.96 | 1.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.13 | 0.48 | 0.77 | 0.94 | 1.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.02 | 0.38 | 0.68 | 0.90 | 1.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.66 | 0.85 | 1.00 |
| 8 | 0.00 | 0.00 | 0.00 | -0.01 | 0.03 | 0.48 | 0.91 | 1.00 |

Mean arrival visit by hop, `pointer_chase`:

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LOOP-T` s1 | 1.82 | 2.92 | 3.69 | 4.11 | 4.47 | 4.82 | 5.15 | 5.51 |
| `LOOP-T` s2 | 1.47 | 2.22 | 2.65 | 3.08 | 3.48 | 3.79 | 4.16 | 4.50 |
| `LOOP-T` s3 | 1.05 | 2.06 | 2.81 | 3.29 | 3.89 | 4.31 | 4.69 | 5.07 |
| `FF-U` s1 | 3.23 | 4.19 | 4.39 | 4.90 | 5.27 | 5.39 | 5.51 | 5.69 |
| `FF-U` s3 | 2.04 | 3.00 | 3.80 | 4.13 | 4.25 | 4.46 | 4.96 | 4.97 |
| `LOOP-T` s1 matched | 2.40 | 3.27 | 4.15 | 4.65 | 5.04 | 5.20 | 5.38 | 5.88 |

Reading:

- The tied loop processes the first two or three hops at about one per visit (mean arrival steps of 0.75
  to 1.1 visits), then several hops per visit: hops 5 to 8 arrive within about one visit of each other.
  That is neither a purely sequential schedule (slope 1 to the end) nor a doubling one.
- The untied stack has hop 1 at the answer position from layer 2 or 3 (recovery 0.38, then 0.51) and
  almost nothing about hops 2 to 8 before layer 4. At layer 4 hops 2 and 3 arrive together (0.69, 0.57)
  and the rest follow over layers 5 and 6. In `reachability` its arrival rises with the hop to hop 4 and
  falls after it (mean arrival 2.6, 3.5, 3.6, 3.6, 3.4, 3.3, 2.6, 2.3), so it has no step-ordered arrival
  there.
- Untied seed 3, at the gate, looks much more like the tied loop. Its `pointer_chase` arrival (2.0, 3.0,
  3.8, 4.1, then 4.3 to 5.0) is within about 0.2 of a layer of tied seed 1's for the first four hops, and
  its `reachability` arrival rises with the hop through hop 5 (2.9, 4.0, 4.8, 5.4, 6.0) before falling at
  hops 7 and 8 (5.5, 4.1). The tied seeds fall at the last hops too (seed 1: 6.4 then 4.7; seed 2: 5.8,
  5.2, 2.7), so that tail is a property of the late `reachability` edits rather than of either arm.
- The matched tied checkpoint keeps its `reachability` ridge (0.45) at untied seed 1's accuracy, and untied
  seed 3 has one at the gate. Untied seed 1's missing ridge goes with that seed, which also failed the
  gate, and not with the untied architecture.

Limits of the control. A random-init model has almost no logit gap between the two answers of a pair: the
median is 0.01 nats (0.000 for the untied model), and 78% to 100% of pairs are under 0.05. Its normalized
recovery is a ratio of small numbers, so its `rho` near zero shows that patching an untrained network does
not produce a step order, but it is a noisy control. The permutation test, which needs no control model, is
the stronger evidence here.

Where the step is held. The same patch at every one of the 64 positions (200 pairs per family, stored in
`e1_arrival.json`) restores the answer almost only at the answer position. In `pointer_chase` the only
other position with mean recovery of 0.2 or more is the start-node token: at visit 2 in `LOOP-T` seed 1
(0.20) and at layer 3 in `FF-U` seed 1 (0.27); in `FF-U` seed 3 no other position reaches 0.2 in any
family. In `reachability` the tied model also carries part of the edit at
its start-node token (position 52) during visits 0 to 4 (0.39 falling to 0.21). The edge tokens themselves
never carry it. Both arms gather the chase at the answer position rather than spreading
intermediate results over the edge list, which fits the Phase 0 reading that the search runs at the last
few positions.

## E2: trace probes

Linear probes at the answer position, fit on `monitor_train` (20,000 examples per family), weight decay
chosen on the selection half of `monitor_calibration`, scored on its reporting half
(`phase1/e2_probes.json`; tables from `python -m research_gym.scripts.report_rmp_phase1`). Site 0 is the
state entering visit 0; site `k` is the state after visit `k - 1`. Probes whose label is constant on the
reporting half (`latch`, `goal_found`, and the Sudoku candidate set at step 0) are left out.

Where each step lives: the site at which `cur_node(s)` peaks, and the first site within 0.01 of the peak for
the monotone flags (peak accuracy in brackets):

| Model | Variable | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 | s=7 | s=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `LOOP-T` s1 | pointer `cur_node` | 3 (0.98) | 4 (0.77) | 4 (0.50) | 6 (0.52) | 7 (0.59) | 8 (0.56) | 6 (0.69) | 8 (0.95) |
| `LOOP-T` s1 | gated `latch` | 3 (0.98) | 3 (1.00) | 4 (1.00) | 5 (1.00) | 6 (0.99) | 7 (0.99) | 8 (0.99) | 8 (0.99) |
| `LOOP-T` s1 | reach `goal_found` | 2 (1.00) | 3 (1.00) | 4 (1.00) | 5 (1.00) | 6 (1.00) | 7 (0.99) | 8 (0.99) | 8 (0.93) |
| `LOOP-T` s2 | pointer `cur_node` | 2 (1.00) | 3 (0.76) | 3 (0.59) | 4 (0.51) | 5 (0.47) | 7 (0.49) | 8 (0.59) | 8 (0.99) |
| `LOOP-T` s2 | gated `latch` | 2 (1.00) | 3 (0.99) | 4 (0.97) | 5 (0.98) | 5 (0.97) | 5 (0.97) | 6 (0.98) | 8 (0.99) |
| `LOOP-T` s2 | reach `goal_found` | 2 (1.00) | 4 (1.00) | 4 (0.99) | 5 (0.99) | 6 (0.99) | 7 (0.98) | 7 (0.97) | 8 (0.96) |
| `LOOP-T` s3 | pointer `cur_node` | 2 (1.00) | 3 (1.00) | 4 (0.80) | 5 (0.59) | 5 (0.59) | 6 (0.62) | 7 (0.69) | 8 (0.99) |
| `LOOP-T` s3 | gated `latch` | 2 (1.00) | 3 (1.00) | 6 (0.99) | 4 (0.98) | 5 (0.98) | 5 (0.98) | 7 (0.99) | 7 (1.00) |
| `LOOP-T` s3 | reach `goal_found` | 3 (0.99) | 4 (0.99) | 5 (0.98) | 6 (0.98) | 7 (0.98) | 8 (0.98) | 8 (0.97) | 8 (0.93) |
| `FF-U` s1 | pointer `cur_node` | 3 (1.00) | 5 (0.80) | 5 (0.62) | 5 (0.58) | 7 (0.56) | 6 (0.54) | 6 (0.59) | 8 (0.99) |
| `FF-U` s1 | gated `latch` | 3 (1.00) | 4 (1.00) | 4 (0.98) | 7 (0.96) | 6 (0.96) | 6 (0.97) | 7 (0.98) | 8 (0.98) |
| `FF-U` s1 | reach `goal_found` | 4 (1.00) | 4 (0.98) | 4 (0.97) | 7 (0.96) | 7 (0.96) | 8 (0.96) | 6 (0.94) | 6 (0.89) |
| `FF-U` s3 | pointer `cur_node` | 3 (1.00) | 4 (1.00) | 4 (0.93) | 8 (0.49) | 8 (0.41) | 5 (0.57) | 8 (0.57) | 8 (1.00) |
| `FF-U` s3 | gated `latch` | 3 (1.00) | 4 (1.00) | 5 (1.00) | 5 (0.97) | 7 (0.98) | 8 (0.98) | 8 (0.96) | 8 (0.98) |
| `FF-U` s3 | reach `goal_found` | 4 (0.99) | 3 (0.99) | 4 (0.99) | 7 (0.98) | 7 (0.98) | 8 (0.95) | 8 (0.95) | 7 (0.91) |
| `LOOP-T` s1 matched | pointer `cur_node` | 3 (0.99) | 4 (0.90) | 5 (0.58) | 6 (0.58) | 7 (0.71) | 8 (0.62) | 6 (0.64) | 8 (0.91) |
| `LOOP-T` s1 matched | reach `goal_found` | 2 (1.00) | 3 (1.00) | 4 (1.00) | 5 (1.00) | 6 (0.99) | 7 (0.99) | 8 (0.99) | 7 (0.86) |

Reading:

- In the tied loop the BFS flag and the latch settle one step per visit: `goal_found(s)` reaches its plateau
  after visit `s` for `s` = 1 to 7 in seed 1, and within one visit of that in seeds 2 and 3. The arrival
  ridge of E1 and this staircase are two views of the same schedule.
- Untied seed 1 has no such order in `reachability` (sites 4, 4, 4, 7, 7, 8, 6, 6), again matching E1,
  while the accuracy-matched tied checkpoint keeps the staircase. Untied seed 3, at the gate, has a coarse
  one: steps 1 to 3 settle by layer 3 or 4 and steps 4 to 8 by layers 7 and 8, ordered in groups rather
  than one step per layer. In the pointer families every model orders the hops; the untied ones start
  later (first hop at site 2 or 3 in the tied loop and 3 in both untied seeds; second hop at site 3 or 4
  in the tied loop, 5 in untied seed 1 and 4 in untied seed 3).
- The current node is transient in the tied loop: `cur_node(1)` peaks at 0.94 to 1.00 and falls to 0.29
  to 0.55 by the last site. The untied seeds keep more of it at their last layer (0.84 and 0.71 in seed 1,
  0.61 and 0.62 in seed 3). The middle hops (3 to 7) peak at 0.28 to 0.93 linearly across the models,
  most of them well under 0.8. A one-hidden-layer probe reads more of them (for `LOOP-T` s1
  `cur_node(3)`: 0.72 against 0.50 linear), so some of the intermediate state is present but not linearly
  accessible. The final node in `pointer_chase`, `cur_node(8)` on depth-8 examples, is read at 0.91 to 1.00
  at the last site.
- The bag of input tokens and the `C0` models stay near chance for every `cur_node(s)` (0.05 to 0.12,
  chance 0.0625).
- The control task relabels the start node, which the answer position reads early: it scores 0.98 to 1.00
  at sites 1 to 3 and fades after. Under the rule fixed in the plan, `cur_node(s)` therefore counts as
  computed only at the sites where it beats start-node decodability by 0.05, which excludes the early sites
  for the first hops. The rule is conservative by design.

Computed, by the plan's rule (accuracy at least 0.05 above the shuffled floor, the control task, the bag of
input tokens, the `C0` model, and the permuted-target model at the same site). First site at which each step
counts as computed:

| Model | pointer `cur_node` s=1..8 | gated `latch` s=1..8 | reach `goal_found` s=1..8 |
| --- | --- | --- | --- |
| `LOOP-T` s1 | 3 4 5 6 5 6 6 6 | 2 3 3 3 3 4 4 4 | 2 2 3 3 3 3 3 3 |
| `LOOP-T` s2 | 3 - 8 - - 6 6 5 | 3 3 3 4 4 4 4 4 | 2 3 3 4 4 4 4 4 |
| `LOOP-T` s3 | 3 3 4 5 5 6 5 5 | 2 3 3 3 3 4 4 4 | 3 4 4 4 4 5 5 5 |
| `LOOP-T` s1 matched | 3 4 5 6 6 6 6 6 | 3 3 3 4 4 5 5 5 | 2 2 2 3 3 3 3 4 |
| `FF-U` s1 | 4 5 6 6 6 6 6 6 | 4 4 5 5 5 5 5 5 | 5 5 5 5 5 5 5 5 |
| `FF-U` s3 | 4 4 4 5 6 5 6 5 | 3 4 4 4 5 5 5 5 | 4 4 4 5 5 5 5 6 |

A dash means the step never clears every baseline. In `LOOP-T` seed 2, hops 2, 4 and 5 of `pointer_chase`
stay under the start-node control at every site, so they never count, although their own accuracy peaks
at 0.47 to 0.76; hop 3 counts only at the last site. The flags rise earlier than the plateaus in the table
above because they need only a 0.05 margin over the class prior, not a plateau. Both permuted-target models
sit at chance or at the class prior: the tied one is the highest baseline at 11 of the 1,368 probe sites of
the tied models, the untied one at none of untied seed 1's 342 and 1 of seed 3's 342. The control task sets
the margin at the early sites.

`sudoku4` never counts as computed, and under this rule it cannot: its control task relabels the queried
cell, which is an input token read at 1.00 at every site, so no variable can exceed it by 0.05. This is a
property of the rule applied to that family, not a finding about the models; the queried cell's candidate
set itself is read at 0.66 to 1.00 from site 2 on in every tied model, and at 0.76 to 1.00 in untied seed 3.

The per-probe tables, with the baseline at the peak, the MLP probe and the online code ratio, are printed by
the script above.

## E3: similarity and Jacobians

4,096 `monitor_train` examples per family, answer position, RMS-normalized
(`phase1/e3_similarity.json`). The state entering visit 0 at the answer position is the same for every
example (it holds only the answer marker and its position), so its CKA and Procrustes entries are 0 or
undefined and are left out below.

Debiased linear CKA, mean along the best monotone alignment between the two models' nine sites:

| Pair | pointer | gated | reach | sudoku |
| --- | ---: | ---: | ---: | ---: |
| `LOOP-T` s1 and s2 | 0.68 | 0.69 | 0.65 | 0.64 |
| `LOOP-T` s1 and s3 | 0.69 | 0.68 | 0.57 | 0.71 |
| `LOOP-T` s2 and s3 | 0.71 | 0.71 | 0.70 | 0.63 |
| `LOOP-T` s1 and `FF-U` s1 | 0.62 | 0.64 | 0.57 | 0.66 |
| `LOOP-T` s2 and `FF-U` s1 | 0.62 | 0.65 | 0.65 | 0.56 |
| `LOOP-T` s3 and `FF-U` s1 | 0.61 | 0.65 | 0.70 | 0.59 |
| `LOOP-T` s1 matched and `FF-U` s1 | 0.62 | 0.64 | 0.59 | 0.66 |
| `LOOP-T` s1 and `FF-U` s3 | 0.61 | 0.68 | 0.64 | 0.71 |
| `LOOP-T` s2 and `FF-U` s3 | 0.61 | 0.67 | 0.67 | 0.62 |
| `LOOP-T` s3 and `FF-U` s3 | 0.62 | 0.65 | 0.69 | 0.71 |
| `LOOP-T` s1 matched and `FF-U` s3 | 0.63 | 0.69 | 0.63 | 0.69 |
| `FF-U` s1 and `FF-U` s3 | 0.62 | 0.67 | 0.69 | 0.60 |
| `LOOP-T` s1 `C0` and `FF-U` s1 `C0` | 0.41 | 0.76 | 0.58 | 0.74 |
| `LOOP-T` s3 `C0` and `FF-U` s3 `C0` | 0.50 | 0.79 | 0.63 | 0.78 |
| `FF-U` s1 `C0` and `FF-U` s3 `C0` | 0.60 | 0.82 | 0.66 | 0.81 |

At the answer position, CKA separates little: tied seeds resemble one another about as much as they
resemble either untied seed, the two untied seeds resemble each other no more than that, and pairs of
random-init networks reach 0.41 to 0.82 on the same inputs. The example permutation null is at most 0.001
in every model.

Procrustes similarity between consecutive visits (1 = the next state is a rotation of this one),
`pointer_chase`, visits 1-2 to 7-8:

| Model | | | | | | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LOOP-T` s1 | 0.75 | 0.79 | 0.82 | 0.85 | 0.91 | 0.96 | 0.98 |
| `LOOP-T` s2 | 0.67 | 0.68 | 0.78 | 0.86 | 0.92 | 0.97 | 0.99 |
| `LOOP-T` s3 | 0.62 | 0.66 | 0.71 | 0.80 | 0.90 | 0.93 | 0.97 |
| `FF-U` s1 | 0.74 | 0.75 | 0.86 | 0.64 | 0.84 | 0.93 | 0.96 |
| `FF-U` s3 | 0.80 | 0.71 | 0.76 | 0.71 | 0.88 | 0.95 | 0.98 |
| `LOOP-T` s1 `C0` | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 |

Trained models write most new content at the answer position in early visits and little in the last two;
a random-init loop barely changes it at all.

Top singular value of each visit's block map, median over 16 examples (whole-sequence state):

| Model | v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LOOP-T` s1 | 37.1 | 19.7 | 9.7 | 8.4 | 7.3 | 6.1 | 5.2 | 4.6 |
| `LOOP-T` s2 | 37.5 | 15.2 | 12.5 | 9.5 | 7.1 | 6.8 | 5.7 | 3.1 |
| `LOOP-T` s3 | 45.7 | 18.2 | 11.6 | 8.4 | 6.5 | 6.0 | 4.8 | 3.8 |
| `FF-U` s1 | 78.4 | 11.9 | 9.8 | 9.8 | 7.4 | 7.2 | 5.1 | 4.6 |
| `FF-U` s3 | 63.9 | 9.5 | 12.3 | 10.6 | 10.7 | 7.7 | 7.3 | 5.8 |
| `LOOP-T` s1 `C0` | 1.9 | 1.9 | 1.8 | 1.7 | 1.6 | 1.5 | 1.5 | 1.4 |
| `FF-U` s3 `C0` | 1.9 | 1.4 | 1.2 | 1.2 | 1.1 | 1.1 | 1.1 | 1.0 |

Trained blocks amplify some direction at every visit, most at the first, so nothing here looks like
convergence to a fixed point within eight visits. This is measured on raw states, and the block normalizes
its input, so the gain falls as the state norm grows (mean answer-position norm rises from about 25 to 50
in `LOOP-T` s1, from 15 to 85 in `FF-U` s1 and from 20 to 121 in `FF-U` s3). The values in the table were
recomputed on 2026-10-12 when `FF-U` s3 was added; they match the first run to within 0.1. A norm-corrected
measure is left for Phase 2.

## Departures from the program and the plan

| # | Specified | Done | Reason |
| ---: | --- | --- | --- |
| 1 | Phase 1 runs on models that passed `G0` in both arms | `FF-U` seed 1 is included below the gate; an accuracy-matched `LOOP-T` checkpoint is added; from 2026-10-12 `FF-U` seed 3, which passed the gate after training seeds 2 and 3 on request, is added beside it | `G0` was not met; the program owner chose this on 2026-10-06, and the program's fairness rule asks for the matched checkpoint. The untied arm still has only one passing seed of three. |
| 2 | Reporting on `locked` | Reporting on the group-disjoint half of `monitor_calibration`; `locked` not written or read | The program reserves `locked` for Phase 3, after registration. |
| 3 | Arrival matrices for every family | Three families; `sudoku4` has none | `sudoku4` has no sibling that records the step at which it diverges. |
| 4 | Residual patching at every position | The ridge test uses the answer position; a position sweep on 200 pairs per family is stored | The answer position is where a step's result must arrive; the full sweep is 64 times the cost. |
| 5 | E1 statistics as first written | Changed after the first, partial run: see the plan | The first form discarded most pairs and could leave no depth to average. |
| 6 | `resid[0]` probe baseline | Mean over positions of the state entering visit 0 | At the answer position that state holds no input. |
| 7 | Program counter `remaining(s)` among the probed variables | Deferred to Phase 2 | It needs a visit-to-step alignment, which E1 supplies. |
| 8 | Permuted-target control, recipe unspecified | Stage-1 recipe without the curriculum or zero-hop pool, 20,000 steps | The curriculum unlocks depths on validation accuracy, which shuffled labels never reach. |
| 9 | Jacobian spectrum | 16 examples per model, raw states | Cost; the norm-corrected version is left to Phase 2. |
| 10 | Probes per variable, visit and position | Answer position only | E1's position sweep puts the chase at the answer position; other positions are left to Phase 2. |
| 11 | Control-task selectivity as one of the "computed" criteria | Kept as declared; for `sudoku4` it makes "computed" unattainable | The rule was fixed before the probes ran. The consequence is reported rather than the rule changed afterwards. |
| 12 | Control models of the same variant as the model they control for | `FF-U` `C0` runs with input re-injection; the `FF-U` permuted-target model is trained and run without it | The permuted-target recipe is the stage-1 recipe of its arm, which has no re-injection. |
| 13 | Representational similarity against trace-variable model matrices, with variance partitioning; dictionary-level similarity | Not done | The first needs the step alignment that E1 and E2 now give; the second needs the dictionaries of E4. |

## What was run and what was only written

- Run: the monitor-split rebuild (hash-checked against Phase 0); the matched-checkpoint search over 123
  checkpoints; E1 on five models and their controls (twice, see departure 5), then on `FF-U` s3; E2 on
  six models, five `C0` controls and both permuted-target models; E3 on six models and five `C0` controls
  (rerun in full on 2026-10-12, matching the first run); the two
  permuted-target control models, 20,000 steps each, ending at chance or the best-constant level on
  validation as intended (pointer 0.066 and 0.060, gated 0.506, reachability 0.511, Sudoku 0.241); all 50
  RMP tests under pytest 9.1.1 (38 from Phase 0, plus 12 new in
  `tests/test_rmp_arrival.py`, `test_rmp_probes.py`, `test_rmp_similarity.py` and `test_rmp_training.py`).
- Not run: noising-direction patching (the program's secondary direction; only denoising was run);
  anything from E4 on; the `COT` arms; the `locked` split; `C4` checkpoints.

## What Phase 2 needs

The program gates Phase 2 on the N1 label (closed-loop sparse replacement) being recorded and on controls 7
(planted positive) and 8 (absent negative) passing. None of that exists yet: it needs the dictionaries of
E4. Open items that Phase 1 leaves for it:

- the program-counter probe `remaining(s)`, now that E1 and E2 give a visit-to-step alignment;
- a norm-corrected Jacobian spectrum, and probes and patching away from the answer position;
- noising-direction patching;
- the literature-registry check against primary sources, still outstanding from Phase 0 (departure 15
  there);
- the untied arm still fails `G0` (one passing seed of three): whether to carry the comparison forward
  on its single passing seed, train more seeds, or change the arm.

## Reproducing

From the repository root, with the `bluebeam` interpreter as `$py` and `$work` set to
`C:\Users\patri\Documents\Codex\rmp_v0`:

```powershell
& $py -m research_gym.scripts.prepare_rmp_task_bundle --out $work\bundle --write monitor_train monitor_calibration
& $py -m research_gym.scripts.rmp_phase1_match --artifact-dir $work\artifacts
& $py -m research_gym.scripts.rmp_phase1_arrival --artifact-dir $work\artifacts --monitor-bundle $work\bundle
& $py -m research_gym.scripts.rmp_phase1_similarity --artifact-dir $work\artifacts --monitor-bundle $work\bundle
powershell -File scripts\run_rmp_permuted_controls.ps1 -Python $py -PoolDir $work\pool -ArtifactDir $work\artifacts
& $py -m research_gym.scripts.rmp_phase1_probes --artifact-dir $work\artifacts --monitor-bundle $work\bundle
& $py -m research_gym.scripts.report_rmp_phase1
```

The probe script skips control models that have not finished training; rerun it with `--reuse` once they
have, and it probes only the missing models before recomputing every "computed" flag.
