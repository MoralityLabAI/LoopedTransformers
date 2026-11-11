# Checkpoint: RMP v0 Phase 2, E5 and E6 (first pass, 2026-10-29)

These are exploratory results under `docs/rmp_phase2_e5_e6_plan.md`. Every rule and change was fixed before
the run it affects. The result files are in `experiments/recurrence_monitorability_v0/phase2/`: `h3.json`,
`e5_def.json`, `n2.json` and `mlp_transcoders.json`.

The transcoders here are the post-hoc MLP-only ones, which stand in for the ill-posed registered `D-trans`.
Every number on rules or H3 rests on that departure.

## H3: description length of the transition function, `unresolved`

The rule asks for the smallest MLP-transcoder size whose replacement model keeps 0.95 or more of clean deep
exact match.

| Model | Size | Replacement | Transcoders | Alive latents | Length (b), bits |
| --- | --- | --- | --- | --- | --- |
| `LOOP-T s1` | w2048 k32 | 0.951 | 1 | 1,713 | 262,486 |
| `LOOP-T s2` | w2048 k32 | 0.968 | 1 | 1,708 | 262,485 |
| `LOOP-T s3` | none (best 0.947 at w8192 k32) | | | | |
| `FF-U s3` | w1024 k16 | 0.959 | 8 | 5,942 | 2,098,365 |

Where both are defined, one tied transcoder is about 3.5 times smaller in latents and about 8 times smaller
in bits than the untied stack's eight. Seed 3 never reaches the bar, so by the registered rule H3 is
`unresolved`. Its best size misses by 0.003.

## E5: fidelity of small explanations (DEF)

Each class is one (family, transition). Its answer-position update was resampled from another example of
the same depth. Explanations were rebuilt from the top-`m` components in each basis, with five controls at
the same `m`. A class is accepted when `S90 <= 16` in some basis and beats every control there.

| Model | Classes with an effect | Accepted | Accepted in `features` | Median S90: features, mixed, rules, heads |
| --- | --- | --- | --- | --- |
| `LOOP-T s1` | 19 of 24 | 12 (63%) | 12 | 4, 8, 64, never |
| `LOOP-T s2` | 23 of 24 | 21 (91%) | 20 | 8, 8, never, never |
| `LOOP-T s3` | 23 of 24 | 18 (78%) | 18 | 4, 8, 64, never |
| `FF-U s3` | 20 of 24 | 13 (65%) | 13 | 8, 8, 64, never |

- **Features carry the result.** A handful of `D-shared` features, 4 to 8, rebuild 90% of what the update
  contributes to the answer, beyond the random, PCA, neuron and orthogonal controls.
- **Rules rarely do.** Transcoder latents almost never get there within 16: `rules` was accepted once, in
  `LOOP-T s3`. The MLP's contribution is spread over many latents.
- **Heads never do.** Single head outputs never reach 90%.
- **Tied versus untied.** The tied seeds' acceptance share (63 to 91%) brackets the untied model's (65%).
  E5 gives no arm difference at this resolution.
- **Excluded classes.** A class whose resampling moves the answer by under 0.1 nats is `no_effect` and left
  out, mostly the last transition.

## E6: N2, abstraction mismatch

Interchange is at the answer position of `resid[v]`, with ranks up to 8. A cell counts only if a
full-residual patch there reaches 0.6 IIA (the site guard).

| Model | `pointer_chase` | `gated_pointer_chase` |
| --- | --- | --- |
| `LOOP-T s1` | `negative`: 1 usable cell (hop/v8), best IIA 0.361 | `unresolved_site` (full patch at most 0.53) |
| `LOOP-T s2` | `negative`: 1 usable cell, 0.543 | `unresolved_site` |
| `LOOP-T s3` | `negative`: 1 usable cell, 0.483 | `unresolved_site` |
| `FF-U s3` | `negative`: 1 usable cell, 0.525 | `unresolved_site` |

**N2 is `negative` on `pointer_chase` by the registered rule, but the result is weak, for three reasons.**

1. **One usable cell.** The only usable cell is the last visit, where `cur_node(8)` is the answer itself.
2. **Still rising at the rank cap.** In that cell IIA climbs steeply with rank: about 0.1 at rank 4 and
   0.36 to 0.54 at rank 8, while the full patch reaches 0.97 to 1.0. A 16-way node code plausibly needs more
   than 8 dimensions, so the rank grid {1, 2, 4, 8} is the likelier limit, not the abstraction.
3. **The site misses earlier visits.** At every earlier visit even the full patch recovers at most 0.40.

So the answer position does not hold `cur_node(s)` mid-loop, and the node is read from elsewhere. Doubling
never exceeds 0.06 in any cell.

The reading supported so far: the registered test, at the answer position, cannot see the hop variable
before the last visit. It is not evidence that no legible trace is causally realized.

## Follow-ups, 2026-10-29 (post hoc where marked; each change recorded in the plan before its run)

**N2 at ranks 16 and 32 (post hoc, `n2_rank16_32.json`).** In the usable cell (`pointer_chase`, hop/v8),
IIA reaches the following against the full-patch ceiling:

| Model | Rank 32 IIA | Full patch |
| --- | --- | --- |
| `LOOP-T s1` | 0.946 | 0.966 |
| `LOOP-T s2` | 0.987 | 0.998 |
| `LOOP-T s3` | 0.979 | 0.997 |
| `FF-U s3` | 0.989 | 1.000 |

So the registered `negative` came from the rank cap. With the refit, `pointer_chase` is `not_negative` on
every model, but on one late cell only. Doubling stays at or below 0.08. `gated_pointer_chase` stays
`unresolved_site`: its full-patch ceiling is at most 0.54. The hop abstraction is causally realized at the
answer position only once the answer is formed. Testing it mid-loop needs other positions.

**H2, the counter: `not_supported` (`h2.json`).**

| Model | Candidates (monotonicity 0.9 or more) | Best clamp shift | Random clamp |
| --- | --- | --- | --- |
| `LOOP-T s1` | 0 | none | none |
| `LOOP-T s2` | 4 | at most 0.006 | 0.002 |
| `LOOP-T s3` | 2 | 0.001 | 0.001 |
| `FF-U s3` | 3 | 0.001 | 0.001 |

The bar is a shift of 0.5. No tied seed has a feature that both counts visits and moves the answer when
clamped.

**Rule ablation (`rule_ablation.json`).** One MLP-transcoder latent is zeroed at a time, top 20 by activity,
in the replacement model (deep exact match 0.95, 0.96, 0.93, 0.98).

| Model | Largest drop | Median drop | Latents dropping more than 0.05 |
| --- | --- | --- | --- |
| `LOOP-T s1` | 0.062 | 0.008 | 1 |
| `LOOP-T s2` | 0.024 | 0.004 | 0 |
| `LOOP-T s3` | 0.088 | 0.044 | 8 |
| `FF-U s3` | 0.004 | 0.000 | 0 |

No single rule carries a step. Tied seed 3, whose replacement is the weakest, leans most on its top
latents. The untied stack is the most redundant.

**Fitted schedule (post hoc, `n2_schedule_*.json`).** For each visit, every step 1 to 8 was scanned with
the full answer-position patch, then a rank-32 subspace was fitted at the best step.

**`pointer_chase`.**

| Phase | Full patch at the best step | What the answer position holds |
| --- | --- | --- |
| Visits 1 to 4 (5 for tied seeds 1 and 3) | at most 0.22 (tied), 0.16 (untied) | no step's node |
| Final visits | full patch 0.81 to 1.00, rank 32 at 0.78 to 0.99 | the final node (step 8) |

- **When the final node appears.** It is held from visit 6 in tied seed 2 and the untied model, and from
  visit 7 in tied seeds 1 and 3.
- **No intermediate node.** No intermediate `cur_node(s)` is ever held there as a patchable variable.
- **One anomaly.** The subspace patch reached 0.64 at visit 2 (step 1) in tied seed 3, against a 0.22 full
  patch. It is a single cell and not interpreted.

**`gated_pointer_chase`.** Nothing passes 0.6 except the untied model's last visit, at exactly 0.60.

**Reading.** The answer position never holds the chase's intermediate nodes as a patchable variable, at any
visit or step. The trace is computed elsewhere, or in a form that a swap of one example's state into another
does not transfer. The answer position then switches to holding the answer one or two visits before the
end, in both arms. This settles the mid-loop N2 question for this site. A hop-level abstraction is not
causally realized at the answer position. Phase 1 found no other position that carries the chase, so the
open question is whether it is realized at all.

**Query-block patch (post hoc, `query_patch_*.json`).** The four query tokens and the answer position are
patched together at visit `v`, from a source with the same hop count. Each cell gives the best step's IIA,
then the "source's own answer" control.

| Model, `pointer_chase` | v1 | v2 | v3 | v4 | v5 | v6 | v7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `LOOP-T s1` | s1 0.21 / 0.05 | **s1 0.68** / 0.06 | s2 0.35 / 0.06 | s4 0.36 / 0.12 | s5 0.47 / 0.52 | s8 0.66 / 0.86 | s8 0.89 / 0.95 |
| `LOOP-T s2` | 0.06 / 0.04 | s1 0.11 / 0.05 | s3 0.13 / 0.06 | s5 0.24 / 0.29 | s6 0.42 / 0.64 | s8 0.78 / 0.92 | s8 0.98 / 0.99 |
| `LOOP-T s3` | 0.08 / 0.06 | **s1 0.77** / 0.05 | s2 0.40 / 0.06 | s4 0.32 / 0.11 | s5 0.39 / 0.38 | s8 0.56 / 0.80 | s8 0.89 / 0.97 |
| `FF-U s3` | 0.09 / 0.06 | s3 0.08 / 0.05 | **s1 0.69** / 0.08 | **s2 0.61** / 0.11 | s6 0.58 / 0.62 | s8 0.97 / 0.99 | s8 1.00 / 1.00 |

**Reading.**

- **The node is spread over the query block.** Once the query tokens are included, the first hop's node
  transfers mid-loop: tied seeds 1 and 3 at visit 2 (0.68, 0.77), and the untied model at layers 3 and 4
  (hops 1 and 2: 0.69, 0.61). The source-answer control stays at or below 0.12 in those cells, so the patch
  moves the intermediate node, not the whole chase.
- **The best step tracks Phase 1's arrival schedule.** In tied seed 1 it climbs 1, 1, 2, 4, 5 over visits 1
  to 5. Later steps transfer less cleanly (0.32 to 0.47).
- **Seed 2 is the exception.** Tied seed 2 transfers no intermediate node above 0.24 at any visit, the same
  seed spread as in the association results.
- **`gated_pointer_chase` stays below 0.6 throughout.** The latch and locks are not covered by this
  counterfactual's site.
- **The hop variable is causally realized, for early hops.** A hop-level variable exists for the first one
  or two hops, spread over the query block and the answer position. That rules out the strong reading of
  N2, on two of three tied seeds and on the untied model. After hop 2 the state stops transferring as a
  clean single node, in both arms.

## Next

- **Rank.** Refit N2's usable cells at rank 16 and 32, as a recorded, post-hoc change.
- **Positions.** Move the interchange to the positions that hold `cur_node`, found from Phase 1's E1
  arrival matrices.
- **E5 reports.** Produce the six-lens reports themselves.
- **E6.** Run H2 (counter clamp) and rule ablation.

**Query-block subspaces (post hoc, `query_das_*.json`).** Distributed alignment search was fitted over the
5-position block at each visit's best step. `pointer_chase` cells where the full block transfers 0.6 or more:

| Model, cell | Full block | Rank 8 | Rank 32 | Rank 128 |
| --- | --- | --- | --- | --- |
| `LOOP-T s1`, visit 2, step 1 | 0.68 | 0.03 | 0.09 | 0.49 |
| `LOOP-T s3`, visit 2, step 1 | 0.77 | 0.06 | 0.41 | 0.66 |
| `FF-U s3`, visit 3, step 1 | 0.69 | 0.09 | 0.56 | 0.63 |
| `FF-U s3`, visit 4, step 2 | 0.61 | 0.04 | 0.20 | 0.34 |

The early node is not carried by a small subspace. Rank 8 transfers almost nothing, and the subspace needs
rank 32 to 128 to approach the full block. That is unlike the final answer, which rank 32 carries at
the answer position alone (0.95 to 0.99). Mid-loop the intermediate node is held in a high-dimensional,
distributed form across the query block: causally present, but not as a compact variable that a low-rank
monitor would isolate. In `gated_pointer_chase` no cell passes 0.6 except `LOOP-T s1` visit 1 at rank 128
(0.71), against a full-block ceiling of 0.43. That subspace probably does more than transfer the node, so it
is not interpreted.

**Phase 3 fresh seeds (started 2026-10-29).** Phase 3 needs model seeds no earlier phase touched. `LOOP-T` and `FF-U` seeds 4, 5 and 6 are queued with the Phase 0 recipe:

- `LOOP-T`: plain 60k steps, then deep supervision 40k.
- `FF-U`: plain, then deep supervision, then deep supervision with input re-injection.

Each stage stops at `G0` and warm-starts from a hash-checked snapshot. Run records go to `rmp_v0\phase3\runs`, apart from the Phase 0 records. This is training only. The `locked` split is not written or read, and H4, H5 and H7 still need registration before Phase 3 reads it.

**Phase 3 fresh seeds: trained 2026-10-30.** Records are in `experiments/recurrence_monitorability_v0/phase3/runs`.

| Arm | Seed 4 | Seed 5 | Seed 6 | Passing |
| --- | --- | --- | --- | --- |
| `LOOP-T` | deep supervision, step 36,250 | plain, step 34,500 | plain, step 54,250 | 3 of 3 |
| `FF-U` | deep supervision, step 23,500 | failed all three stages | re-injection, step 12,750 | 2 of 3 |

Unlike Phase 0, two tied seeds reach `G0` without deep supervision. The untied arm again loses one seed of three. Phase 3 evaluation needs H4, H5 and H7 registered first, and the `locked` split is still untouched.

## Successor experiment: features and the named variable (registered 2026-10-31, `feature_link.json`)

These are the registered rules from the plan, run on `monitor_calibration` `pointer_chase`. The cell is chosen
and the features are ranked on the selection half. The table reports the reporting half: 1,000 collision-free
pairs per model. Feature columns give selected and random alive features.

| Model | Dictionary | Cell (visit, step) | Full block | All features | m=16 | m=32 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `LOOP-T s1` | w1024 k16 | 2, 1 | 0.670 | 0.640 | 0.405 / 0.003 | 0.605 / 0.003 | linked |
| `LOOP-T s2` | w1024 k8 | 2, 2 | 0.131 | 0.111 | 0.031 / 0.000 | 0.086 / 0.000 | ineligible |
| `LOOP-T s3` | w1024 k16 | 2, 1 | 0.773 | 0.719 | 0.008 / 0.000 | 0.215 / 0.001 | not linked |
| `LOOP-T s4` | w1024 k16 | 1, 1 | 0.968 | 0.976 | 0.427 / 0.119 | 0.942 / 0.000 | linked |
| `LOOP-T s5` | w1024 k8 | 2, 1 | 1.000 | 1.000 | 1.000 / 0.000 | 1.000 / 0.021 | linked |
| `LOOP-T s6` | w1024 k8 | 3, 1 | 0.938 | 0.939 | 0.938 / 0.048 | 0.939 / 0.104 | linked |
| `FF-U s3` | w8192 k32 | 3, 1 | 0.733 | 0.722 | 0.178 / 0.000 | 0.277 / 0.000 | not linked |
| `FF-U s4` | w1024 k8 | 2, 1 | 0.459 | 0.509 | 0.310 / 0.000 | 0.488 / 0.006 | ineligible |
| `FF-U s6` | w4096 k8 | 3, 1 | 0.860 | 0.814 | 0.176 / 0.000 | 0.386 / 0.012 | not linked |

- **Tied: linked in 4 of 5 eligible seeds.** With 32 or fewer features, the patch recovers 0.90 to 1.00 of
  the full-block transfer. Random features stay at 0.12 or below. `LOOP-T s3` needs 64 features (0.652),
  which is outside the registered m ≤ 32.
- **Untied: linked in 0 of 2 eligible seeds.** At m = 32 the patch recovers 0.38 and 0.45 of the full block.
  At m = 64, `FF-U s6` reaches 0.565.
- **Caveats.**
  - The untied models' selected dictionaries are 4 to 8 times wider, which confounds the arm comparison.
  - The features are ranked by code difference, not taken from the E5 DEF sets.
  - Only two untied models are eligible.
