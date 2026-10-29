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

## Next

- **Rank.** Refit N2's usable cells at rank 16 and 32, as a recorded, post-hoc change.
- **Positions.** Move the interchange to the positions that hold `cur_node`, found from Phase 1's E1
  arrival matrices.
- **E5 reports.** Produce the six-lens reports themselves.
- **E6.** Run H2 (counter clamp) and rule ablation.
