# AF-14 local research report | November 15, 2026

**Registered scientific verdict: FAIL.** **Local engineering verdict: PASS after one source-limited layout repair.** Both independently replayed seed evaluations and the exact cross-seed aggregate completed in Python 3.13 / Torch 2.10 CPU. The pinned hosted Python 3.12 / Torch 2.5.1 and Lean 4.19.0 job **did not execute** because both GitHub Actions runs failed before runner assignment; this result must be read as a complete *local* experiment, not a completed hosted replication.

- Original frozen protocol: [PROTOCOL.json](../../experiments/activation_funnel_af14/PROTOCOL.json), SHA-256 `cd7b7e283ea8508d34b2f9c1648060185b7a4231b4b3c3462ffbab7eba1fe65a`.
- First source registration: [`9a565b7`](https://github.com/MoralityLabAI/LoopedTransformers/commit/9a565b7196e9a7dd99a220b2c6e91c7f2c0d16c3).
- Repaired scored source: [`f6a5b11`](https://github.com/MoralityLabAI/LoopedTransformers/commit/f6a5b11f10026b6a6ab4d905930d5c584b808846).
- [Numerical engineering record](AF14_ENGINEERING.md): first seed-17 training reached 100% development accuracy, but the evaluator rejected AF-14's new physical layouts. The repaired experiment re-fitted every model from its original SHA-verified AF-06 checkpoint without modifying the registered loss or thresholds.
- [AF-14 PR #70](https://github.com/MoralityLabAI/LoopedTransformers/pull/70), draft, stacked on `main` after AF-01 through AF-13 were merged.
- Frozen prospective corpus: 672 train, 224 dev and 656 held-out cases on new layout IDs 26–31. All 1,552 cases and **8,024 reachable-node deadlines** agreed between two independent Horn minimal-depth algorithms.

## Experimental setup

Three arm types, `relay_final`, `async_sound`, and `bounded_progress`, were trained independently from identical original AF-06 width-two 480-update qualified source checkpoints for seeds 17 and 29. Each arm consumed 160 optimizer updates with identical training cases, donor sampling, recurrence counts, step-size and optimization settings. Training split contains proof depths 2 and 3. The bounded arm adds 0.35 times a softplus true-class margin penalty on every reachable atom from its bounded deadline `max(1,2*d(v))` through the final upstream call. Earlier successful derivations remain valid. All arms preserve the original clean, patched-goal, subgoal and irrelevant-branch supervision. Auxiliary backward arithmetic is additional; this is a forward-call-matched study.

The source model has 15 atom and 14 rule tokens with 16 decoded working coordinates per token, carried through two float32 coordinates per token (58 carried floats). The original native TypedLoop class is pinned. Lean's Boolean theorem-generation program remains part of the source; **the hosted Lean checks did not run** in this iteration. Neural predictions have no kernel proof credit.

## Frozen eligibility

| Seed | Arm | Development goal | Exact joint-Q |
|---:|---|---:|---:|
| 17 | relay_final | 1.000000 | 1.000000 |
| 17 | async_sound | 1.000000 | 1.000000 |
| 17 | bounded_progress | 1.000000 | 1.000000 |
| 29 | relay_final | 0.906250 | 1.000000 |
| 29 | async_sound | 1.000000 | 1.000000 |
| 29 | bounded_progress | 0.758929 | 1.000000 |

The registered comparison requires both `async_sound` and `bounded_progress` arms to pass 90% development and training goal and exact-Q accuracy. Seed 17 is eligible and **fails**. Seed 29 is ineligible and its held-out comparisons are **inconclusive**. An eligible failure determines the aggregate FAIL verdict.

## Held-out depth-generalization

Exact upstream joint-Q (Q0 and Q1 both correct):

| Seed | Heldout depth | relay_final | async_sound | bounded_progress | bounded minus async |
|---:|---:|---:|---:|---:|---:|
| 17 | 4 | 1.000000 | 1.000000 | 0.765625 | -0.234375 |
| 17 | 5 | 1.000000 | 0.562500 | 0.562500 | 0.000000 |
| 29 | 4 | 0.562500 | 0.562500 | 0.250000 | -0.312500 |
| 29 | 5 | 0.250000 | 0.250000 | 0.250000 | 0.000000 |

Exact all-atom final closure and post-deadline persistence (seed 17):

| Heldout depth | Arm | Final all-atom accuracy | Stable-after-deadline fraction | Clean goal | Informative patched-goal accuracy |
|---:|---|---:|---:|---:|---:|
| 4 | async_sound | 1.000000 | 0.083333 | 1.000000 | 1.000000 |
| 4 | bounded_progress | 0.816667 | 0.291667 | 0.750000 | 0.593750 |
| 5 | async_sound | 0.958333 | 0.000000 | 0.765625 | 0.648438 |
| 5 | bounded_progress | 0.733333 | 0.392857 | 0.781250 | 0.593750 |

The increased persistence is real under the independent metrics, but remains below the registered 0.85 requirement and coincides with severe loss of final semantic accuracy. Seed 17's bounded model increases unsupported intermediate truth counts from **7,040 to 9,920** at depth 4, while decreasing missing-fact counts from 1,664 to 960. At depth 5, unsupported true counts rise from **8,320 to 11,520** and the bounded arm makes **20 false-positive patched goal attempts** where the async baseline makes zero. Both arms score zero on fully admissible intermediate trajectories. The deadline penalty encourages earlier, stickier positive states without having established rule-level source soundness.

The registered gate required, for every eligible seed, at least +0.05 in **both** joint-Q and final all-atom accuracy at both new depths, ≥0.85 persistent deadlines, and preserved task/causal negative controls. Seed 17 fails multiple required conditions. The scientific result is a negative outcome of this fixed objective, not evidence of depth-stable causal proof inference.

## Independent evidence, checks, and limitations

- **Six complete trained models** with exact Torch `.pt` and named-array `.npz` exports, 5,376 train/dev fit predictions, true-length source outputs, and frozen source/checkpoint SHA-256 receipts.
- **3,936** held-out clean prediction rows and **7,872** donor-counterfactual rows, with recurrence-resolved raw logits, per-atom registered deadline arrays, original host truth and independent donor identities.
- **144** sampled source-bound forward replays, split between seeds 17 and 29. Independent auditors recompute all Horn oracle labels and physical permutation maps, reject tampered and rehashed deadline rows, reproduce all registered metrics and verify original Torch weights against their named NumPy arrays.
- The two independently produced local proof-depth algorithms agree for all 1,552 train/development/held-out cases and 8,024 reachable positive milestones.
- The original qualified AF-06 training ZIPs match their pre-pinned SHA-256 digests. This run uses local Python 3.13/Torch 2.10 CPU. The hosted preflight workflows [#43698202729](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43698202729) and [#43698810409](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43698810409) failed with zero executed steps; no hosted training or Lean compilation is claimed.

## Mechanistic direction

The [bounded-progress proof](AF14_BOUNDED_PROGRESS.md) assumes each enabled Horn rule fires within a bounded number of opportunities. A class-logit deadline penalty does not enforce that premise. The AF-14 negative result motivates a **structurally sound propagation operator**: start from the given facts and update an atom only through enabled prerequisite rules. Boolean OR accumulation then guarantees no unsupported atoms and no erasure. If rule gates are additionally bounded-fair, the same `bD` convergence guarantee applies. Trainable gating could be studied in a separately preregistered AF-15 experiment, with ordinary TypedLoop/soft-constraint controls and a fresh prospective layout corpus.
