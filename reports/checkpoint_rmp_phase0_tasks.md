# Checkpoint: RMP v0 Phase 0, Step 1 (Trace-Annotated Task Bundle)

Date: `2026-10-01`

## Completed

- `research_gym/interp/tasks.py`: six task families. Each instance is a JSON-able
  description that the family turns into tokens, an answer, and a per-step
  ground-truth trace. Minimal-pair siblings are edits of that description which
  are solved again from scratch.
- `research_gym/interp/bundle.py`: hash-partitioned splits, sibling generation,
  a deterministic training stream, shard hashing, and the manifest.
- `research_gym/scripts/prepare_rmp_task_bundle.py` and
  `configs/rmp_task_bundle_v0.json`.
- The full bundle was generated and hashed in 172 seconds. Only `validation`
  and `ood_stress` were written to disk (27 MB); they are what gate `G0` and the
  depth 9-12 report need. The other shards are sealed by hash in the manifest
  and regenerate from the config.

## Bundle

| Family | Trace variables | Depths | Best constant (validation) |
| --- | --- | --- | ---: |
| `pointer_chase` | `cur_node`, `remaining` | 1-8, OOD 9-12 | 0.0725 |
| `gated_pointer_chase` | `cur_node`, `remaining`, `latch` | 1-8, OOD 9-12 | 0.5062 |
| `masked_pointer_chase` | `cur_node`, `remaining`, `unknown_touched` | 1-8, OOD 9-12 | 0.5092 |
| `reachability` | `frontier`, `visited`, `goal_found` | 1-8, OOD 9-12 | 0.5108 |
| `sudoku4` | `candidates`, `determined` | 1-3 | 0.2597 |
| `modular_recurrence` | `acc`, `remaining` | 1-8, OOD 9-12 | 0.0628 |

Base examples per family, uniform over depth:

| Split | Requested | Structure-hash share | Siblings attached |
| --- | ---: | ---: | --- |
| train (generated on the fly) | unbounded | 0.50 | no |
| `validation` | 4,000 | 0.06 | no |
| `monitor_train` | 20,000 | 0.28 | no |
| `monitor_calibration` | 5,000 | 0.06 | yes |
| `locked` | 5,000 | 0.06 | yes |
| `ood_stress` | 2,000 | 0.04 | no |

All four core families filled every quota. `modular_recurrence` fell short at
depth 1 only (validation 289, monitor_train 1,455, monitor_calibration 515,
locked 466): there are 272 one-step programs, so the instance space is too
small for the quota.

Siblings in `locked`, of 5,000 bases per family (4,534 for modular):

| Family | Sibling kinds and counts |
| --- | --- |
| `pointer_chase` | `edge_rewire` 4,924; `hops` 4,954; `start` 5,000 |
| `gated_pointer_chase` | `grant_flip` 5,000; `lock_move` 4,352; `edge_rewire` 4,927; `hops` 4,967; `start` 5,000 |
| `masked_pointer_chase` | `mask_move` 4,362; `edge_rewire` 4,078; `hops` 4,984; `start` 5,000 |
| `reachability` | `horizon` 5,000; `goal` 5,000; `start` 5,000; `edge_rewire` 3,821 |
| `sudoku4` | `clue_move` 4,506; `query` 5,000 |
| `modular_recurrence` | `op_change` 4,487; `hops` 3,133; `start` 4,513 |

Each sibling records its base, the edit, whether the answer changed, and for
every trace variable the first step at which the two traces differ.

## Split guarantee

A structure hash (the mapping, graph, puzzle, or program) decides which split a
base example belongs to. The training stream draws only structures in the
train region. A sibling inherits its base's split, and a sibling whose edit
changes the structure is kept only if its own hash also falls outside the train
region. No fixed-split example, base or sibling, can therefore be produced by
the training stream.

## Tests

```text
tests/test_rmp_tasks.py: 9 passed (pytest 9.1.1)
```

The repository's full suite is run as a baseline before step 2 edits shared
model code; its result is recorded in the Phase 0 checkpoint.

- An independent oracle per family reads only the token sequence and recomputes
  the answer. It agrees with the stored answer at every depth each family
  supports, including the OOD depths.
- Three deliberate breakages (a gated solver that ignores locks, a reachability
  horizon off by one, a biased refusal rate) each made a test fail.
- A second full build to a separate directory is the determinism check; its
  result is recorded under Provenance.

## Departures from the program document

| # | Document | Built | Reason |
| ---: | --- | --- | --- |
| 1 | `rewrite_normalization` as a replication family | dropped | Depth cannot be controlled. Requiring a rewrite at every step, 77% of depth-8 instances revisit a state and 59% are a pure two-state oscillation (150 samples); sampling costs about 30 ms each. |
| 2 | `sudoku4` with `stalled(s)` and depths 1-8 | depths 1-3, no `stalled`, no OOD split | In 29,103 sampled unique-solution 4x4 puzzles, naked-single propagation resolved every one within three rounds. No stall state exists to label, so search state rests on `reachability` alone. |
| 3 | `reachability` on 12 nodes | 16 nodes | A negative at horizon 12 whose goal sits one step beyond needs 14 breadth-first layers. |
| 4 | `masked_pointer_chase` hides one entry | hides two | The mapping is a permutation, so a single hidden destination is the one node missing from the others and can be recovered by elimination. |
| 5 | Lock counts made uninformative by the grant-flipped sibling | made uninformative in the sampler | Refusal moves one unheld lock from an off-path edge onto the path, so both labels share a lock multiset. Every base still has a one-grant sibling; it flips the answer in 8,908 of 10,000 cases, and cannot when an allowed path uses no held lock. |
| 6 | Pointer families reuse the v1 generator | first `k` hops must leave `k` distinct nodes | Otherwise a path can revisit an edge, and "the edge used at step `s`" is not well defined for traces or rewires. |
| 7 | No validation split | `validation`, 4,000 per family | Gate `G0` may not read `locked`. |
| 8 | Siblings on `locked` only | also on `monitor_calibration` | Patching code needs a split with siblings that is not the locked one. |
| 9 | One module | `tasks.py` for families, `bundle.py` for splits and manifest | Size. |

Two details the document left open were fixed here. Gated instances use three
lock classes and all eight grant sets; holding every class can never refuse, so
the other seven refuse at 4/7 to keep refusals at one half overall. Reachability
labels are split evenly over four cases: goal exactly at the horizon, goal
closer, goal one step beyond, and goal far or unreachable.

## Decision

Step 1 is complete. The system Python on this host has no `torch`, but an
existing virtual environment outside the repository
(`%USERPROFILE%\.venvs\bluebeam`: Python 3.12, torch 2.11.0+cu128, pytest 9.1.1)
does. It belongs to another project and is used read-only; nothing was
installed. Steps 2 to 5 proceed with that interpreter. The GPU is an RTX 5080
Laptop with 16 GB, shared with other work on this machine.

## Provenance

- Bundle hash: `f1d81a3afde4fab4cc07d70949e6f87eb8c1ff5766e1b0cace206a7456e4d265`.
- Config hash: `e322799e952d08d82f8c1b40b2cbf089d1b56f6ab945e31cf92571a2a7099578`.
- Manifest: `experiments/recurrence_monitorability_v0/datasets/manifest.json`,
  which also records each shard's SHA-256 and the generator source hashes.
- Rebuild check: a second full build, written to a separate directory,
  reproduced the same bundle hash.
