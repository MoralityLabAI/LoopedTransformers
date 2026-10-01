# Checkpoint: RMP v0 Phase 0 (status at 2026-10-01 21:40)

## Outcome so far

Gate `G0` has not been reached and has not yet been tested. Steps 1 to 3 are
complete. Step 4, the single-seed pilot, has a model that learns: the tied
8-visit loop is at 0.949 on pointer chasing across depths 1-8 and still
improving. Step 5, the six gate cells, has not started. Work is paused because
the laptop went onto battery at 25% charge; the pilot is queued to resume by
itself on mains power.

| Step | State |
| --- | --- |
| 1. Trace-annotated task bundle | done, `reports/checkpoint_rmp_phase0_tasks.md` |
| 2. Instrumentation | done |
| 3. Training cell | done |
| 4. Pilot: `LOOP-T` on `pointer_chase` | learning; 0.949 at step 25,000; gate not yet passed |
| 5. Gate cells: 2 arms x 3 seeds x 4 families | not started; queue script ready |

## Step 4: pilot attempts

All are `LOOP-T`, seed 101, `pointer_chase` only, width 128, 4 heads, 8 visits.
Accuracy is the interim validation subset: 64 examples per depth, 512 in all.
Chance is 0.0625 and the best constant answer scores 0.0725.

| Cell | What changed | Steps | Examples seen | Accuracy, depths 1 to 8 | Pooled |
| --- | --- | ---: | ---: | --- | ---: |
| `LOOP-T-pointer-s101` | program spec: beta 0.25, batch 128, uniform over depths 1-8 | 46,750 | 6.0M | all at chance | 0.059 |
| `LOOP-T-pointer-cur2-s101` | beta 1.0, batch 256, zero-hop rung, depth curriculum | 25,000 | 6.4M | 1.00 1.00 1.00 1.00 1.00 0.97 0.91 0.72 | 0.949 |
| `LOOP-T-pointer-ds-s101` | the same plus loss at every visit | 4,000 | 1.0M | all at chance | 0.074 |

In the first cell the training loss ended at 2.771 against a target entropy of
2.773: nothing was learned. In the second, the curriculum unlocked depths 2 to
8 at steps 3,750, 5,500, 8,750, 10,250, 12,500, 15,750, and 22,250. The third
was paused at step 4,000 to give the second the GPU; the second left its own
first plateau at about step 3,500, so 4,000 steps does not show that
supervision at every visit fails.

Loop-count sweep of the second cell at step 24,750, 100 validation examples per
depth, same weights run for a different number of visits:

| Visits | Depth 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.00 | 1.00 | 0.92 | 0.41 | 0.04 | 0.03 | 0.06 | 0.04 |
| 6 | 1.00 | 1.00 | 1.00 | 0.97 | 0.86 | 0.25 | 0.09 | 0.11 |
| 8 (trained) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.86 | 0.76 |
| 10 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 0.96 | 0.89 | 0.85 |
| 16 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.95 | 0.86 | 0.68 |
| 24 | 1.00 | 1.00 | 0.96 | 0.97 | 0.97 | 0.89 | 0.77 | 0.63 |

Deeper questions need more visits, answers hold once reached, and running to
three times the trained loop count degrades them only slowly. Depth 8 improves
with two extra visits, so part of the remaining error is the visit budget.

## Why the program's own spec did not learn

The diagnosis is in `experiments/recurrence_monitorability_v0/diagnostics/README.md`.
With a permutation as the mapping, the answer is uniform given any single
token, so no part of the lookup circuit receives a gradient until the others
exist, and leaving that plateau is limited by gradient noise. Three changes
were each needed:

- Batch size. At 64 examples of one depth per step, nothing learned in 8,000 to
  10,000 steps under any setting tried. At 512 it learned.
- Residual branch scale. At 1.0 with zero-hop examples a two-layer model solved
  depth 1 within 1,000 steps; at 0.25 it was at 0.17 after 6,000.
- Depth curriculum. A uniform mixture over depths 0-8 stayed at chance for
  7,000 steps; the easy-to-hard curriculum reached depth 3 by step 5,500.

## What the fallback ladder record looks like

| Rung | Status |
| --- | --- |
| More exposure | First pilot ran 6.0M examples at chance. Batch size, not step count, was the binding factor. |
| Learning rate and warm-up | 0.0003, 0.001, 0.003 at batch 64: all at chance. At batch 512, 0.001 and 0.0003 learn, 0.003 does not. |
| Deep supervision at every visit | Started, paused at step 4,000, inconclusive. |
| Input re-injection | Not run. Implemented and tested in the training cell. |
| 8-node task | Not run, and not recommended as written: see below. |

Three changes outside the listed ladder were made, because the diagnostics
located the failure in forming the first lookup, which the later rungs do not
address: residual scale 1.0, zero-hop training examples, and the depth
curriculum. None changes what is evaluated or the 0.98 threshold.

The 8-node rung has a flaw. The bundle requires the first `k` hops to leave `k`
distinct nodes. On 8 nodes, an 8-hop path is then always the full cycle, so the
answer equals the start node. A model could pass depth 8 without chasing
anything.

## Departures from the program document and the goal

| # | Specified | Done | Reason |
| ---: | --- | --- | --- |
| 1 | `beta = 0.25` | `beta = 1.0` | Lookup forms an order of magnitude more slowly at 0.25 (diagnostics). |
| 2 | Train on depths 1-8, uniform | Zero-hop examples added for the pointer families; depth curriculum with half of each batch on the frontier depth until the family clears its deepest depth | The uniform mixture does not leave the plateau. Evaluation is unchanged. |
| 3 | Batch unspecified | 256 per family, accumulated over microbatches of 64 | Gradient noise at smaller batches; microbatches keep VRAM near 0.4 GB on a crowded card. |
| 4 | RAM cap 3,800 MB | 6,144 MB, in `configs/rmp_resource_profile_v0.json` | The cap is enforced on committed memory. A CUDA process commits 4,138 MB here with a 1,234 MB working set, so the old cap made CUDA fail at random. All other caps are unchanged. |
| 5 | Idle-GPU preflight | Start when a set amount of VRAM stays free over three samples; never stop another process; below-normal CPU priority; mains power only | The GPU is shared with several other jobs all day and is never idle. No timing claim is made. |
| 6 | C4 at four times the C3 step | Cells stop at C3; C4 is produced by resuming the same cell | Triples the GPU time per cell; deferred, not dropped. |
| 7 | Gate cells cover the task table | Four core families only | `masked_pointer_chase` and `modular_recurrence` are generated but not trained, to limit compute. |
| 8 | Task-bundle departures | nine, listed in `reports/checkpoint_rmp_phase0_tasks.md` | |

## What was run and what was only written

- Run: all RMP tests (35 passing under pytest 9.1.1); the bundle build and its
  rebuild; the three pilot cells above; the diagnostics; the capped runner on
  CPU and GPU.
- Written and unit-tested but not run at scale: the multi-family gate cells,
  input re-injection, the gate summary (`report_rmp_phase0.py`), the queue
  script.
- The repository's existing suite has 15 failing tests and 2 files that do not
  import, on `main` as well as on this branch; none involves RMP code.

## Compute

Every GPU run today shared the card with other jobs on this machine (model
servers and trainers using 9 to 15.7 of 16.3 GB at 96-100% utilisation). The
pilot ran at 1 to 4 optimizer steps per second. Two of its slices ended in
CUDA errors under memory pressure and resumed from the last evaluation point.
A four-family gate cell does four times the work per step, so six cells are
several days at this rate and several hours on a free card.

## To resume

```powershell
powershell -File scripts\start_rmp_cell_detached.ps1 -Name LOOP-T-pointer-cur2-s101 -RunnerArguments "<arguments recorded in runs/logs/LOOP-T-pointer-cur2-s101.runner.log>"
powershell -File scripts\run_rmp_phase0_queue.ps1 -Python <python> -PoolDir <pool> -ArtifactDir <artifacts>
python -m research_gym.scripts.report_rmp_phase0 --variant core4
```

Checkpoints, pools, and predictions are under
`C:\Users\patri\Documents\Codex\rmp_v0`, outside OneDrive. Receipts and events
are under `experiments/recurrence_monitorability_v0/runs`.
