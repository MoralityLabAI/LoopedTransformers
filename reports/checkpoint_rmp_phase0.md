# Checkpoint: RMP v0 Phase 0 (status at 2026-10-02 08:20)

## Outcome so far

Gate `G0` has not been reached. No cell has passed it. Steps 1 to 3 are
complete. Step 4, the single-seed pilot, produced a model that learns: the tied
8-visit loop scores 0.948 on pointer chasing across depths 1-8. Step 5 is
running: the first gate cell, `LOOP-T-core4-s1`, is at step 28,500 of 60,000.

In that cell three of the four families are at or near the gate and one is
not:

| Family | Interim accuracy | Depths 1 to 8 (`sudoku4`: 1 to 3) |
| --- | ---: | --- |
| `sudoku4` | 1.000 | 1.00 1.00 1.00 |
| `gated_pointer_chase` | 0.988 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.91 |
| `pointer_chase` | 0.973 | 1.00 1.00 1.00 1.00 0.98 1.00 0.92 0.88 |
| `reachability` | 0.838 | 1.00 0.97 0.94 1.00 0.89 0.77 0.67 0.47 |

Interim accuracy uses 64 validation examples per depth; the gate is decided on
the full 4,000-example split. `reachability` is a yes/no answer, so 0.5 is
chance, and its depths 7 and 8 are there. Its pooled accuracy has moved from
0.809 at step 20,000 to 0.814 at step 28,000.

The reason appears to be the visit budget. Reading the answer after every
visit of this cell's checkpoint
(`experiments/recurrence_monitorability_v0/diagnostics/visit_cost.py`) shows
that a depth-`d` answer first becomes correct at visit `d + 2` for both pointer
families and at visit `d + 3` for reachability (depth 5 at visit 10). Eight
visits therefore cover pointer chasing to depth 6 and reachability to depth 5
at the learned rate. The pointer families reach depths 7 and 8 only through a
computation that is right at exactly visit 8 and wrong one visit later.
Reachability has found no such shortcut, and its depths 6 to 8 do not improve
with up to 16 visits. One plausible cause is the encoding: the question comes
after the edge list and attention is causal, so the search has to run at the
last few positions, which must carry the whole reached set.

This is evidence, not proof: a model could learn a faster scheme, such as
composing pointers to double the hop length per visit. A diagnostic training
the same recipe on `reachability` alone with 16 visits
(`LOOP-T-reach-v16-s101`) is queued to test whether the budget is what binds.

Progress stalled on 2026-10-02 from about 07:50 because the GPU was held at a
15 W software power limit (its default is 80 W), running at 180 MHz. This was
reported to the user; no system setting was changed.

| Step | State |
| --- | --- |
| 1. Trace-annotated task bundle | done, `reports/checkpoint_rmp_phase0_tasks.md` |
| 2. Instrumentation | done |
| 3. Training cell | done |
| 4. Pilot: `LOOP-T` on `pointer_chase` | 0.948 on full validation at step 27,500; paused, resumable |
| 5. Gate cells: 2 arms x 3 seeds x 4 families | `LOOP-T-core4-s1` at step 28,500; five cells waiting |

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

Full evaluation of the second cell when it was paused at step 27,500
(`runs/LOOP-T-pointer-cur2-s101/manual_full_evaluation.json`, 4,000 validation
and 2,000 held-out-depth examples):

| Split | Pooled | By depth |
| --- | ---: | --- |
| validation, depths 1-8 | 0.948 | 1.000 1.000 0.998 1.000 0.986 0.966 0.890 0.746 |
| held-out depths 9-12 | 0.010 | 0.006 0.012 0.010 0.012 |

The model does not extrapolate past the depths it was trained on: at depths 9
to 12 it is below the 0.0625 chance rate.

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
| 3 | Batch unspecified | 256 per family; families run one after another within a step, and the pilot split each batch into microbatches of 64 | Gradient noise at smaller batches; accumulation keeps VRAM to one batch on a crowded card. |
| 4 | RAM cap 3,800 MB | 6,144 MB, in `configs/rmp_resource_profile_v0.json` | The cap is enforced on committed memory. A CUDA process commits 4,138 MB here with a 1,234 MB working set, so the old cap made CUDA fail at random. All other caps are unchanged. |
| 5 | Idle-GPU preflight | Start when a set amount of VRAM stays free over three samples; never stop another process; below-normal CPU priority; mains power only | The GPU is shared with several other jobs all day and is never idle. No timing claim is made. |
| 6 | C4 at four times the C3 step | Cells stop at C3; C4 is produced by resuming the same cell | Triples the GPU time per cell; deferred, not dropped. |
| 7 | Gate cells cover the task table | Four core families only | `masked_pointer_chase` and `modular_recurrence` are generated but not trained, to limit compute. |
| 8 | Task-bundle departures | nine, listed in `reports/checkpoint_rmp_phase0_tasks.md` | |

## What was run and what was only written

- Run: all RMP tests (35 passing under pytest 9.1.1); the bundle build and its
  rebuild; the three pilot cells above; the diagnostics; the capped runner on
  CPU and GPU.
- Started but far from finished: the first multi-family gate cell.
- Written and unit-tested but not run at scale: input re-injection and the gate
  summary (`report_rmp_phase0.py`).
- Paused early and not informative: the deep-supervision pilot (step 4,000) and
  an `FF-U` pointer pilot (step 1,000).
- The repository's existing suite has 15 failing tests and 2 files that do not
  import, on `main` as well as on this branch; none involves RMP code.

## Compute

Every GPU run today shared the card with other jobs on this machine (model
servers and trainers using 9 to 15.7 of 16.3 GB at 96-100% utilisation). The
pilot ran at 1 to 4 optimizer steps per second. Two of its slices ended in
CUDA errors under memory pressure and resumed from the last evaluation point.
A four-family gate cell does four times the work per step, so six cells are
several days at this rate and several hours on a free card.

## Step 5: the gate queue

`scripts/run_rmp_phase0_queue.ps1` was started detached at 22:11. It runs
`LOOP-T` and `FF-U` at seeds 1, 2, 3 on the four core families, one cell at a
time, each with a 60,000-step budget, stopping at C3. Each runner waits for
mains power and 2 GB of free VRAM, retries after GPU errors, and resumes from
the cell's last evaluation point.

The queue lost the night of 2026-10-01 to standby (22:58 to 05:23) and was
restarted at 05:24 as `phase0_queue_3.log`.

Trajectory of the first cell, `LOOP-T-core4-s1` (interim pooled accuracy):

| Step | `pointer_chase` | `gated_pointer_chase` | `reachability` | `sudoku4` |
| ---: | ---: | ---: | ---: | ---: |
| 4,000 | 0.164 | 0.141 | 0.529 | 0.990 |
| 8,000 | 0.293 | 0.400 | 0.602 | 0.995 |
| 12,000 | 0.518 | 0.738 | 0.676 | 0.995 |
| 16,000 | 0.869 | 0.928 | 0.773 | 1.000 |
| 20,000 | 0.969 | 0.977 | 0.809 | 0.995 |
| 28,000 | 0.977 | 0.984 | 0.814 | 1.000 |

Curriculum events: `sudoku4` cleared all depths by step 1,750;
`gated_pointer_chase` reached depth 8 at step 15,250 and cleared it at 17,750;
`pointer_chase` reached depth 8 at 15,500 and cleared it at 26,000;
`reachability` unlocked depths 2 to 6 at steps 5,500, 11,750, 13,500, 15,500 and
21,000, and had not unlocked depth 7 by step 28,500.

Diagnostics run beside the queue, not gate cells: `LOOP-T-reach-v16-s101`
(reachability alone, 16 visits) is running. `LOOP-T-reach-v8-s101`
(reachability alone, 8 visits) was stopped at step 0 to reduce contention.

## To monitor, stop, or resume

```powershell
Get-Content experiments\recurrence_monitorability_v0\runs\logs\phase0_queue_2.log -Tail 5
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_rmp' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
powershell -File scripts\run_rmp_phase0_queue.ps1 -Python <python> -PoolDir <pool> -ArtifactDir <artifacts> -MaxSteps 60000 -Microbatch 256
python -m research_gym.scripts.report_rmp_phase0 --variant core4
```

Rerunning the queue resumes every cell where it stopped. The pilot resumes with
`scripts\start_rmp_cell_detached.ps1` and the arguments in
`runs/logs/LOOP-T-pointer-cur2-s101.runner.log`.

Checkpoints, pools, and predictions are under
`C:\Users\patri\Documents\Codex\rmp_v0`, outside OneDrive. Receipts and events
are under `experiments/recurrence_monitorability_v0/runs`.
