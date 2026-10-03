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
composing pointers to double the hop length per visit.

To test whether the budget is what binds, a first diagnostic trained the same
recipe on `reachability` alone with 16 visits from scratch
(`LOOP-T-reach-v16-s101`). It never left the first plateau: depth 1 was still
at chance after 8,000 steps, and it was stopped. In the four-family cell,
reachability got started on the back of the other families. The replacement
uses the fact that a tied loop has the same weights at any visit count: two
arms start from a snapshot of `LOOP-T-core4-s1` at step 32,840
(`snapshots/LOOP-T-core4-s1-step32840.pt`, SHA-256 `9b799802...`), which
already solves reachability to depth 5, and continue on `reachability` alone
for up to 20,000 steps, one at 16 visits (`LOOP-T-warm-reach-v16-s1`) and a
matched control at 8 (`LOOP-T-warm-reach-v8-s1`). If the 16-visit arm clears
depths 6 to 8 and the control does not, the 8-visit budget is the reason G0
fails on reachability.

Result, full validation (4,000 examples):

| Arm | Outcome | Steps | Depths 1 to 8 | Pooled | Depths 9-12 |
| --- | --- | ---: | --- | ---: | ---: |
| `LOOP-T-warm-reach-v8-s1` | budget exhausted | 20,000 | 1.000 1.000 0.996 0.994 0.992 0.884 0.834 0.706 | 0.926 | 0.657 |
| `LOOP-T-warm-reach-v12-s1` | gate passed | 8,750 | 1.000 0.996 0.988 0.976 0.980 0.976 0.986 0.974 | 0.985 | 0.641 |
| `LOOP-T-warm-reach-v16-s1` | gate passed | 6,750 | 1.000 0.996 0.982 0.964 0.980 0.964 0.976 0.978 | 0.980 | 0.644 |

From the same weights, 12 and 16 visits clear reachability and 8 does not,
even with 2.3 times the steps. The 8-visit budget is what blocks G0 on
reachability. The proposal to revise G0 to 12 visits is
`docs/rmp_gate_g0_revision_proposal.md`.

Progress stalled on 2026-10-02 from about 07:50 because the GPU was held at a
15 W software power limit (its default is 80 W), running at 180 MHz. This was
reported to the user; no system setting was changed.

| Step | State |
| --- | --- |
| 1. Trace-annotated task bundle | done, `reports/checkpoint_rmp_phase0_tasks.md` |
| 2. Instrumentation | done |
| 3. Training cell | done |
| 4. Pilot: `LOOP-T` on `pointer_chase` | 0.948 on full validation at step 27,500; paused, resumable |
| 5. Gate cells: 2 arms x 3 seeds x 4 families | `LOOP-T` seed 1 passes G0 after its deep-supervision stage; `FF-U` seed 1 in its deep-supervision stage; seeds 2 and 3 waiting |

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

## Fallback ladder on reachability at 8 visits (2026-10-02)

Each rung starts from the same snapshot (`LOOP-T-core4-s1` at step 32,840) and
trains `reachability` alone with the same data and curriculum, so it compares
directly with the plain 8-visit control. Full validation, 4,000 examples:

| Rung | Cell | Steps | Depths 1 to 8 | Pooled | Depths 9-12 |
| --- | --- | ---: | --- | ---: | ---: |
| more exposure (control) | `LOOP-T-warm-reach-v8-s1` | 20,000 | 1.000 1.000 0.996 0.994 0.992 0.884 0.834 0.706 | 0.926 | 0.657 |
| learning rate 0.0003, warm-up 1,000 | `LOOP-T-warm-reach-v8-lr3e4-s1` | 20,000 | 0.998 0.994 0.984 0.980 0.980 0.948 0.876 0.682 | 0.930 | 0.639 |
| deep supervision at every visit | `LOOP-T-warm-reach-v8-ds-s1` | 20,000 | 1.000 1.000 0.994 0.996 1.000 0.994 0.996 0.802 | 0.973 | 0.705 |
| deep supervision, budget extended | same cell, gate passed | 36,000 | 1.000 1.000 0.998 0.996 0.998 0.996 0.982 0.880 | 0.981 | 0.696 |
| input re-injection | `LOOP-T-warm-reach-v8-inj-s1` | 20,000 | 1.000 0.998 0.990 0.992 0.994 0.988 0.848 0.838 | 0.956 | 0.680 |
| 8-node task | not applicable | | | | |

Input re-injection disrupts the inherited weights (accuracy 0.000 at step 0) but
recovers within 2,500 steps; it ends with the best depth 8 of the 8-visit
rungs (0.838) and depth 7 at 0.848. The learning-rate rung matches the
control. Deep supervision solves depths 6 and 7 at 8 visits, so the visit cost measured on `LOOP-T-core4-s1` reflects how
that model was trained, not a hard limit; only depth 8 stays short. Its
training loss (0.345) is averaged over all eight visits, including early ones
that cannot yet be right, so it is not comparable with the other rungs'. Deep
supervision, extended to 40,000 steps, passed the gate on reachability at step
36,000 (0.981; depth 8 at 0.880), so G0 as written appears reachable at 8
visits through the goal's own ladder. The margin is thin: interim accuracy
moved between 0.967 and 0.982 over its last 1,500 steps.

The gate queue was therefore restarted on 2026-10-02 at 16:43
(`phase0_queue_4.log`) with a second stage: every gate cell that exhausts its
60,000-step budget is continued from its final weights, snapshotted with a
SHA-256 receipt, with deep supervision at every visit, for up to 40,000 more
steps (variant `core4-ds`). The 8-node rung does not apply
to reachability: a depth-`d` question needs a chain of breadth-first layers
reaching one past the horizon plus an unreachable node, at least `d + 3`
nodes, so an 8-node graph cannot hold a depth-8 question.

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
| 9 | Fallback ladder applied to a model that will not reach G0 | Applied to every gate cell as a fixed protocol: stage 1 plain training for 60,000 steps; if it exhausts, stage 2 deep supervision from its final weights for up to 40,000 steps; if that exhausts, stage 3 deep supervision plus input re-injection for up to 40,000 steps | Every seed-1 cell failed on reachability alone, so the same rungs were applied to all cells in the same order rather than chosen per cell after seeing its result. Each stage warm-starts from a SHA-256-checked snapshot of the previous stage. |
| 10 | Learning-rate rung | Tested once, on reachability alone, as a diagnostic; not part of the gate protocol | It matched the plain control (0.930 against 0.926), so it was not carried into the queue. |
| 11 | 8-node rung | Not run | A depth-8 reachability question needs at least 11 nodes, and on 8 nodes an 8-hop pointer path always returns to its start. |
| 12 | 8 visits only | Diagnostics at 12 and 16 visits, warm-started on reachability alone; the gate stays at 8 | To test whether the visit budget limits reachability. Both passed and 8 did not; deep supervision later passed at 8. The gate was not changed. |

## What was run and what was only written

- Run: all RMP tests under pytest 9.1.1; the bundle build and its rebuild; the
  pointer pilots; the plateau diagnostics; the visit-cost measurement; the
  visit-budget diagnostics (8, 12, 16 visits); the reachability ladder rungs
  (learning rate, deep supervision, input re-injection); the gate cells listed in
  the final tables; the capped runner and queue on GPU.
- Written and unit-tested but not used for any result: the per-variant gate
  summary (`report_rmp_phase0.py`, superseded by `report_rmp_phase0_final.py`)
  and C4 production (cells stop at C3).
- Paused early and not informative: the deep-supervision pointer pilot (step
  4,000), an `FF-U` pointer pilot (step 1,000), and reachability-alone training
  from scratch at 8 and 16 visits (steps 0 and 8,000).
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

Final result, budget exhausted at step 60,000 without passing G0 (full
validation, 4,000 examples per family):

| Family | Validation | Depths 1 to 8 (`sudoku4` 1 to 3) | Depths 9-12 | Training loss | Target entropy | Best constant |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `sudoku4` | 1.000 | 1.000 1.000 1.000 | none | 0.002 | 1.386 | 0.260 |
| `gated_pointer_chase` | 0.994 | 1.000 1.000 1.000 1.000 1.000 1.000 1.000 0.952 | 0.354 | 0.018 | 2.229 | 0.506 |
| `pointer_chase` | 0.992 | 1.000 1.000 1.000 1.000 1.000 1.000 0.998 0.936 | 0.005 | 0.018 | 2.773 | 0.073 |
| `reachability` | 0.945 | 1.000 0.996 0.998 0.984 0.990 0.980 0.830 0.778 | 0.664 | 0.205 | 0.693 | 0.511 |

Every family learned (training loss below target entropy, accuracy above the
best constant). Three pass the gate; `reachability` does not, because of depths
7 and 8. The added exposure, the first rung of the fallback ladder, raised it
from 0.836 at step 32,750 to 0.945, mostly by fixing depth 6. The pointer
families do not extrapolate (0.005 on depths 9-12 for `pointer_chase`); the
gated family's 0.354 there is largely the refusal answer, which is half the
labels.

Second stage, deep supervision from the step-60,000 weights
(`LOOP-T-core4-ds-s1`, snapshot `snapshots/LOOP-T-core4-s1-final.pt`): the
gate passed at step 8,500 on every family. This is the first cell to pass G0.

| Family | Validation | Depths 1 to 8 (`sudoku4` 1 to 3) | Depths 9-12 | Loss, mean over visits | Target entropy |
| --- | ---: | --- | ---: | ---: | ---: |
| `sudoku4` | 0.999 | 1.000 1.000 0.997 | none | 0.131 | 1.386 |
| `gated_pointer_chase` | 0.997 | 1.000 1.000 1.000 1.000 0.998 0.998 0.994 0.982 | 0.365 | 0.621 | 2.229 |
| `pointer_chase` | 0.995 | 1.000 1.000 1.000 1.000 0.994 1.000 0.998 0.968 | 0.010 | 0.960 | 2.773 |
| `reachability` | 0.987 | 1.000 1.000 1.000 0.998 0.998 1.000 0.978 0.924 | 0.591 | 0.279 | 0.693 |

Under deep supervision the recorded training loss averages all eight visits,
including early visits that cannot yet be right; it is below each family's
target entropy even so. Deep supervision first cost the pointer families
accuracy at depths 7 and 8 (pointer chasing 0.967 and gated 0.971 at step
2,500), and both recovered by step 5,000.

The untied arm's first seed, `FF-U-core4-s1` (1,839,616 parameters), also
exhausted its 60,000-step budget without passing G0:

| Family | Validation | Depths 1 to 8 (`sudoku4` 1 to 3) | Depths 9-12 | Training loss |
| --- | ---: | --- | ---: | ---: |
| `sudoku4` | 0.999 | 0.999 0.998 0.998 | none | 0.002 |
| `gated_pointer_chase` | 0.990 | 1.000 1.000 1.000 0.998 1.000 0.986 0.966 0.966 | 0.337 | 0.017 |
| `pointer_chase` | 0.998 | 1.000 1.000 1.000 1.000 1.000 1.000 0.992 0.988 | 0.009 | 0.004 |
| `reachability` | 0.885 | 0.998 0.972 0.928 0.906 0.896 0.868 0.842 0.668 | 0.645 | 0.181 |

Both arms fail on `reachability` alone. The tied loop falls short only at
depths 7 and 8; the untied stack falls short from depth 3 onward.

Curriculum events: `sudoku4` cleared all depths by step 1,750;
`gated_pointer_chase` reached depth 8 at step 15,250 and cleared it at 17,750;
`pointer_chase` reached depth 8 at 15,500 and cleared it at 26,000;
`reachability` unlocked depths 2 to 6 at steps 5,500, 11,750, 13,500, 15,500 and
21,000, and had not unlocked depth 7 by step 28,500.

Diagnostics run beside the queue, not gate cells: `LOOP-T-reach-v8-s101` and
`LOOP-T-reach-v16-s101` (reachability alone from scratch) were stopped at steps
0 and 8,000; the warm-start pair described above replaced them.

## To monitor, stop, or resume

```powershell
Get-Content experiments\recurrence_monitorability_v0\runs\logs\phase0_queue_6.log -Tail 5
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_rmp' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
$py = "C:\Users\patri\.venvs\bluebeam\Scripts\python.exe"; $work = "C:\Users\patri\Documents\Codex\rmp_v0"
$cmd = 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_rmp_phase0_queue.ps1 -Python {0} -PoolDir {1}\pool -ArtifactDir {1}\artifacts -SecondStage -ThirdStage >> experiments\recurrence_monitorability_v0\runs\logs\phase0_queue_7.log 2>&1"' -f $py, $work
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = (Get-Location).Path }
python -m research_gym.scripts.report_rmp_phase0_final
```

Run these from the repository root. The queue is started through
`Win32_Process Create` so it outlives the session that started it. Rerunning it
resumes every cell where it stopped and passes quickly over finished stages. The pilot resumes with
`scripts\start_rmp_cell_detached.ps1` and the arguments in
`runs/logs/LOOP-T-pointer-cur2-s101.runner.log`.

Checkpoints, pools, and predictions are under
`C:\Users\patri\Documents\Codex\rmp_v0`, outside OneDrive. Receipts and events
are under `experiments/recurrence_monitorability_v0/runs`.
