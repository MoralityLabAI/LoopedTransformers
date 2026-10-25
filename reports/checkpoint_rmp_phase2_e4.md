# Checkpoint: RMP v0 Phase 2, E4 (paused 2026-10-12; N1 and controls 7-8 on LOOP-T seed 1 added 2026-10-13)

Phase 2 was paused by the program owner on 2026-10-12, early in E4, to give the shared GPU back after a week
of RMP training. On 2026-10-13 the gate's three items were run on `LOOP-T` seed 1 alone:

- N1 labels it `faithful_state`, met by a margin of a few examples (below).
- Control 7 (planted positive) passes.
- Control 8 (absent negative) passes on the two models it could be run on, seed 1 and its planted copy.

The Phase 2 gate is therefore met for `LOOP-T` seed 1. Control 8 still has to be run on the other three
gate-passing models once they have dictionaries. The plan, with every rule fixed before any result, is
`docs/rmp_phase2_plan.md`.

## Artifact loss and rebuild (2026-10-24 and 25)

When Phase 2 was resumed on 2026-10-24, the heavy artifact directory `C:\Users\patri\Documents\Codex\rmp_v0`
was gone. That meant the training pools, the monitor bundle, every checkpoint and snapshot, and the Phase 2
dictionaries and planted model. No session transcript records the deletion, and nothing was in the Recycle
Bin. The repository records survived: run records with each checkpoint's SHA-256, the pool and bundle
hashes, and every result JSON.

Everything is being rebuilt and checked against those hashes:

| Item | How | Check |
| --- | --- | --- |
| Monitor bundle | `prepare_rmp_task_bundle` | bundle hash `f1d81a3a...` matches |
| Six training pools | `prepare_rmp_train_pool` (four core families, two staircase) | all six SHA-256 match |
| `C0` of the four plain cells (control 1) | `rmp_regenerate_c0` (seeded initialization) | all four SHA-256 match |
| Trained models | retraining, `scripts/run_rmp_queue.sh` | every checkpoint against its run record (`rmp_verify_rebuild`) |
| Dictionaries, planted model | retraining after their parent models | dictionary SHA-256 recorded as `reproduced`; the planted model through its committed run record |

Training on this machine is bit-for-bit deterministic, so retraining recovers the same models rather than
new seeds. A fresh run of `LOOP-T-core4-s1` from the rebuilt pools reproduced its cell hash and every
recorded checkpoint through step 512. Paused at step 997 and resumed, it reproduced steps 1024 and 2048
too, so resuming is exact and the Phase 0 runs' 20-minute slicing does not matter. A model is used again
only after every one of its recorded checkpoints has matched. A mismatch stops the rebuild queue.

The rebuild runs from a dedicated worktree of this branch outside OneDrive
(`C:\Users\patri\Documents\Codex\LoopedTransformers-rmp`): another session had switched the OneDrive
checkout to another branch. Rebuilt run records go to `rmp_v0\rebuild\runs`, so the committed Phase 0 records
are not overwritten. The queue (`rmp_v0\queue\main.queue`, log `main.log`) then runs the rest of E4 in the
order of the plan. A new checkout needs the git-ignored `validation` and `ood_stress` splits under
`experiments/recurrence_monitorability_v0/datasets` (`prepare_rmp_task_bundle --out <that dir> --write validation
ood_stress`, or a copy; `read_shard` checks every shard's hash); training evaluates on them.

## What exists

Code, all tested (`tests/test_rmp_dictionaries.py`, `tests/test_rmp_association.py`; 8 tests):

- `research_gym/interp/dictionaries.py`: TopK autoencoders and transcoders, trained on states generated on
  the fly from a model.
- `research_gym/interp/replacement.py`: open-loop and closed-loop splices, the transcoder replacement model,
  and the matched-noise and PCA-truncation controls for N1.
- `research_gym/interp/association.py`: select-then-confirm feature-variable association for controls 1, 2,
  7 and 8.
- `research_gym/interp/phase2.py`: model list (including the planted model to be made), data splits and the
  dictionary selection rule.
- Scripts: `rmp_phase2_dictionaries.py` (grid and splice scoring), `rmp_phase2_n1.py` (N1 and its label),
  `rmp_phase2_plant.py` (control 7's planted model), `rmp_phase2_controls.py` (controls 7 and 8), and
  `scripts/run_rmp_phase2_grids.sh` (all grids, one `vramorama`-leased job at a time).

Results: 9 of the 12 `D-shared` configurations for `LOOP-T` seed 1
(`experiments/recurrence_monitorability_v0/phase2/dictionaries.json`; dictionaries under
`C:\Users\patri\Documents\Codex\rmp_v0\phase2\dictionaries`). Width 8192 had not run.

| Width | k | Calibration FVU | Cross-entropy recovered, minimum over visits 1-8 |
| ---: | ---: | ---: | ---: |
| 1024 | 8 | 0.029 | 0.929 |
| 1024 | 16 | 0.018 | 0.984 |
| 1024 | 32 | 0.013 | 0.989 |
| 2048 | 8 | 0.025 | 0.962 |
| 2048 | 16 | 0.015 | 0.990 |
| 2048 | 32 | 0.010 | 0.995 |
| 4096 | 8 | 0.022 | 0.969 |
| 4096 | 16 | 0.013 | 0.994 |
| 4096 | 32 | 0.008 | 0.996 |

The worst visit is visit 1 in every configuration. By the plan's selection rule, the selected configuration
for this seed is width 1024 with `k` 16, the smallest width and then the smallest `k` at 0.95 or more at every
visit; the missing width-8192 runs cannot change that. These are open-loop numbers only.

## N1 on LOOP-T seed 1 (2026-10-13)

`experiments/recurrence_monitorability_v0/phase2/n1.json`; command
`vramorama run --vram 3G -- python -m research_gym.scripts.rmp_phase2_n1 --model "LOOP-T s1" --artifact-dir
...\rmp_v0\artifacts --monitor-bundle ...\rmp_v0\bundle`, about 10 minutes. Scored on `validation`, depths 5 to
8 of the three graph families (6,000 examples, 2,000 per family). Clean exact match there is 0.986 (pointer
chase 0.990, gated pointer chase 0.993, reachability 0.975). Each entry is exact match as a fraction of
clean. "PCA" is the truncation control with the stated number of components (of 128), the fewest whose error
on the calibration states is at most the dictionary's.

| Width | k | Calibration MSE | PCA components | Closed loop | Noise control | PCA truncation | Open loop, worst single visit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 8 | 2.42 | 87 | 0.704 | 0.373 | 0.556 | 0.928 |
| 1024 | 16 | 1.54 | 96 | 0.810 | 0.503 | 0.749 | 0.968 |
| 1024 | 32 | 1.06 | 102 | 0.844 | 0.643 | 0.874 | 0.976 |
| 2048 | 8 | 2.06 | 90 | 0.718 | 0.408 | 0.646 | 0.923 |
| 2048 | 16 | 1.23 | 100 | 0.808 | 0.571 | 0.868 | 0.974 |
| 2048 | 32 | 0.84 | 106 | **0.902** | 0.715 | 0.900 | 0.988 |
| 4096 | 8 | 1.84 | 93 | 0.717 | 0.441 | 0.685 | 0.911 |
| 4096 | 16 | 1.05 | 103 | 0.874 | 0.623 | 0.871 | 0.983 |
| 4096 | 32 | 0.69 | 109 | **0.916** | 0.762 | 0.915 | 0.986 |
| 8192 | 8 | 1.67 | 95 | 0.719 | 0.464 | 0.715 | 0.916 |
| 8192 | 16 | 0.93 | 104 | 0.846 | 0.659 | 0.888 | 0.980 |
| 8192 | 32 | 0.60 | 110 | 0.923 | 0.805 | 0.926 | 0.990 |

Label, by the frozen rule: **`faithful_state`**, through width 2048 and width 4096 at `k` 32. The label is
met by the narrowest margin the rule allows. Each qualifying dictionary beats its truncation control by
0.002 of clean: 5,336 against 5,324 and 5,422 against 5,411 correct of 6,000, margins of 12 and 11
examples. Pooled over the three families, a sparse dictionary is no more faithful in closed loop than PCA at
the same error.

The width-8192 rows were added on 2026-10-25, after the rebuild; the other nine reproduced exactly (same
dictionaries by SHA-256, same numbers). Width 8192 at `k` 32 is the most faithful dictionary in closed loop
(0.923 of clean), but it does not qualify: its truncation control is 17 examples better (5,477 against
5,460 correct). The label is unchanged.

The pooled tie hides a split by family (raw exact match on depths 5 to 8, dictionary against PCA):

| Width | k | Pointer chase | Gated pointer chase | Reachability |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 (selected) | 0.741 / 0.664 | 0.776 / 0.799 | 0.879 / 0.751 |
| 2048 | 32 | 0.868 / 0.911 | 0.878 / 0.936 | 0.922 / 0.814 |
| 4096 | 32 | 0.905 / 0.923 | 0.887 / 0.945 | 0.918 / 0.837 |
| 8192 | 32 | 0.920 / 0.927 | 0.888 / 0.949 | 0.921 / 0.863 |

The dictionary keeps reachability better than PCA in all twelve configurations, by 0.06 to 0.18. PCA keeps
gated pointer chasing better in eleven of twelve, and plain pointer chasing better in seven of the eight
configurations with `k` 16 or more (all but width 1024, `k` 16); at `k` 8 the dictionary is better on plain
pointer chasing. One reading, untested: the sparse dictionary keeps the near-binary search
variables (goal found, latch) that Phase 1 found for reachability, but loses some of the dense node-identity
code that pointer chasing carries; PCA keeps the high-variance node code. Two other observations:

- Errors compound around the loop. Every dictionary keeps 0.91 or more of clean when spliced in at any single
  visit, but only 0.70 to 0.92 when spliced in after all eight. The worst single visit for exact match is
  usually `resid[4]`, while for cross-entropy it was `resid[1]`.
- The model tolerates the dictionary's error far better than random error of the same size. The noise
  control keeps 0.37 to 0.76 of clean, 0.15 to 0.33 below the dictionary. Most of the dictionary's error lies
  in directions the loop does not read; PCA truncation shares this property.

No transcoder had been trained, so the replacement-model row is empty.

One change to the N1 script was made before this run. The plan sets the truncation control's component count
by its error on `monitor_calibration`. The script had used the residual variance of the 400 `monitor_train`
examples the PCA was fitted on. It now measures the truncation error on the same states the dictionary's error is
measured on: 300 examples from the selection half of `monitor_calibration`, all positions, visits 1 to 8.

## Controls 7 and 8 on LOOP-T seed 1 (2026-10-13)

`experiments/recurrence_monitorability_v0/phase2/controls_7_8.json`.

**Planted model.** Made by `rmp_phase2_plant` (cell `LOOP-T-core4-plant-s1`): 3,000 steps from the parent's
`C3` with the stage-2 recipe plus the auxiliary loss, about 80 minutes on the shared card. The auxiliary
loss fell from 1.33 to 0.09 within 250 steps and ended at 0.0005. The task loss, averaged over visits,
ended at 0.53 against 0.51 at the start. Task accuracy on `validation` is essentially unchanged from the
parent:

| Family | Parent (`C3`) | Planted |
| --- | ---: | ---: |
| `pointer_chase` | 0.995 | 0.994 |
| `gated_pointer_chase` | 0.997 | 0.998 |
| `reachability` | 0.987 | 0.984 |
| `sudoku4` | 0.999 | 1.000 |

Its `D-shared` at the parent's selected size (width 1024, `k` 16) has calibration FVU 0.017 and minimum
open-loop cross-entropy recovered 0.972, so it meets the selection bar.

**Control 7, planted positive: passes.** The variable is the parity of the input token at position 4.

| Model | Selected site, feature | Report AUROC | p | Found | Decoder cosine to `u` | Largest absolute cosine, any feature |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Planted | `resid[7]`, 875 | 0.987 | 0.001 | yes | 0.838 | 0.866 |
| Parent | `resid[1]`, 127 | 0.379 | 0.001 | no | -0.174 | 0.299 |

The pipeline finds the planted variable in the planted model, on a feature whose decoder lies along the
planted direction. In the parent it does not find the variable, and no feature of the parent's dictionary
comes within 0.5 of the direction. The parent does carry a weak trace of the token's parity at the first
visit, which is to be expected of an input token. Its AUROC of 0.38 is significant, but it is 0.12 from chance
against the 0.35 the rule requires.

**Control 8, absent negative: passes on both models run.**

- Seed 1: selected `resid[6]`, feature 904, report AUROC 0.500, p 0.90.
- Planted model: selected `resid[4]`, feature 367, report AUROC 0.502, p 0.54.

`LOOP-T` seeds 2 and 3 and `FF-U` seed 3 were skipped because they have no dictionaries yet. The plan runs
control 8 on every model, so this control is incomplete until they do.

Two notes on the run:

- **The first attempt failed silently.** It was the last step of a chained job, behind a `--quiet` lease. It
  ended after 85 minutes with exit code 0, no output and no result file. The cause is unknown. A lease that
  times out exits with code 2 and prints a message, both checked afterwards, and no orphaned process was
  left. The rerun, with lease messages logged, was admitted in 16 seconds and finished in 76.
- **A summary bug was fixed first.** The control-8 summary now lists the models it covers, and it no longer
  reports a pass when no model was run; `all([])` had made that possible.

## What remains, in order

1. The rest of the grids: `bash scripts/run_rmp_phase2_grids.sh /c/Users/patri/Documents/Codex/rmp_v0`
   (it skips finished configurations). About 6 minutes per `D-shared` configuration on a quiet card, so
   roughly a day for all four models and the transcoders. On a busy card it is much slower: the planted
   model's single configuration took 36 minutes on 2026-10-13.
2. `D-tok` and `D-visit` at each model's selected size (`rmp_phase2_dictionaries --kind tok` / `--kind visit
   --sites 1 ... 8 --widths W --ks K`).
3. N1 on the other models (`rmp_phase2_n1 --model ...`), which records each label. `LOOP-T s1` is done; rerun
   it once its width-8192 dictionaries and transcoders exist.
4. Control 8 on `LOOP-T` seeds 2 and 3 and `FF-U` seed 3 (`rmp_phase2_controls`; it skips models without
   dictionaries and recomputes the others in about a minute each). Control 7 is done.
5. Controls 1, 2, 3, 9, 10.

Every GPU step should go through `vramorama run`, one at a time, without `--quiet`, so that lease waits are
logged. Training jobs have used `--vram 3G`, but stage-2 training peaked at 535 MB allocated. The controls
ran in `--vram 1G`.
