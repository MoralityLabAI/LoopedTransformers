# Checkpoint: RMP v0 Phase 2, E4 (paused 2026-10-12; N1 on LOOP-T seed 1 added 2026-10-13)

Phase 2 was paused by the program owner on 2026-10-12, early in E4, to give the shared GPU back after a week
of RMP training. On 2026-10-13 N1 was run on `LOOP-T` seed 1 alone, with the nine dictionaries it already had
(one short evaluation job); its label is `faithful_state`, met by a margin of a few examples (below). Controls 7
and 8 have not run, so the Phase 2 gate is still not met. The plan, with every rule fixed before any result,
is `docs/rmp_phase2_plan.md`.

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

Label, by the frozen rule: **`faithful_state`**, through width 2048 and width 4096 at `k` 32, the only two
dictionaries at 0.90 or more of clean in closed loop. The missing width-8192 runs cannot remove the label,
which needs only one dictionary. The label is met by the narrowest margin the rule allows. Each qualifying
dictionary beats its truncation control by 0.002 of clean: 5,336 against 5,324 and 5,422 against 5,411
correct of 6,000, margins of 12 and 11 examples. Pooled
over the three families, a sparse dictionary is no more faithful in closed loop than PCA at the same error.

The pooled tie hides a split by family (raw exact match on depths 5 to 8, dictionary against PCA):

| Width | k | Pointer chase | Gated pointer chase | Reachability |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 (selected) | 0.741 / 0.664 | 0.776 / 0.799 | 0.879 / 0.751 |
| 2048 | 32 | 0.868 / 0.911 | 0.878 / 0.936 | 0.922 / 0.814 |
| 4096 | 32 | 0.905 / 0.923 | 0.887 / 0.945 | 0.918 / 0.837 |

The dictionary keeps reachability better than PCA in all nine configurations, by 0.08 to 0.18. PCA keeps
gated pointer chasing better in eight of nine, and plain pointer chasing better in five of the six
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

## What remains, in order

1. The rest of the grids: `bash scripts/run_rmp_phase2_grids.sh /c/Users/patri/Documents/Codex/rmp_v0`
   (it skips finished configurations). About 6 minutes per `D-shared` configuration on a shared card, so
   roughly a day for all four models and the transcoders.
2. `D-tok` and `D-visit` at each model's selected size (`rmp_phase2_dictionaries --kind tok` / `--kind visit
   --sites 1 ... 8 --widths W --ks K`).
3. The planted model (`rmp_phase2_plant`), then its `D-shared` at width 1024, `k` 16.
4. N1 on the other models (`rmp_phase2_n1 --model ...`), which records each label. `LOOP-T s1` is done; rerun
   it once its width-8192 dictionaries and transcoders exist.
5. Controls 7 and 8 (`rmp_phase2_controls`), then 1, 2, 3, 9, 10.

Every GPU step should go through `vramorama run --vram 3G`, one at a time.
