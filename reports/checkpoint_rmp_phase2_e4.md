# Checkpoint: RMP v0 Phase 2, E4 (paused 2026-10-12)

Phase 2 was paused by the program owner on 2026-10-12, early in E4, to give the shared GPU back after a week
of RMP training. Nothing in the Phase 2 gate has been measured: N1 has not run, and controls 7 and 8 have not
run. The plan, with every rule fixed before any result, is `docs/rmp_phase2_plan.md`.

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
visit; the missing width-8192 runs cannot change that. These are open-loop numbers only. Whether the loop can
run on its own reconstructions is what N1 measures, and it has not run.

## What remains, in order

1. The rest of the grids: `bash scripts/run_rmp_phase2_grids.sh /c/Users/patri/Documents/Codex/rmp_v0`
   (it skips finished configurations). About 6 minutes per `D-shared` configuration on a shared card, so
   roughly a day for all four models and the transcoders.
2. `D-tok` and `D-visit` at each model's selected size (`rmp_phase2_dictionaries --kind tok` / `--kind visit
   --sites 1 ... 8 --widths W --ks K`).
3. The planted model (`rmp_phase2_plant`), then its `D-shared` at width 1024, `k` 16.
4. N1 on each model (`rmp_phase2_n1 --model ...`), which records the label.
5. Controls 7 and 8 (`rmp_phase2_controls`), then 1, 2, 3, 9, 10.

Every GPU step should go through `vramorama run --vram 3G`, one at a time.
