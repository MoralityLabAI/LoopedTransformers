# Results: tied-loop transformer recipe on two new evals, 2026-11-26

Question: does the Phase 0 training recipe for the tied-loop transformer (LOOP-T, 8 visits, 459,776 parameters) transfer from the four core families to `masked_pointer_chase` and `modular_recurrence`, and how does it compare with the untied 8-layer stack (FF-U, 1,839,616 parameters)? Gate G0 is 0.98 exact match on the full 4,000-example validation split.

## Result

| Eval | LOOP-T | FF-U | Verdict |
|---|---|---|---|
| `masked_pointer_chase` | Passes on 3 of 3 seeds in the plain stage, at 6,250 / 18,750 / 15,750 steps | Passes on 3 of 3 seeds, only after the deep-supervision stage, at 71,250 / 66,750 / 87,750 total steps | Recipe transfers. The tied loop needs 3.6x to 11x fewer steps per seed. |
| `modular_recurrence` | Does not train, with or without the depth curriculum (0.054 exact after 30,000 steps without it) | Not run | Recipe does not fit this family at this size and budget. |

## `masked_pointer_chase`, gate-time full validation

| Cell | Seed | Steps to gate | Exact (4,000) | Depth 8 | Held-out depths 9-12 |
|---|---:|---:|---:|---:|---:|
| `LOOP-T-mask2` | 1 | 6,250 | 0.9805 | 0.900 | 0.269 |
| `LOOP-T-mask2` | 2 | 18,750 | 0.9843 | 0.922 | 0.319 |
| `LOOP-T-mask2` | 3 | 15,750 | 0.9847 | 0.910 | 0.286 |
| `FF-U-mask2` plain stage (60,000 steps, budget exhausted) | 1 | no gate | 0.6345 | 0.224 | 0.221 |
| `FF-U-mask2` plain stage | 2 | no gate | 0.9772 | 0.852 | 0.336 |
| `FF-U-mask2` plain stage | 3 | no gate | 0.8075 | 0.306 | 0.278 |
| `FF-U-mask2-ds` deep-supervision stage | 1 | +11,250 | 0.9843 | 0.904 | 0.391 |
| `FF-U-mask2-ds` deep-supervision stage | 2 | +6,750 | 0.9812 | 0.902 | 0.336 |
| `FF-U-mask2-ds` deep-supervision stage | 3 | +27,750 | 0.9835 | 0.936 | 0.352 |

All six passing cells clear the gate by 0.05 to 0.47 points, so the margins are thin. Held-out depths 9-12 are low for both arms (LOOP-T 0.27 to 0.32, FF-U 0.34 to 0.39): neither arm extrapolates in depth on this family. `FF-U-mask2-s2` stopped 0.3 points short of the gate at the end of its plain stage and passed after 6,750 deep-supervision steps.

## `modular_recurrence`

- Under the depth curriculum the cell never leaves depth 1. The depth-1 training pool has 131,435 examples but only 2,448 distinct inputs, and none of the 211 validation depth-1 inputs occur in it. A checkpoint scores 0.996 on the training pool and 0.09 to 0.10 on validation, so it memorises. Deeper pools do not have the problem (112,982 distinct inputs in 124,458 examples at depth 2, all 123,910 distinct at depth 8).
- Without the curriculum (`LOOP-T-modnc-s1`, all depths from step 0, 30,000 steps, 256 examples per batch) full-validation exact is 0.054, between 0.034 and 0.078 at every depth, and training loss stays at 2.83. The gate is therefore not the only blocker.
- The CPU twin `LOOP-T-modnccpu-s1` was stopped at step 4,500 (0.043 interim) and is not used in the conclusion.

## Throughput on this machine

GPU fp32 runs about 10 steps/s alone on the RTX 5080 Laptop (5.6 steps/s while sharing the card). CPU fp32 peaks at about 0.3 steps/s at 8 to 12 threads; bf16 autocast on CPU is 1.9x faster (about 0.8 steps/s sustained over 20,000 steps), and bf16 on the GPU is 0.93x, so it is off by default (`RMP_BF16=1` to enable). About 68% of each 64-token sequence is padding; skipping padded positions was not tried. A LOOP-T cell trained on both new families on CPU in bf16 reached 0.943 interim on `masked_pointer_chase` (0.047 on `modular_recurrence`) at 22,000 steps and was paused there, so it is evidence that CPU/RAM training works, not a gate result.

## Limits

- Three seeds per arm, one 0.98 gate, one model size. The FF-U versus LOOP-T step comparison counts optimizer steps, and the FF-U model has 4x the parameters.
- Gate-time numbers come from each cell's own `result.json`; interim accuracy in the logs is on a 512-example subset and runs a few points away from them (FF-U seed 3 shows 0.803 interim against 0.8075 full).
- The `c4` checkpoint step of the Phase 0 protocol is still pending for every passing cell (`gate_passed_c4_pending`).
- The LBF-20M local reproduction (5 of 5 repetitions) is separate and documented in the reference report; its bundled verifiers were not run because the bundle zip is missing.

## Receipts

- Reference report with throughput, profiles and the LBF-20M reproduction: [`checkpoint_aux_gym_masked_modular_20261126.md`](checkpoint_aux_gym_masked_modular_20261126.md)
- Per-cell events, results and queue logs: [`experiments/recurrence_monitorability_v0/aux_gym_20261126/`](../experiments/recurrence_monitorability_v0/aux_gym_20261126) (`cells/<cell>/events.jsonl`, `result.json`, `cell_status.md`, `logs/`)
- Runner and queue scripts: `scripts/run_rmp_training_cell.ps1` (stall watchdog, memory gate, one-worker default), `aux_gym_20261126/scripts/run_aux_queue_seed.ps1`
