# Results: tied-loop transformer recipe on two new evals, 2026-11-26

Question: does the Phase 0 training recipe for the tied-loop transformer (LOOP-T, 8 visits, 459,776 parameters) transfer from the four core families to `masked_pointer_chase` and `modular_recurrence`, and how does it compare with the untied 8-layer stack (FF-U, 1,839,616 parameters)? Gate G0 is 0.98 exact match on the full validation split (4,000 examples for `masked_pointer_chase`, 3,711 for `modular_recurrence`).

Revised after an external review of `cc07e5ef`. Every number in the first version reproduced from the receipts; three verdicts claimed more than the numbers carry and are corrected below.

## Result

| Eval | LOOP-T | FF-U | What the receipts support |
|---|---|---|---|
| `masked_pointer_chase` | Crosses the gate on 3 of 3 seeds in the plain stage, at 6,250 / 18,750 / 15,750 steps | Does not cross in 60,000 plain steps on any seed (full-validation exact 0.634 / 0.977 / 0.808); crosses on 3 of 3 seeds after deep supervision, 11,250 / 6,750 / 27,750 steps later | LOOP-T crosses in at most 18,750 plain steps where FF-U does not cross in 60,000. The two arms were not run under the same schedule, so no step ratio is claimed. |
| `modular_recurrence` | One seed, no depth curriculum: at chance after 30,000 steps (0.054 exact; chance is 1/17 = 0.059) | One seed, no depth curriculum, same settings: at chance after 30,000 steps (0.063 exact) | Nothing beyond one seed per arm at chance. |

## `masked_pointer_chase`, gate-time full validation

| Cell | Seed | Steps to gate | Exact (4,000) | Examples over the 3,920 line | Depth 8 | Held-out depths 9-12 |
|---|---:|---:|---:|---:|---:|---:|
| `LOOP-T-mask2` | 1 | 6,250 | 0.9805 | +2 | 0.900 | 0.269 |
| `LOOP-T-mask2` | 2 | 18,750 | 0.9843 | +17 | 0.922 | 0.319 |
| `LOOP-T-mask2` | 3 | 15,750 | 0.9847 | +19 | 0.910 | 0.286 |
| `FF-U-mask2` plain stage (60,000 steps, budget exhausted) | 1 | no gate | 0.6345 | -1,382 | 0.224 | 0.221 |
| `FF-U-mask2` plain stage | 2 | no gate | 0.9772 | -11 | 0.852 | 0.336 |
| `FF-U-mask2` plain stage | 3 | no gate | 0.8075 | -690 | 0.306 | 0.278 |
| `FF-U-mask2-ds` deep-supervision stage | 1 | +11,250 | 0.9843 | +17 | 0.904 | 0.391 |
| `FF-U-mask2-ds` deep-supervision stage | 2 | +6,750 | 0.9812 | +5 | 0.902 | 0.336 |
| `FF-U-mask2-ds` deep-supervision stage | 3 | +27,750 | 0.9835 | +14 | 0.936 | 0.352 |

How much weight the crossings carry:

- **Margins are within noise for some cells.** One standard error at 0.98 on 4,000 examples is about 9 examples. LOOP-T seed 1 (+2) and FF-U deep-supervision seed 2 (+5) are inside it, and each pass is a first crossing chosen by the stop rule.
- **Selection.** `FF-U-mask2-ds-s1` had seven interim readings at or above 0.98 (steps 7,500 to 10,750) before the step where it passed. The events log records no full-validation result for any trigger that failed, so those evaluations are not in the receipts.
- **No confirming reading yet.** The `c4` checkpoint is the one full evaluation not chosen by the stop rule. It has not run (`gate_passed_c4_pending`), so every passing cell has one crossing and no confirmation.
- **FF-U needs deep supervision: two seeds clearly, one within noise.** FF-U seed 2 missed the plain-stage gate by 11 examples on a single full reading at step 60,000. On seeds 1 and 3 the depth curriculum stalled at depths 4 and 6 in the plain stage, so those cells never trained on deeper examples. Neither arm ran deep supervision from scratch, which is why the table gives step counts per stage and no ratio.
- **Held-out depths 9-12 show no extrapolation by either arm** (LOOP-T 0.27 to 0.32, FF-U 0.34 to 0.39). The FF-U edge is probably abstention: the generator puts a hole on the path about half the time, so about half of the targets are ABSTAIN. By the reviewer's calculation from `tasks.py`, a model that tracks 8 hops and abstains at a hole scores about 0.44, 0.40, 0.36 and 0.33 at depths 9 to 12 with no correct pointer answers, and `FF-U-mask2-ds-s1` scores 0.47, 0.42, 0.35 and 0.32. The shipped data supports the premise: one target token covers 0.494 of the held-out depths (0.509 of validation), and every trained model scores below that constant at depths 9-12 (see [`datasets/README.md`](../experiments/recurrence_monitorability_v0/aux_gym_20261126/datasets/README.md)). The calculation itself is an inference: the prediction files are not in the repo. Splitting exact match into ABSTAIN targets and node targets would settle it.

## `modular_recurrence`

- **Under the depth curriculum** the cell never leaves depth 1. The depth-1 training pool has 131,435 examples but only 2,448 distinct inputs, and none of the 211 validation depth-1 inputs occur in it. A checkpoint scores 0.996 on the training pool and 0.09 to 0.10 on validation, which is memorisation. The reviewer traces the cause to the bundle's split: it splits by structure hash, which for this family is the operation list. At depth 1 that is 272 operations, 144 of them in training and the rest in validation, so unlocking depth 2 requires generalising to unseen operations. I have not checked the 272 / 144 counts.
- **Without the curriculum** (`LOOP-T-modnc-s1`, all depths from step 0, 30,000 steps, 256 examples per batch, one seed) full-validation exact is 0.0544 on 3,711 examples, with 0.034 to 0.078 at every depth. Training loss is 2.83 against a recorded target entropy of 2.8332 nats (ln 17), and exact match is about 1/17. That is chance. The cell did not even memorise the depth-1 pool that the curriculum cell memorised, so this is an optimisation plateau on one seed. Whether model size is the limit was not tested, because size was never varied.
- **FF-U on this family** (`FF-U-modnc-s1`, same settings: one seed, no depth curriculum, 30,000 steps, 1,839,616 parameters) is also at chance: full-validation exact 0.0628 on 3,711 examples (0.054 to 0.085 at every depth), 0.0575 on held-out depths 9-12, against 1/17 = 0.0588 (one standard error at chance is about 0.004). Its final training loss is 2.67 against the 2.8332 entropy, a small drop that has not carried over to validation. Four times the parameters and an untied stack do not change the outcome in this setting; I did not run deep supervision or the curriculum for FF-U on this family.
- The CPU twin `LOOP-T-modnccpu-s1` stopped at step 4,500 (0.043 interim) and is not used in the conclusion.

## Throughput on this machine

GPU fp32 runs about 10 steps/s in the isolated benchmark. Per-cell rates (optimizer steps over `training_seconds`) range from 5 to 24 steps/s across the GPU cells, depending on how many families a cell trains and what else held the card. CPU fp32 peaks at about 0.3 steps/s at 8 to 12 threads. bf16 autocast is 1.9x faster on CPU in the 30-step benchmark (deep supervision on) and 0.93x on the GPU, so it is off by default (`RMP_BF16=1` to enable). The long CPU cell `LOOP-T-aux2cpu-s1` ran at about 0.8 steps/s early on without deep supervision, fell to about 0.24 steps/s once the GPU queue started, and averaged 0.66 steps/s over its run. It reached 0.943 interim on `masked_pointer_chase` and 0.047 on `modular_recurrence` at 22,000 steps and was paused, so it is evidence that CPU/RAM training runs, not a gate result. About 68% of each 64-token sequence is padding; skipping padded positions was not tried.

## Limits

- **Seeds:** three seeds per arm for `masked_pointer_chase`; one seed for `modular_recurrence` (plus a paused CPU cell). All seeds share one validation sample, which the curriculum and the stop rule both read.
- **One gate, one model size per arm.** FF-U has 4x the parameters of LOOP-T.
- **Interim versus full:** interim accuracy in the logs is on a 512-example subset and runs a few points from the full evaluation (FF-U seed 3 shows 0.803 interim against 0.8075 full).
- **LBF-20M reproduction (separate):** the bundled verifiers (20L identity, fresh reference, diagram fixture, CI archive) were not run because the bundle zip is missing. `verify_local.py` checks the run's own files against counts written alongside the run, the five repetitions differ only in timing fields, and the "CI" column in the LBF `REPORT.md` is hardcoded in `make_report.py` with no source cited. `SHA256SUMS.txt` fails as committed for 7 of 75 entries that name paths from the original folder layout (`launcher.done`, `launcher.err`, `theory.stdout`, `theory.stderr`, `report/*`); the hashes match the files where they now sit.

## Receipts

- Reference report with throughput, profiles and the LBF-20M reproduction: [`checkpoint_aux_gym_masked_modular_20261126.md`](checkpoint_aux_gym_masked_modular_20261126.md)
- Per-cell events, results and queue logs: [`experiments/recurrence_monitorability_v0/aux_gym_20261126/`](../experiments/recurrence_monitorability_v0/aux_gym_20261126) (`cells/<cell>/events.jsonl`, `result.json`, `cell_status.md`, `logs/`)
- Runner and queue scripts: `scripts/run_rmp_training_cell.ps1` (stall watchdog, memory gate, one-worker default), `aux_gym_20261126/scripts/run_aux_queue_seed.ps1`
