# Aux gym datasets, 2026-11-26

The data behind the `masked_pointer_chase` and `modular_recurrence` runs in [`../cell_status.md`](../cell_status.md) and the reports [`results_aux_gym_masked_modular_20261126.md`](../../../../reports/results_aux_gym_masked_modular_20261126.md) and [`checkpoint_aux_gym_masked_modular_20261126.md`](../../../../reports/checkpoint_aux_gym_masked_modular_20261126.md). Bundle `rmp_task_bundle_v0`, bundle hash `f1d81a3afde4fab4cc07d70949e6f87eb8c1ff5766e1b0cace206a7456e4d265` (from `../../datasets/manifest.json`). Byte counts and SHA-256 hashes for every file are in [`dataset_index.json`](dataset_index.json). Total size 60 MB; files are stored byte-exact (`.gitattributes` is `-text` one level up).

## Contents

| Path | What | Size |
|---|---|---:|
| `validation/<family>.jsonl` | Gate split: 4,000 examples per family (3,711 for `modular_recurrence`), six families | 18.3 MB |
| `ood_stress/<family>.jsonl` | Held-out depths 9-12, 500 examples per depth, 2,000 per family, five families (no `sudoku4`) | 8.5 MB |
| `train_pools/masked_pointer_chase.pt.gz`, `modular_recurrence.pt.gz` | First 1,000,000 training-stream examples per family (int16 token tensors, saved with `torch`), gzip -9 | 20.4 MB, 12.3 MB |
| `train_pools/masked_pointer_chase.staircase.pt.gz` | 125,000 examples of the easier-than-evaluated depths for the staircase pool | 2.4 MB |
| `train_pools/pool_receipt.json` | `sha256` of each uncompressed `.pt`, example count, `config_sha256` of the bundle config | |
| `dataset_index.json` | Hashes and sizes of every file, plus the per-split statistics below | |

Each shard record is one JSON object per line: `tokens` (64 ids), `target_token`, `difficulty` (depth, or the sudoku level), `semantic` (hops / mapping / operations / start), `fingerprint`, `group_id`, `example_id`, `facts`, `sibling`, `split`. The training pools are uncompressed int16 tensors, so unpack before loading: `gunzip -k masked_pointer_chase.pt.gz`, then check `sha256sum masked_pointer_chase.pt` against `pool_receipt.json`. Every `.pt.gz` was decompressed and matched to the receipt hash before this commit. Cast tokens with `.long()` before use.

## Statistics

| Split / family | Examples | Per depth | Distinct targets | Most common target (share) |
|---|---:|---|---:|---:|
| validation `masked_pointer_chase` | 4,000 | 500 at each of depths 1-8 | 17 | 0.509 |
| validation `modular_recurrence` | 3,711 | 211 at depth 1, 500 at depths 2-8 | 17 | 0.063 |
| validation `pointer_chase` | 4,000 | 500 at each of depths 1-8 | 16 | 0.073 |
| validation `gated_pointer_chase` | 4,000 | 500 at each of depths 1-8 | 17 | 0.506 |
| validation `reachability` | 4,000 | 500 at each of depths 1-8 | 2 | 0.511 |
| validation `sudoku4` | 4,000 | 1,334 / 1,333 / 1,333 at levels 1-3 | 4 | 0.260 |
| ood_stress `masked_pointer_chase` | 2,000 | 500 at each of depths 9-12 | 17 | 0.494 |
| ood_stress `modular_recurrence` | 2,000 | 500 at each of depths 9-12 | 17 | 0.071 |
| ood_stress `pointer_chase` | 2,000 | 500 at each of depths 9-12 | 16 | 0.072 |
| ood_stress `gated_pointer_chase` | 2,000 | 500 at each of depths 9-12 | 17 | 0.490 |
| ood_stress `reachability` | 2,000 | 500 at each of depths 9-12 | 2 | 0.504 |

Every example in every shard has a distinct fingerprint and group id, and every sequence is 64 tokens.

## What the statistics change in the results

- **Constant baseline on `masked_pointer_chase`.** One target token covers 0.509 of validation and 0.494 of the held-out depths (consistent with ABSTAIN, which the generator emits when the path meets a hole). On held-out depths 9-12 every trained model scored 0.22 to 0.39 exact, below the 0.494 constant. The held-out comparison between LOOP-T and FF-U therefore says little about extrapolation, which supports the abstention reading in the results report.
- **`modular_recurrence` has no dominant target.** The best constant scores 0.063 on validation, and the trained cells score 0.054 (LOOP-T) and 0.063 (FF-U), so both sit at the constant and at 1/17.
- **`modular_recurrence` validation is smaller at depth 1** (211 examples), which is why the validation total is 3,711 and not 4,000. The training pool at depth 1 has 2,448 distinct inputs (see `../modular_check2.out.txt`).

## Rebuild the training pools

From a checkout with `configs/rmp_task_bundle_v0.json` (the commands the pools were built with; the validation and held-out shards come from the bundle builder and are checked against `../../datasets/manifest.json`):

```powershell
python -m research_gym.scripts.prepare_rmp_train_pool --families masked_pointer_chase modular_recurrence --size 1000000 --workers 4 --out <pool-dir>
python -m research_gym.scripts.prepare_rmp_train_pool --families masked_pointer_chase modular_recurrence --size 125000 --workers 4 --staircase --out <pool-dir>
```

`modular_recurrence` has no staircase pool: generating it fails with `IndexError: Cannot choose from an empty sequence`, and the trainer skips a missing staircase pool, so there is no `modular_recurrence.staircase` entry. Rebuilding was not repeated for this commit; the hashes in `pool_receipt.json` are those of the files shipped here.
