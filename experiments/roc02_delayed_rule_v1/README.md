# ROC-02 prospective delayed-rule source and exact-oracle gym

**Source registered 16 November 2026.** Frozen source implementation: [`configs/roc02_delayed_rule_source_impl_v1.json`](../../configs/roc02_delayed_rule_source_impl_v1.json). This is an implementation supplement to MathCan ROC-01 Part 4 and Amendment C, preserving the original six model seeds, split seeds and dataset sizes, nine observer ranks, 128 rank-null seeds and physical intervention families.

## Inputs, audit and baseline

An active source A binds one of eight bijective modular-affine lookup rules over values `0..7`. Distractor events rewrite sources B/C during a 2–16-step delay. A final query asks for A's mapped output. Every source event is encoded into **79 binary features** (event type, source, 8×8 lookup table and query). Neural inputs contain the table itself, while the hidden generator family ID is excluded. The independent dictionary interpreter and affine arithmetic oracle both verify every labeled case.

The original registered seeds generate 1,024 training cases (families 0–3), 256 calibration cases (families 4–5) and 384 test inputs (families 6–7), with 24 separate invalid no-rule/out-of-domain fixtures. Source-family partitions and case identities are SHA-256 verified, and exact-rule overlap is rejected. All 1,664 valid labels are verified against the independent symbolic interpreter. The symbolic baseline achieves **100%** on the generated valid cases and correctly rejects 24/24 invalid fixtures by construction.

The compressed `roc02_source_cases.tar.gz` contains four named JSONL members; `source_manifest.json` pins the SHA-256 of each uncompressed file. The public sealed-test input member omits `label` fields. This is a *procedural* holdout: because the input contains the lookup table and the generator is public, the labels can be reconstructed. Authors must refrain from using sealed inputs for rank, penalty or network selection. The prospective evaluation must declare source/checkpoint and observer freeze commits before labels are scored.

## Run

```sh
python -m research_gym.analysis.roc02_delayed_rule_source --out /tmp/roc02_cases
python -m unittest tests.test_roc02_delayed_rule_source tests.test_roc02_dataset_archive -v
```

The source generator is deterministic, with no model weights trained by this commit. The registered model trains at tied depths 8/16 on six fresh seeds, with 240 optimizer updates of batch size 8. The next staged tasks are checkpoint training, exact syntax/label QA, held-out task selection, then prospective ROC report cards and physically norm-matched causal interventions. Refrain from promoting any future result without the source, null and evidence gates in ROC-01 Amendment C.

## First six-seed training cohort (calibration only)

The registered 240-update training campaign has completed for all six model seeds at depths 8 and 16. [Per-seed calibration results](roc02_train_calibration_by_seed.csv), the [summary](roc02_train_calibration_summary.json), [step-zero retrospective reference](roc02_stepzero_diagnostics.json), and a [critical limitations report](ROC02_CALIBRATION_REPORT.md) are included here. The original 12 model/optimizer checkpoints and individual full training receipts are available as six [SHA-locked ZIP bundles](checkpoints/roc02_checkpoint_archive_index.json); the [pure-stdlib verification and unpacker](../../research_gym/analysis/roc02_checkpoint_archive.py) checks all saved model hashes. All final models were trained from scratch after the trainer source was committed at `a4f912b`.

The held-out source-family task labels have not been opened. No ROC causal intervention or rank-optimization conclusion is asserted from these early, low-accuracy models.
