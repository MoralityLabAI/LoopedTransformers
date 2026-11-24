# Audit scripts for the independent LBF-14–17 referee review

Reviewed commit: `3d8243818858c19c5031c36e2bf7e328f831cf25`.
Pinned simulator: `research/lean_cgrt_hrm400/ppt_gym/rewrite_gym.py` at `cd9ac784c5804ec2fe595bb6a01d2697fc016610` (git blob `204a1d012a4ebe8e80d8bd722590742b7e7f02ba`).

## Custody rules the scripts enforce

- Nothing under `results/` is written. Archives are decompressed into a scratch directory.
- The sealed `ordinary` / `ood` files are never parsed. `common.read_jsonl_gz` refuses them; `audit_05` hashes their decompressed byte stream only. The generator is only ever asked for `train` and `calibration`, which are drawn before any sealed split.
- No optimiser is constructed; checkpoints are loaded read-only after their registered SHA-256 is verified.
- `audit_01` draws extra rule systems from audit-only generator seeds `990000001–990000004`, which appear in no registration. Exclude them by lineage from any future fresh cohort.

## Setup

```bash
REPO=$(git rev-parse --show-toplevel)
A=$REPO/research/lite_braille_fresco/review/independent_referee_20261124/audit
mkdir -p /tmp/lbf_pinned /tmp/lbf_scratch
git -C "$REPO" fetch --depth 1 origin cd9ac784c5804ec2fe595bb6a01d2697fc016610
git -C "$REPO" show cd9ac784c5804ec2fe595bb6a01d2697fc016610:research/lean_cgrt_hrm400/ppt_gym/rewrite_gym.py > /tmp/lbf_pinned/rewrite_gym.py
SIM=/tmp/lbf_pinned/rewrite_gym.py
LEAN=/path/to/lean-4.19.0/bin/lean          # commit 6caaee842e94
export LCGRT_LEAN_HOME=$(dirname "$(dirname "$LEAN")")
pip install torch                            # the audit ran on torch 2.15.0, CPU, one thread
```

## Scripts

| Script | Question | Runtime (2 cores) |
|---|---|---|
| `audit_01_soundness_counterexample.py` | Is root-backtracking suppression sound? Counterexample `R*` on the symbolic interpreter and all archived `rel_prov_*` checkpoints; scan of the open cohorts; generator Monte Carlo; a minimal repair. | ~12 min |
| `audit_02_lean_recheck.py` + `LBF17AuditAppendix.lean` | Recompile the registered Lean file; kernel-check the counterexample against the registered definitions. | ~15 s |
| `audit_03_replication_factorial.py` | Re-run every registered B = 128 episode from the archived checkpoints; per-root budget curves; the joint `env_mat_zero` intervention; implicit one-hop enumeration counts. | ~40 min (`--shard 0/2`, `--shard 1/2`, then `--merge`) |
| `audit_04_statistics.py` | Record-level recomputation of the manifest; exact closure bounds; sign-flip tests with Holm adjustment; attribution-ratio interval; AUROC decomposition; sample sizes. Standard library only. | ~1 min |
| `audit_05_custody_and_certificates.py` | Every archived hash; suite determinism; regeneration of `Proofs.lean` and (with `--lean`) kernel recheck of all 2,481 declarations. | ~3 min |
| `audit_06_symbolic_controls.py` | Missing zero-parameter controls (`absint_mm`, `absint1_mm`, `exact_mm`) and bidirectional BFS at larger budgets. | ~5 min |
| `audit_07_precision_horizon_collisions.py` | float32 support versus symbolic support for K = 1…12; horizon overshoot; fiber collisions; fiber-count sweep. | ~6 min |
| `audit_08_efficiency_profile.py` | Instrumentation inside the timed region; decision cost on identical states; FLOP-counter coverage; measured versus estimated memory; time split. Run on an idle machine. | ~2 min |
| `audit_09_synthesis.py` | Replication rates, AUC intervals, factorial, equalised successor accounting, cache reuse, wall clock. | seconds |
| `audit_10_horizon_examples.py` | Word-level horizon witness; first concrete distance of K = 3 "false positives". | seconds |
| `audit_11_information_and_window_accounting.py` | Rule-window matching in one unit for all arms; concrete ball sizes versus abstract state size. | seconds |
| `audit_12_trajectory_equivariance.py` | Alphabet-permutation equivariance at the level of whole episodes. | ~2 min |
| `audit_13_prior_archives.py` | Record-level recomputation of the LBF-14, 15 and 16 archives. | seconds |

```bash
python $A/audit_01_soundness_counterexample.py --repo $REPO --pinned-sim $SIM --scratch /tmp/lbf_scratch --out $A/outputs/audit_01_soundness_counterexample.json
python $A/audit_02_lean_recheck.py --repo $REPO --lean $LEAN --scratch /tmp/lbf_scratch --out $A/outputs/audit_02_lean_recheck.json
for i in 0 1; do python $A/audit_03_replication_factorial.py --repo $REPO --pinned-sim $SIM --scratch /tmp/lbf_scratch_$i --out-dir /tmp/lbf_a03 --shard $i/2 & done; wait
python $A/audit_03_replication_factorial.py --repo $REPO --pinned-sim $SIM --out-dir /tmp/lbf_a03 --merge
python $A/audit_04_statistics.py --repo $REPO --out $A/outputs/audit_04_statistics.json --fsi-export-dir $A/outputs/fsi_receipts
python $A/audit_05_custody_and_certificates.py --repo $REPO --pinned-sim $SIM --scratch /tmp/lbf_scratch --lean $LEAN --out $A/outputs/audit_05_custody_and_certificates.json
python $A/audit_06_symbolic_controls.py --repo $REPO --pinned-sim $SIM --out $A/outputs/audit_06_symbolic_controls.json
python $A/audit_07_precision_horizon_collisions.py --repo $REPO --pinned-sim $SIM --scratch /tmp/lbf_scratch --out $A/outputs/audit_07_precision_horizon_collisions.json
python $A/audit_08_efficiency_profile.py --repo $REPO --pinned-sim $SIM --scratch /tmp/lbf_scratch --out $A/outputs/audit_08_efficiency_profile.json
python $A/audit_09_synthesis.py --repo $REPO --rows /tmp/lbf_a03/audit_03_rows.jsonl --symbolic $A/outputs/audit_06_symbolic_controls.json --out $A/outputs/audit_09_synthesis.json
python $A/audit_10_horizon_examples.py --repo $REPO --pinned-sim $SIM --out $A/outputs/audit_10_horizon_examples.json
python $A/audit_11_information_and_window_accounting.py --repo $REPO --out $A/outputs/audit_11_information_and_window_accounting.json
python $A/audit_12_trajectory_equivariance.py --repo $REPO --pinned-sim $SIM --scratch /tmp/lbf_scratch --out $A/outputs/audit_12_trajectory_equivariance.json
python $A/audit_13_prior_archives.py --repo $REPO --out $A/outputs/audit_13_prior_archives.json
```

`outputs/` holds the results of the run this review is based on. `outputs/audit_03_rows.jsonl.gz` is the merged per-episode table (6,432 rows); pass it to `audit_09` with `--rows`. `outputs/fsi_receipts/` holds per-case paired outcomes in the column format expected by `research_gym/scripts/evalmath_fsi_gate.py` (`case_id`, `root_id`, `candidate_correct`, `control_correct`); the gate itself was not run because the EvalMath package is not vendored at the reviewed commit.

## Status of the added controls

`absint_mm`, `absint1_mm`, `exact_mm` and `env_mat_zero` are post hoc and unregistered. They were defined before being run and were not tuned; one correctness fix (the no-intersection bound for `absint1_mm`, `2K + 1` instead of `7`) was made after a three-root smoke test and before the 48-root run. They use calibration roots that had already been scored, so they are evidence about alternative explanations, not confirmatory results.

Timings are machine-dependent. Outcomes (`closed`, `work`, `path`) are deterministic and matched the archive on all 1,680 compared records.
