# PPT-G04 CPU Lean reasoning gym

Run from `research/lean_cgrt_hrm400`. Install a CPU-only PyTorch wheel, NumPy and `psutil`, build `lgym` with Lean 4.19.0, and set `LCGRT_LEAN_HOME` to that toolchain directory. The historical 588-case archive is outside the input path. The gym consumes only its own frozen, regenerated Horn theorem roots.

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.1.3 psutil==7.2.2
lake -d lean build lgym
New-Item -ItemType Directory runs/ppt-ram -Force
python -m ppt_gym.ram freeze runs/ppt-ram/cohort.json --seed 904 --families 2 --queries 2
python -m ppt_gym.ram --ram 6 run runs/ppt-ram/cohort.json runs/ppt-ram/session.sqlite --plan a --preset micro --audit runs/ppt-ram/memory.json
python -m ppt_gym.ram --ram 6 replay runs/ppt-ram/cohort.json runs/ppt-ram/session.sqlite runs/ppt-ram/replay.json
python -m ppt_gym.ram report runs/ppt-ram/cohort.json runs/ppt-ram/session.sqlite runs/ppt-ram/report
python -m ppt_gym.ram export runs/ppt-ram/cohort.json runs/ppt-ram/session.sqlite runs/ppt-ram/datagoblin --replay runs/ppt-ram/replay.json
```

The `--ram 2`, `--ram 4` and `--ram 6` profiles set the same process-tree limit for the Python worker, the Lean server and standalone Lean compilers. The default is 6 GiB and two CPU threads. Windows uses the existing Lean-CGRT Job Object. On Linux use a dedicated cgroup v2 scope. Replace both occurrences of the profile size together.

```bash
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  python -m ppt_gym.ram --ram 6 run runs/ppt-ram/cohort.json runs/ppt-ram/session.sqlite
```

| Preset | Plan A parameters | Plan B parameters |
|---|---:|---:|
| Micro | 175,011 | 162,403 |
| Tiny | 1,730,243 | 1,757,955 |
| Small | 6,737,283 | 6,907,395 |
| Medium | 26,875,011 | 23,026,435 |

Counts cover the specialist backbone and its ordinary heads. Fast learners and their optimizer state are reported separately in the checkpoint. The native pilot ran Micro and Tiny.

Windows enforces aggregate private commit. The audit separately samples private and total resident working sets, including file-backed mappings. Linux enforces cgroup charged memory and records the kernel peak. It has no equivalent per-cgroup commit counter. The Python process rejects accelerator-enabled PyTorch builds, sets the CPU thread pools before importing PyTorch, and uses CPU FP32 tensors. Operator wall time and process CPU time are measured for each visited pass. The dense MAC count covers the convolution, attention and linear operations and is an analytical estimate. Certificate compilation and verifier RPC latency are recorded separately from model inference.

Stop after a bounded number of committed entries with `run --steps N` or interrupt the process. Repeat the same command with the same session and remove `--steps` to resume. Model and optimizer bytes, adapter covariance and momentum, retrieval state, RNGs and the ledger cursor share a SQLite transaction. The frozen registration, options, runtime and native verifier fingerprints bind every run. Replay checks Lean actions and rollback, recompiles certificates, recomputes encoder and adapter updates, and checks FP32 weight hashes. Keep the SQLite session for full resume. The export emits a local specialist bundle and replayed train-only SFT, transition, outcome, repair and preference streams. Import eligibility remains gated on DataGoblin ancestry and rights review.

## Pilot registration

The primary unit is the independent generated theorem lineage. The primary contrast is the paired difference in independently certified proof completion after 16 support roots for SGD versus frozen and equal-support retrieval. A five-percentage-point improvement is the minimum meaningful effect. Use eight tactic checks and a common per-proof CPU allowance, with the public-state Horn shortest-proof baseline as the structural ceiling. Report paired wins and losses, separate same-family unseen roots from out-of-family roots, and treat repeated measurements of the same query theorem as paired observations.

This pilot uses fixed seeds and cohorts with a descriptive readout. The small sample leaves the sham variation and minimum-effect resolution uncalibrated. There is no confirmatory hypothesis test or promotion decision. A confirmation needs independently generated development worlds to calibrate the matched sham, then a prospectively sized, untouched evaluation set and familywise inference. The pilot keeps all scored arms and raw verifier outcomes. The manifest includes the source generator hash and a hash of the full run schedule.

| Stage | Registered pilot |
|---|---|
| RAM-01 | Seed 904. Micro Plan A, one fresh support root, pause and resume under 2 GiB. Replay the exact adapter updates and native Lean receipts. |
| RAM-02 | Seed 905. Micro and Tiny Plan A, independent roots for support and query, with the 0, 1, 2, 4, 8 and 16 support-dose readouts. Include SGD, RLS, rank-8, retrieval, frozen and shuffled feedback. |
| RAM-03 | Seed 906. Micro Plan A at 1, 2, 4, 8 and 16 passes, under common CPU and estimated-MAC ceilings. Include its one-pass static control and eight-check automation. |
| RAM-04 | Same seed and roots as RAM-03. Compare matched-size Plan B with enabled, zero and slot-permuted previous-pass messages. Train message gates through the same train-only differentiable support/query procedure. |
| RAM-05 | Seed 907. Two families and two rounds with persistent Micro weights, frozen and Horn controls, and checkpointed train-only encoder fine-tuning. Score retention on earlier families at completed support doses. |
| RAM-06 | Independently replay RAM-05 and produce its local DataGoblin export. |

The recorded pilot used one query root per source family. All stages used two CPU threads and a six CPU-second search ceiling. The fixed-compute stages used a 90 million estimated-MAC allowance per proof. Their backbone preparation used two train-only meta-training steps. The persistence stage used four such steps and an additional fit every eight support roots.

On Windows, run the complete registered sequence with `./configs/run_ppt_g04_pilot.ps1`. Set `-Python` to an absolute interpreter path if the CPU environment is outside the current shell.

The [budget-feasible follow-up](configs/freeze_ppt_g04_followup.py) froze seed 908 against the four original cohort files before scoring. It selected four fresh same-family query roots per training family and two per disjoint family, each with a reference proof of three to eight tactics. Candidate screening happened before Lean evaluation and supplied zero supervision to the model. The frozen follow-up cohort and exact run options are in [the results](PPT_G04_RESULTS.md).

The support scheduler can repeat family visits without repeating source roots. `--persistent` carries adapters across families. Independent episodes reset all fast learners. `--meta-steps` differentiates the training-query loss through a support update, and `--backbone-every` enables a separate periodic encoder experiment. Query outcomes never update the model. A fixed support probe collects actual Lean rejection evidence. Delayed positive credit is assigned only to the surviving path of a separately compiled proof. Exhausted branches supply negative search credit and unresolved branches are censored.

```powershell
$env:PPT_NATIVE_TEST = "1"
python -m unittest discover -s ppt_gym/tests -v
```
