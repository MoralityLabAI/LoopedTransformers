# Lean-CGRT Local HRM Gym

## v2 (`lcgrt/`): Lean-authoritative gym

Start with these documents:
- [`LOCAL_GYM_AUDIT.md`](LOCAL_GYM_AUDIT.md): what v1 got wrong, with receipts;
- [`LEAN_DATA_PIPELINE.md`](LEAN_DATA_PIPELINE.md): Lean decides every goal, tactic and proof;
- [`HRM_CGRT_ARCHITECTURE.md`](HRM_CGRT_ARCHITECTURE.md);
- [PPT Plan A: DRAM-plastic proof specialist](PPT_PLAN_A_DRAM_PLASTIC_PROOF_TRANSFORMER.md): proposed typed latent recurrence, CPU-updated fast weights, meta-learning and Lean gates (2026-11-21).
- [PPT Plan B: Inkling-derived block/message specialist](PPT_PLAN_B_INKLING_BLOCK_MESSAGE_MEMORY.md): updated architecture after the 2026-11-20 release and related papers, with gated cross-pass memory, block-level lenses, variable depth and DRAM adapter controls.
- [PPT gyms: runnable Plan A/B, DataGoblin importer, DRAM accretion and replay](PPT_GYM_RUNBOOK.md): CPU PyTorch reference implementation, checked-data custody, support/query controls, synthetic smoke tests and CI.
- [LSA-02B block-order probe](../../docs/lsa02b_block_order_protocol.md): frozen Plan-B AB/BA interventions, mirror-order training, label-blind commutator measurement and native Lean receipt gates.
- [LSA-02B native Lean bridge](../../docs/lsa02b_native_lean_bridge.md): paired AB/BA whole-proof search on frozen Plan B weights with independent Lean 4.19 kernel, standalone certificate compilation and model-free replay.
- [LSA-02B cloud CPU training pilot](../../docs/lsa02b_cpu_training_pilot.md): native shortest-continuation teacher, paired-seed AB/BA CPU fits and Lean-replayed calibration, with GitHub Actions evidence artifacts.
- [LSA-02B saved-checkpoint CPU diagnostics](../../docs/lsa02b_cpu_diagnostics_protocol.md): read-only source-fork census, root-level train-label accuracy, and shared-state AB/BA action disagreement.
- [PPT-G03 live Lean search and verifier-checked data accretion](PPT_G03_RUNBOOK.md): native Lean 4.19 stepper, frozen theorem suites, independent compiled certificates, train-support DRAM updates and replay-gated export.
- [PPT-G03B bounded backtracking gym](PPT_G03B_BACKTRACKING.md): saved Lean-state rollback on unprovable branches, separate tactic and host-cost accounting, and exact independent replay.
- [`LOCAL_SCALING_RESULTS.md`](LOCAL_SCALING_RESULTS.md): campaign c1, Lean-certified results;
- [`SILICO_400M_HANDOFF.md`](SILICO_400M_HANDOFF.md): decision and ready-to-run 400M specs.
- [`RLT_ABLATION_MATRIX.md`](RLT_ABLATION_MATRIX.md): RLT-01 feedback controls and Q5 transfer caveats; [completed two-lane pilot results](results/rlt01/RLT-comparison.md) found no RLT-specific Lean completion gain.
- [RLT-02 longer-training results](results/rlt02/RLT-long-report.md): two parallel continuations, 64 fresh goals / 62 certifiable, no consistent RLT completion advantage; one direct-model added solve, with no combined-policy gain over automation. [Frozen registration and amendments](RLT_LONG_REGISTRATION.md).

```powershell
# one-time: Lean 4.19.0 (pinned, as in experiments/lean_reasoner) and the proof-state server
$env:PATH = "D:\lean_cgrt\toolchains\lean-4.19.0-windows\bin;$env:PATH"   # or set LCGRT_LEAN_HOME
cd research\lean_cgrt_hrm400\lean; lake build; cd ..
python -m unittest discover -s tests                     # 36 tests
python -m lcgrt.dataset build --name lcgrt-v1 --workers 3
python -m lcgrt.train --data lcgrt-v1 --preset S --steps 3000 --out D:\lean_cgrt\runs\demo
python -m lcgrt.evaluate --ckpt D:\lean_cgrt\runs\demo --policies greedy best --out D:\lean_cgrt\runs\demo_eval
python -m lcgrt.report --campaign c1
```

**Unattended:**
- `.\keep_running.ps1` probes the hardware, picks a queue by VRAM, and runs it one GPU job at a time through vramorama (`-ProbeOnly` for the hardware receipt alone).
- `.\install_task.ps1` registers an at-logon task; run it yourself to enable automatic startup.
- Jobs resume from hash-verified checkpoints. The queue stops after two consecutive failures.

**Resume campaign c1/c2 after a stop:** run `.\keep_running.ps1 -Queue configs\campaign_c1.jsonl -Campaign c1`. Finished jobs are skipped and unfinished ones resume from their checkpoints.

With at least 40 GB of RAM the worker runs two lanes: GPU training (one vramorama job at a time), and Lean-bound evaluation on the CPU with the model in RAM. Each evaluation starts as soon as its checkpoint is complete. To run a lane by hand: `python -m lcgrt.runq --campaign c1 --queue configs\campaign_c1.jsonl --lane train|eval`.

**No GPU, 9 GB of RAM** (for example while the GPU and most RAM run something else):
`python -m lcgrt.cpugym --campaign c1 --queue configs\campaign_c1.jsonl --ram-cap 9G`.
- It puts the whole gym in one Windows job object, with a hard cap on committed memory summed over every runner, trainer, evaluator and Lean process, at Idle priority.
- It runs one CPU-only runner per lane. The train lane takes only queue lines marked `"cpu": true` (c3-cpu); GPU lines wait.
- Killing its one process stops the whole tree. Live usage is in `D:\lean_cgrt\runs\c1\cpugym_status.json`.
- See [`lcgrt/cpugym.py`](lcgrt/cpugym.py) and [`lcgrt/ramcap.py`](lcgrt/ramcap.py). On Linux, the same cap is `systemd-run --scope -p MemoryMax=9G`.

**Rapid gym (exploratory):** `python -m lcgrt.rapid` runs rounds of short training runs until `D:\lean_cgrt\runs\rapid\STOP` exists.
- Each round trains about 6 specs for 1,000-2,000 steps, one GPU lease at a time. Knobs come from `--cfg FIELD=VALUE` over a preset, for example `trm_sup=4`.
- It then scores all of them in one Lean-certified pass on 120 `val` states (never a test split), and writes `ledger.md`, with a copy in `results/rapid/`.
- The better half survives at twice the steps on a new seed, and one-knob mutants of the best spec fill the round. Each mutant is paired with its parent's rerun, and `knob_effects` averages those pairs.
- See [`lcgrt/rapid.py`](lcgrt/rapid.py).

**9 GB of RAM and a share of the GPU:** add `--gpu-vram 9G --gpu-duty 0.6`.
- The train lane becomes a GPU runner: one job at a time, each admitted by a vramorama lease (the line's `"vram"`) and with torch's allocator capped at that lease.
- A training job keeps the card busy at most 60% of the time. It idles after each step, and while the card is at 82 °C or more it halves its share, then climbs back slowly.
- `"cpu": true` lines run there in fp32 (`--precision fp32`: no autocast, no TF32), so c3-cpu keeps its recipe. An eval line with `"lane": "train"` runs on the GPU too; `"priority"` orders the lane.
- Windows charges a CUDA process's VRAM to its commit, so the job's commit cap is 9 + 9 GiB. The RAM actually held is `resident_gib` in the status file.

Heavy artifacts live outside the repository, under `D:\lean_cgrt` (data, runs, logs, toolchain).

## v1 pilot (`gym.py`)

The section below describes the original v1 gym, kept runnable as the pilot.

This additive research gym targets an MSI laptop with expanded RAM and a small NVIDIA GPU. It trains a **two-clock, shared-weight proof-frontier model** on synthetic Horn-logic proof trajectories, with separately compiled Lean proof terms and a basic cross-context CGRT diagnostic. The full 400M Silico model remains a planned scaling stage.

### Start on Windows / PowerShell

```powershell
cd research/lean_cgrt_hrm400
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python gym.py generate --count 64 --out runs/corpus
python gym.py audit --source runs/corpus/Generated.lean --receipt runs/corpus/audit.json
python gym.py train --preset smoke --device cpu --steps 10 --eval-every 5 --save-every 5
python gym.py eval --checkpoint runs/local/latest.pt --count 24 --depth-min 6 --depth-max 7
```

The `audit` command needs `lean` on PATH (install Lean via elan and use the toolchain set for this project). A missing Lean executable exits **2** and produces `lean_kernel_checked: false`. Passing the local graph checks is a separate status. Generated proof source is a restricted propositional fragment, not Mathlib extraction. The name CGRT in this repository means **Cross-Generation Reasoning Tree**; the current implementation is a *cross-context invariance diagnostic* using rule permutations at a fixed prefix. Full prefix-controlled branching and learning from Mathlib are later milestones.

## Once the laptop RAM passes diagnostics

```powershell
python gym.py train --preset 5m --batch 1 --accumulate 8 --device auto --steps 10000 --out runs/overnight --resume
# Or supervised restart loop with GPU temperature gate (nvidia-smi):
.\keep_running.ps1 -Preset 5m -TotalSteps 10000 -Output runs/overnight
```

Increase to `15m` then `30m` only after recorded same-depth/depth-OOD curves, verified checkpoints, and VRAM/thermals checks. Presets set model width; the script records the **actual** parameter count in `config.json`. CUDA memory is separate from installed system RAM. A 4GB RTX 3050 favors the `smoke` or `5m` presets, small batches and gradient accumulation. The model is a research experiment, with no guaranteed training speed or temperature behavior on the MSI.

## What the gym does

- Streams newly generated problems instead of caching a huge dataset, with deterministic per-step seeds for resumability.
- Learns proof-rule selection using set-valued supervision, with fast attention called twice for each slow attention update. Slow and fast blocks have distinct weights.
- Saves model, optimizer, configuration and RNG state atomically to `latest.pt`, with `metrics.jsonl` and checkpoint hashes.
- Evaluates new synthetic problems at training depths and depths beyond training, counting full goal closure and model calls under a proof-length-plus-one action budget.
- Exports explicit Lean proposition/implication proof terms. `audit` requires compiler success and `#print axioms` reports for every declaration.
- Supports `eval --mode cgrt --branches 3` to average equivalent-context predictions. This diagnostic costs extra model calls and **cannot establish full CGRT benefit**; compare against `greedy` and cost-normalized controls.

## Gating and measurement

Start with `smoke` on CPU and a small Lean audit. Freeze held-out seeds; evaluate a simple non-recurrent or single-pass baseline with equal parameter and runtime budgets before interpreting recurrence as useful. Track accepted-action accuracy, verified theorem completion, independent problem-family splits and GPU-seconds per success. Current `eval` computes synthetic goal closure only and explicitly returns `kernel_checked: false`; compile its proof terms to promote a mathematical correctness claim.

For all-night operation, Windows sleep/hibernate, charger policy, ventilation, CPU thermal readings and persistent paths still need checking. The restart helper reads `nvidia-smi` temperature where available; it does not monitor CPU temperature, and it cannot interrupt an active training process when the GPU heats up. Use an external hardware monitor to establish safe sustained settings first. `--resume` requires an unchanged checkpoint configuration. Store `runs/` off the repository and back up experiment logs periodically.

## Provenance

The local predecessor LR-01 trained 5,161 parameters on generated propositional proofs, finding a strong role for shared attention visits on deeper dependencies. Its reported results were based on a Python fragment checker and await a Lean-kernel audit. This gym has independent generated data and new model code; no LR-01 pretrained checkpoint is loaded, and no 400M-parameter run has been performed.
