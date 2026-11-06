# Lean-CGRT Local HRM Gym

## v2 (`lcgrt/`): Lean-authoritative gym

Start with these documents:
- [`LOCAL_GYM_AUDIT.md`](LOCAL_GYM_AUDIT.md): what v1 got wrong, with receipts;
- [`LEAN_DATA_PIPELINE.md`](LEAN_DATA_PIPELINE.md): Lean decides every goal, tactic and proof;
- [`HRM_CGRT_ARCHITECTURE.md`](HRM_CGRT_ARCHITECTURE.md);
- [`LOCAL_SCALING_RESULTS.md`](LOCAL_SCALING_RESULTS.md): campaign c1, Lean-certified results;
- [`SILICO_400M_HANDOFF.md`](SILICO_400M_HANDOFF.md): decision and ready-to-run 400M specs.

```powershell
# one-time: Lean 4.19.0 (pinned, as in experiments/lean_reasoner) and the proof-state server
$env:PATH = "D:\lean_cgrt\toolchains\lean-4.19.0-windows\bin;$env:PATH"   # or set LCGRT_LEAN_HOME
cd research\lean_cgrt_hrm400\lean; lake build; cd ..
python -m unittest discover -s tests                     # 16 tests
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
