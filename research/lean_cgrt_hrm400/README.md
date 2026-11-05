# Lean-CGRT Local HRM Gym

Portable CPU/GPU gym for the MSI Stealth A18 AI+ and the planned 400M HRM Silico campaign.

The complete tested Python source, PowerShell runner and test suite live in [local_gym.zip](local_gym.zip). The bundle includes a two-clock recurrent proof-frontier model, 5M/15M/30M width presets, reproducible synthetic proof generation, explicit Lean proof export and a fail-closed kernel audit, atomic checkpoint/restart, and a CGRT context-invariance diagnostic.

## Unpack and run on Windows

From the repository root:

```powershell
Expand-Archive -Path research\lean_cgrt_hrm400\local_gym.zip -DestinationPath research -Force
cd research\lean_cgrt_hrm400
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python gym.py generate --count 64 --out runs/corpus
python gym.py audit --source runs/corpus/Generated.lean
python gym.py train --preset smoke --device cpu --steps 10 --eval-every 5 --save-every 5
```

Once RAM is installed, hardware diagnostics pass, and Python/Lean are available:

```powershell
.\keep_running.ps1 -Preset 5m -TotalSteps 10000 -Output runs/overnight
```

Start at 5M with batch 1 and gradient accumulation 8 on 4GB RTX 3050 VRAM. Run 15M and 30M only after measuring VRAM, temperatures and sustained throughput. System RAM helps corpus generation and CPU jobs; CUDA VRAM limits GPU training. The PowerShell runner performs a prelaunch temperature check when nvidia-smi is present; monitor active training separately.

## Experiment boundaries

A smoke test on CPU passed four Python unit tests and three optimizer steps. The Lean compiler was absent in the original sandbox and the required audit returned `lean_kernel_checked: false` (exit 2). Synthetic goal closure is a proxy until the Lean kernel accepts emitted terms. CGRT mode presently measures agreement under rule permutations, not a trained cross-generation branch search. The 400M model remains a separate Silico training proposal. The bundle is committed as a portable archive; unpack before editing individual source files.

For origin and roadmap see the accompanying LR-CGRT-400 plan from the current research conversation.
