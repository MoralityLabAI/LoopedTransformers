# Two-pass skill gym: local pilot

A small, additive pilot for math-informed recurrent skill compilation. Uses PyTorch and NumPy, with CPU by default and optional CUDA. Original full research bundle with experiment receipts and trained modules remains in the user's ChatGPT Library as `two_pass_skill_gym.zip`. This branch contains a minimal runnable trainer, not the full bundle.

## Design

For a contractive affine Bellman system `z* = b + Mz*`, visit `t` takes an exact residual `r=b+Mz-z` and an embedded latent workspace. The network learns a residual correction while the solver's ordinary update is preserved. Two visits are the default, and each visit shares weights.

The error certificate is `||z-z*||_inf <= ||b+Mz-z||_inf/(1-q)` when `||M||_inf <= q < 1`.

## Run

```powershell
python -m pip install torch numpy
python research/two_pass_skill_gym/gym.py train --steps 400 --device cpu
python research/two_pass_skill_gym/gym.py eval --checkpoint two_pass_runs/last.pt
python research/two_pass_skill_gym/gym.py export --checkpoint two_pass_runs/last.pt
python research/two_pass_skill_gym/gym.py train --steps 100000 --device cuda --resume
python -m unittest discover -s research/two_pass_skill_gym/tests -v
```

Use a different run directory for each ablation. The trainer checkpoints periodically and responds to SIGINT after the active update. CUDA selection is explicit; no automatic GPU contention management is implemented in this pilot.

## Registered evaluation

Sweep visits = 1, 2, 4, 8; seeds = 211, 223, 227. Compare exact one- and two-step fixed-point updates, independent direct MLP, and an untied model at a matched parameter and block-evaluation budget. Evaluate on held-out random matrices and composition sizes. Always report raw residual, certified error bound, direct target MSE, peak RAM/VRAM, examples per second, and wall clock. Keep validation selection separate from final test.

Treat two visits as a hypothesis. No result in this initial pilot establishes superiority over TRM/HRM, direct solvers, or deeper recurrence.

## Follow-on

Port the full queued trainer, trained module export with measured INT8/FP32 sizes, and RMP adapters from the Library bundle. Add pointer chasing and verified program repair tasks where proposals have nontrivial search cost and checks are cheap. Import the existing RMP depth 9–12 evaluation before claiming extrapolation.
