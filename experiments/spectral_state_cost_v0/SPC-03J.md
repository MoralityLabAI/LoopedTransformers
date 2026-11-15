# SPC-03J: an actual full-state Jacobian check of transient convergence

The SPC-03 scalar fit estimated `rho≈0.43536` from one-step differences of
an untrained `LoopedChase` answer-position state. This follow-up
differentiates the complete recurrent update, including *all* prompt
and answer token positions. It is still an untrained control.

The actual `LoopedChase` modules (`tok`, `dst`, `pos`, `inject`, `core`)
reconstruct its recurrence `h_{t+1}=F_e(h_t)`, with fixed prompt input
`e`. The test checks that this reimplementation exactly matches the
native model's forward hidden state at every loop. All Jacobian-vector
products use PyTorch's explicit math scaled-dot-product-attention
backend, as the default CPU flash-attention backward path does not
implement the double backward used by `autograd.functional.jvp`.

At the terminal 42-loop state, a fixed-start-vector SciPy Arnoldi
routine estimates the leading eigenvalues of the full-state Jacobian.
The 576-dimensional state is 9 tokens by 64 channels.

Across four fixed initialization seeds (one prompt each), the largest
estimated eigenvalue modulus averages `0.46561` (range 0.4570..0.4702).
The terminal fixed-point residual relative to the full state is below
`1.1e-16`. The endpoint-Jacobian linear prediction of the next state
difference has mean **relative error `4.47e-7`** at difference depth
12, decreasing markedly from depth 4. The exact-state forward
reconstruction has max discrepancy zero in this run.

This validates a local linear contraction mechanism on the visited
untrained trajectories, with a dominant mode in the vicinity of the
scalar transient fit. Arnoldi eigenvalues are numerical approximations,
and spectral radius alone does not bound a nonnormal operator's transient
amplification in Euclidean norm. The result supplies a concrete
mechanistic test and a direct baseline for checking whether training
changes the Jacobian, rather than an interpretation of hidden reasoning.

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -p test_jacobian_monitorability.py -v
python -m experiments.spectral_state_cost_v0.jacobian_monitorability --out /tmp/spc03j.json
```

The companion mathematical proof in `MathCan/research/spectral_aliasing/SPC-03J.md`
quantifies the endpoint-Jacobian secant error. Native tests freeze seeds,
full token-state shape and forward consistency; the workflow emits a
regenerated receipt. No training or capability result is claimed.
