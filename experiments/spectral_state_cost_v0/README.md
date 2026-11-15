# SPC-01: spectral state-cost / phase-probing gym

A dependency-free CPU reference for a new pairing from OpenAI Math (September 2026): family **129**, two-way automaton state-cost lower bounds, and family **130**, exact Fourier circuits. Here we study the distinct question of *complex-linear recurrent state cost* of sampled loop-activation streams. Neither source paper's headline claim is assumed by the code.

A periodic length-`n` loop stream is sampled at stride `d` using selected phases `offsets`, with `n=m*d`. The rank of the sampling map is `m*len(offsets)`; its nullity is `m*(d-len(offsets))`. For unknown spectral support with at most `k` Fourier modes in every frequency-alias class, consecutive phases identify the spectrum uniformly exactly when at least `min(2*k, d)` phases are sampled. Reconstruction here is exhaustive over supports, suitable only for small `d`.

Run the deterministic campaign:

```bash
python -m experiments.spectral_state_cost_v0.spectral_state_cost --json
python -m unittest discover -s tests -p test_spectral_state_cost.py -v
```

The pinned constructed control uses `n=24`, `stride=6`, `k=2`, offsets `[0,1,2,3]`, with five nonzero Fourier coefficients. The reference oracle should report rank 16, nullity 8, full spectral order 5, sampled multi-output order 4, and near-zero recovery error. A separate missing-phase impulse has full order 24 and sampled order 0, demonstrating severe probe invisibility when the sparsity promise fails.

This measures exact periodic linear signals, **not** full transformer capability or safety. Probe placement, approximate recurrence, finite-precision spectral estimates and nonlinear dynamics require held-out experiments. See the [pure mathematics and proofs](https://github.com/patrickdugan/MathCan/blob/feat/spc01-sparsu-phase-vandermonde-20261115/research/spectral_aliasing/SPC-01.md).

The exact theory also distinguishes the phase budget for merely detecting an
occupied Fourier alias class (at least `k` consecutive phase probes) from the
budget for uniquely identifying its unknown `k`-sparsu contents (at least
`min(2*k, d)` probes). Non-coprime arithmetic-progression schedules create

distinguishable frequency columns even when many phases are measured. See
`MathCan/research/spectral_aliasing/SPC-01.md` for the proofs.
