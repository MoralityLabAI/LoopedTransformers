# SPC-03: transient contraction, precision floor and damped phase selection

Continuation of SPC-01/02 using OpenAI Math families #129 (state
separation) and #130 (Fourier-coordinate methods). The mathematical
proof package is `patrickdugan/MathCan/research/spectral_aliasing/SPC-03.md`.

## What we measured

An untrained one-layer `LoopedChase` network was replayed with seeds
`13012902` through `13012905`, four frozen prompts per seed, for 42
recurrent loops. Models were initialized once in float32, then cloned
with identical weights cast to float64; token sequences were identical.
The model has width 64 and eight pointer-chase nodes. No learning,
checkpoint, activation intervention, safety claim or extrapolation is
performed by the native campaign.

Across 16 trajectories, the mean temporal DC energy fraction was
approximately 0.9959, despite a centered periodic-proxy residual
about 0.716. A stationary mean cannot explain the changing trajectory;
it can conceal its residual in a raw relative norm.

A preregistered scalar transient model was trained only on difference
pairs 3..8 (zero based), excluding the first input-injection jump,
and evaluated on pairs 9..14. Float64 held-out difference prediction
error averaged `0.07453` relative to the observed test differences.
The zero-difference forecast has error 1 by definition and persistence
averaged `1.2358`. The fitted scalar coefficient averaged `0.43536`
over 16 inputs. This is an empirical scalar proxy, **not** a measured
Jacobian eigenvalue or a proof of a one-dimensional internal state.

Float32 initially agrees with float64 but reaches a state-difference
noise floor near `7.4e-7`. Float64 continues to decay, with mean
state differences near `9.45e-15` at the final depth. The late-
window float32 AR(1) failure in SPC-02 therefore includes a finite-
precision effect. A stable dynamics claim must be evaluated above the
precision floor and on held-out time steps.

## Why probe phases change

SPC-02 optimized undamped Fourier modes and preferred the irregular
four-phase schedule `[0,1,2,4]` over `[0,1,2,3]` for `d=7,k=2`.
For an explicitly defined damped Fourier family
`x_t=Σ_r a_r (rho*exp(2πi*r/d))^t`, the measurement matrix multiplies
its `s`th phase row by `rho^(L+s)`. At `rho≈0.43536`, exhaustive finite
SVD checks show consecutive probes `[0,1,2,3]` are best; gains are
`0.05114` vs `0.03252` for the previously preferred schedule.
The two schedules exchange order around `rho≈0.58820` in this bounded
example. The true native LoopedChase nonlinear modes have not been
identified with damped Fourier eigenmodes. The numeric schedule flip
is a testable **designed model** conclusion.

A new theorem in MathCan's paper proves the exact `rho^L` loss in
sparse identification gain from delaying every phase observation, and
proves early consecutive probes are the unique optimal `p`-phase
schedule for sufficiently strong damping when `q=p`.

## Reproduction

```bash
python -m pip install numpy==2.3.5
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -p 'test_dynamic_controls.py' -v
python -m unittest discover -s tests -p 'test_precision_dynamics.py' -v
python -m unittest discover -s tests -p 'test_damped_phase_schedule.py' -v
python -m experiments.spectral_state_cost_v0.precision_dynamics --out /tmp/spc03-native.json
python -m experiments.spectral_state_cost_v0.damped_phase_schedule --out /tmp/spc03-damped.json
```

The committed JSON receipts are deterministic CPU references; GitHub
CI regenerates independently. Source snapshots are the existing repo
`LoopedChase` and `phase_stability` implementations, not the offline
minimal copy used to test the development environment.

## Limitations and follow-up experiment

Centering removes a static anchor and first differences isolate
transients. Differencing also suppresses modes near eigenvalue one.
A trained model, multi-token Jacobian or causal perturbations, and
higher-precision validation remain necessary before using this probe
as a mechanistic control or a training reward.
