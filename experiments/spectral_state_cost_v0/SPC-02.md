# SPC-02: robust sparse Fourier probing of recurrent trajectories

Continuation of SPC-01 for `MoralityLabAI/LoopedTransformers`.

**Research sources (OpenAI Math 2026):** family #129 on deterministic
state costs for future tests; family #130 on exact Fourier circuits.
These papers motivate the mathematical formulation, with no inference
that their automata complexity or Fourier operation bounds hold for
transformer models. Complete mathematical proofs and Lean/F\(_7\)
finite certificates are in `patrickdugan/MathCan/research/spectral_aliasing/SPC-02.md`.

## New experiment

A periodic length-\(md\) looped signal is observed at stride \(d\)
and a set of \(q\) phase offsets. Each alias class holds up to \(k\)
nonzero Fourier frequencies. SPC-01 proved sharp noiseless detection
and identification thresholds. SPC-02 computes their **restricted
singular-value gain**, which controls amplification of observation
noise, and searches alternative phase schedules within a fixed budget.

The independent, intentionally small CPU implementation can:

* Demodulate real sampled trajectories into the exact alias equations;
* calculate a rigorous conservative gain floor for consecutive phases;
* enumerate all support minors and all phase schedules for small \(d\);
* fit a noisy \(k\)-sparse spectrum by exhaustive support enumeration;
* test adversarial pairs attaining a poorly conditioned four-column
  minor, including midpoint noise that makes them indistinguishable;
* compare observed model-depth traces against their best periodic
  proxies and report residual plus spectral-sparsity tail.

For \(d=7,q=4,k=2\), consecutive phases give worst gain `0.2961051161`;
non-arithmetic schedule `[0,1,2,4]` gives `0.6311266657`, a `2.131428`
factor. Composite `d=6` includes an irregular schedule with a singular
minor. Coprime arithmetic progressions preserve the consecutive
schedule's *exact* worst-case singular spectrum up to relabeling.

**Noise certificate.** With \(p=\min(2k,d)\), q consecutive phases
where \(q\ge p\), and \(\gamma\) from the MathCan theorem, a
block-\(k\)-sparse fit satisfying observed residual \(\le\varepsilon\)
obeys \(\|x-\hat x\|_2\le 2\sqrt d\varepsilon/\gamma\).
This is a worst-case guarantee for an exactly periodic signal, measured
in Euclidean norms with the SPC-01 unnormalized Fourier convention.
The numerical optimized schedule has a better measured uniform gain,
but its SVD estimate is floating-point rather than a formal certificate.

## Run

```bash
python -m pip install numpy==2.3.5
python -m unittest discover -s tests -p 'test_phase_stability.py' -v
python -m experiments.spectral_state_cost_v0.phase_stability
```

Feed a cached hidden-state trajectory, with depth as the first dimension:

```bash
python -m experiments.spectral_state_cost_v0.phase_stability \
  --trace /path/to/loop_activations.npz --key activations \
  --period 21 --stride 7 --max-modes 2
```

Input shape can be `(depth,)`, `(depth, hidden_dim)`, or
`(batch, depth, hidden_dim)`. There is no GPU requirement; the default
search uses 1,225 four-by-four SVDs. The script caps larger searches.

The trace diagnostic reports how well a *fitted periodic proxy* matches
the original trace, then calculates the fraction of proxy spectral
energy outside its best per-class \(k\)-mode approximation. For any
model whose traces exhibit phase drift, damping, Jordan transients,
or unmodeled nonlinear dynamics, the residual measures a failure of
the exact-periodic premise. The near-periodicity and sparsity
statistics must be included with any extrapolation or safety claim.

## Experiment interpretation

An optimized acquisition schedule is selected by worst-case gain, not
by a model-specific average improvement. An adversarial two-spectrum
pair attains the weakest consecutive schedule response; the same pair
has a substantially larger response under the optimized schedule.
This tests probe design before probing any real network.

No trained-model checkpoint or native transformer activations were
included in this bounded CPU replay. The adapter is designed for an
explicit user-supplied cached activation array; successful synthetic
checks are not evidence of capability gains or mechanistic control.

## Native LoopedChase falsification control

The supplementary `native_phase_smoke.py` imports the repository's
actual `LoopedChase` class and generates 42 recurrent answer-position
state vectors from a small **untrained** tied-layer pointer-chase model.
It computes the same proxy periodic-residual and spectral-tail metrics
for two or four frozen synthetic prompts. The smoke is CPU-only, does
not train or access paid models, and records `trained_checkpoint: null`.
The controls ask whether exact periodicity and per-alias sparsity are
reasonable assumptions even before testing a trained checkpoint. A
large periodic-proxy residual means the SPC-01/02 exact bounds should
not be applied directly to the raw native trajectory.

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -p 'test_phase_native_smoke.py' -v
python -m experiments.spectral_state_cost_v0.native_phase_smoke
```
