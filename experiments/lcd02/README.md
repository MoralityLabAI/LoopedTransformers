# LCD-02: spectral recurrent extrapolation

Status: implemented source pilot; native results pending. Parent: [LCD-01](../lcd01/README.md). Source inspiration: Liu et al., LoopCD (arXiv:2610.02185v1). The spectral estimator is new experimental work; it is not part of LoopCD.

## Registered native design

Use the pinned Huginn-0125 BF16 loader from LCD-01. One prompt per batch, 16 recurrent steps, reference step 6, three-shot variable-copy families with 1, 2, and 3 hops. Exact 10-way digit choice. 48 independent fit roots, 48 selection roots, and 192 test roots; equal counts per family. Seeds 2026112504, 2026112505 and 2026112506. Test roots are generated only in the test stage, after fit selection. Check no duplicates/overlap and fail on native/manual recurrence parity. No model weights change. Use the same latent start for every arm on a given root.

Let d = h16 - h6. U is the leading uncentered right singular-vector basis of the final-token d across fit roots. It is applied to all positions at inference. For a fixed scalar omega=.5, the correction is

    g = omega*d + (d @ U * lambda) @ U.T
    g = min(1, ||d|| / max(||g||, eps_norm)) * g

Use ranks 4, 8, 16; coefficients are constrained to [-.5,.5]. The effective linear map has spectral norm at most 1. The cap is a guard and norms are over all positions of one prompt. Quantized/BF16 rounding can change achieved norm; record pre-cast and post-cast correction norms. The basis is geometry, not causal evidence.

Fit coefficients using a finite-difference model of decoder behavior. At each fit root, around the scalar-guided state, estimate each mode's derivative of the correct-option logit minus logsumexp of other options by +/- .25 perturbations. Fit bounded ridge regression from these derivatives to the clipped nonnegative margin deficit (target zero margin, cap 4). Ridge grid {.1,1}. Clamping an unconstrained ridge solution is an approximation to constrained fitting, not a globally optimal solution. Same estimator and hyperparameter budget for an independently drawn orthonormal random basis. All coefficients and bases frozen before test. Fixed label-free projection controls use rank 8 and .5 projection multiplier, without fitting labels. Scalar grid {0,.25,.5,.75,1}. Hyperparameters selected by macro accuracy then conditional candidate NLL with deterministic tie break.

Held-out arms:
- fitted PCA (primary);
- independently selected random subspace;
- random at PCA's chosen rank/ridge;
- random at PCA's chosen rank/ridge rescaled per prompt to PCA correction norm;
- calibrated scalar and scalar at the PCA correction norm per prompt;
- fixed scalar LoopCD-Hidden (.5);
- fixed label-free PCA and random projection;
- unguided 16- and 32-loop baselines.

The exact-norm diagnostics require the PCA correction at inference to obtain its norm and are excluded from cost-superiority ratios. Other timings include actual correction arithmetic and synchronized native forward. Each timed arm has an independent same-seed trajectory. Record raw predictions, conditional NLL, correction norms, peak memory, latency and paired root bootstrap intervals. Fit cost includes 48 basis trajectories, 48 repeated fit trajectories and 3,120 coda calls for scalar reference plus finite differences (48 x 65); selection costs are additional. Measure end-to-end fitting elapsed time, and record selection wall time separately when running. No amortization claim absent measured fit/selection time and projected inference volume.

Primary pilot hypothesis: positive paired accuracy delta versus both selected scalar and norm-matched random, with latency <=110% of fixed scalar. Secondary: R16 PCA within 2pp of R32 unguided and latency <=70% of R32. Report all arms even on failure. Inspect the baseline and oracle-free three-digit heuristic (1/3) to distinguish weak task competence from lack of directional benefit. These are pilot screens, not powered confirmatory hypothesis tests. A fresh confirmation must freeze rank, ridge, coefficient vector, basis, and primary comparisons before unsealing, and use independently derived sample size. Repeat random bases and model initialization seeds in confirmation. Any transfer claim requires native verifier receipts, quantizer admission, and independent tasks.

## Run

Run LCD-01's smoke and native parity/pilot first. Check CUDA memory, power and model cache. Fit and test use distinct output paths, and any partial/failing results must be retained rather than overwritten.

```sh
python -m research_gym.scripts.lcd02_spectral --stage smoke
python -m research_gym.scripts.lcd02_spectral --stage fit --out results/lcd02/fit.json
python -m research_gym.scripts.lcd02_spectral --stage test --freeze results/lcd02/fit.json --out results/lcd02/test.json
```

Fit receipt includes frozen bases and coefficients, derivatives, margin targets, singular values and selection predictions. Test checks source and environment digests before evaluating. Only a successful native GPU execution can support an empirical claim. CPU smoke checks algebra and synthetic basis/ridge recovery.

## Follow-on experiments

LCD-02B: separately verify Sudoku full-grid and Lean kernel closure, then compare quantized direction retention at equal compute. Preserve full-state recurrence and legality checks. Fit basis separately for each architecture; avoid transferring a coordinate basis across incompatible models.

LCD-03: residual-based adaptive depth and Anderson extrapolation. Require native full-state stationarity and exact resume admission before calling the method a fixed-point accelerator. Store conditioning and correction norm, charge extra recurrence and readout passes, and compare prefix-only controller to fixed depth at matched average cost. Keep separate from LCD-02 so a failure of spectral reweighting cannot be hidden by an adaptive policy.
