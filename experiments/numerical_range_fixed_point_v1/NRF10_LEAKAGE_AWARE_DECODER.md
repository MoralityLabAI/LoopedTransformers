# NRF-10: leakage-aware frozen-model observer optimization

**Study date:** 2026-11-15. **Preregistration:** `configs/nrf10_leakage_aware_observer_v1.json`, committed before training or opening final evaluation folds 10/11. The signed-permutation task family, recurrent architecture, and objective weight λ=1 had been examined in NRF-04–09, so this is an exploratory within-family intervention.

## Intervention

NRF-09's twelve final CPU checkpoints are pinned by their SHA-256 values, with tied residual recurrence depth 16 or 32, six independently initialized seeds per depth, and 32 model-training optimizer updates. The model weights remain untouched. Each 128D midpoint representation is mapped to an eight-dimensional orthogonal observer. The supervised baseline takes eight train-fold cross-covariance columns. The task-only and leakage-aware variants start from this same subspace and both receive 80 Adam updates at learning rate 0.02 with QR orthonormalization every update. All arms refit the same ridge-regularized linear task head on fold 1.

Training fold 1 has 160 input/label pairs. Penalty candidates γ∈{0.05,0.2,1.0} are selected on fold 9, using 96 calibration task inputs and 12 calibration Jacobian shadow states. A candidate is eligible if its calibrated task MSE is at most 1.10 times the calibration MSE of the unoptimized supervised baseline. The eligible candidate with lowest calibrated J=MSE+(mean shadow leakage)^2 is selected, with smaller γ breaking ties. All twelve selected candidates were eligible. The optimization uses *mean squared per-state leakage* as a smooth upper bound for the registered squared mean-leakage objective. The task-only optimizer has identical training exposure with γ=0.

Folds 10 (96 task inputs) and 11 (12 fresh shadow states) were opened for final evaluation only after each decoder penalty was selected. Eight separately initialized equal-rank random subspaces provide the median of complete random-arm J scores. The held-out gate per depth-32 seed requires task MSE ≤1.10× supervised, shadow leakage ≤0.90× supervised, and J_joint < min(J_supervised, median(J_random)); the registered cohort threshold is at least 4/6.

## Measurements

| Depth | Arm | Mean held-out task MSE | Mean local leakage | Mean J |
|---|---|---:|---:|---:|
| 16 | Supervised baseline | 0.237942 | 0.416945 | 0.411819 |
| 16 | Task-only, 80 Adam updates | 0.126244 | 0.375939 | 0.267592 |
| 16 | Joint leakage-aware, 80 Adam updates | 0.154010 | 0.315753 | **0.254382** |
| 32 | Supervised baseline | 0.276102 | 0.838128 | 0.978825 |
| 32 | Task-only, 80 Adam updates | 0.149298 | 0.512220 | 0.411739 |
| 32 | Joint leakage-aware, 80 Adam updates | 0.247608 | 0.318440 | **0.352556** |

At depth 32, the eight-random-arm complete-J median averages **0.915933**. The joint observer beats the baseline, the random median, and the equal-budget task-only observer on J for **all six seeds**. It satisfies the preregistered joint task/leakage gate **5/6**, passing the required 4/6. Seed 149 failed the strict task criterion by a small amount: final task-MSE ratio 1.1011 versus the allowed 1.10. This remains a failed seed in the gate. Depth 16 passes the same per-seed gate 6/6.

The mean depth-32 joint decoder's leakage is **62.04% lower** relative to the mean per-seed supervised leakage ratio (the arithmetic mean of ratios). The mean combined J is **64.0% lower** than the supervised baseline and **14.4% lower** than the task-only optimization. On depth 16 the joint decoder has lower J than task-only in 6/6 seeds; depth 32 has the same 6/6 sign.

## Mechanism and limitations

For an orthonormal basis U (128×8) and Jacobian J at a fixed midpoint state, with Q=I−UUᵀ, the exact local squared leakage is

```
L(U,J)^2 = (||UᵀJ||_F² − ||UᵀJU||_F²)/8.
```

This algebra supports differentiation through a QR orthogonalization without allocating the full complement. The optimization uses the **mean** of these squared per-state leakage values. Evaluation uses the square of the **mean** per-state leakage, consistent with NRF-09. Jensen's inequality gives the surrogate an upper bound on that evaluation penalty for a fixed U.

Every source `.pt` checkpoint hash was verified against the recorded NRF-09 receipt before calculation. A dense-Jacobian versus reverse-JVP test on a real 128D checkpoint matched within 1e-9. The twelve optimized projections were orthonormal to approximately machine precision. All eleven evaluation-arm scores and selection details are recorded separately per cell, with a 12-row compact CSV and a reproduction bundle containing the original checkpoint bytes.

The model itself does not acquire any new task capability through this intervention; only the observation map and linear readout are learned. The target is a synthetic task, with an empirically chosen λ and finitely sampled local Jacobians. The result neither establishes a global invariant quotient nor certifies safety or reasoning performance in language models.
