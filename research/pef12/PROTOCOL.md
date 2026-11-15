# PEF-12 | Probe-only Jacobian forecast of the precision frontier

**Registered November 15, 2026.** This file is committed as its own commit **before any PEF-12 heldout noise or quantization scoring**. Previous PEF-07 through PEF-11 results, including the 0.25° fragility and 4° robustness, are known. This is a conditional calibration experiment on the same six frozen synthetic models, not an independent validation of earlier findings.

## Frozen provenance and condition family

Branch from PEF-11 draft PR #85, head `53035957720538e0c35e6cf80f21af8c5819b0f4`. Require original PEF-07 training ledger SHA256 `1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714`, PEF-09 source-only selection SHA256 `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157`, PEF-10 angle/ridge ledger SHA256 `3ef6ce605540a5009f6896cf02a938c200bde23c7a5104d378aace7388430177`, and PEF-11 precision selection SHA256 `f0bd80d8dc7b4af99dbc24be2723828bcf85be5b6f60f46452581e4a53661741`. Verify each original 450-step checkpoint byte hash and weight digest. Keep the source's 160/48/48 context split with all 16 focal states per context; 768 probe and 768 heldout examples. No updates to models, model architecture, decoder coefficients, source groups, or previously frozen angle directions.

Select from PEF-11 frozen selection ledger exactly two tangents (`random`, `pair01`) and two **positive-sign** angle settings (0.25° and 4°). All six task/seed combinations are included. At h8 apply the exact PEF-10 rank-one affine ablation using its serialized direction `v` and probe-only center `mu`; retain the PEF-11 `w` and `offplane` unit vectors and PEF-10 fitted h12 ridge head. These choices are fixed by prior protocols and cannot be changed after looking at PEF-12 outcomes. PEF-09 source / PEF-10 and PEF-11 hashes are the input trust chain.

## Precommitted analytical forecast (no noisy heldout inference)

For each pair (task,seed), tangent and angle, obtain the *natural h8 probe states*, apply the original ablation, and propagate the frozen tied recurrent network for natural visits 9–12. Write `f(h8)` for the unchanged PEF-10 h12 ridge margin after these four natural steps. Compute the exact autograd **input directional derivative** `J_i(u)=grad_(h8) f(h8_i) dot u` for both `u=w` (retained transverse) and `u=o` (equal-norm off-plane). No model or decoder parameter has gradients updated. For probe labels `t_i = 2*y_i-1` in ±1, `m_i=t_i f(h8_i)` is the signed clean margin.

The local-linear Gaussian model for injected `epsilon_i=sigma*z_i*u`, independent `z_i ~ N(0,1)`, yields exact *linearized* correct-class probability
`p_i(sigma)=Phi(m_i/(sigma*abs(J_i(u))))`.
For `J_i(u)=0`, set p=1,0 or 0.5 by the sign of m (tie=0.5). The sigma scale is `s = sqrt(mean_probe(((h8-mu) dot global_PC)^2))`; evaluate fixed relative `sigma/s in {0.002,0.006,0.02}`, new levels distinct from PEF-11. Average all 768 probe probabilities to produce a forecast for *new heldout-context average accuracy*. Preserve the probe signed margin and directional Jacobian arrays, their SHA256 and predictions for independent auditing. This is a probe-distribution forecast, not an input-specific heldout prediction.

For scalar lattice quantization, use the PEF-11 retained-coordinate `q=(h8-mu)·w` and step `delta/s in {0.005,0.02,0.05}`. Register the **dither/uniform-phase approximation** `epsilon ~ Uniform(-delta/2,delta/2)`; with frozen local derivative J(w), predicted accuracy is `clip(0.5 + m_i/(delta*abs(J_i(w))),0,1)` for nonzero J, with the same degenerate limit. Quantization is deterministic without dithering, so this is an explicitly labeled approximation and is secondary to the Gaussian test. The real implementation applies nearest-even undithered quantization, without refitting readouts.

## Strict ledger order

1. Verify input hashes, source ownership, checkpoint bytes and all serialized directions / readouts.
2. Compute **all 24 probe-only model/angle/tangent configurations** and full probe margin/J arrays, both Gaussian directional forecasts, and quantization forecast. Freeze these in a single canonical JSON prediction ledger with SHA256 before **ever computing heldout h8 states**.
3. Deterministic heldout Gaussian noise: independently draw `z` for 16 replicas per condition and level, in FP64 on CPU using `seed=120120000 +100000*seed +50000*[task=threshold]+1000*tangent_idx+300*angle_idx+50*level_idx+replica`, tangent_idx=0 random, 1 pair01; angle_idx=0 0.25°, 1 4°. Use exactly the same z for transverse and off-plane, each added separately to the unmodified ablated h8 state. The entire source-independent seed recipe is frozen here. No PEF-11 z values or previous noise levels are reused.
4. For each perturbation, replay visits 9–12 naturally with unchanged weights, evaluate the frozen PEF-10 h12 ridge, record exact accuracy, mask-level errors, and per-replicate scores. Do the same for the three deterministic retained-coordinate quantizer steps, with no head refit. Report all task/seed pairs, both tangents and angles, all conditions. After scoring, recheck original weight and checkpoint hashes.

## Decisions and controls

**Primary forecast calibration:** for each coalition seed and **random tangent** at both angles (6 Gaussian transverse predictions: 2 angles x 3 levels), the mean absolute prediction-versus-16-replica heldout accuracy error must be <=0.07 on fraction-correct scale AND must be at least 25% lower than the mean absolute error of a constant *clean probe-accuracy* predictor at those six conditions. A seed passes only when both hold; criterion_met requires >=2/3 coalition seeds. Report failed decisions and all threshold arms.

**Directional control:** calculate separate probe-Jacobian forecasts and 16-replica heldout outcomes for off-plane directions with matched perturbation norm and z. Report transverse vs off-plane model error and empirical accuracy difference; these comparisons are descriptive with no alternative pathway to primary success.

**Quantization audit:** report predicted and actual correct /768 for all three steps, including absolute error. No quantization criterion is promoted posthoc. Compare observed deterministic quantizer error with the uniform-phase approximation to reveal phase correlation or nonlinear dynamics.

## Mathematical bounds

For differentiable downstream frozen margin `f` and fixed direction u, if second directional derivative is bounded by H along the perturbation segment, Taylor gives `|f(x+epsilon*u) - f(x) - epsilon*J(u)| <= H*epsilon^2/2`. Therefore a sign flip disagreement with the linearized model is possible only when `|f(x)+epsilon*J(u)| <= H*epsilon^2/2`. `Phi(m/(sigma*abs(J)))` is exact for the **linearized** margin under independent Gaussian noise, not guaranteed for the original nonlinear recurrent network. Numerical float32 operations and nonlinearity may produce calibration errors. A success measures forecast agreement across heldout **context IDs within this already explored 12-bit task family**, not fresh operator transfer, Lean capability, or generalization to pretrained LMs.

## Reproduction

```sh
python -m pytest -q research/pef12/test_pef12.py
python -m research.pef12.pef12 --evidence /path/to/registered \
  --selection-ledger /path/to/pef09_results/frozen_selection_ledger.json \
  --angle-ledger /path/to/pef10_results/frozen_angle_and_ridge_ledger.json \
  --precision-ledger /path/to/pef11_results/frozen_precision_ledger.json \
  --out /tmp/pef12
```

Unit-only CI tests the mathematics without the external binary checkpoints. Hosted Python 3.12 may remain unavailable independently of local Python 3.13 testing.
