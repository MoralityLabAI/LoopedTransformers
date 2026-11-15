# PEF-13 | Phase-aware lattice forecast after ungated ablation

Registered 2026-11-15 **before PEF-13 heldout quantization replay**. Earlier PEF-12 results, including discrepancies at the pre-existing 0.005/0.02/0.05 lattice steps, are already known. New prospective step levels and scoring rules are fixed here. PEF-12's successful Gaussian prediction and imperfect uniform-phase quantization are motivation, not independent validation.

## Pinned model and source

Fork the PEF-12 PR #90 exact head `b8d71579e61b770f707fca45872b1ae7ec55fa8e`. Read the original six PEF-07 checkpoints from the local evidence archive, and fail closed on the checkpoint bytes, parameter digests, PEF-07 training ledger SHA256 `1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714`, PEF-09 selection ledger SHA256 `f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157`, PEF-10 angle/ridge ledger SHA256 `3ef6ce605540a5009f6896cf02a938c200bde23c7a5104d378aace7388430177`, PEF-11 precision ledger SHA256 `f0bd80d8dc7b4af99dbc24be2723828bcf85be5b6f60f46452581e4a53661741`, and PEF-12 *already frozen probe Jacobian/margin forecast ledger* SHA256 `fa9a3f3727b9551be721ca08e307082b81d1d2da2e359a34d88d26c9e7185b49`. Use the original 768 probe vs 768 heldout histories over context-disjoint 48+48 source groups, containing all 16 focal-bit combinations per group. No weight updates, no decoder refitting, no evaluation hyperparameter search.

## Frozen grid and models

Use all six task/seed combinations (coalition and threshold; seeds 103/131/167). At visit 8, exactly the two PEF-12 tangents `random` and `pair01`, and the *positive* PEF-12 PEF-10 angles `0.25°` and `4.0°` (four interventions/model). For each condition, load its serialized PEF-11/10 frozen projector v, retained transverse direction w, center mu, scale s, and untouched PEF-10 h12 ridge; check its direction and ridge digests against both the PEF-11 ledger and PEF-12 frozen forecast ledger. Project each original natural h8 state onto the same ungated affine hyperplane: `a=h8-((h8-mu)·v)v`. Use the original tied recurrent steps 9–12 and the frozen ridge margin, exactly as PEF-12.

The **prospective** *relative quantization step* grid is fixed to `{0.0075, 0.0125, 0.035, 0.08}` times s. PEF-12's already evaluated `{0.005, 0.02, 0.05}` may be reread only to cross-check implementation and will not enter the PEF-13 success rule.

## Probe-only phase-aware forecast

For each of the 24 conditions, load PEF-12's sealed *probe-only* clean scalar margins f_i and input directional derivatives J_i(w) (768 records), with source context ordering and checkpoint verification. Compute q_i=((a_i-mu)·w) from **probe h8 states only**. For each fixed step Δ=s*relative, compute the **deterministic nearest-even quantizer residual**
`d_i(Δ) = Δ*round(q_i/Δ)-q_i`.
The linearized phase-aware post-quantization signed margin is
`mhat_i(Δ) = (2y_i-1) * (f_i + J_i(w)*d_i(Δ))`.
The forecast is the fraction of the 768 probe rows where the predicted class `1[f_i+J_i d_i>=0]` matches the true label; no parameters are fitted on the heldout and no noise is injected in the prediction.

Independent comparators computed on the same 768 probe rows:
* Uniform-phase (PEF-12) approximation `mean clip(0.5 + m_i/(Δ |J_i|),0,1)` with the exact zero-derivative limit.
* Frozen clean-accuracy constant from unperturbed probe predictions.
* **Phase-shuffled negative control:** deterministically permute the 768 probe residuals d_i with `torch.randperm(768,generator=CPU Generator().manual_seed(1313000+seed*10000+1000*[task=threshold]+100*tangent_idx+10*angle_idx+step_idx))`; apply permuted d to the original probe margins/J. This preserves the marginal lattice-error distribution but destroys the per-input phase dependence.

For every condition, save the exact probe q, d, f, J, labels, all predictions and controls; SHA256-seal **all 96** model-condition-step probe forecasts before any heldout states are evaluated. A second run must match that forecast ledger hash with the same environment. Keep exact PEF-12 probe row ordering for label association. The model's observed postquantized probe accuracy may be measured after freezing these predictions, as a supplementary local linearity check, but it cannot enter the primary decision or change the forecast.

## Heldout scores and preregistered criteria

Only after the forecast ledger is written and hashed may the code generate heldout natural h8 states. Apply the same *nearest-even* scalar quantization to heldout projected states at each fixed step, replay the natural suffix, and score the **original frozen PEF-10 h12 ridge**. Report correct/768, per-class recalls, 16-focal-mask error histogram, per-condition probe-vs-heldout prediction difference, difference to the uniform-phase and clean predictors, and comparison with phase-shuffled control. Preserve the raw per-condition receipts.

For each coalition seed, compute mean absolute fraction-correct error (MAE) across its 2 tangents × 2 angles × 4 new steps = 16 forecasts for each predictor. **Primary calibration rule:** phase-aware MAE `<=0.035` AND phase-aware MAE `<=0.75 * uniform-phase MAE`; the criterion passes if at least two of the three coalition seeds pass both. Report failure just as prominently if it occurs. Every threshold model is an independent matched task control and appears in tables regardless of outcome. The phase-shuffled negative control and clean constant comparisons are descriptive; neither is an alternate success pathway.

**Exploratory geometry diagnostics (registered without a pass criterion):** for each case report phase-error mean/std, normalized phase concentration, Pearson corr(signed margin, signed J*d), and fraction of probe samples with `abs(m)>abs(J*d)` (a first-order sign robustness certificate). Also report the per-step signed forecast error and class-conditional scores. No inference about calibration over unseen Boolean operators or pretrained LMs.

## Conditional mathematical guarantee

Let x be an h8 state after affine rank-one ablation, w a fixed unit vector, f the differentiable frozen h12 scalar margin, q=(x-mu)·w, and d=Δ round(q/Δ)-q. If along the interval from x to x+d*w the directional second derivative has magnitude at most H, Taylor's theorem gives
`|f(x+d*w) - (f(x)+d*J_w f(x))| <= H*d*d/2 <= H*Δ*Δ/8`.
Thus whenever the **linearized signed margin** has magnitude strictly greater than `H*Δ²/8`, the linearized and nonlinear classifications agree. This is a conditional, single-example certificate; H is not asserted to be known for the trained network. Error probability across contexts also depends on the joint distribution of (m,J,q), which motivates the empirical probe-to-heldout transfer test.

## Execution and governance

Run `python -m pytest -q research/pef13/test_pef13.py`, then `python -m research.pef13.pef13 --evidence /.../registered --selection-ledger /.../pef09_results/frozen_selection_ledger.json --angle-ledger /.../pef10_results/frozen_angle_and_ridge_ledger.json --precision-ledger /.../pef11_results/frozen_precision_ledger.json --jacobian-ledger /.../pef12_results/sealed_probe_forecasts.json --out /tmp/pef13`. Binary checkpoints stay external. Hosted Python 3.12 verification is separate from local Python 3.13 results. Leave PR unmerged.