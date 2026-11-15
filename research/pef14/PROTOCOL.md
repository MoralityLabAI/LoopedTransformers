# PEF-14 | Conditional phase mixtures on fresh trained models and a new context partition

Registration 2026-11-15. This protocol is committed **before PEF-14 model training, probe fitting, or heldout evaluation**. PEF-13's negative phase-aware result and its posthoc probe-versus-heldout diagnosis are known and motivate this experiment. PEF-13 outcomes remain unchanged.

## Source and new independent training

Fork PEF-13 PR #94, head `2bb8ec38edfadaac1c6927e6577bf21563d5af9c`, keeping every earlier PEF subtree untouched. New independent training seeds **211, 223, 239** and a newly hashed context split SALT=`b"pef14-independent-v1:2026-11-15:"`. Enumerate all 256 eight-bit context words, sort by (SHA256(SALT+bytes(context)), bytes(context)), assign the first 160 contexts to trunk training, next 48 to probe calibration, and final 48 to heldout. All 16 four-bit focal counterfactuals travel together with their context; no input history appears twice. Disjoint source sizes 2560/768/768.

Train separately for both PEF-07 tasks: `coalition=(a AND b) XOR (c AND d)`, and `threshold=1[a+b+c+d>=2]`. Use the exact PEF-07 32-channel causal two-token recurrent attention architecture with 12 tied visits, optimizer AdamW (lr=0.003, weight_decay=0), batch 128, 450 optimizer updates, clip 1, same initial RNG arithmetic (seed + 10000 for threshold), and independent train-only sampling generator seeded `seed*7919+59+100000*[task=threshold]`. CPU, PyTorch 2.10, deterministic algorithms, torch threads=1, single-thread BLAS. Train and hash **all six checkpoints and their initial/final weights** before any probe or heldout evaluations. Never reuse pretrained PEF-07 checkpoints in PEF-14. Preserve all models even if underperforming. Record untouched heldout native h12 task scores for context only *after* the full forecast ledger is frozen.

## Frozen probe-only representation, readout and precision

On **probe contexts only**, compute natural h8 states. Use their mean `mu` and top right singular vector `g` of centered states (top PC), with sign fixed by the maximum absolute coordinate. Independently extract one pair-derived direction: stack all four-corner hidden second differences for pair (0,1) across all four assignments of bits 2,3 and all 48 probe contexts; take its top right singular vector, sign-align with g, and remove its component parallel to g to obtain tangent `u_pair` (unit). Also generate one random unit tangent `u_random` from one FP64 CPU Gaussian vector with seed `14014+1000*seed+100000*[task=threshold]`, orthogonalized to g. For degenerate singular values / tangent norm <1e-9, use a deterministic Gram-Schmidt canonical coordinate fallback and flag degeneracy. Do not use model labels or task head weights to choose directions.

Fixed positive rotation angles `theta in {0.25,4.0}` degrees; for both tangents define `v=cos(theta)g+sin(theta)u`, `w=-sin(theta)g+cos(theta)u`. Remove one ungated affine rank-one direction from *every* h8 state: `z=h8-((h8-mu) dot v)*v`. Replay natural input visits 9..12. For each condition fit a single PEF-07 intercept-augmented standardized ridge classifier with L2 penalty 1 on the **projected probe** h12 states only, then freeze it. For each projected probe input i, record frozen clean ridge margin f_i, label y_i, gradient `J_i=grad_z f_i dot w` through the four natural recurrent visits (weights fixed), retained coordinate `q_i=(z_i-mu) dot w`, and PC scale `s=sqrt(mean_probe[((h8-mu) dot g)^2])`. Store every raw probe statistic and its SHA256 before any heldout inference.

## Four newly selected quantizer steps and probability forecasts

Fixed relative step sizes `{0.009,0.018,0.045,0.07}` multiplied by probe PC scale s. They differ from PEF-13's registered step values. In FP64 calculate exact nearest-even lattice error `d_i=Delta*round(q_i/Delta)-q_i` and linearized postquantization margin `f_i+J_i d_i`.

Four **fully probe-only** predictors are reported:

1. **Exact-probe phase:** average the indicator that `f_i+J_i*d_i` predicts the correct y_i on its own probe source.
2. **Uniform-phase:** average `clip(1/2 + ((2*y_i-1)*f_i)/(Delta*|J_i|),0,1)`, using exact sign limit for J=0.
3. **New conditional phase mixture:** group probe contexts into 48 complete 16-mask families. Within each focal pattern k (0..15), independently pair each example's (f,J,y) with the **47 other contexts'** quantization errors d_{c',k}. Let `p_cross` be the mean of the 47 cross-context class-correct indicators. Shrink with **eight pseudo-contexts** toward the uniform phase predictor: `p_mix = (47*p_cross + 8*p_uniform)/55`. Weights and grouping are fixed before training/analysis. This samples a *per-focal-mask conditional phase distribution* rather than reusing each example's exact phase.
4. **Clean-accuracy constant:** fraction correct under unquantized fitted probe ridge.

For each condition, save all context-level probability contributions, predictions, raw q/f/J/labels/phase residuals, SHA256 identities, random tangent and fitted ridge. Seal one canonical JSON **forecast ledger** before evaluating any heldout source state. No model exclusion, hyperparameter fitting to heldout, angle/step changes, or threshold tuning. Comparisons to PEF-13 are contextual, not the PEF-14 preregistered test.

## Evaluation and registered decision

Evaluate all 6 independently trained models × 2 tangents × 2 angles × 4 quantizer steps = **96 heldout conditions**, each on the 768 context-disjoint heldout histories. Apply the same ungated projection at h8, then apply nearest-even quantization **only along w**, via `z'=z+(Delta*round(q/Delta)-q)w`; replay natural visits 9..12 and score the *frozen probe-fit ridge*. Record exact correct /768, per-class recall, 16-mask error counts, unquantized fitted ridge accuracy, and untouched native h12 accuracy. Include a direct same-input heldout Jacobian evaluation only as a **posthoc diagnostic after all 96 scores are saved**, if performed.

For each coalition seed compute MAE on 16 conditions (2 tangents × 2 angles × 4 steps) for each probe forecast versus heldout accuracy fraction. **Primary descriptive criterion:** conditional mixture MAE <=0.030 and <= 0.90× uniform-phase MAE, in at least two of the three *fresh coalition model seeds*. Report `criterion_met` or `criterion_not_met` and all three seed outcomes even if failure. Threshold arm is a matched negative/control task, never omitted. Secondary aggregate differences over 6 seeds, per-step breakdown and all relative errors are reported without alternate success thresholds.

A fixed 90% uncertainty interval uses the **48 probe-context-level** contributions to the mixture estimator: `p_mix ± 1.645*sqrt(2)*sd(p_context)/sqrt(48)`, clipped to [0,1]. The factor sqrt(2) accounts approximately for variation from both source and destination cohorts; this is an exploratory exchangeability-based normal interval, not a rigorous coverage guarantee. Report coverage over each task/seed's 16 heldout conditions. No pseudo-replication of the 16 focal masks as independent observations.

## Mathematical statement to verify

Let K denote a focal mask, and A=(f,J,Y) and E=Delta*round(q/Delta)-q. For any bounded classification indicator `phi(A,E) in [0,1]`, true conditional risk is `R=E_K E[phi(A,E)|K]`. The independent phase mixture `R_ind=E_K E_{A|K} E_{E'|K} phi(A,E')` differs by at most `E_K TV(P_{A,E|K},P_{A|K} tensor P_{E|K})`, where total variation is sup_event|P(event)-Q(event)|. For our 8-pseudocount convex mixture with an ideal uniform phase prior, the population bias is bounded by the corresponding convex combination of these two phase-law TV errors. The bound concerns **distributional phase dependence**; it does not bound the recurrent model's nonlinear linearization remainder, which must be audited separately.

## Verification and limitations

Run `python -m pytest -q research/pef14/test_pef14.py`, then `python -m research.pef14.pef14 --out /tmp/pef14`. Separate training / probe ledger / heldout in code and on disk. Confirm all checkpoint bytes and weight hashes after evaluation. CI Python 3.12 tests invariant algebra; full training is a CPU-only optional job and must not be claimed to have run from green invariant tests. New independent seeds and new context split test replicability within **the same known Boolean tasks and architecture**, not novel operators or pretrained model transfer.
