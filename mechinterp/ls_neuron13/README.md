# LS-NEURON-13: discrete codebook repair and continuous scale fitting

**Primary development gate FAILED (3 new certificates, target 6).** Discrete repair recovers exact certificates at 15% perturbation for the first time in this series. Adaptive paired multi-anchor search yields 4/12 exact certificates and 96.26% sampled T128 correctness. Parent [LS12 #384](https://github.com/MoralityLabAI/LoopedTransformers/pull/384); verified parent archive SHA256 5f4573f166ebc08c874c87bb00828d803aede866901f5ca55ed26fc0ee3b06b2. No paid compute, native TRM or protected cohorts.

## Fixed development experiment

See protocol.md and lock.sha256. Eight-GELU scalar modulo-four cell, 52 integer entries across skip/up/up-bias/down tensors, four positive tensor scales, two FP32 offsets; 54 dense recurrence parameter slots. Twelve fresh paired seed groups 433..444 at 5% and 15% tensor-RMS noise. All five arms share identical decoded initial weights within seed/dose. Regions R_p=[p-2-1/8,p-2+1/8], exact-real GELU and beta=1/5, disturbance 1/32. No repair after certificate scoring. Full QAT is the parent 800-update control with Taylor-bound penalty; continuous-only and the two discrete repair arms use a point-grid objective with no trajectory data (32 cases x 9 equally spaced points, MSE plus ten times mean squared positive excess over 3/32). Different optimization algorithms and compute budgets; this is not an equal-FLOP causal contrast. Fixed code range [-7,7]; stored scales and decoded products FP32, not packed four-bit execution.

### Alternating search

For a codebook C=(C_s,C_u,C_b,C_v), positive scales alpha,beta_u,gamma,delta and combined offset c, write

    F(h,x)=h+alpha*(C_s q)+0.2*delta*C_v GELU(beta_u*C_u q+gamma*C_b)+c,
    q=(h,a,b,n).

Here beta_u is the up-weight scale, distinct from the residual gain 1/5. At fixed C, analytic-Jacobian bounded least squares optimizes four log scales and the combined offset. Down bias remains fixed. On each code search round, fix the two nonlinear preactivation scales and construct the two features C_s q and 0.2*C_v GELU(...). Profiling the linear skip/output scales and intercept gives a cheap proposal for every candidate. The profiler checks an interior and box-edge least-squares fit with bounded positive scales and offset; it ranks proposals by the point loss, but does not globally solve that nonlinear loss. Refit all continuous variables after an improving edit. Single-coordinate search enumerates +/-1 moves. Paired search additionally knows the template's ramp pairing: copy mate input codes, change paired incoming/bias codes jointly, align opposite down codes, or set the bias-code gap to two. Neither search consults unperturbed target codes. The prior supplies known ramp geometry, not unaided architecture discovery. Sixteen rounds plus initial fit, maximum 40 residual evaluations per fit, deterministic ties.

## Primary results

Final correctness averaged across twelve seeds and 256 sampled histories per horizon. The initial certificate count is the same for every arm in a dose. New requires a rational point violation at init and exact certificate at endpoint.

| Dose | Arm | Init cert | Final cert | New | Lost | T8 % | T128 % | T128 all-prefix % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|5|frozen|10|10|0|0|100.00|100.00|100.00|
|5|full QAT|10|2|0|8|46.48|37.89|16.67|
|5|continuous|10|12|2|0|100.00|100.00|100.00|
|5|single|10|12|2|0|100.00|100.00|100.00|
|5|paired|10|12|2|0|100.00|100.00|100.00|
|15|frozen|0|0|0|0|32.32|21.13|0.00|
|15|full QAT|0|0|0|0|29.43|20.48|0.00|
|15|continuous|0|0|0|0|45.93|25.91|3.74|
|15|single|0|3|3|0|84.18|65.89|54.85|
|15|paired|0|2|2|0|92.58|67.48|57.65|

Primary 15% single-minus-continuous new-certificate contrast = +3 seeds, satisfying the difference gate but not >=6/12 absolute new certificates. Thus the combined gate fails. Single recovered seeds 435,436,440; paired recovered 440,441. The 5% continuous recovery (seeds 436,438) needs no code edits; the original low-dose checkpoints were already 100% on sampled rollouts. All frozen, continuous and code-search 5% arms have identical codes within each seed. Full QAT loses eight initial certificates and changes a mean of 3.5 code positions. These are matched-start observations under distinct objectives/compute, not population comparisons.

## Adaptive multi-anchor extension

After the initial numerical endpoint sweep, before exact certificate scoring, freeze extension.md and extension_lock.sha256. Repeat all twelve 15% seeds with three additional arms. Profile candidates at both current and initial preactivation scales; shortlist the top four per anchor plus current codebook. Refit all five continuous variables per shortlist entry, retain the strict best improvement over the incumbent. Continuous_restart uses only the incumbent codebook and both anchors. No exact proof or rollout signal enters search. Higher additional cost; adaptive diagnostic, not a new primary gate.

| Arm | Final cert | New | T8 % | T128 % | T128 all-prefix % | Mean changed code positions |
|---|---:|---:|---:|---:|---:|---:|
| continuous_restart |0|0|46.48|25.33|3.74|0.00|
| single_restart |3|3|97.75|81.09|75.88|2.67|
| paired_restart |4|4|99.93|96.26|95.87|4.75|

Paired_restart certified seeds 434,440,441,444. Single_restart certified 435,440,444. The best paired restart has 4/12 certificates; the remaining eight are exact violations of the fixed region, even with high numerical accuracy. A fixed-region violating state may be unreachable; it is not by itself an all-history counterexample. The combined union of successful seeds across adaptive methods must not be counted as a single method's success rate. Positive template control passes.

### Search cost (15%, summed over 12 seeds)

| Arm | Candidate profiles | Least-squares residual evaluations |
|---|---:|---:|
| continuous |0|2007|
| single |19829|1266|
| paired |28846|1324|
| continuous_restart |384|8236|
| single_restart |39732|32779|
| paired_restart |57744|27336|

Full QAT has 9600 Adam updates across seeds, not comparable units. The paired restart gain is accompanied by much greater search cost. No claim that code search is more compute-efficient than QAT at matched FLOPs. The structure-aware proposal family is an intervention on representation, so causal attribution to search alone is limited.

## Exact verification and audit

Use the unmodified parent LS12 verifier: independently evaluate a numerical grid's proposed violating point with rational GELU enclosures, or prove the full region using affine-ReLU tail bounds and exact rational endpoint interpolation with |GELU''|<=1. All 32 state/input combinations must pass for each certificate. Induction from -2 with nearest-center readout gives arbitrary finite horizon under per-step absolute disturbance <=1/32. Exact-real arithmetic with stored FP32-decoded weights, no floating-point kernel or Lean proof. The four recovered paired-restart models require 32 accepted boxes each (one per case). Across 156 final method/dose/seed entries: 60 certificates and 96 exact point violations (includes repeated weights); 24 distinct initial conditions: 10 certified, 14 exact violations. No unresolved cases. 180 stored receipt files, 576 stored numerical horizon evaluations, 147 accepted discrete edits across primary and extension. Audit recomputes code decoding, initialization equality, legal accepted moves, monotone logged loss and count consistency, stored rollout vectors and fresh exact endpoint verification. It does not replay every rejected candidate's optimization or independently retrain full QAT. Raw ledgers preserved. Parent certificate instrumentation and source hashes retained.

## Reproduction

PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0; four one-thread CPU workers. In this directory:

    sha256sum -c lock.sha256
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python run13.py
    sha256sum -c extension_lock.sha256
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python restart13.py
    sha256sum -c audit_lock.sha256
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit13.py

The separate qat13.py allows individual frozen/full QAT replay. Companion archive contains raw receipts, full exact certificates, logs and hash manifest. FSI-01 REDESIGN_OR_PILOT_ONLY: independent unit paired seed group n=12 per dose, repeat observations correlated, adaptive extension, no calibrated physical sham or prospective population power. Explicit labeled state regions and structural prior are supplied. Native TRM and formal kernel validation remain outstanding.

Next: prospectively test multi-anchor paired repair on fresh 15% seeds with a fixed search budget and stronger matched-compute controls; analyze certificate-aware interval loss versus point-grid loss. The exact verifier is a terminal test, not a training oracle. A separate width/precision experiment can test whether codebook recovery transfers to binary shared-carry cells.
