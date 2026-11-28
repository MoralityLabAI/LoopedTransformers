# LCD-03: bounded Anderson recurrent extrapolation

Status: registered native pilot, implementation awaiting GPU admission. Parent experiments LCD-01 (scalar LoopCD), LCD-02 (spectral). This is a new experiment motivated by LoopCD, not a result from the paper.

## Mechanism

For a stationary recurrent map F and successive states x_i, use residuals f_i = F(x_i)-x_i = x_(i+1)-x_i. For m consecutive residuals form Gram G_ij = <f_i,f_j>. Solve for alpha minimizing alpha^T (G/s + ridge I) alpha subject to sum(alpha)=1, with s = trace(G)/m. The candidate is y=sum_i alpha_i x_(i+1). Apply correction g = dose*(y-x_t), clipped to 2*||x_t-x_(t-1)||. Abort correction for zero residuals or unstable coefficients (L1 > 8); record condition, coefficients, clipping and fallback. Ridge > 0. The recurrence is BF16; Gram and correction FP32, small solve FP64, final state cast back. The implementation uses one batch-one full-state residual vector per transition.

Two distinct interventions:
* AA12: apply once immediately after 12 native transitions, then execute 4 more native transitions using the true next step/index.
* AA16: extrapolate at the 16th native transition and read out directly.

No recurrent calls are added during inference. Each main arm uses 16 recurrent steps and one coda. The method is bounded single-intervention Anderson-style mixing, not a full repeated Anderson solver.

## Native admission

For one calibration root from each hop family, audit same-seed native forward versus manual unroll at R16 and R32 (including top-1 equality), the R16 prefix of R32, and exact native continuation from saved full h8 and block index. Test the same complete h6 and input embeddings at actual step/index pairs 6, 8, 12 and 16, varying the seed to check stochasticity. Maximum absolute error <= .125 and relative L2 <= .005 for every tensor check. If this fails, write BLOCKED_NATIVE_MAP and stop before calibration. This finite audit can establish approximate stationarity on admitted cases; it is not a proof of a global fixed-point map. The implementation reuses native no-cache paths. Before native execution, inspect model revision for additional mutable state or cache side effects; reject or extend the admission audit if needed.

## Prospective protocol

Huginn-0125 pinned revision from LCD-01, cached locally, BF16, batch one. Fresh three-shot variable-copy tasks (1/2/3 hops) with exact forced-choice 10-digit answers. 48 calibration roots, 192 held-out roots; seeds 2026112507 and 2026112508. Test roots are generated only after checking the frozen calibration record. Separate output paths. Frozen config bound to code, hardware/model environment, calibration data hash, and calibration summaries. No target labels enter the extrapolation or diagnosis; labels select calibration hyperparameters.

Grid for AA12 and AA16 independently: m={3,5}, ridge={.001,.01}, dose={.25,.5,1}. Select family-macro accuracy then candidate NLL, deterministic config tie-break. Scalar reference is h_(t-4) with the same dose grid. Primary frozen arms AA12 and AA16 are retained separately; AA12 is the primary hypothesis. Compare to:
* unguided R16 and R32;
* tuned scalar extrapolation at the same intervention time (h_t-h_(t-4));
* fixed scalar LoopCD at R16 with h6 and dose .5;
* scalar using the same full-state correction norm as AA at the intervention time;
* orthogonal random full-state correction at the same norm, applied at the same time with identical remaining recurrence.

The last two controls compute the AA correction to obtain its norm and are diagnostic, not fair latency baselines. Pre-cast norms match unless the scalar axis degenerates; retain the flag and actual post-cast norm. No test-time selection or family exclusions.

Report raw paired correctness, conditional NLL, state-correction norm, coefficient L1/condition, clipping/fallback rates, synchronized forward time, peak allocated memory. One additional native transition at the *final* state measures relative residual ||F(h)-h||/||h||, after prediction/timing, equally for every test arm; charge it separately as one diagnostic call. Never use this diagnostic to choose an answer. Reduced residual and improved correctness are separate hypotheses. In nonstationary cases do not call it fixed-point acceleration.

Pilot screen: AA12 has positive paired accuracy delta versus R16, tuned scalar, norm-matched scalar and norm-matched random, with forward time <=110% of R16. Also report R32 recovery within 2pp and latency <=70% of R32. Residual delta is a secondary mechanistic outcome, and may have opposite sign to accuracy. AA16 is a separate terminal extrapolation comparison. Use descriptive paired family-stratified bootstrap intervals. No powered claim on 192 roots; confirmation requires fresh seeds, sample size from pilot discordance, and frozen primary comparisons. Report baseline competence and negative outcomes unchanged. No end-to-end generation or quantization claim.

## Run

```sh
python -m research_gym.scripts.lcd03_anderson --stage smoke
python -m research_gym.scripts.lcd03_anderson --stage calibrate --out results/lcd03/cal.json
python -m research_gym.scripts.lcd03_anderson --stage test --freeze results/lcd03/cal.json --out results/lcd03/test.json
```

Prerequisites: LCD-01 native admission, existing Huginn-compatible local environment, GPU power/free VRAM and cached snapshot. Record environment and power/load. CPU smoke checks numerical mechanics on synthetic contractions and degenerate histories; it does not measure language-model accuracy. Keep partial/failure receipts; do not overwrite them.

## LCD-04 follow-up

Use prefix-only residual norms, alignment and conditioning, optionally paid decoder confidence, to allocate recurrence on fresh calibration roots. Require a fixed-cost baseline and a depth-distribution-matched random assignment as comparators; account for every coda and native transition. If stationarity admission fails, label the controller as trajectory-based rather than fixed-point. Do not choose a stopping policy from the LCD-03 test set.
