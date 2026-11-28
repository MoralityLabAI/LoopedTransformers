# LCD-01: native recurrent contrast pilot (registered design, 2026-11-25)

Status: implementation awaiting native execution. Source: Liu et al., [LoopCD](https://arxiv.org/abs/2610.02185v1), sections 3, E.1, F.2 and Tables A4/A8. Existing native model loader and task generator: research_gym/scripts/rmp_huginn_depth_sweep.py. Prior depth sweep is exploratory; fresh prompts here use distinct seeds.

## LCD-01A

Frozen Huginn-0125 revision bb6621b65e90b6a4b9b29ef88dc83866d450470c, RTX 5080 BF16, batch one, cache disabled, three-shot variable-copy chains with 1/2/3 hops. 48 calibration items (16 per hop); 144 new held-out items (48 per hop). Exact generated digit answers, 10 single-token candidates. Candidate-restricted NLL is conditional on those ten candidates; it is not whole-vocabulary perplexity. No weight updates. Each prompt has a deterministic common latent initialization across all arms and independently repeated forward passes for timing. Reject native API if manually unrolled logits fail native-forward parity or the prefix trajectory differs by more than 0.125 max absolute units; require identical top-1. Store all discrepancies. No results yet.

Run from repo root using the existing Huginn-compatible environment:
```sh
python -m research_gym.scripts.lcd01_huginn --stage smoke
python -m research_gym.scripts.lcd01_huginn --stage calibrate --out results/lcd01/cal.json
python -m research_gym.scripts.lcd01_huginn --stage test --freeze results/lcd01/cal.json --out results/lcd01/test.json
```
Model is cache-only by default. Use --allow-download only if approved. Check GPU power, free VRAM and disk beforehand. Preserve failures and partial receipts; start a new output path for retries. Pin dependency versions and record the actual GPU power and driver in the accompanying run log.

At R=8,16,32: unguided baseline, fixed logit contrast, margin-adaptive logit contrast, hidden contrast, independently calibrated norm-matched random hidden direction, orthogonal logit re-ranking, positive logit temperature scaling. Hidden reference grid {1,5,6,7}; logit reference grid {1,5}; dose {.25,.5,1}. Calibration chooses family-macro accuracy then candidate NLL (temperature chooses NLL). Frozen choices and code/data digests must match test. Test additionally evaluates the exact reference/dose-matched random hidden arm and Apple's fixed R=16/32 settings: R16 reduced-budget logit h5 and hidden h6; R32 full logit h1 and hidden h6; dose .5. The fixed and fitted arms are distinct. Random contrast is orthogonal to the genuine contrast per position before rounding; numerical norm matching is approximate after BF16 casting.

Primary pilot comparison: fixed R16 paper-hidden against R32 and R16 unguided, with calibrated hidden versus its exact matched sham as a mechanistic secondary. R16 is also compared with R8 via the R8 arms; report R16 separately because prior exploratory data had a peak at R16. Temperature is a rank-invariance sanity check and NLL control; orthogonal logit re-ranking removes the centered parallel component, as in the paper. Adaptive logit strength pays for two output passes. Report every family including near-chance three-hop outcomes; no post hoc family exclusions.

Record paired wrong-to-right/right-to-wrong, family-macro accuracy and conditional NLL, stratified paired bootstrap 95% descriptive intervals, synchronized end-to-end forward latency, max allocated VRAM, recurrent loops and coda passes. Layer-token work proxy counts 2 prelude + 4R core + 2 per coda; it omits LM-head and vector arithmetic and is not measured FLOPs. Latency excludes tokenization, disk and scoring; no throughput or long-generation claim. Compare same-depth and next deeper baselines. Report timings with machine/load details. The calibration trajectory is shared; test timing is independent per arm and arm order randomized deterministically.

Pilot screen: R16 hidden accuracy within 2 points of R32 baseline, positive delta over same-depth baseline, positive matched-sham delta, latency ratio <=0.70. Also report absolute task competence and the 1/3 digit-in-program heuristic. Screen is descriptive; 144 examples cannot establish a tight noninferiority claim. If baseline is weak or the effect is null, retain negative result; no expansion of the test grid after reading outcomes. Confirmatory sample size must be determined from calibration/pilot discordance with fresh seeds. Freeze a single primary method/comparator; require lower confidence bound >= -0.02 versus deeper baseline, superiority over same depth and sham, and measured cost gate. Quantization, longer contexts, independent initialization seeds and external tasks need their own fresh confirmation.

## LCD-01B: verifier transfer (next gate, not implemented)

Use existing Sudoku roster/verifier and Lean-CGRT proof-search interfaces. Sudoku: fresh uniquely solved roots, verify full grid or logically determined cell as appropriate; stratify difficulty and retain invalid output. Lean: pre-declaration environment, remove target theorem and all future declarations, enforce legal tactics, exact kernel closure, reject sorry/axioms outside whitelist; evaluate both tactic rank and proof closure at equal verifier calls and matched wall clock. Do not treat a tactic match as a proof. Separate model-family adapters and checkpoint/precision parity. Report depth/precision frontiers BF16 vs native quantized paths only after dtype/quantizer admission; quantized results are unimplemented.

## LCD-02: spectral extension (registered proposal, not implemented)

Define d=h_R-h_k. Learn a rank-r basis U from *calibration-only* contrast covariance, with equal-rank random basis and scalar contrast baselines. A=(omega I + U diag(lambda) U^T), constrained operator norm and guidance norm. Fit lambda with calibration-only verifier labels, choose rank {4,8,16} and ridge from calibration, freeze all choices; compare to matched-norm scalar and random-subspace guides, including training and per-token operator costs. Require positive held-out verifier delta against scalar guidance at matched compute. PCA is a variance basis, not proof of causal task modes. Separate label-free projected guidance from supervised fitted coefficients.

For Anderson-style state extrapolation, fit coefficients to minimize norm of a window of fixed-point residuals subject to sum(alpha)=1, regularize, cap correction norm and report ill-conditioning. Only call it fixed-point acceleration if the full state and loop map are stationary; otherwise label trajectory extrapolation. Adaptive depth must use observable prefix-only features (successive residual alignment/norm and optional paid coda confidence), never future states or verifier oracle. Charge all extra passes and use abstention as failure unless separately scored. Compare with calibrated fixed depth and equal expected compute. These are successor experiments, not claims from LCD-01A.

## Handoff

Run smoke, native admission, calibration, then frozen held-out evaluation; stop on parity failure. Do not tune on test, silently alter tokenizer or claim paper replication from these custom tasks. Publish per-item receipts and failure cases on this branch; defer larger GPU campaign until pilot is audited. 
