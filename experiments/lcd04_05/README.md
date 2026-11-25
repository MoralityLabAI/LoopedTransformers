# LCD-04 / LCD-05: adaptive depth and selective state precision

Status: source implemented; no native results.

## LCD-04
Frozen Huginn BF16, 48 calibration / 192 held-out roots (three equal hop families). Decide after eight transitions, exit at 8 or continue to 24; one final coda. Features only use full states 6,7,8: relative residual, residual growth and successive cosine. Calibration quantiles {.25,.375,.5,.625,.75} and both threshold directions; choose highest macro accuracy then NLL with mean AND token-weighted recurrence <=16. Always-short feasible fallback. No gold/future state in routing. Test policy can drift over budget; record failure, never retune on test. Baselines fixed 8/16/24/32; random depth schedule exactly matches counts within family and 64-token length bucket. Token-weighted work need not match exactly; report it. Route prepass (8 steps per test root) used solely to create shuffled control, excluded from timed forward and counted separately. Each actual arm reruns same initial state in shuffled order, charge feature sync and full forward. Primary pilot gate: positive delta over random and fixed16, both test budget caps, latency <= fixed16. Report the full depth distribution and baseline competence; no confirmatory claim.

    python -m research_gym.scripts.lcd04_adaptive --stage smoke
    python -m research_gym.scripts.lcd04_adaptive --stage calibrate --out results/lcd04/cal.json
    python -m research_gym.scripts.lcd04_adaptive --stage test --freeze results/lcd04/cal.json --out results/lcd04/test.json

## LCD-05
Fit contrast and state right-singular bases from final-token vectors of 48 calibration roots, independent 48-root selection, 192-root test. R=16, reference 6, rank grid {4,8,16}, random orthonormal basis at matched rank and independently tuned. Real nibble-packed symmetric signed int4 residual, groups of 64, FP16 scale per group, BF16 protected coefficient vector per token, shared FP32 basis. Uniform int4/int8 and BF16 comparators. Quantize carried state after each of 16 native transitions; unpack before next transition or coda. Initial latent same across arms. All model weights, input embedding, recurrent matmuls and decoder remain native BF16. Residual is reprojected orthogonally before restoring protected coefficients. State codec only; there is no quantized kernel or claim of weight compression.

For D coordinates, L tokens, group size G=64, padded D'=ceil(D/G)*G, rank r, int4 logical payload is L*(D'/2+2D'/G+2r) bytes. Shared U costs 4Dr bytes. This counts actual tensor payload, excludes metadata/allocator overhead; report separate actual peak GPU allocation and incremental peak. Basis upload excluded from steady-state timing, codec included, last roundtrip diagnostic charged. Scratch may dominate, so smaller payload does not imply lower peak VRAM or latency. Primary pilot screen: accuracy within 2pp of BF16, positive paired delta over uniform Q4 and rank-matched random Q4, payload <=40% of BF16. Compare state PCA to contrast PCA at same rank; inference latency and peak are separate outcomes. No claim of hardware acceleration absent actual peak and wall-clock gains. Selection and test source/environment frozen.

    python -m research_gym.scripts.lcd05_precision --stage smoke
    python -m research_gym.scripts.lcd05_precision --stage fit --out results/lcd05/fit.json
    python -m research_gym.scripts.lcd05_precision --stage test --freeze results/lcd05/fit.json --out results/lcd05/test.json

Both pilots use exact generated digit labels and report candidate-conditional NLL, stratified paired intervals, raw receipts. Keep negative outcomes. No test-driven tuning. Confirm on fresh seeds with powered sample size and native verifier transfer.
