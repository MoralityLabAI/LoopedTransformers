# LS-NEURON-08: minimax region loss and learned quantization scales

**Primary development gate FAILED.** Fresh 12 paired seed groups, 48 model endpoints; no learned certificate at the prospectively fixed disturbance 1/32. Stacked on [LS07 #375](https://github.com/MoralityLabAI/LoopedTransformers/pull/375). No paid compute or native/protected NEC04/Sudoku data. LS07 parent archive SHA256 de297d39f21b88ce6623179719729fb8f702418208c4d09089459b407f2f885b verified; no parent weights used.

## Design

[Protocol](protocol.md) and source hashes locked locally before training. Scalar recurrent state, four GELU channels, 30 trainable weights/offsets and three fixed init/readout slots. Four factorial arms: mean vs maximum squared region excess, dynamic max-abs vs learned positive log scales for four quantized tensors. Signed Q4 integer codes in [-7,7], FP32 decoded products, two scalar offsets stored FP32. The learnable-scale arms have four extra optimized variables during QAT. 2400 Adam updates, matched fresh T8 minibatch streams within seed (batch64), FP32 600 / Q6 300 / Q4 1500, all arms reset optimizer at precision switches, per-seed gradient clipping. Both region losses use 16 cases x 16 Taylor subintervals, the same target 3/32, same weight 10. Training uses |GELU''| <= 1; derivative and enclosure evaluated numerically and not taken as a proof. All 48 endpoints scored on fresh numerical development histories and exact fixed-region verification. No weaker radius or repair allowed in this experiment.

## Results

Equal-seed mean Q4 final correctness (%), 512 histories per checkpoint and horizon. All certificates at disturbance 1/32.

| Arm | T4 | T8 | T16 | T32 | T64 | T128 | Certified |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean_fixed |71.65|66.86|63.27|60.97|58.22|56.41|0/12|
| max_fixed |56.45|52.36|52.41|50.67|51.17|50.29|0/12|
| mean_learn |80.50|74.48|69.52|66.94|63.62|61.75|0/12|
| max_learn |55.29|53.87|54.05|54.54|54.38|53.17|0/12|

FP32 latent-weight accuracy (not deployed Q4):

| Arm | T4 | T8 | T16 | T32 | T64 | T128 |
|---|---:|---:|---:|---:|---:|---:|
| mean_fixed |71.13|61.98|55.88|54.51|50.93|48.93|
| max_fixed |56.58|53.16|52.08|50.29|52.03|49.93|
| mean_learn |65.10|59.98|56.01|53.45|51.90|50.16|
| max_learn |53.99|51.38|51.32|49.95|49.92|49.58|

Fixed-region maximum error on a 257-point grid in each parity/input case (descriptive lower-bound diagnostic, not an enclosure of the continuous maximum):

| Arm | Mean | Median | Best | Worst | Q4 T128 all-prefix accuracy (%) |
|---|---:|---:|---:|---:|---:|
| mean_fixed |.645789|.627873|.258863|1.112350|15.28|
| max_fixed |.603706|.610948|.467627|.754550|0.24|
| mean_learn |.529871|.483875|.247968|1.148315|23.91|
| max_learn |.588444|.608536|.289532|.771947|7.81|

The allowed transition error is 3/32 = .09375. Every checkpoint has a rational point violation of the inclusion, not merely a failed search; positive control passes all 16 cases using 248 boxes and 132 leaves. Primary comparison max_learn vs mean_fixed = 0 additional certified seeds; both thresholds (+3, >=6/12) fail. Mean-learn Q4 T128 advantage over mean-fixed +5.34pp; mean-learn exceeds max-learn by 8.58pp. These are exploratory paired-seed summaries, not population effects. Max loss has lower observed worst across-seed grid error, but its grid median and task accuracy do not establish superior provability.

## Exact witnesses and optimization interpretation

Mean_fixed seed163 and mean_learn seed227 both score all prefixes of all 512 sampled T128 histories correctly. Seed227 nevertheless has exact fixed-region point violation: parity p=0, h=1/8, input (1,0,0), target odd. Its next state is enclosed by [909151136197911302272699/1208925819614629174706176, 909151136197911302272709/1208925819614629174706176], approximately .7520321937434694, outside the required [29/32,35/32]. Seed163 likewise maps h=-1/8 with input (0,1,0) to ~1.258862900906128, above 35/32. These disprove the prescribed fixed-region inclusion even though numerical rollouts succeed. The witness states need not be reachable from init, so no all-history counterexample is established by these point violations.

The constant transition F(h,x)=1/2 is representable by cancellation of the skip and an affine offset. Its maximum transition error is 1/2 on the balanced parity cases. It illustrates a competing noncomputational solution to the minimax training landscape. This does not prove the observed optimizer converged to that exact constant function or to a stationary point. The observed best max-loss grid errors remain far above the threshold; use direct casewise gradient diagnostics to evaluate the hypothesis that early hard-max training concentrates updates on a few constraints before discovering parity structure. A prospective curriculum from mean to smooth-max to hard max, or explicit active-constraint balancing, is a reasonable next comparison. Keep the original radius and fresh seeds. QAT scales improved the mean arm's sampled performance, but there is no certifiable improvement on the primary endpoint.

## Audit and reproduction

In this directory with PyTorch 2.10.0+cpu:

    sha256sum -c lock.sha256
    for arm in mean_fixed max_fixed mean_learn max_learn; do python train08.py --arm "$arm"; done
    sha256sum -c verifier_lock.sha256
    python certify08.py
    python audit08.py

Four one-thread workers used for training. Audit passed 48 checkpoint receipts, 576 numerical replays (FP32/Q4 x six horizons x 48), 1,548,288 scalar XOR prefix labels, 48 rational point violations and one positive-control certificate replay. Code/scale decoding, integer ranges, paired sampler hashes, two scalar offsets and numerical enclosure receipts verified; max float bound replay difference zero. Audit does not independently retrain. All 48 endpoints preserved. Exact-real arithmetic with stored FP32 dyadic coefficients and beta=1/5; no Lean or floating-point hardware certificate. See LS06 for CDF interval bounds. `verify.py` includes older adaptive proposal helpers from LS07, but `certify08.py` calls only the fixed-region functions and no adaptive search.

Independent statistical unit is paired seed/checkpoint, n=12; 512 histories per horizon, repeated intervals and horizons correlated. No prospective population power, calibrated matched physical sham or sealed task family. FSI-01 REDESIGN_OR_PILOT_ONLY. No exact total training FLOP matching. All four arms receive region-state supervision; no task-only baseline in this factorial (LS07 provides a separate, non-matched comparison). The term fixed scale denotes dynamic max-abs scaling rather than a permanently frozen numeric scale. The prior certified LS07 cell remains a separate result.

Raw receipts: certificates.json SHA256 580f3b6b31729ea5053e7fa0d3b5a2dbabdff94ff1f4cf0e85b5f020c198654d; audit_results.json SHA256 7630d110f05a5e728900dde0bf54558510aa2d0b2608fc4ffe89b57700024adf. Companion archive includes all 48 weights and raw correctness vectors, logs, rational witnesses, source and hash manifests. No checkpoint binaries needed to replay.
