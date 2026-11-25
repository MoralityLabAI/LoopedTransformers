# LS-NEURON-11: fresh parity recovery and four-state modular transfer

**Parity development replication gate PASS; modular transfer exploratory.** Parent [LS10 #381](https://github.com/MoralityLabAI/LoopedTransformers/pull/381), archive SHA256 fc3acfa32b1cdf56759573a00e382799fcf172ffa3c73d14e84033d8ee35c30d verified. No paid GPU, native TRM or protected cohorts. All 108 training runs preserved.

## Constructive modular recurrence

Let the four state centers be c_p=p-5/2 for p=0,1,2,3. Regions R_p=[c_p-1/8,c_p+1/8], initial state c_0, nearest-center readout. For s=h+a+b define r_j(s)=(GELU(30(s-j)+10)-GELU(30(s-j)-10))/20. A ten-neuron scalar cell is

    F(h,a,b,n)=-5/2 + r_(-2)(s)+r_(-1)(s)+r_0(s)-3*r_1(s)+r_2(s).

The nuisance input n is ignored by the construction. On each region and binary input, the ReLU counterpart has plateau values 0/1 for every ramp, and its output is exactly c_((p+a+b) mod 4). The 32 cases include both nuisance values. Each preactivation has absolute value >=5/4. Mills' inequality gives -1/5 < GELU(z)-ReLU(z) <=0: phi(5/4)<(2/5)/2=1/5 follows from pi>25/8 and exp(25/32)>2 (first three positive Taylor terms). A paired difference of errors therefore has absolute value <=1/5. Sum of absolute ramp coefficients is 7, yielding |F-c_target|<=7/100. Adding a disturbance of magnitude <=11/200 gives total error <=1/8. The readout margin is >=3/8. Induction from c_0 proves correctness for all finite histories under those disturbances. In particular epsilon=1/32 is valid. The direct rational construction check verifies all 32 affine plateau cases and the exact weight grids. The separate endpoint interpolation verifier also passes at epsilon=1/32 (508 boxes, 270 leaves).

The same residual architecture uses skip coefficient -1, skip bias -5/2, MLP gain 1/5, up weights (30,30,30,0), biases 70,50,40,20,10,-10,-20,-40,-50,-70, and down weights 1/4,-1/4,1/4,-1/4,1/4,-1/4,-3/4,3/4,1/4,-1/4. Four tensor scales (1/4,5,10,1/4) give exact signed Q4 codes (absolute values <=7), with two scalar offsets FP32. This is 66 recurrence slots plus three scalar init/readout conventions (69 bookkeeping slots); multi-class decoding uses fixed nearest-center arithmetic, not a learned readout. An explicit width upper bound, not a minimum or native-transformer result. Scales are explicit rather than max-abs. Exact-real semantics with stored dyadic coefficients and beta=1/5; no packed kernel or hardware rounding theorem.

## Frozen experiment

See protocol.md and lock.sha256. Parity: fresh seeds 379..402, same LS10 5% tensor-RMS perturbation and disrupted outgoing-weight control, same learned-scale initialization policy and region-only training. Modular: fresh seeds 403..414, structured/disrupted 5% and 15% plus random, explicit template scales for all arms. 800 Adam updates with normalized parameter weights, learned log scales, matched schedules and deterministic region constraints. Both tasks use known target-state labels. Modular has 512 constraints vs parity 256 and 10 channels vs 4, so cross-task recovery rates do not isolate state count. Controls match scales/noise but not initial function/loss. Numerical evaluation uses 256 fresh sampled histories per seed/horizon, T8/T128; nonfinite states count incorrect. No endpoint selection, repair, weaker radius or adaptive region search.

| Task / arm | Initially certified | Final certified | Newly certified | Final T8 (%) | Final T128 (%) | Final T128 all-prefix (%) |
|---|---:|---:|---:|---:|---:|---:|
| parity struct_05 |0/24|15/24|15|93.457|87.826|75.618|
| parity disrupt_05 |0/24|0/24|0|85.384|67.513|30.778|
| mod4 struct_05 |1/12|2/12|2|53.841|41.764|20.508|
| mod4 disrupt_05 |0/12|0/12|0|26.953|25.000|0.000|
| mod4 struct_15 |0/12|0/12|0|24.707|24.382|0.000|
| mod4 disrupt_15 |0/12|0/12|0|25.553|24.349|0.000|
| mod4 random |0/12|0/12|0|23.893|25.163|0.000|

Parity primary developmental gate (>=12/24 newly certified and >=6 additional vs disrupted) passes: 15 new vs zero, all initial failures are exact point violations. Prior LS10 9/12 is separate; descriptive combined recovery 24/36, not a pooled prospective population estimate. Modular newly certified seeds 405 and 414; seed412 initially certified but loses its certificate. Both recovered cells return to template-like integer codes with adjusted scales and offsets; this demonstrates local recovery, not unaided discovery. All 198 non-certified endpoints have exact point violations of the specified inclusion, no unresolved cases. These states need not be reachable. Mod4 random arm has ten nonfinite T128 initial runs and one final; mod4 struct_15 has one nonfinite initial run. Finite flags and per-history correctness retained. Numerical accuracy for such arms should be interpreted with the failure flags.

## Verifier and audit

Verifier improves dependency overestimation with endpoint interpolation. GELU''(z)=(2-z^2)phi(z), |GELU''|<=1 (for z^2<=2 use 2phi(0)<1, otherwise maximize (z^2-2)exp(-z^2/2) at z^2=4). Thus |F''|<=C=(1/5)sum |down_i| up_i,h^2. On [l,u], exact rational endpoint enclosures plus C(u-l)^2/8 bound the full image by the interpolation remainder theorem. Split dyadically until every leaf fits target inner region; point violations are verified with original rational GELU enclosure. The numerical grid proposes a witness but is not a proof. This differs from LS10's original naive interval extension while preserving exact semantics and disturbance target. Both handcrafted controls pass. Across 216 endpoints, 18 certificates (5,740 boxes, 3,038 leaves), 198 exact violations, 432 numerical replays. Code/scale decoding and initial weights replayed. No independent retraining.

Disturbed FP32 stress (2048 histories, 512 visits each, per-step random sign magnitude 1/32) for all final certified modular models and handcrafted control:

| Model | Errors / transitions | Max region-center error |
|---|---:|---:|
| handcrafted |0/1,048,576|0.0312653|
| seed405 |0/1,048,576|0.0615902|
| seed414 |0/1,048,576|0.0539529|

Numerical stress is descriptive, not a floating-point proof. Exact certificates separately establish arbitrary-depth correctness under the specified real semantics.

## Reproduction

PyTorch 2.10.0+cpu, deterministic single-thread workers, four concurrent workers. In this directory:

    sha256sum -c lock.sha256
    bash run.sh
    sha256sum -c verifier_lock.sha256
    python verify11.py
    sha256sum -c audit_lock.sha256
    python audit11.py

certificates.json SHA256 1ab6059f0f66ae4abd860ea63880e1c4869bedd6527e27b2ea5a75315138e673; construction.json a5c9fff6f411f8198a2af189202dd8d1027e91623684f9d7a83d43d1624f7880; audit_results.json 388e1cf9d4a4f4efd37146516101a3d9f00833bd696b97e9cf263b5262ae7b2c. Companion archive includes all raw model/receipt JSON, logs and sources. Both tasks remain development-only (FSI-01 REDESIGN_OR_PILOT_ONLY): 24 and 12 paired seed groups, no calibrated physical sham, prospective power or sealed population. Mathematical certificate of a particular model is exact evidence, independent of statistical recovery rates.

Next: exploit the exact integer-code structure with scale-only recovery, compare with unrestricted QAT and frozen controls at matched budgets, and test if a smaller width is sufficient for robust modulo-four recurrence. A stronger general result would characterize how the number of states and input alphabet constrain robust scalar neural automaton width. Native TRM and Lean formalization remain outstanding.
