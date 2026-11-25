# LS-NEURON-06: exact-real recurrent parity certificate

**Separate results:** a reachable certified counterexample for the selected LS05 learned Q4 checkpoint, and a constructive arbitrary-depth theorem for a handcrafted four-GELU cell. No trained-model population claim, no paid compute or sealed cohorts.

## Learned checkpoint

Seed 37 staged QAT checkpoint SHA256 8bc1d3319d65fa26693589e381b2e092243df3cdd2581bc6e72d3a4c587b2731. FP64 evaluation of its stored FP32 fake-quantized tensors scores all 2,396,744 prefixes through depth 7 without error. A deterministic selective beam finds 129 errors in 126,800 expansions at depth 11. It does not establish shortest failure depth. Witness binary triples (a,b,nuisance):

    101 000 000 110 110 110 111 011 100 101 011

True parity 1, final logit -2.0510699995885195. The self-contained rational interval verifier decodes witness_data.json (hexadecimal dyadic weights including signed zeros), checks its decoded tensor JSON SHA256, and encloses the final logit in [-2.051069999588531,-2.051069999588530] under exact-real GELU and residual coefficient 1/5. First ten margins are strictly positive. The CDF enclosure uses Machin's identity, alternating arctangent sums, Taylor's theorem with integrated remainder, and Mills' inequality. Outward rounding uses a 2^-80 grid. This rules out global all-history correctness for these weights in the specified semantics.

## Constructive theorem

Define r(s)=[GELU(32s-8)-GELU(32s-24)]/16 and F(h,a,b)=b+r(h+a-b)-r(h+a+b-1). Four GELU channels process both focal bits per visit. In the 16D residual architecture, cancel the identity residual with skip=-I, add b on coordinate 0, set preactivations as above, down weights (5,-5,-5,5)/16 on coordinate 0, all other coordinates reset to zero. Init=0 and readout=h-1/2. All parameters lie exactly on their respective max-abs symmetric Q4 grids and survive the original PyTorch fake quantizer bitwise. 513 dense parameter slots; the scalar-state restriction has 33 slots, with the same three input coordinates.

For p in {0,1}, R_p has |h_0-p|<=1/8 and |h_j|<=1/8 for j>0. For any binary triple (a,b,n) and any additive state disturbance e with infinity norm <=511/4096, the next state belongs to R_(p XOR a XOR b), with correct readout margin >=3/8. Correctness for every finite horizon follows by induction from zero.

Proof: replace GELU by ReLU. The ramp becomes 2[ReLU(s-1/4)-ReLU(s-3/4)], flat at 0 for s<=1/4 and 1 for s>=3/4. Both arguments remain within 1/8 of integers. Enumeration of all parity/input cases shows the ReLU expression equals p XOR a XOR b throughout the region. Each preactivation has absolute value >=4. Mills' inequality gives |GELU(z)-ReLU(z)|=|z|Phi(-|z|)<=phi(|z|)<exp(-8)<1/1024; the last bound follows from the positive Taylor partial sum of exp(8) through degree 9. Absolute coefficients sum to 1/4, yielding an output error <=1/4096. Adding 511/4096 gives 1/8. Other coordinates are reset and the readout margin is >=3/8.

Contraction: the derivative of the ReLU counterpart vanishes throughout each region. For |z|>=4 the derivative difference is bounded by (|z|+1/|z|)exp(-z^2/2) <=17/(4*1024); the envelope is decreasing. The absolute derivative prefactors sum to 8. The induced infinity-norm Jacobian bound is therefore 17/512. This establishes regional contraction, not contraction between parity states.

## Reproduction

Run, in this directory:

    python certificate.py
    python audit_certificate.py
    python interval_verify.py

All three are stdlib-only. The exact arithmetic certificate checks 16 region/input cases, Q4 levels and bounds; the independent matrix auditor derives the ReLU transitions directly from the generated tensors. Numerical stress on 2048 histories of length 512, with independent random-sign coordinate disturbances of magnitude 511/4096, gives zero wrong decisions in 1,048,576 FP32 transitions; max parity-coordinate error 0.1247636079788208, min margin 0.3752440810203552. Full companion archive SHA256 8d23fddfe7d6c9e4da9f88a277c4e1a0a5c6f62273ab1f57ae852e2cfb57b95a includes numerical scripts, checkpoint, raw receipts and source manifest. No Lean kernel proof or hardware rounding certificate. Q4 denotes symmetric quantized weight grids with per-tensor scales, not packed execution.

Next: use the constructive transition as a positive control for region-inclusion training, with plain QAT, curriculum and randomized initialization controls. Report certificate rates for fresh independent seeds.
