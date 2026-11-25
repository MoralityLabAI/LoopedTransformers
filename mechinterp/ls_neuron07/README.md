# LS-NEURON-07: trained and repaired three-GELU parity certificate

**Primary development gate FAILED.** A secondary exact-real arbitrary-depth certificate succeeds on one trained-and-repaired checkpoint. Synthetic parity, no paid compute, no population generalization, native TRM or hardware rounding theorem. Stacked on [LS06 #373](https://github.com/MoralityLabAI/LoopedTransformers/pull/373).

## Training

12 paired seeds (101,103,107,109,113,127,131,137,139,149,151,157), 36 endpoints. Scalar state, four GELU channels, 33 slots including three fixed init/readout slots. Skip-biased random initialization. Adam 1800 updates, 64 random eight-visit histories/update, FP32 600 steps, Q6 300, Q4 900. Task prefix-state MSE shared by task-only QAT, sampled region loss, and Taylor-bound region loss. Auxiliary arms cover 16 parity/input cases and eight subintervals/case, and penalize excess above 3/32 with coefficient 5. Training uses the global bound |GELU''|<=1 and numerical Taylor enclosures; independent rational verification is separate. See [protocol](protocol.md). All region labels explicitly supplied.

| Arm | FP32 T4 | FP32 T8 | FP32 T16 | FP32 T32 | FP32 T64 | Q4 T4 | Q4 T8 | Q4 T16 | Q4 T32 | Q4 T64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qat |53.96|50.49|50.03|49.38|49.74|57.29|52.26|51.35|50.18|49.76|
| sample |64.83|60.19|56.92|54.51|52.23|65.48|61.69|60.38|56.41|55.29|
| bound |61.12|58.20|56.46|55.29|54.31|65.95|62.71|60.43|56.07|55.92|

Equal-seed final accuracy %, 512 histories per horizon per seed. All 36 original endpoints have exact point violations of the required region inclusion at disturbance 1/32; every arm 0/12 certificates. The proposed +3 seed and >=6/12 bound success gates fail. A numerical adaptive interval hull search proposes no admissible separated interval pair at epsilon 1/32 or zero. This is a failed search, not an impossibility theorem. Handcrafted LS06 control passes (248 boxes, 132 leaves).

## Post-hoc repair

Every endpoint receives deterministic SLSQP minimax optimization of four tensor scales and the affine offset, with fixed integer codes and down bias. Scalar skip bias may change. 33 sampled points/case, 120 iterations maximum, 35/36 optimizer success (sample seed137 status8 retained). This is a proposal only; exact rational verification tests decoded FP32 dyadic weights. No evaluation-stream optimization.

| Arm | repaired Q4 T4 | T8 | T16 | T32 | T64 |
|---|---:|---:|---:|---:|---:|
| qat |55.29|53.99|52.49|51.35|50.39|
| sample |67.33|63.56|60.07|56.27|54.35|
| bound |66.88|65.66|65.72|64.13|63.02|

All 36 still fail the original epsilon=1/32 inclusion. Post-hoc weaker epsilon grid yields 0/36 at 1/64, 0/36 at 1/128, 1/36 at 1/256, and 1/36 at zero. Bound seed107's sampled maximum transition error improves from .182199899 to .117563365 and all its sampled prefixes through T64 are correct before and after repair.

## Exact theorem for the selected compact checkpoint

For fixed decoded dyadic weights in [compact_weights.json](compact_weights.json), exact GELU and beta=1/5, define R0=[-1/8,1/8], R1=[7/8,9/8]. For every p,a,b,n in {0,1}, the transition from any h in R_p, with any additive scalar disturbance |e|<=1/256, belongs to R_(p XOR a XOR b). Initial h=0; readout h-1/2 has signed margin >=3/8. Induction proves correctness for every finite input history. The verifier establishes all sixteen cases with 270 rational boxes and 143 leaves. The repaired four-neuron model's fourth down-projection weight is exactly zero; physical extraction reduces it to 27 slots, including three fixed, and three active GELU neurons. The extracted model passes a standalone exact verifier:

    python certificate_only.py

Only Python stdlib needed; see LS06 for the mathematical CDF enclosure using Machin/Taylor/Mills bounds. This is an exact-real interval proof, not a Lean kernel or floating-point rounding certificate. Q4 denotes signed integer codes with FP32 scales, not end-to-end four-bit arithmetic. Numerical stress with 2048 histories of 512 visits and random-sign perturbations of magnitude 1/256 yields 0/1,048,576 wrong decisions; max region-center error .1123440266. This numerical result is separate from the proof.

## Reproduction

    for arm in qat sample bound; do python train.py --arm "$arm"; done
    python verify.py
    python repair.py
    python reduced_margin.py
    python audit.py
    python certificate_only.py

Torch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0. Audit passed 540 numeric horizon replays, 214 exact point violations, three certificate replays, code/scale decoding and sampler receipts, plus compact certificate and numerical stress. The audit uses stored training/repair weights; no independent retraining or optimization. Companion archive SHA-256 de297d39f21b88ce6623179719729fb8f702418208c4d09089459b407f2f885b includes all 36 raw endpoints and repaired outcomes, proof receipts, logs, hashes, and full report.

12 independent seed groups; multiple histories/visits correlated within each checkpoint. Matched updates and source streams, not exact total compute. Additional region labels and optimizer repair supplied. No native or sealed NEC04/Sudoku data. FSI-01 REDESIGN_OR_PILOT_ONLY: no calibrated matched physical sham or prospective confirmation. Primary failure preserved. Exact finite interval verification + induction for one selected repaired model; no statistical inference of superiority. Next: prospectively optimize worst-case region excess and explicit quantizer scales; fresh seeds, frozen disturbance radius, and matched computation.
