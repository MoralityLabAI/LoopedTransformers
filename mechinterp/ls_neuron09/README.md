# LS-NEURON-09: mean-to-maximum objective homotopy

**Primary development gate FAILED.** Fresh 12 paired seed groups, 48 endpoints, zero learned certificates at the fixed disturbance 1/32. Stacked on [LS08 #377](https://github.com/MoralityLabAI/LoopedTransformers/pull/377). Parent source reused, no parent model weights, paid compute or protected native cohorts.

## Design

See the [locally frozen protocol](protocol.md) and `lock.sha256`. Scalar recurrent state and four GELUs, 33 parameter slots (30 trained, fixed initial state/readout); four independently learned quantization scales during QAT. All four arms have the same initialization, T8 batch64 source streams, 2400 Adam updates, precision/learning-rate schedules and 256 Taylor-bound region constraints. The squared excess above 3/32 is averaged, maximized, or annealed (weight 10, plus prefix state MSE). Delayed hard and annealed use the mean objective for steps 0–899. Annealing uses normalized log-sum-exp with temperature .25 to .002 geometrically over 1200 updates, constant thereafter. Both scalar offsets are FP32; Q4 weights are signed integer codes times FP32 scales. No actual low-bit kernel. Numerical development histories use a fresh fixed seed domain (512 histories per seed/horizon). Exact verifier uses decoded dyadic tensors, real GELU and exact beta=1/5.

## Numerical and exact endpoints

Equal-seed final accuracy (%). Q4:

| Arm | T4 | T8 | T16 | T32 | T64 | T128 | T128 all-prefix | Certified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean |61.62|56.12|53.66|51.30|51.04|50.23|1.14|0/12|
| hard |54.46|51.01|50.36|50.21|50.05|49.84|0.00|0/12|
| delayed_hard |67.33|60.53|56.98|54.56|51.61|50.47|2.34|0/12|
| annealed |66.94|60.84|57.31|56.85|54.28|53.82|8.35|0/12|

FP32 latent weights (not deployed Q4):

| Arm | T4 | T8 | T16 | T32 | T64 | T128 |
|---|---:|---:|---:|---:|---:|---:|
| mean |55.65|54.31|53.16|49.89|49.37|50.29|
| hard |51.86|50.52|50.37|50.47|50.11|51.77|
| delayed_hard |55.22|54.05|53.40|50.13|49.74|50.10|
| annealed |57.65|54.39|53.12|50.03|49.93|49.37|

Fixed-region grid maximum error (257 points per case; grid is a lower-bound diagnostic, not a continuous enclosure):

| Arm | Mean | Median | Best | Worst |
|---|---:|---:|---:|---:|
| mean |.673615|.703873|.326418|.944688|
| hard |.554753|.573261|.356462|.630926|
| delayed_hard |.551257|.553806|.311639|.824017|
| annealed |.536618|.578329|.244290|.814241|

The required transition error is <=3/32=.09375. All 48 have exact point violations. The LS06 handcrafted positive control passes 248 rational boxes/132 leaves. The frozen primary gate (annealed >=6/12 and >=3 additional certificates vs mean) fails with 0/12 vs 0/12. No relaxed radius, region search or repair. Annealed T128 vs mean +3.597pp mean and +.0977pp median, 6 wins, 5 losses, 1 tie; vs delayed hard +3.353pp mean, 0 median, 4 wins, 4 losses, 4 ties. No inferential claim.

Annealed seed263 achieves 93.9453% T128 final and 87.5% all-prefix correctness. Its exact point violation is at p=1, h=9/8, input (0,0,1), target 1; output lies in [114199608612621139791629/151115727451828646838272,456798434450484559166523/604462909807314587353088] (~.7557096176440318), below 29/32. The witness is in the proposed region; reachability from zero is unestablished. Hence it disproves the specified inclusion, not arbitrary-depth correctness by itself.

## Gradient diagnostics

Medians across seed groups; diagnostics recorded before the named optimizer update, not independently reconstructed from intermediate weights. Mean and max gradient vectors include active learned scale variables where present, and refer to the auxiliary losses alone. Task-vs-aux cosine is also recorded. The first three diagnostic records are exactly identical across mean, delayed hard and annealed for every seed, as expected before their objective diverges.

| Step | Annealed effective constraint count | Annealed max weight | Annealed mean/max gradient cosine | Annealed task/max cosine |
|---:|---:|---:|---:|---:|
|0|256|.0039|.9013|.8186|
|599|256|.0039|.6855|.2063|
|899|256|.0039|.7920|.7310|
|1199|76.4816|.0262|.9101|.8653|
|1799|13.7432|.1288|.9331|.8597|
|2399|4.2006|.3413|.8401|.8315|

Mean objective always weights 256 constraints uniformly; the median hard-max effective count is 1 throughout, including after the delayed switch. Effective count is inverse Simpson of the reduction weights, not the count of nonzero parameter gradients. Some seed/step gradient cosines are negative; at the last recorded step none are negative in annealed or delayed-hard. The data do not establish gradient conflict as the cause of poor training; local alignment can coexist with a poor transition representation. Temperature concentrates gradient weighting as intended, yet does not produce a certificate.

For normalized log-sum-exp L_tau and nonnegative squared excesses ell_i, L_tau <= max ell_i <= L_tau + tau log N. With N=256 and tau=.002 the additive gap is ~.01109 in squared-excess units. Reducing temperature alone gives no certificate. Certification needs an actual bound on the transition over entire input regions. The robust target is unchanged.

## Reproduction and evidence

    sha256sum -c lock.sha256
    for arm in mean hard delayed_hard annealed; do python train09.py --arm "$arm"; done
    python certify09.py
    python audit09.py

PyTorch 2.10.0+cpu, deterministic single-thread workers (four concurrent). Audit passed all 48 weight receipts, 576 endpoint evaluations, 1,548,288 scalar XOR labels, 48 rational point violations, one positive-control certificate, code/scale and sampler identity checks, and endpoint float Taylor-bound replay (max discrepancy 0). Intermediate gradients were logged but not independently replayed; no independent retraining. Raw per-model weights and correctness vectors, diagnostics and exact witnesses in companion archive. `certificates.json` SHA256 da92a51941aa4861a017488a819a71c3131052ea8d4af105e580e0fb404e8b22; `audit_results.json` SHA256 7630d110f05a5e728900dde0bf54558510aa2d0b2608fc4ffe89b57700024adf.

FSI-01 REDESIGN_OR_PILOT_ONLY. Twelve independent paired seed groups, repeated histories, constraints and gradient snapshots correlated. No prospective power or calibrated physical sham; matched update budgets and batches, not total FLOPs; no native transformer or floating-point kernel proof. Preserved all arms and seeds. The prior LS07 trained-and-repaired certificate remains separate.

## Next direction

Loss reweighting is now a weak candidate for the main bottleneck. Test structural initialization or an explicit two-state transition basis (with matched random controls) and train away from that basis under a fixed Q4 certificate margin. Include exact LS06 and LS07 witnesses as verifier controls, distinguish preservation of an already-certified construction from discovery, and use fresh seeds. A separate synthetic task family beyond parity is essential before attempting any general architecture claim. No change to the sealed cohorts.
