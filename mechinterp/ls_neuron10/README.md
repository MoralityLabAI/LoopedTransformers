# LS-NEURON-10: structural initialization and certified recovery

**Primary development gate FAILED.** Secondary finding: nine newly certified scalar four-GELU cells at the original disturbance 1/32, starting from 5% tensor-RMS noise around the LS06 constructed circuit. Stacked on [LS09 #379](https://github.com/MoralityLabAI/LoopedTransformers/pull/379). Parent archive SHA256 2caeae349d8e6bc03b7d2934f59c2651f95dffb4974c892dcbffa48943862ec2 verified. No paid compute or protected native data.

## Design

See [frozen protocol](protocol.md) and lock.sha256. Nine arms, twelve paired fresh seed groups, 108 training runs, 800 Adam updates per model. The certified LS06 scalar cell is perturbed by iid Gaussian noise scaled by each tensor's RMS (floor 1/8); sigma=.01,.05,.15,.30. Disrupted controls permute only outgoing MLP weights [2,3,0,1] before adding the same noise. Random arm uses Gaussian at template RMS with skip coefficient biased by -1. Every arm uses the same normalized parameterization, direct Q4 with learned per-tensor scales and two FP32 offsets, fixed initial state and readout. The region-only objective combines mean squared transition error and ten times mean squared positive Taylor-enclosure excess above 3/32. No rollout loss or random training batches: all 256 deterministic region constraints are used at each update. This differs from LS09. The controls match tensor scales, noise and updates, not initial loss or distance from the correct computational function. No adaptive radius, repair, or choice of endpoint.

## Outcomes

Exact certificates at epsilon=1/32, fixed regions [-1/8,1/8] and [7/8,9/8]. Both step 0 and 800 use the deployed decoded Q4 dyadic weights and exact-real GELU with beta=1/5. Newly certified means an exact point violation at step 0 followed by an exact certificate at step 800.

| Arm | Initial certificates | Final certificates | Newly certified | T8 final % | T128 final % | T128 all-prefix % |
|---|---:|---:|---:|---:|---:|---:|
| struct_01 |12/12|12/12|0|100.00|100.00|100.00|
| struct_05 |0/12|9/12|9|98.05|92.19|83.76|
| struct_15 |0/12|0/12|0|74.45|51.33|2.18|
| struct_30 |0/12|0/12|0|61.72|50.26|0.26|
| disrupt_01 |0/12|0/12|0|72.85|49.87|3.12|
| disrupt_05 |0/12|0/12|0|79.59|63.05|26.79|
| disrupt_15 |0/12|0/12|0|75.29|55.73|14.03|
| disrupt_30 |0/12|0/12|0|62.37|50.13|0.55|
| random |0/12|0/12|0|51.24|51.11|0.00|

Each numerical accuracy is an equal-seed mean over 256 sampled histories per horizon; final endpoint rollouts all finite. At initialization, five random-arm and one disrupt_30 T128 rollouts have nonfinite states, recorded with a finite flag; their threshold scores should not be interpreted as valid reasoning. Primary struct_15 vs disrupt_15: zero additional certificates and zero new certificates; fails all registered thresholds. Secondary struct_05: 9/12 newly certified, vs 0/12 disrupted. Initial/final median numerical Taylor bound .617066/.039446. Initial/final sampled T128 accuracy 56.90%/92.19%; numerical bounds are training diagnostics, not the exact proof. No unresolved exact-verifier cases. The recovered seeds are 307,313,317,331,337,347,353,359,367. Their certificates together cover 2,600 rational boxes and 1,372 accepted leaves (per-seed counts in certificates.json). Across all 33 certified endpoints, 8,584 boxes and 4,556 leaves. The LS06 positive control separately uses 248 boxes/132 leaves.

## Meaning

The result demonstrates local recoverability of a specific supplied computational circuit under the given Q4 optimizer and perturbation distribution. It establishes exact-real inductive correctness under additive state disturbance up to 1/32 for each recovered model: every input triple and every point of either parity region maps inside the corresponding target region after accounting for disturbance, and the readout has signed margin at least 3/8. It does not prove that 5% is a basin boundary, that recovery is monotone in the noise scale, or that a randomly initialized model can discover the circuit. Point violations concern the prescribed regions; the violating states are not necessarily reachable. The four-neuron template supplies explicit parity structure and the loss supplies parity-labeled states. No hardware low-bit kernel or rounding certificate, Lean kernel theorem, native TRM or general task result. Q4 denotes integer codes with FP32 scales and float scalar offsets.

## Reproduction and audit

In this directory with PyTorch 2.10.0+cpu:

    sha256sum -c lock.sha256
    for arm in struct_01 struct_05 struct_15 struct_30 disrupt_01 disrupt_05 disrupt_15 disrupt_30 random; do
      python train10.py --arm "$arm"
    done
    python verify10.py
    sha256sum -c audit_repair.sha256
    python audit10.py --audit

The original audit failed because its initial-scale equality check omitted the actual log/exp and [1e-6,10] clamp in decode(). The separate audit10.py corrects only that check; the training and certificate code, weights, and certificates are unchanged. Original failure log retained. Corrected audit passed 216 endpoint hashes, 432 numerical evaluation replays, 33 certificate replays, 183 exact point violations and the positive control; initialization, code/scale and offset identities also checked. Replays use stored weights, not independent retraining. certificates.json SHA256 46d04a4b219d434070343969307e6f865c2d188c9248fdc9415be8feeb43742b; audit_results.json SHA256 1f19ae6cd0fa27c6b6a03c7ae6f2e224aa7a5f53bae007d21a0ded33d538cd48. All source hashes retained in the original and repair locks. Companion archive contains all weights, outcomes, certificate leaves/enclosures, witness intervals, source and logs.

FSI-01 REDESIGN_OR_PILOT_ONLY. Independent statistical unit paired seed group, n=12. No prospective power, sealed confirmation or calibrated physical sham; no population inference. Nine-arm dose comparisons secondary to the failed primary gate. Next: freeze an independent confirmation at sigma=.05 and compare continuous training against discrete-codebook repair, then test transfer to a new finite-state task family with a separate exact certificate. No native protected cohorts opened.
