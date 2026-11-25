# LS-NEURON-12: modular automaton width bounds and fixed-codebook recovery

**15% primary development gate FAILED.** Secondary 5% scale-only result: eight new exact certificates versus one under unrestricted QAT. Constructive results and empirical recovery are separate. Parent [LS11 #382](https://github.com/MoralityLabAI/LoopedTransformers/pull/382). No paid GPU or native/protected data.

## Scalar reset/wrap theorem

For integer modulus m>=2, centers c_p=p-m/2, region radius rho=1/8, define R_K(z)=[GELU(K(z+1/4))-GELU(K(z-1/4))]/(K/2). The recurrence is

    F(h,a,b)=-m/2+a+b+sum_{j=1}^{m-1} R_K(h+m/2-j+1/2)
             -m R_K(h+a+b+m/2-m+1/2).

The ReLU counterpart reconstructs p from h, adds the binary inputs a+b via the affine branch, and subtracts m precisely on wraparound. The nuisance bit is ignored. Each ramp is constant on every prescribed state/input region. The skip cancels the outer residual. Width=2m, state dimension=1, readout margin 3/8. K>=32 gives |preactivation|>=4 everywhere. Mills' bound implies each GELU-minus-ReLU error lies in [-1/1024,0], so the paired difference is bounded by 1/1024. Output error <=(2m-1)/(512 K); a disturbance of size <=rho minus this error preserves the invariant region indefinitely. Choose sufficiently large dyadic K for each m. Mod4 at K=32: width 8, 54 dense recurrence slots versus 66 in LS11, certified disturbance 2041/16384. Tensor scales (1,K,K/4,10/K) give maximum integer-code magnitude 2m-1; signed symmetric b>=1+ceil(log2(2m)) bits suffice. Two offsets are exact floating values. The precision bound is for integer codes, not scales, offsets or arithmetic.

## Distributed shared-carry theorem

For m=2^k use k coordinates with centers h_i=L(q_i-1), q_i the bits of p. Readout threshold -L/2. Define S(z)=[GELU(K(z+L/8))-GELU(K(z-L/8))]/(K L/4). Reset bits B_i=S(h_i+L/2). Carry C_j=S(h_0+2 sum_{i=1}^j h_i+L(a+b)-L/2), j=0..k-1. Let C_-1=a+b. Output coordinate i is -L+L(B_i+C_(i-1)-2 C_i). The ReLU counterpart gives the carry from the lower binary word: a missing higher lower-order bit prevents overflow with increment<=2; otherwise the carry is determined by q_0+a+b>=2. Each carry ramp pair feeds two adjacent output bits, saving duplication. The exact ReLU plateaus hold if (2k-1)rho<=3L/8. Width=4k, state dimension=k, dense recurrence slots=9k^2+21k. Per-coordinate GELU error <=32/(5K), using the uniform Mills bound phi(0)<2/5 and total absolute ramp coefficient <=4L. K sufficiently large provides a positive disturbance radius rho-32/(5K). The checker tightens this bound with the 1/1024 tail where possible. For L=1,2,4, four-bit signed codes suffice at scales (1,K,KL/8,20/K). Absolute rho=1/8 permits k<=2 at L=1 and k<=6 at L=4. The larger state separation changes the geometry; this is not a uniform fixed-relative-noise theorem or a general minimum-width result.

| States | Scalar width | Scalar bits | Binary width | Binary bits | Binary separation |
|---:|---:|---:|---:|---:|---:|
|2|4|3|4|4|1|
|4|8|4|8|4|1|
|8|16|5|12|4|4|
|16|32|6|16|4|4|
|32|64|7|20|4|4|
|64|128|8|24|4|4|

For 64 states: 774 vs 450 dense recurrence slots. Q4 binary construction disturbance 4095/32768, state separation 4. Independent matrix checks cover 2088 state/input cases across 15 constructed cells, including seven scalar and eight binary variants. Stress: 512 sampled histories x256 visits per cell, coordinate random-sign disturbance 1/32 each visit, zero errors in 1966080 transitions. Stress is numerical, not a hardware proof.

## Restricted lower bound

A univariate ridge ReLU model of s=h+a+b with m+2 consecutive sum intervals, exactly constant on every interval and unequal adjacent plateaus, needs at least two slope changes in each of the m+1 disjoint gaps. Hence W>=2(m+1), because each ReLU contributes at most one breakpoint. The LS11 shared-sum construction reaches this bound for m=4 in that restricted ReLU class. No such lower bound is asserted for general GELU networks, approximate plateaus or separate input directions. The reset/wrap construction uses separate directions to reach 2m.

## Scale-only recovery

See frozen protocol. Seeds 419..430, 5%/15% tensor-RMS Gaussian perturbations, 800 Adam updates, same paired starts and 512 regional constraints. Frozen control, six-variable scale-only (four scales plus two offsets), 58-variable full QAT. Both learned arms use identical update schedules but different trainable counts and computational budgets. Explicit initial scales and cap 128 replace the old cap 10. Fixed target disturbance 1/32. T8/T128 numerical accuracy over 256 sampled histories each. All nonfinite states count incorrect.

| Dose | Arm | Init cert | Final cert | New | Lost | T8% | T128% | T128 all-prefix% |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|5%|frozen|3|3|0|0|93.75|94.08|91.67|
|5%|scale|3|11|8|0|98.05|93.52|91.76|
|5%|full|3|2|1|2|53.61|49.19|33.33|
|15%|frozen|0|0|0|0|24.71|20.90|0.00|
|15%|scale|0|0|0|0|26.63|25.72|0.00|
|15%|full|0|0|0|0|34.11|25.07|0.20|

Initial certified seeds 420/421/422 at 5%; scale final all except 426; full final 420/429. Scale-only retains all 3 initial certificates and adds 8; full loses 2 and adds 1. The original 15% primary threshold fails. Secondary 5% contrast +7 new certificates, paired seed n=12, development-only. Point witnesses are rigorous interval violations of prescribed regions, not necessarily reachable. FSI-01 REDESIGN_OR_PILOT_ONLY. No powered population claim, no unaided circuit discovery, no native transformer claim.

## Reproduction

PyTorch 2.10.0+cpu, deterministic one-thread workers. From this directory:

    sha256sum -c lock.sha256
    python construction.py
    for dose in 5 15; do
      for arm in frozen scale full; do python experiment.py --dose "$dose" --arm "$arm"; done
    done
    sha256sum -c audit_lock.sha256
    python audit.py

Audit: 2088 actual-matrix transition checks; 120 endpoint receipts, 240 numerical replays, 22 certified endpoints and 98 exact point violations (including repeated initial states). Positive control passes. Code/scale decoding and frozen tensors checked. Stored decoded dyadics, exact real GELU, beta=1/5, explicit scales/offsets. No Lean kernel or floating-point rounding proof; audit does not independently retrain. Numerical stress and exact certificates remain separate evidence.

Raw receipt SHA256:
- construction_results.json: 30c85c511decaba81de4c3b9d42e0f5079abb57bd562e1d7bb47344fcb40383f
- certificates.json: ac92897879d3ae32936d8bf597c4321281fadfafb92ed7ea4122b6d35deb2030
- audit_results.json: 3206439eeae5e716ea16a906e25b6d4a647efa247f1e1c6f69ceb0215e0b49d8

Companion archive LS-NEURON-12-automaton-complexity.zip (SHA256 5f4573f166ebc08c874c87bb00828d803aede866901f5ca55ed26fc0ee3b06b2) contains all raw weights, proof ledgers, exact executed source, full report, logs and hashes. Next: lower-width unrestricted GELU constructions, fresh paired comparison of code-preserving scales against discrete code repair, followed by native headroom-qualified checkpoints.
