# LS-NEURON-05: recurrent dynamics and quantization-aware training

**Development only; FSI-01 REDESIGN_OR_PILOT_ONLY.** Stacked on [LS04 #369](https://github.com/MoralityLabAI/LoopedTransformers/pull/369). Verified source archive SHA256 7f5641001c6a99e213a8be34c014e5d2d0e72484d9af082d5301164d4de4985b and all 21 extended parent checkpoints. [Protocol](protocol.md) and source hash lock were frozen locally before the run. No native NEC04/Sudoku or sealed confirmation populations accessed; no paid compute.

## QAT

Each of the three extended LS04 curriculum checkpoints gets 1200 AdamW updates (LR .0015, batch128, clip1, no decay) on identical per-seed fresh T8 source batches. FP32 continuation, direct Q4 STE, staged Q6 for 400 updates then Q4 for 800. Symmetric per-tensor max-abs scaling and rounding on every parameter, including biases; 15 representable levels in Q4, three in Q2, floating-point activations. STE identity backward derivative; no packed kernels, activation quantization, memory or latency measurement. Fresh deterministic sampled development histories (2048 per seed and horizon), same nuisance-prefix families as LS04, no untouched confirmation.

Mean final accuracy (%) across seeds 11/23/37:

| Arm | Evaluation weights | T4 | T8 | T12 | T16 | T24 |
|---|---|---:|---:|---:|---:|---:|
| FP continuation | FP32 |98.73|95.10|90.28|83.14|62.16|
| FP continuation | Q8 |99.72|95.98|90.79|83.71|62.00|
| FP continuation | Q4 |77.38|62.92|54.07|50.33|50.05|
| FP continuation | Q2 |45.20|49.46|50.23|50.28|51.03|
| Direct QAT | FP32 |79.88|71.26|67.33|63.22|56.04|
| Direct QAT | Q8 |79.57|70.88|67.50|63.74|56.79|
| Direct QAT | Q4 |90.98|84.13|73.26|67.82|61.49|
| Direct QAT | Q2 |48.08|49.84|49.77|50.10|50.47|
| Staged QAT | FP32 |75.34|68.51|68.60|66.89|64.86|
| Staged QAT | Q8 |75.52|68.46|68.44|66.85|64.68|
| Staged QAT | Q4 |89.81|81.80|77.39|72.15|66.72|
| Staged QAT | Q2 |49.22|49.69|49.95|49.82|50.55|

| Seed | Initial parent Q4 T12 | FP continuation Q4 T12 | Direct QAT Q4 T12 | Staged QAT Q4 T12 | Staged QAT Q4 T24 |
|---|---:|---:|---:|---:|---:|
|11|48.05|48.63|60.16|65.72|53.96|
|23|51.12|52.88|59.72|66.46|58.11|
|37|48.93|60.69|99.90|100.00|88.09|

Primary direct-QAT gain +19.19pp exceeds the 10pp floor, but absolute Q4 T12 73.26% <80%, FP32 T12 loss 22.95pp >5pp, and FP32 T4 79.88% <95%. **Primary gate fails.** Staged QAT is secondary and also misses the 80% mean competency target; its best seed is a descriptive selected case. Additional matched optimizer steps and source batches, not hardware cost matching.

## Dynamics

For each parent and final student in FP32 and Q4, sample 512 calibration and 512 development T16 histories, 128 per each of four disjoint nuisance-prefix groups. At T8 fit two Euclidean parity centroids from calibration data. Replace the entire development hidden state with same/opposite parity centroid, compare with clean and norm-matched random direction (original state plus random unit vector times opposite-centroid displacement norm). Preserve original eight-input suffix. Opposite intervention scored against inverted parity target; random direction scored both ways. Four independently generated directions per history used at each perturbation dose. Report final and entire-suffix accuracy.

Parent FP32 results (equal-seed means in percent; gain is median across seed-wise median eight-step product norms):

| Parent arm | Clean T16 | Same centroid | Opposite centroid flipped | Random flipped | Centroid parity decode T8 | Product gain |
|---|---:|---:|---:|---:|---:|---:|
| fixed4 |53.19|56.51|52.67|47.46|61.46|31.341|
| curriculum |83.33|75.85|75.52|40.76|90.56|6.090|
| long8 |79.23|58.59|58.59|38.15|89.45|1.952|
| kd_long8 |63.22|48.50|49.61|48.05|71.48|2.509|
| prog_salience |64.13|55.73|53.84|45.83|89.84|12.968|
| prog_random |74.35|40.23|40.76|43.62|78.32|7.371|
| budget_curriculum |84.24|74.22|73.44|42.71|90.04|5.770|

All seven parent Q4 clean T16 means lie between 48.63% and 52.93%. Parent centroid intervention can fail even where clean behavior is competent (e.g. seed11 curriculum: clean 74.80%, same 33.59%, opposite flipped 32.23%). Full-state contraction is neither necessary nor sufficient for these task observations.

Final staged QAT seed37, Q4: T8 nearest-centroid accuracy 100%, centroid gap 2.55478, within-class RMS 0.91680, gap ratio 2.78664. Clean T16 96.29%, clean all eight suffix visits 95.70%; same centroid 98.44% final and all eight; opposite centroid 98.44% inverted final and all eight; equal-norm random 33.59% inverted final and 10.35% inverted all eight. State replacement is a causal intervention on this model, not proof of two invariant regions. Centroids can be off-manifold; equal-norm random is not a calibrated inert sham. Mean Q4 T16 clean/same/opposite/random-flipped across three final staged seeds = 73.50/65.04/63.22/39.91%; direct QAT =66.02/66.80/68.03/44.21%; FP continuation Q4 =50.46/52.93/48.44/48.57%.

For final staged QAT seed37, conditional clean-correct T16 survival at perturbation norms 0/.05/.1/.25/.5/1 times centroid distance = 100/99.44/99.09/97.82/90.47/69.57%. Unconditional accuracy at the same doses =96.29/96.00/96.04/95.65/88.92/68.75%. Four random directions per history, no worst-case guarantee.

## Mathematical interpretation

Write the actual 16D width-four recurrence as h'=h+A h+B x+c+0.2 V GELU(U h+W x+d), absorbing the down-projection bias into c. The analytic Jacobian is

    J = I + A + 0.2 V diag(GELU'(U h + W x + d)) U.

The state-dependent correction has rank at most four, a deterministic algebraic fact for this architecture. For staged Q4 seed37, median local spectral radius 1.01253, median local largest singular value 1.37797, median eight-step product operator norm 3.46066; product p90 3.84300 and max 17.77666. Parent seed37 curriculum Q4 product median 13.92758; FP continuation Q4 12.09046. No global contraction. A future sufficient certificate would find disjoint parity regions R0,R1 with correct readout margin, such that for every permitted input x and perturbation e, F_x(R_p)+e is contained in R_(p XOR parity(x)). The empirical centroids motivate this test but do not establish inclusion. Analyze task-relevant readout and transverse directions separately; full-state norm alone is insufficient.

## Evidence and reproduction

Independent checkpoint-seed groups n=3. Within-seed contexts, repeated histories, interventions, horizons and directions correlated. 21 parent and nine student checkpoints; 60 FP/Q4 dynamics conditions; 360 initial/final precision-horizon evaluations (initial measurements repeated). Four autograd comparisons per dynamics condition. No calibrated physical-sham floor or prospective power estimate. No inference of equivalence, significance or generalization to native models. Readout exposure remains development only. All 9 arms and all 21 parents retained.

Archive includes verified parents and manifest, trained checkpoints, raw per-case outcomes, source lock, run/audit logs and aggregate JSON. Reproduction in that directory with Torch 2.10.0+cpu:

    sha256sum -c lock.sha256
    for seed in 11 23 37; do
      for arm in fp qat4 qat6to4; do
        python experiment.py --task qat --seed "$seed" --arm "$arm" --out qat
      done
      for arm in fixed4 curriculum long8 kd_long8 prog_salience prog_random budget_curriculum; do
        python experiment.py --task diag --seed "$seed" --arm "$arm" --out diag
      done
    done
    python audit.py

Audit passed all 30 checkpoint hashes, 60 dynamics replays, 360 precision-horizon replays, 442368 scalar XOR prefix labels, matched sampler hashes, maximum analytic/autograd Jacobian error 2.38418579e-7. Dynamics replay uses original code, independent checks are scalar parity and autograd; no independent retraining. Source Git blob IDs match locally executed bytes for all five committed source/protocol files. Raw archive SHA256 96720175bd4bb8680eab5adaebada3a5bc5310f9925d377b905f11c637f8c1f3.

Next: prospective fresh seeds and task families; explicit region-invariance/interval bounds for the selected four-neuron checkpoint, then randomized robustness curricula or task-sensitive Jacobian regularization with matched budgets and native MLP-necessary model qualification. Do not claim a general compactification law from this toy task.
