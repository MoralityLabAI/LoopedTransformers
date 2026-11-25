# LS-NEURON-16: breadth versus depth under a fixed fitting budget

## Question and scope

Does retaining three distinct integer codebooks improve the rate of exact repair relative to spending the same expansion budget on a single greedy path? The sole prospective contrast is `beam` minus `greedy`. LS15 obtained 11/24 new certificates in both its conditional minimax arms, with identical successful seeds. Its final report and stagnation diagnostics motivated this allocation experiment.

This study is a bounded CPU pilot. The unit is an independently seeded perturbation of the fixed eight-GELU mod4 template. The 32 known transition cases guide fitting and selection throughout. They are observations within each seed. This experiment does not sample new tasks or architectures.

Parent: `MoralityLabAI/LoopedTransformers` commit `c198af6b3302849e91eec5d5462a626559fbc780`, draft PR #388. The source archive SHA-256 is `22ecf568dd145d2b9fc1d953c2c2be08d4ce60d17ba2659c8ce3a779b543b5dd`. Parent search, fitting, decoding, verifier and construction sources are copied without modification. LS16 replaces search allocation and adds receipts with LP dual vectors.

## Cohort and decision rules

The assigned cohort is 24 fresh seeds, 493 through 516 inclusive, at the inherited 15-percent tensor-RMS perturbation dose with RMS floor 0.125. The Torch generator uses `12010000 + seed`. All assigned seeds stay in both arm denominators. Development uses previously exposed seeds 433 through 492 only. The current runtime must reproduce archived initial states on development seeds before the new cohort is drawn.

The primary binary outcome is one exactly when an initial model has a rationally proven `POINT_VIOLATION` and the corresponding final model is `CERTIFIED`. An initially `UNRESOLVED` model has primary outcome zero; any subsequent certificate acquisition is reported separately. Final `UNRESOLVED` outcomes also score zero. Initial certificates, losses, and all verdict counts remain visible.

The operational gate requires at least 12/24 new certificates for beam and at least six more repairs than greedy. The scientific effect floor is 0.25. There is one planned paired comparison, with familywise alpha 0.05. The official FSI-01 bridge reports exact conditional McNemar inference, descriptive paired bootstrap uncertainty, marginal per-arm Clopper-Pearson intervals, and its conservative effect-floor gate. With 24 units and one comparison, the Hoeffding radius is about 0.499644; exceeding the 0.25 floor requires at least 18 net repairs. That gate is separate from the operational threshold.

No calibrated beam effect or matched-sham distribution is available before this study. The strict official preflight is `REDESIGN_OR_PILOT_ONLY`. The historical LS15 bound-LP versus max-LP contrast has zero effect across all 24 seeds. Its identity checks and universal binary-difference support bound do not constitute empirical sham calibration. The study remains `PILOT_ONLY` regardless of operational gate outcome. A null result does not establish equivalence.

## Fixed model and exact endpoint

There are four scalar state regions centered at -2, -1, 0, 1, each with radius 1/8. Inputs `a`, `b` and nuisance bit `n` are binary; the state target is `(p+a+b) mod 4`. The cell has eight GELU units, residual coefficient 1/5, and 52 integer codes in [-7,7]. Four positive scales lie in [1e-6,128]. The fitted combined offset is bounded by [-8,8]; the initial down-bias is fixed. `repair13.decoded` constructs the stored FP32 weights. The model has 54 decoded parameter slots.

For each of the 32 state/input cases, the inherited exact verifier must prove that every hidden state in its full source interval maps within 3/32 of the target center. A further additive disturbance of magnitude at most 1/32 keeps the next state in its radius-1/8 region. Induction then covers every finite input sequence from a valid initial region under the declared ideal real recurrence. The verifier treats each stored FP32 weight as an exact rational and uses rational GELU enclosures. It can return a point violation or unresolved result.

This scope concerns the real recurrence with the stored weights. Floating-point rollout results provide separate numerical observations. This work supplies no Lean theorem or hardware-kernel rounding proof.

## Shared candidate fitter

Each expanded parent produces the inherited legal paired-code proposals. The ordered list is padded with no-ops to 177 profile slots. The profiler runs twice, using the parent's scales and the original initialization scales as anchors. At each anchor the shortlist contains its no-op and the two highest-ranked distinct edited codebooks, with stable ordering under the inherited nine-point mean/hinge profile objective.

Every shortlisted candidate receives the inherited analytic-Jacobian least-squares fit, capped at 40 function evaluations. Its output then receives a conditional bound LP, capped at 1,000 HiGHS dual-simplex iterations. The LP fixes the decoded FP32 up weights and up biases while optimizing positive skip and down scales, combined offset, and a maximum-error variable. It uses 1,088 inequalities from 32 cases and a 17-point grid. Every grid endpoint receives the larger adjacent local GELU-curvature interpolation pad.

The LP's second state wins only if its decoded numerical bound is lower than the least-squares state's bound by more than 1e-12. Otherwise the first state remains the candidate. A failed LP retains the least-squares parameters and records its failure status. All candidate scores are recomputed on the decoded FP32 weights. The numerical bound guides search; terminal rational verification supplies the exact certificate.

LP receipts retain primal variables, sparse inequality dual vectors, and all lower/upper-bound duals. The independent auditor reconstructs constraints, checks feasibility, stationarity, dual signs and gaps, and compares independent decoded interval scores. These floating-point checks do not constitute an exact proof of LP optimality.

## Search allocation and deterministic ties

A common initial least-squares fit and one common expansion run once per seed. The pool contains the initial fitted state followed by the six selected candidate states. Shared selection sorts by raw numerical bound in ascending order, using a stable sort so exact ties preserve earlier insertion. It takes the first state for each full 52-code tuple until the requested width is reached. This deliberately keeps only one scale basin per codebook. Greedy starts from the pool's best state; beam starts from its best three distinct codebooks.

Both arms then run 12 comparison rounds, each with three expansions:

* Greedy expands its current best state, selects the best single state from that parent and its six children, and repeats this process twice more within the round.
* Beam freezes its three parents at the beginning of the round, expands each once, and selects the best three distinct codebooks from all old parents followed by all children in expansion order.

The same selection function governs the shared bootstrap and both arms. Exact ties favor an old parent over a newly inserted child. Every retained set includes the previous best as an option, so its best numerical bound is nonincreasing. A beam child can survive with a worse score than its own parent when it outranks another codebook. All parent IDs are preserved so those paths can be inspected. The final output is the lowest-bound retained state. Search allocation, including retention of alternate codebooks, is the treatment.

Each arm is charged 37 expansions, 13,098 profile slots, 223 least-squares calls including its common initial fit, and 222 LP calls. Reusing the shared bootstrap saves compute without changing either arm's available states. Across the full cohort, actual execution comprises 1,752 expansions, 620,208 profile slots, 10,536 least-squares calls and 10,512 LP calls. These are equal opportunity budgets per arm; solver work and wall time can differ. Duplicate candidates still consume their registered fit calls. No adaptive early stopping or certificate-directed extension is allowed.

## Exposure, audit, and reproducibility

The source, protocol, audit, development validation receipts and evidence registration are hashed in `lock.sha256`. That lock is committed on a branch based on LS15 before any fresh initialization is generated. `freeze_receipt.json` records the resulting commit and lock digest. A run manifest records initialization and search timestamps. All 48 searches must finish before exact terminal verification or rollout scoring begins.

The auditor checks each initial state against the deterministic generator, all shared and per-arm budget totals, profile arrays and shortlists, decoded first and second candidate states, LP checks, the entire selection sequence, and endpoint hashes. Exact verification covers a known positive construction, 24 initial models, and all 48 final models. Rollouts evaluate each initial and final model at lengths 8 and 128 using the inherited evaluation generator, giving 144 horizon conditions in total. These are descriptive observations within the same seed.

Frozen descriptive diagnostics include the bootstrap numerical bound, final bound distribution, retained and expanded codebook counts, repeated expansions, same-weight endpoint counts, final path depth, worse-parent edges on retained paths, last best-score improvement, per-arm solver evaluations, and elapsed time. None adds a hypothesis test. LP or audit failures, unresolved certificates and any source deviations must be reported. Post-lock repairs require preserving the original lock and documenting whether fresh outcomes had been exposed; changing a treatment after exposure prevents a prospective claim for that version.

The archive includes complete initial states, shared bootstrap records, every shortlisted fit and profile, final certificates, evaluation receipts, source hashes and official evidence outputs. A draft PR carries the report and reproducible sources. The protected ARG-03-N cohort stays sealed. This run uses at most four CPU workers, one thread per numerical worker, and no paid compute.
