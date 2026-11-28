# LOGRANK-BPD-01: bit-budgeted reasoning interfaces

Status: exact finite fixture replayed; native learned experiment proposed, not trained or registered for confirmation.

Source: [Cogentic, Breaking the Square-Root Barrier for the Log-Rank Conjecture, pinned README](https://github.com/google-research/google-research/blob/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/logrank_conjecture/README.md). The full paper and announced Lean formalization were not audited here. The asymptotic bound is motivation, not a numerical prediction at gym-scale ranks.

## Replay

Run `node experiments/logrank_bpd_01/exact_fixtures.mjs`. The committed receipt was produced by executing the same JS functions in V8. No external dependencies. The protocol uses Bob-only local output; every communicated binary branch costs one bit. Local output is free. With a public-output convention, append a final output bit if needed. No stochastic messages or hidden side channels.

The exact DP minimizes worst-case cost over all binary splits of either party's remaining input set. Leaf condition: all remaining rows agree on every remaining column. This is valid because Bob knows his column. For nonempty row/column sets A,B, its recurrence is
`d(A,B)=0` at a valid leaf; otherwise take the minimum over all nontrivial row and column bipartitions of one plus the larger child cost. Each child strictly reduces the rectangle. Exhaustive replay checks every input pair and worst-case depth. Runtime exponential; DP capped at 8x8. Matrix rank is exact over Q (equal to real rank for integer matrices), using fraction-free BigInt elimination with GCD reductions. Rank zero is treated separately with bound zero. One-way optimum is ceil(log2(number of distinct rows)). Every Bob-output protocol leaf has rank at most one when extended by zero outside its rectangle, so the sum of leaf matrices gives rank(M) <= 2^d. Rectangle density is computed separately with monochromatic leaf semantics.

| Fixture | Rank | One-way | Exact interactive | Largest mono rectangle |
|---|---:|---:|---:|---:|
| Equality4 | 4 | 2 | 2 | 1/4 |
| Greater-than4 | 3 | 2 | 2 | 3/8 |
| Inner-product2 | 3 | 2 | 2 | 1/4 |
| Index3 | 3 | 3 | 3 | 1/6 |

Index4: rank 4, 16 distinct rows, one-way 4 bits; the explicit two-bit query plus one-bit answer protocol replays all 64 cases with zero errors. Its cost is a certified upper bound, not a DP optimum. 512 out of 512 3x3 Boolean matrices passed rank lower bound, one-way upper bound and exact tree replay.

## Feature positive control

x,y in {-1,+1}^4, M=(1+x0*x1*y1*y3)/2. Train on anchor-column/anchor-row labels with nuisance-coordinate holdout on each side; evaluate the cross product of disjoint held-out rows and columns. Both agents independently choose one signed local feature. First-order feature selection 32/64, a fixed wrong quadratic pair 32/64, degree-two feature selection 64/64. All send one binary symbol from Alice to Bob. The feature selector is an exact enumerative learner, not a trained transformer or a BPD implementation; pair search has six candidates versus four for first order and the fixed wrong pair is a sanity control, not a statistically matched sham. This one hand-built fixture cannot establish transfer or mechanistic mediation.

## Native experiment design (development proposal)

For a state/action validity matrix, enforce a real split: Alice receives s alone; Bob receives a alone. Independent local recurrent states h^A_t,h^B_t may consume only local input, public fixed metadata and the recorded bit transcript. A designated public speaking schedule (or a charged explicit speaker/STOP code) controls who sends each bit. No float activations, shared state, dynamic tensor shapes, timing, uncharged routing indices, or verifier access during execution. The verifier provides training labels and evaluation only. Use a fixed B-bit packet padded and charged in full first; add variable-cost prefix transcripts only after an independent decoder/replay audit. Compare Bob's output to exact labels; do not confuse accuracy with exact deterministic correctness.

Suggested BPD-style factorization at each sender:
z_t=P h_t, c_t=U z_t, q_t=V[(z_t)_i(z_t)_j]_{i<j}, m_t=1{w^T(c_t+q_t)+b>=0}.
This is proposed notation, not an existing BPD API. Use selected/sparse pairs to avoid quadratic memory cost. Include linear-only, parameter- and FLOP-matched linear, quadratic, fixed random pair, trainable shuffled-pair and exact classical baselines. For KL arm use a per-root development-calibrated penalty on conditional message probabilities relative to a specified reference; it is not the theorem's KL stability criterion. Report all weights, activations and actual transmitted bits. Hard binary forward path; straight-through or relaxation allowed for training only. Log finite-precision replay and enforce no extra input-dependent messages. Compare fixed one-way and alternating recurrent schedules at each equal bit cap. Measure exact all-pair success per matrix, mean case accuracy as secondary, worst/mean bits, compute/RAM, and recovery/corruption versus baseline for causal perturbations. Parameter-count parity does not imply equal search freedom.

Candidate development family: fresh rank-two parity matrices with random local feature pairs and nuisance coordinates, indexing k=2..8, modular predicates, exact verifier matrices exported from Lean/Sudoku with leakage audits. Hold out entire generating roots, not random matrix cells alone; no labels or fitted factorization from test roots at inference unless explicitly charged as an adaptation experiment. Public family-specific protocol synthesis is a distinct comparison track. Do not silently count rank approximations as exact: use exact rank on small integer matrices, modular lower bounds or certified upper bounds separately on larger data. For actual BPD use native APIs and compare selected-pair intervention to dose-matched sham, match radius and displacement, and use correct recovery/corruption accounting. Follow LoopedTransformers AGENTS.md and EvalMath FSI-01 before any causal or large training claim. No protected cohorts touched.

Tentative development target: exact success on at least one fresh structured family at lower bit cap than a matched one-way learned protocol; do not change classical lower bounds. Preregister endpoint, unit (independent matrix generator/checkpoint root), minimum effect, sham variation, power and multiplicity before confirmation. No GPU training or native integration performed.

## Repo interfaces

* BPD: causal pair-interaction ablation; distinguish geometry from communicated symbols.
* Compactification-Theory: Pareto frontier (parameters, local precision, depth, communicated bits, compute, exact success), separate local memory from communication.
* HybridTRMLDT: bit-only message ABI and deterministic transcript replay between modules.
* DataGoblin: schema with root ID, generator/version, split, private inputs, public metadata, verifier labels and certificate, exact-rank status; do not leak test matrices in training.
* MathCan: formalize rank-one leaf sum, DP recurrence and one-way row-count result before tackling Cogentic's stability theorem.

This single folder is the canonical fixture and receipt; downstream adapters should import or pin it rather than duplicate evidence.
