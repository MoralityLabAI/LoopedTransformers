# PPT Plan A: DRAM-plastic proof specialist for Lean-CGRT

**Design registration:** 2026-11-21  
**Status:** architecture and training proposal; no PPT checkpoint or online adaptation run claimed.  
**Parent project:** Lean-CGRT local HRM gym.  
**Target integration:** MoralityLabAI/DataGoblin RLT gym and ALEPH-1B training corpus.

## Research question

Can a compact, weight-tied transformer learn to improve its Lean proof-search policy *within a session* by updating a small, DRAM-resident specialist memory from kernel-verifier feedback, while a typed latent proof workspace supplies adaptive test-time reasoning?

The proposal is a **Plastic Proof Transformer (PPT)**. It separates three clocks:

1. The **inference token clock**, reading a goal, hypotheses, retrieved lemmas, and search history.
2. The **algorithm clock**, repeatedly applying the same transformer blocks to typed latent slots before choosing an action.
3. The **learning clock**, changing a CPU/DRAM-resident fast-weight head after verified outcomes and periodically consolidating a specialist adapter.

An action is a tactic ID, a syntactically constrained tactic proposal, lemma retrieval, explicit delegation, or STOP. The Lean kernel remains the source of truth. An accepted tactic that opens subgoals is distinct from a certificate closing the entire target.

## Local evidence informing the design

- In research/lean_cgrt_hrm400/LOCAL_SCALING_RESULTS.md, flat-S often matched or exceeded two-clock HRM at matched parameters/Lean calls. CGRT branching spent about 4x inference compute and did not improve close@b1. Graph incidence and sufficient training-time recurrent depth mattered; additional inference loops helped a single-clock S-loop on the shortcut-free mirror set for one seed, with mixed behavior elsewhere. **Do not presuppose an HRM or CGRT advantage.**
- The RLT-01/02 Mathlib experiments in RLT_ABLATION_MATRIX.md and results/rlt02 did not establish a reliable benefit of rectified feedback over direct/matched damping. Treat feedback schedule as a tested hyperparameter with necessary controls.
- DataGoblin PR #4 (https://github.com/MoralityLabAI/DataGoblin/pull/4) replays 588 *historical* certified c4 goals. Automation@4 closed 300, automation@8 closed 307, the archived menu had witnesses for 20 of the 281 automation@8 misses, and its 327/588 menu ceiling is retrospective. The archive is EVALUATION-ONLY, training_eligible=false; it must never enter PPT meta-train or adapter selection.
- DataGoblin PR #3 provides an independent OxiBonsai/Lean tactic-execution path, while PR #2 provides checked RLT episode exports and split custody. Fresh, lineage-disjoint collection is the intended input.
- The Algernon Circuit Mirror research/algernon/circuit_mirror in Compactification-Theory demonstrated a *toy* recurrence-specific slot signal under intermediate supervision and some residual-conditioned corrections. The AF-17 TypedLoop receptive-field analysis in this repository shows why the minimum loop count must cover the required dependency radius. These do not yet demonstrate generalized Lean-state emulation or online learning.

## Architecture: proof-state embedding, latent recurrence, and plastic memory

### Input and typed slots

A proof packet contains a canonical problem lineage ID, public printed Lean goals, context/hypotheses, available lemmas, candidate tactics (when using candidate-ranking mode), verifier-visible failed attempts, and the remaining check/compute budget. Hide source reference tactics, labels and certificate receipts until after the decision.

Use 16-64 learned latent slots with explicit roles:
GOAL, LOCAL_CONTEXT, RULE, PROOF_FRONTIER, PLAN, LEMMA, ERROR, EVIDENCE, VALUE, MEMORY, DELEGATE, HALT.
Role IDs determine masked attention and head projections; the role assignments are trainable embeddings, not claims of faithful semantic decomposition.

An illustrative *teacher-visible* structured trace is:

~~~text
<GOAL> ∀ n : Nat, n + 0 = n
<LOCAL_CONTEXT> n : Nat
<RETRIEVED_LEMMA> Nat.add_zero
<PLAN> use the retrieved lemma
<ACTION> exact Nat.add_zero n
<LEAN_VERDICT> closed; kernel_certified
~~~

The plan is attributed teacher prose. Only independently verified tactic actions, goal transitions and the kernel certificate create authoritative task targets. Train the student on response-only targets and keep verdicts outside predecision features.

### Shared recurrent computation

Let c encode the immutable public proof packet, z_k the typed slots after inner pass k, M_t the episodic/fast-weight memory at outer time t, and phi_t the trainable specialist adapter:

u_(k+1) = F_theta(z_k, c, Read(M_t,z_k); phi_t)
z_(k+1) = z_k + eta_k [u_(k+1) - z_k].

Reuse the same 2-4 decoder/attention blocks at each inner pass. Read the goal context again on every pass through dedicated cross-attention or controlled reinjection. Initial candidate K is {1,2,4,8,16}; an ACT-style stopping head should predict remaining benefit and cost, trained on observed continuation improvements. A kernel-closed proof is a hard external stop.

Compare direct feedback (eta=1), rectified feedback (eta_k = 1/(K-k), zero-based k), and matched damping on every budget. RLT states depend on the requested total K; recalculate a trajectory at each K, rather than reading a K=4 prefix out of a K=16 RLT trajectory.

Use three explicit readouts: action policy over legal candidates, predicted verifier outcome/remaining proof effort, and structural successor-goal prediction. Candidate pool coverage and ranker quality are independent metrics. Include an out-of-pool tactic generation path, but verify every generated action separately.

### DRAM-resident online plasticity

**Fast weights**: a small CPU-resident head W_t scores (encoded goal, tactic, budget, feedback history). A simple linear/bilinear head begins with W in FP32; use SGD, OGD or recursive least squares with measured convergence and numerical stability. For binary verifier labels, an example SGD update is

W_(t+1) = W_t - alpha (sigma(W_t h) - y) h^T.

This does not require backprop through the entire transformer at update time if the feature encoder is frozen. Record success/failure/accepted-with-subgoals as distinct targets and mark timeout, server failure and out-of-budget results as censored.

**Plastic adapter**: rank-8/16 LoRA or a low-rank MLP conditioned on typed slots. Store FP32 master weights and optimizer in DRAM; load active low-precision weights to the GPU when available. Periodically train from replayable, certified batches, then validate and switch via an atomic versioned adapter snapshot. CPU-offloaded training still needs gradient compute and activation memory.

**Episodic store**: content-hashed source goals, candidate actions, kernel feedback, exact environment/toolchain versions, and model identity. Retrieval uses an index separate from the weights. This is a comparator for assessing actual learned weight updates versus look-up.

The memory states are different objects: (a) within-proof latent slots, (b) between-proof online parameters, (c) persistent retrieved episodes. Every ablation must change exactly one of them.

### Sizing and resource constraints

- PPT-20M (approximately): width 512, two shared blocks, 16k tokenizer, 16-32 slots. CPU smoke, small GPU training.
- PPT-55M (approximately): width 768, four shared blocks, 32k tokenizer, up to 64 slots. Main specialist experiment.
- ALEPH integration: export role/state interfaces and graft/adapters into ALEPH-1B only after the standalone result clears held-out learning gates.

All parameter counts are target ranges; run the actual counter, memory and throughput measurement before scaling. For illustration, a 256x256 FP32 value matrix uses 256 KiB; its RLS covariance uses another 256 KiB. A separate 100k x (256-key + 256-value) BF16 episodic memory uses ~102 MB of payload excluding indexes. Two million FP32 trainable parameters with weights, gradients and two Adam moments use ~32 MB, excluding activations/optimizer buffers. These are DRAM storage, not GPU execution costs.

## Training program

**PPT-00 / verifier and hygiene.** Reuse Lean-CGRT's pinned Lean/Mathlib environment and DataGoblin's checked source manifests. Freeze family-/source-file-/theorem-ancestry-disjoint train, calibration, ordinary test, OOD, and sealed lanes. Maintain a quarantined c4 archive. Include negative examples, abstentions, candidate pool misses, parse failures and timeouts with distinct evidence classes.

**PPT-01 / syntax and action semantics.** Train proof-state encoder and legal action mask on fresh Lean goals, valid next-state transitions and local candidate pools. Include non-terminal tactics and multi-goal states. Measure closings and intermediate validity separately. Compare against the existing automation portfolio and shallow transformer.

**PPT-02 / intermediate latent supervision.** First train with explicit successor-goal / action / remaining-budget targets at each recurrence depth. Use distillation of teacher *actions*, optionally attributed public explanations; audit textual plans against actual Lean traces. Add hidden-state probes and causal patching of GOAL/ERROR slots to test whether they carry task-specific information. Controls: equal-FLOP feedforward expansion, extra visible CoT tokens, shuffled slots, constant slots, and matched token budget.

**PPT-03 / verifier-feedback repairs.** Use independently checked failed tactic and successful revision sequences. Train outcome prediction and policy improvement on fixed predecision states. Avoid rewarding a false confident STOP. Compare closed-goal counts and verifier calls at a fixed max of 8, including an automation-first four-call allocation.

**PPT-04 / online meta-learning.** Construct *tasks* as proof families: each task offers several support theorems with outcomes, followed by unseen query goals in the same family. Reset W_t at task boundaries. Run the inner DRAM update only on support labels. Train the encoder and an optional update controller on query loss, never directly on query feedback. Group by theorem ancestry and avoid repeated variants across meta-train and meta-test. Compare gradient/RLS fast weights, retrieval-only, no-update, shuffled-update, same-cost extra-inference, and equal-parameter static heads.

**PPT-05 / routing and continuation.** Learn whether to take another inner pass, rank a tactic, synthesize a new tactic, query retrieval or delegate to a larger local Bonsai model. Price each route by measured wall time, tokens and Lean calls; cost-aware oracle/policy must see the same evidence. Report end-to-end certified success under fixed checks and fixed total compute.

**PPT-06 / consolidation and ALEPH graft.** Produce replayable fast-to-slow adapter training snapshots. Re-evaluate retention on old theorem families, transfer to fresh ones, and compatibility with RLT/Hypergame task packets before grafting into ALEPH.

## Online-learning objective and readout

Outer meta-objective for a proof-family task tau, with frozen pre-query fast-weight state phi_0:

min_(theta, psi) E_tau [L_query(theta, U_psi(phi_0, D_support_tau))
                         + lambda ||phi_tau - phi_0||^2
                         + mu C_compute].

The update U may initially be a fixed optimizer; a learned update controller is a separate experiment. The query split is unseen during the inner update. Training the feature encoder to make small, local updates effective is the novel test.

A candidate value head is Q(s,a,b) = P(kernel_certified_completion | s,a,b) - lambda E[cost | s,a,b]. Disaggregate immediate valid tactic, terminal proof, horizon-limited chance of closure and solver error; treat uncertain outcomes as censored.

For structural evidence, learn a typed projection pi and measure the commuting error of pi(F(z,a)) against the *real* Lean successor state T_Lean(pi(z),a). Such approximate agreement on held-out states is an empirical property, not a formal correctness theorem.

## Evaluation and promotion

Primary endpoint: pairwise *incremental certified close@8* on fresh automation@4 misses, matched for proof pool, model FLOPs, retrieval, memory exposure and total Lean calls. Secondary endpoints: valid nonterminal tactics, fresh out-of-pool rescues, success/query before versus after support updates, memorization leakage, adaption/forgetting on unrelated families, recurrent visits, DRAM bytes written, wall time, peak host commit and VRAM.

Analyze independent theorem lineages as the unit, not repeated actions or loop passes. Preregister effect threshold, sample size, checkpoint/seed roster, budget and sealed-test opening. Every accepted result must include a Lean kernel receipt and permitted axiom audit. Cross-family benefit, positive gain over retrieval-only and same-compute static controls, and stable retained skills are necessary before calling DRAM plasticity useful.

Do not promote a run based solely on historical replay, high tactic-validity on train, synthetic Horn success, or test-time loops on an already saturated split.

## Concrete ownership

- This repository: recurrent core, typed workspace, test-time depth controls, latent causal probes and proof-graph ablations.
- DataGoblin: fresh Lean rollout collection, kernel receipts, deduplicated corpus, fast-weight updater, episode replay, ALEPH adapters and resource ledger.
- EvalMath/BPD: held-out power/uncertainty controls and causal-intervention corroboration.

## First deliverables

1. Freeze a new PPT experiment registration and proof-family training/eval rosters (keep c4 quarantined).
2. Implement a CPU-usable PPT-20M reference with reproducible forward, recurrent masks, and fast-weight update; record exact parameter count and resource cap.
3. Export one proof-family support/query collection with exact replay, independent Lean certification and no future-label leakage.
4. Run paired no-update/retrieval-only/DRAM-fast-weight experiments before moving to PPT-55M.

**Status:** Proposed. No trained PPT, DRAM-plasticity gain, new Lean closures, or GPU spend are asserted.

## Related local documents

- [Lean-CGRT architecture](HRM_CGRT_ARCHITECTURE.md)
- [Mathlib extraction and certification](LEAN_DATA_PIPELINE.md)
- [Local scaling and negative findings](LOCAL_SCALING_RESULTS.md)
- [RLT ablation and negative findings](RLT_ABLATION_MATRIX.md)
- [DataGoblin PR #4](https://github.com/MoralityLabAI/DataGoblin/pull/4)
- [Compactification Algernon](https://github.com/MoralityLabAI/Compactification-Theory/tree/main/research/algernon)
