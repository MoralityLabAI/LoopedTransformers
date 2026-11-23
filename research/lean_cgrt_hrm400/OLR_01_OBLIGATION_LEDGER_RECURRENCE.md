# OLR-01: Obligation Ledger Recurrence for a compact looped transformer

**Status:** Architectural research proposal, not an experimental success claim  
**Date:** 2026-11-23  
**Codebase:** MoralityLabAI/LoopedTransformers, research/lean_cgrt_hrm400  
**Source checkpoint:** 3df3d5efaff77686bcc46190fe01c05dd053da2b  
**Primary source:** [PPT_RESIDUAL_TRANSFER.md](PPT_RESIDUAL_TRANSFER.md)  
**Experiment classification:** Exploratory design. Any implementation first needs a frozen prospective protocol; earlier opened cohorts remain development/audit data.

## Why pursue this architecture?

The registered 1217 repaired prospective transfer favored the 7,369-parameter, tied, gated two-slot proof model over a 10,393-parameter full-history TRM on **directed BPD proof residuals**. At two target-call budgets over 32 roots, its source-stopped reward checkpoint completed 20/192 scored scenarios at root utility 0.079167. The TRM reward checkpoint completed 5/192 at -0.010677. A same-architecture imitation checkpoint completed 20/192 at 0.046354, and a native search-tree reference completed 32/192 at 0.158333. The reward fit's utility increment over imitation was +0.032813, primarily from fewer sibling/check expenditures, **not additional proofs**. The reward/TRM utility increment was +0.089844, descriptive root bootstrap [0.060547, 0.119922].

These are one-seed, specialized transfer results. All learned/calibrated arms closed the 84 equality scenarios; the reward, TRM, and imitation arms each closed 11/92 Horn scenarios. The earlier development study's paired message-status sham was inconclusive, and the prospective 1217 report does not demonstrate a direction-specific causal mechanism. See [PPT_RESIDUAL_SEARCH.md](PPT_RESIDUAL_SEARCH.md) and [PPT_RESIDUAL_TRANSFER.md](PPT_RESIDUAL_TRANSFER.md).

The nearby [PPT_HYBRID_RL_TRANSPORT.md](PPT_HYBRID_RL_TRANSPORT.md) supplies a second clue. On a reused 12-pair graph pilot, a frozen-backbone transport readout reached 19/24 certified worlds while fully trained transport RL remained at 12/24. Incoming messages had been averaged over premises; that average can transmit partial support for a rule that requires *every* premise. The development results show architecture and training-objective bottlenecks; they leave the role of learned graph propagation unsettled.

The [LOCAL_SCALING_RESULTS.md](LOCAL_SCALING_RESULTS.md) control studies also favor simple tied loops in some shortcut-free cases while recording strong seed variance and negative depth extrapolation on other settings. Thus OLR must be tested against an ordinary tied transformer with equal information, FLOPs, and verified search calls.

## Hypothesis

**Represent the task remaining after each action as an explicit obligation ledger, couple it to a forward evidence ledger, and refine both through a shared recurrent block.** A loop should transport *requirements* backward from the goal as well as *verified support* forward from the environment.

A useful OLR result would show that these two coupled channels improve native Lean proof completion over the two-slot actor, ordinary loop, frozen readout, and exact structural planning controls on **new** theorem lineages. Better calibration and call efficiency are separately valuable endpoints.

## Architecture

Public state -> prelude -> {EVIDENCE ledger, OBLIGATION ledger} -> repeated tied update K times -> action/value/continue coda -> native Lean verifier. Real feedback enters the next decision. Hidden target scripts, hindsight oracle values, proof certificates, and query labels never appear in a predecision observation.

The minimal state contains:

- **Evidence:** available local facts, verified tactic outcomes, current public goals, hypothesized rule applicability, and a confidence/status mask separating native facts from predictions.
- **Obligation:** active goal and all pending subgoals, candidate prerequisites, action costs, remaining Lean-call budget, speculative continuation costs, and branch rollback state.
- **Budget and control:** terminal-success mask, checked/censored/rejected feedback, kernel status, a local step counter, and an optional continue/STOP readout.

Treat proof rules as **AND-OR hyperedges**. An edge r -> v may require a set P(r) of premises; multiple edges into v are alternative derivations. This is an explicit distinction absent from simple mean aggregation over incoming facts.

### Structural tree-cost recurrence

For an acyclic, fully observed Horn hypergraph with positive action costs, define:

    D_0(v) = 0 if v is a given fact; +infinity otherwise
    D_{k+1}(v) = min(
        D_k(v),
        min_{r -> v} [c(r) + sum_{u in P(r)} D_k(u)]
    ).

An empty premise set has sum 0. Given an acyclic graph and enough rounds to reach every dependency, the fixed point gives **minimum derivation-tree cost** under these assumptions. Shared premises may be counted repeatedly, so this recurrence is not automatically the least-cost proof DAG or the actual Lean tactic-call count. Cyclic, dependent, or computational side conditions require explicit treatment. For integer proof depth, an analogous Boolean reachability update is exact when source facts and rule truth are known.

Use the structural channel as a predictable reference and train an auxiliary neural channel for real Lean state semantics:

    z_{k+1} = SharedBlock(z_k, context, evidence_k, obligations_k)
    delta(r,k) = CorrectionHead(z_k, edge_r, budget)
    C(r,k) = c(r) + sum_{u in P(r)} D_hat_k(u) + softplus(delta(r,k))
    D_hat_{k+1}(v) = SoftMin_{r -> v} C(r,k).

The learned correction is *an estimated difficulty penalty*, not a certified lower bound. SoftMin can undercut the hard minimum; no bound claim follows from it. Preserve a separately computed hard structural channel for exact fragments and use strictly learned estimates only as tactic-rank features for broader Lean. Mask any infeasible edge and always include STOP as an action with its own measured penalty.

### Forward channel

For the restricted Horn/AND-OR pilot, a Boolean evidence recurrence gives an interpretable oracle:

    E_{k+1}(v) = E_k(v) OR any_{r -> v} all_{u in P(r)} E_k(u).

At inference, only actual public and verifier-confirmed facts receive certified E=1. Predicted applicability lives in a separate channel and never overwrites a certified status. A smooth surrogate can train message weights, but the exact Boolean channel remains available as a control. Direction-specific message gates let evidence and obligations influence each other while retaining separately accessible slots.

### Recurrent budget allocation

Each repeated block reads the same public packet and the updated two-ledger state. Initially train K in {2,4,8} with budget-specific targets and test K=1,2,4,8,12,16 after freezing the checkpoint. Record actual per-pass compute and coda expense. Early exit should use measured improvement from an extra pass and calibrated wall/compute cost; a hard Lean certificate ends a proof externally.

Cross-pass memory is ephemeral within one decision. Persisted verifier feedback is a separate object. DRAM/fast-weight updates are excluded from OLR-01 so causal interpretation remains focused on graph structure and tied recurrence; G04 found no clear online-SGD advantage at its reported final dose.

## Pilot model and intervention ladder

Start with a small sidecar in **ppt_gym** that consumes the already public Horn graph and candidate menu, produces an action-value correction, and leaves the native Lean server as the only proof authority. Retain the existing [ppt_gym/live_quote_policy.py](ppt_gym/live_quote_policy.py) two-slot reward actor and [ppt_gym/residual_search.py](ppt_gym/residual_search.py) search controller for comparison. The gate must permit a true zero-correction identity; trainable value weights and expected FLOPs need reported values.

Register these arms **before seeing new scores**:

| Arm | Purpose |
| --- | --- |
| Source-frozen two-slot checkpoint | Reproduce the published reference on audited data |
| Newly trained two-slot actor | Same train distribution, seed roster, and optimizer as OLR |
| OLR, true graph, learned correction | Main candidate |
| OLR with randomized/premise-permuted edges | Structure-specific control |
| OLR with recurrent body frozen | Head/calibration explanation |
| Exact structural Horn cost/search controller | Strong algorithmic baseline, using the same available graph |
| Ordinary tied transformer, equal input | Architecture/representation control |
| FLOP- and parameter-matched nonrecurrent model | Additional-compute control |

For a mechanism test, patch an evidence bit that is native-known, or switch a single distant fact between paired worlds that require *opposite actions*. Compare action-margin perturbations with same-margin, norm-matched random interventions and an edge-shuffled sham. Recalculate and count full Lean proofs under the unchanged budget; feature or logit sensitivity alone is an auxiliary observation.

### Data and experimental gates

1. **Training and dev:** Use separate new graph fingerprints and family-disjoint theorem ancestries. Archived 1201/1211/1217 tapes may serve only as development and reproducibility material unless a fresh protocol explicitly assigns some to training; they cannot also become held-out transfer.
2. **Reachability:** Independently execute all planned witness scripts and validate menu identifiers *before* scoring. The 1211 binder-grammar mismatch previously left just two of 96 primary residuals reachable at the registered depth.
3. **Unsaturated evaluation:** Include paired source-fact-switch graphs with equal local lexical cues, joins with 2-4 prerequisites, dead-end decoys, two simultaneously pending Lean goals, and independent held-out dependency distances. Also include computational-guard and symbol-binding stress cases, separately stratified.
4. **Primary endpoint:** root-macro, Lean-certified completion at fixed native-check and model-compute budgets. Secondary: root utility, calls, exact attainable coverage, verified successor-goal prediction, false STOP rate, halting passes, wall time, memory, and robust seed variance.
5. **Threshold:** preregister a +0.025 root-utility floor over the best learned comparator as a pilot target. Require prospective root-paired and seed-level intervals and a direction-specific improvement exceeding the matched-sham envelope before claiming a mechanism. Use EvalMath's FSI-01 floor rather than repeating the earlier power-limited sham design.
6. **Kernel provenance:** retain theorem statement, independent script compilation, Lean/compiler hashes, allowed axiom audit, root lineage, structural graph fingerprint, exact checkpoint/memory hash, all scored failures, and resource budget receipts.

### Failures that should redirect design

- Exact structural planning wins on the same graph at a fraction of the compute. Keep OLR as a general-domain proposal only if it adds transfer on tasks where structural semantics are incomplete.
- Newly trained two-slot or frozen-body OLR matches the full recurrence. Investigate encoding and value supervision before allocating parameters to a larger recurrent core.
- The true-edge model behaves like the edge-shuffled sham. Revise message routing and proof dependencies.
- Added loops preserve or decrease certified success at matched FLOPs. Test recurrence dynamics rather than scaling depth by default.
- The selected cohort saturates or the witness menu fails. Register a repaired disjoint cohort before seeing any model outcomes, preserving the failed cohort and its results.

## Stage gates and possible scaling

**OLR-01A:** Implement an instrumented, tiny AND-OR cost/evidence module with deterministic unit tests on 1-4 premise joins, cycles, unreachable subgoals, and shared dependencies. Test exact hard recurrence against a small dynamic-programming oracle.

**OLR-01B:** Replace the existing learned/graph messages one interface at a time, retaining the complete two-slot reward baseline and source-selected stopping. Check parameter counts, gradient paths, zero gate identity, CPU memory, native tactic grammar, and ablations.

**OLR-01C:** Run matched 3-seed pilot on preregistered fresh Lean roots. Save all outcomes. If significant and causally supported on independent roots, register a sealed confirmatory study.

**OLR-02:** Test another domain with the same structural interface: DataGoblin's storyworld action prerequisites, causal facts, and budgeted interventions. Separate observed effects from model predictions and evaluate actual environment moves.

**Scale decision:** Consider a ~5M tied-loop prototype only after a clear improvement in certified completions over simple and structural controls. A larger 20M-55M PPT or DataGoblin graft is a later research choice.

## Nearby evidence to keep in view

- [PPT_RESIDUAL_TRANSFER.md](PPT_RESIDUAL_TRANSFER.md): targeted 1217 two-slot reward transfer and budget-calibrated stopping.
- [PPT_RESIDUAL_SEARCH.md](PPT_RESIDUAL_SEARCH.md): source 1201 fit, mixed-source prior, and inconclusive sham.
- [PPT_HYBRID_RL_TRANSPORT.md](PPT_HYBRID_RL_TRANSPORT.md): mean-premise transport and paired credit experiments.
- [PPT_HYBRID_RL_PENDING_RECOVERY.md](PPT_HYBRID_RL_PENDING_RECOVERY.md): pending goals; strong structural solver; RL/imitate/calibrate variability.
- [PPT_COMPUTATIONAL_GUARDS.md](PPT_COMPUTATIONAL_GUARDS.md): rule applicability depends on ground computation; direct-reader and exact-fold controls outperform the two-slot actor.
- [PPT_G04_RESULTS.md](PPT_G04_RESULTS.md): no measured improvement from early DRAM SGD pilot, despite successful verifier-backed replay.
- [PPT_PLAN_A_DRAM_PLASTIC_PROOF_TRANSFORMER.md](PPT_PLAN_A_DRAM_PLASTIC_PROOF_TRANSFORMER.md), [PPT_PLAN_B_INKLING_BLOCK_MESSAGE_MEMORY.md](PPT_PLAN_B_INKLING_BLOCK_MESSAGE_MEMORY.md): existing typed slots and message bank; OLR adds explicit AND-OR semantics.
- [LOCAL_SCALING_RESULTS.md](LOCAL_SCALING_RESULTS.md): simple-tied-loop wins, negative results, and strong seed variance.

**Current evidence level:** Proposal derived from registered and development runs. OLR has not been trained, benchmarked, or verified to improve any proof completion.
