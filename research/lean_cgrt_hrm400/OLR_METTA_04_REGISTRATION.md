# OLR-METTA-04 registered dependent-proof and learned repair experiment

**Date:** 2026-11-23  
**Status:** source-only prospective development registration, committed before 04 results.  
**Continuation of:** [OLR-METTA-03](OLR_METTA_03_RESULTS.md), with source-free independent Lean 4.19 kernel and native Hyperon 0.2.10.

## Scientific question

Can a small weight-tied recurrent controller trained on independently verified **multi-state Lean trajectories** select a first branch, interpret an observed guard failure, restore a saved proof state, and either request MeTTa's bounded STOP repair or continue manually? The key scientific outcome is extra native-certified proofs or lower native call cost over equal-information direct, one-pass and scripted controls. No native proof labels may appear in a policy's predecision packet.

The study is **not a claim that numerical arithmetic is hidden from Lean**. Only the model observation intentionally omits exact guard numerals. A fully informed arithmetic planner remains the exact upper bound.

## New dependent proof family

Each source statement binds unary predicates \`P Q : Nat → Prop\`, the goal \`R : Prop\`, a witness \`h0 : P (a+b)\`, and a dependent rule \`h1 : ∀ n : Nat, P n → Q n\`. Both candidate implications have the shape

    rᵢ : (a+b=cᵢ) → P cᵢ → Q cᵢ → R

where precisely one guard is provable, and c0/c1 swap between paired worlds. The identity of pending obligations depends on the numerical witness index. A true proof takes

    apply rᵢ; rfl; apply h0; apply h1; apply h0

in five native tactic calls. The alternative **MeTTa repair** at the first guard-closed state proposes

    apply h0; exact h1 _ h0

and can complete those two obligations in two tactic calls, subject to independent Lean kernel checking. The fast continuation is also recorded as an algorithmic baseline; its origin is not assumed to require MeTTa.

## Frozen development design

- Cohort seed **20261130**, source 24 pairs (48 worlds), development 8 pairs (16 worlds), range 17–30, OOD 8 pairs (16 worlds), range 31–44. Separate complete (a,b,delta) lineages across splits. Claims use fresh namespace \`olr-metta-04\`, so previous OLR-METTA-03 claims cannot be reused as exact examples.
- All actions receive a strict **six-native-tactic ceiling**, with \`focus\`, MeTTa match queries and host restores metered separately. Kernel and standalone compilation do not count as policy actions and are always performed.
- The learner must operate on three public stages: (1) initial branch choice/probe allocation, (2) guard status with pending \`P c, Q c\` and budget, (3) failed guard with an optional saved-root restore. Source-only training examples are derived from *native Lean checked complete returns*, including failed trajectories. STOP before kernel closure must remain unsuccessful unless the offered MeTTa continuation is itself fully certified.
- Native MeTTa is restricted to advisor/eligibility. It accepts a proposed two-call dependent repair from public obligation types and a current, unique source claim. Ineligible or over-budget proposals consume zero Lean tactic checks and commit zero state. A failure on either tactic returns the original immutable state.
- Set repair-capability eligibility by a **public source-only rule** independent of the hidden correct branch, so some worlds need a five-step manual proof. The exact same eligibility evidence must be available to every model and scripted baseline.
- Arms: tied attention recurrence with three visits, equal-input one-visit recurrence, direct MLP, train-calibrated finite-state controller, fixed early-probe, fixed late-probe, exact full-information planner and deterministic MeTTa-only repair baseline. Training actions, fitted checkpoints and root-level evaluation decisions must be committed before any query-native outcome.
- Train three seeds **17, 29, 43**, at most 80 CPU optimizer updates each, same 24 source pairs, same optimizer and minibatch rosters. Match both native calls and model inference FLOPs when making a causal architectural claim; report actual passes when only a development comparison is possible.
- A causal intervention must alter learned intermediate recurrent state or admission source identities with matched random/norm controls, while keeping the public claim multiset, exact proof goal and physical budget unchanged.
- Outcomes: paired native-certified complete proofs, STOP errors, first-action correctness, guard checks, spent native tactic calls, focus operations, restores, MeTTa offers/vetoes and trained model forward passes. Each success must replay from a fresh Lean server state and compile independently, with empty axioms.

## Gates

1. **04A native grammar preflight:** before any model training, verify that true/false paired branches create one indexed equality and two indexed predicate goals. Confirm canonical five-step and two-step repair proofs in Lean 4.19 and standalone source compilation. Record any Lean definalitional-equality or tactic grammar failure transparently and repair only the source implementation.
2. **04B source trajectory / policy integration:** fit using TRAIN-only native feedback, freeze source/checkpoint hashes, and evaluate fresh paired roots. Preserve all three seeds and failed actions. Do not open a sealed or prior held-out cohort.
3. **Scientific decision:** call a model mechanism success only if native proof completions improve over the strongest same-information direct or scripted controller at matched compute, and a prospective shammed intervention clears an FSI-01 sample/power gate. A good Lean/MeTTa engineering interface alone is an implementation result.

**Current claim:** an experiment plan, not a measured improvement.
