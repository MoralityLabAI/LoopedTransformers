# LBF-20E registration: context-scoped support DAGs for native branching tactics

Parent 55e729cf, LBF-20D. Development engineering only; no new fits or old sealed cohorts. Fixed seed 2026112512. 24 public generated fixtures (eight each shared, witness_choice, reuse_tradeoff); paired implication branches share the same conclusion but receive different local hypotheses. Shared templates are deliberate, not a claim of lineage-disjoint generalization.

## Fragment

Public hypotheses are first-order indexed Horn clauses over arbitrary predicates Nat→Prop. Each rule has one or two explicit Nat variables, constant offsets, and a finite list of premises. Instantiation domain {0,1,2}; offsets can make goal indices larger. The goal is (A k→G(k+4)) ∧ (B k→G(k+4)); a shared derivation chain creates two branching obligations. The final two-variable rule requires a witness index j whose two auxiliary facts both hold. All variables are explicitly instantiated in the main search, so no dependent metavariable aliasing is assumed. Separate native audits test full and partial Lean inference. General first-order or dependent Lean completeness is outside scope.

## Exact finite reference

Each ground rule instance is a hyperedge. Given a local context Γ, min-tree costs dΓ(a)=0 for facts, otherwise min over instances with conclusion a of 1+sum dΓ(premises). Least positive-cost closure retains an acyclic witness.

For DAG cost, let FΓ(a) be an antichain of sets of instance IDs sufficient to derive a. Initial facts have {∅}. Each rule inserts its own ID unioned with one sufficient support per premise; discard supersets. Monotonicity of union under inclusion ensures pruning preserves subset feasibility. A finite fixed point covers every finite Horn derivation; each retained set supports a concrete forward derivation. Minimum cardinality support gives minimum number of distinct ground rule applications needed for this one target, with shared derived facts. Reconstruct the reachable selected hypergraph and remove irrelevant firings. The finite set universe bounds strict coverage updates, exponential in instance count. A hard 5000-label cap causes explicit failure, not approximation. No cross-context cache sharing except public facts and rules.

The Lean file states abstract derivation monotonicity, cache inlining/extension, context counterexample and conditional support-query exactness. It does not certify Python indexing, finite saturation or general Lean search completeness. Runtime audit checks every finite support's forward closure, antichain and closure inequalities. Independent exact audit: 96 seven-edge five-atom ground hypergraphs, two local contexts each, all 128 active subsets, compare every atom's minimal supports and min-DAG cost against exhaustive subset enumeration. Include a strict min-tree=7 vs min-DAG=6 fixture.

## Native arms

1. native_order: exact public instantiated head retrieval, stable rule order, live apply and fact tactics, bounded DFS rollback. No reference-plan access.
2. native_tree: same native engine, exact min-tree rank and unreachable premise pruning.
3. native_dag: compute minimum-support DAG per local context; introduce each named proof once using live have; finish branch with exact.
4. deferred_tree: minimum proof tree rendered as a final explicit Lean term.
5. deferred_dag: minimum support DAG rendered with local let bindings, single final exact.

All arms use same public formulas/facts, maximum 96 native tactic requests, four-second soft cap before each request; wall includes reset, init, actual planner computation, native tactics and kernel. Each separately recomputes its needed planner; reference audit is outside timed episodes. Actual live constructor/intro/apply successors must match expected ground obligations (order-independent); scope verified via actual local context. Every successful proof checked by Lean kernel with zero axioms and independently compiled. Rotate arm order. Report 120 episodes, closure curves, native calls, actual timings, rollbacks, rule applications in symbolic reference witnesses, script and displayed kernel-term character counts. Character counts are descriptive, not binary proof sizes or kernel complexity guarantees.

Separate audit after all arms on each root: native apply of bare rule, partial witness instantiation, full instantiation; record actual acceptance/goals, including unresolved Nat/Prop metavariables. Prove target with a local named cache in first implication branch, close it, enter second branch and try exact of the now-out-of-scope cache. Require rejection and then successful correctly scoped proof. Audit requests separate from controller budgets. No labels fed to controllers. These scopes remain static within each branch; the native environment changes through introduction and branching, but the chosen fragment admits exact symbolic planning.

Engineering gate: independent audit and axiom checks green, all 120 episode receipts, 24 separate context audits, explicit outcomes including failures. No registered claim of neural superiority. A neural follow-up needs a domain with nontrivial symbolic inference cost and matched exact/deferred controls; this pilot should first establish what cost remains after optimal symbolic factoring.
