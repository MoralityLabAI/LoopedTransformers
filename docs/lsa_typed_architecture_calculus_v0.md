# LSA-02: A typed calculus of blocks, recursions, and lattices

**Status:** mathematical framework and finite combinatorial checks, 2026-11-23. This extends [Loop Schedule Algebra v0](../reports/looped_transformer_algebra_v0.md); it leaves the earlier measurements, sealed evaluations and their conclusions unchanged. The propositions below are ordinary mathematical arguments. Python tests check representative finite instances. There is no Lean kernel proof of this document and no new neural-performance result.

## 1. Existing mathematical pieces and the missing connection

The repository already describes a recurrent architecture as
\[
A=(S,\Phi,w,\rho,g,\sigma,\pi),
\]
where the registers \(S\), physical blocks \(\Phi\), schedule word \(w\), residual rule \(\rho\), gradient visibility \(g\), loss sites \(\sigma\), and carry rule \(\pi\) are separate architectural choices. See [`lsa/schedule.py`](../lsa/schedule.py), [`lsa/instances.py`](../lsa/instances.py), [`lsa/gradient_policy.py`](../lsa/gradient_policy.py) and [`lsa/topology.py`](../lsa/topology.py).

The extension separates three distinct mathematical structures:

1. **Neural computation** is typed operator composition over explicit registers.
2. **Parameter identity** is a lattice of equivalence relations among compatible block visits, or a more general parameter-incidence hypergraph.
3. **Certified knowledge** is a separate lattice of candidate states, whose refinement requires an authority-checked membrane.

An architecture also has an ordered *dispatch word*, a directed dependency graph, clocks, and resource constraints. The words themselves generally form a free monoid, **not** a lattice. This distinction matters for architecture generation: more sharing, more visits, and stronger deduction are three different operations.

### Source-grounded block taxonomy

| Family | Maps and registers | Sharing / clock | Repository implementation or contract |
| --- | --- | --- | --- |
| Standard decoder block | causal attention and MLP act on the residual stream | tied, untied, or schedule-word dispatch | [`research_gym/neural/looped_decoder.py`](../research_gym/neural/looped_decoder.py) |
| Typed graph-attention block | graph nodes and typed incidence edges | tied message-passing steps | [`research/lean_cgrt_hrm400/HRM_CGRT_ARCHITECTURE.md`](../research/lean_cgrt_hrm400/HRM_CGRT_ARCHITECTURE.md) |
| Fast/slow slot block | fast node state, slow memory slots, cross-attention | nested \(T\)-step fast and \(k\)-step slow clocks | [`HRM_CGRT_ARCHITECTURE.md`](../research/lean_cgrt_hrm400/HRM_CGRT_ARCHITECTURE.md) |
| Two-state recursive TRM | \(y,z\) states, repeated latent updates and answer updates | nested recursions, selectable gradient path | [`lcgrt/tactic_lm.py`](../research/lean_cgrt_hrm400/lcgrt/tactic_lm.py) |
| Gated prior-pass message block | current stream plus earlier-pass block messages | tied core, gated cross-pass transport | [`PPT_PLAN_B`](../research/lean_cgrt_hrm400/PPT_PLAN_B_INKLING_BLOCK_MESSAGE_MEMORY.md) |
| Plastic adapter / fast weights | learned core plus episode-updated small parameters | overlapping shared and private parameter sets | [`PPT_PLAN_A`](../research/lean_cgrt_hrm400/PPT_PLAN_A_DRAM_PLASTIC_PROOF_TRANSFORMER.md) |
| Proposer GRU | latent recurrent proposal state | one tied GRU applied repeatedly | [`research_gym/neural/trm.py`](../research_gym/neural/trm.py) |
| LDT candidate membrane | set-valued public state with exact meet/refinement | deterministic symbolic transition | [`research_gym/core/hybrid.py`](../research_gym/core/hybrid.py) |
| Interval / box deduction | product of integer intervals | parameter-free meet and join | [`research_gym/core/lattice.py`](../research_gym/core/lattice.py) |
| Hierarchical module flow | fast manager, slow manager, module-owned states, typed receipts | event-driven scheduling across module graph | [`docs/hrm_conductor_architecture.md`](hrm_conductor_architecture.md) |
| Controller-mesh/sheaf | typed local controller outputs with restrictions and compatibility maps | directed port graph, spectral Laplacian | [`docs/controller_mesh_sheaf_forward.md`](controller_mesh_sheaf_forward.md) |

These are *axes of composition*, not mutually exclusive model species. A typed graph-attention block can be recurrent, have cross-pass memory, and sit behind a lattice membrane.

## 2. Typed state and block semantics

Let \(S\) be a finite collection of registers. Register \(s\) has type \(X_s\), and the full state is \(X=\prod_{s\in S}X_s\). A block \(b\) has a declared read set \(R_b\), write set \(W_b\), and parameter schema \(T_b\). For a fixed parameter value \(\theta_b\), its typed local map lifts to a total-state map
\[
U_{b,\theta_b}:X\to X
\]
that preserves registers outside \(W_b\). This includes graph operators, attention blocks, slot updates, learned gates, and deterministic membrane operators provided that their registers and authority are declared. An input re-injection is an explicit read of the fixed input register rather than an invisible extra edge.

For visits \(w=(b_1,\ldots,b_L)\), define
\[
\llbracket w\rrbracket_\theta
=U_{b_L,\theta_L}\circ\cdots\circ U_{b_1,\theta_1}.
\]
Concatenation corresponds to function composition in execution order. The empty word is the identity; integer powers give repeated execution. The existing `lsa.schedule.Word` already supports concatenation and powers.

A *block* is a local typed function; a *layer* is an execution stage or collection of such functions; a *visit* is one application of a physical block; a *recursion* is a schedule containing repeat visits or coupled state updates. Consequently, counting layers, unique weights, and forward visits yields different quantities.

For multiple clocks, include all fast, slow, slot, and input registers in \(X\). For example,
\[
f_{t+1}=F_\theta(f_t,s_t,x),\qquad
s_{t+1}=
\begin{cases}
G_\phi(s_t,f_{t+1},x),& t+1\equiv0\pmod k,\\
s_t,&\text{otherwise}.
\end{cases}
\]
One *macro-update* is the composition of \(k\) fast ticks and the scheduled slow update. Claims about one-clock convergence or fixed points must apply to this combined map or to a justified invariant subsystem.

An input-dependent stopping policy turns the schedule into a state-dependent program. That policy belongs in the architecture record with its exit decision and consumed budget.

## 3. The parameter-sharing partition lattice

Fix \(L\) visit positions with compatible parameter schemas. A sharing pattern \(P\) partitions \(\{0,\ldots,L-1\}\) into groups that use the **same physical parameter object**. Two names for the same partition have the same identity under first-appearance canonicalization.

Define \(P\preceq Q\) when \(P\) *refines* \(Q\): every tied class of \(P\) is contained in a tied class of \(Q\). The untied pattern is the bottom (finest) element; a fully tied pattern is the top element when all visits have compatible schemas. Higher in this order means fewer independent blocks, not greater reasoning power.

**Proposition P1 (sharing lattice).** For fixed visit positions and a fixed compatibility schema, the allowed sharing partitions form a finite lattice. Its meet \(P\wedge Q\) is the set of nonempty intersections of classes. Its join \(P\vee Q\) consists of the equivalence classes of the transitive closure of the union of the two tie relations.

*Proof.* Intersections produce the greatest relation contained in both equivalence relations. The smallest equivalence relation containing both is obtained by transitive closure, giving the least upper bound. Reflexivity, symmetry and transitivity establish valid partitions. When visits only tie within an identical parameter schema, each union edge and thus each closure path stays in one schema, so both operations preserve compatibility. \(\square\)

The partition lattice is generally **non-distributive**. A finite four-visit counterexample is checked in [`tests/test_lsa_partition_lattice.py`](../tests/test_lsa_partition_lattice.py). This rules out treating tied and untied designs as independent Boolean switches.

The new [`lsa/partition_lattice.py`](../lsa/partition_lattice.py) implements canonicalization, refinement, meet, join, schema validation, independent-block counts, tie-pair distance, and exhaustive small-\(L\) enumeration. It can be used alongside `ScheduleTopology.macro_word` without changing existing architecture hashes or experiment registries.

**Extension to adapters.** A partition alone describes one parameter identity per visit. A shared recurrent core with visit-specific LoRA adapters uses several parameter objects per visit. Model this by a bipartite incidence relation
\[
I\subseteq\{\text{visits}\}\times\{\text{parameter objects}\}.
\]
The partition lattice is a special case in which each visit has one primary block identity. Parameter incidence captures shared cores, untied routers, gated message banks, and episodic plasticity. It will require a separate serialization and identity gate before use in a frozen experimental campaign.

## 4. Ordered schedules and a composition obstruction

A schedule \(w\) is an ordered word of visits. There is no canonical join or meet for two arbitrary words that preserves their execution semantics. In particular, changing visit order while holding the parameter partition fixed can change the computed function.

**Proposition P2 (residual block commutator).** Let
\[
A_\varepsilon(x)=x+\varepsilon f(x),\qquad
B_\varepsilon(x)=x+\varepsilon g(x).
\]
Their exact order gap is
\[
B_\varepsilon(A_\varepsilon(x))-A_\varepsilon(B_\varepsilon(x))
=\varepsilon[g(x+\varepsilon f(x))-g(x)-f(x+\varepsilon g(x))+f(x)].
\]
For twice continuously differentiable \(f,g\), on a bounded neighborhood where the derivatives are bounded, this becomes
\[
\varepsilon^2(Dg(x)f(x)-Df(x)g(x))+O(\varepsilon^3).
\]

*Proof.* Substitute the two residual definitions, cancel the common \(x+\varepsilon f(x)+\varepsilon g(x)\) terms, and Taylor-expand the remaining increments. \(\square\)

The leading term is the familiar vector-field bracket with the stated orientation. It suggests a **schedule-order sensitivity descriptor**: estimate commutator norms on calibration states for selected adjacent block pairs, while controlling residual scale and block type. An elevated commutator norm establishes order sensitivity at those states, not better task performance.

A simple exact witness is \(A(x)=x+1\), \(B(x)=2x\), with \(x=1\). Executing \(A\) then \(B\) yields \(4\); executing \(B\) then \(A\) yields \(3\). Both architectures use the same two physical parameter blocks.

For a fixed typed node set, the *set of allowed computational edges* is a Boolean lattice under inclusion. Restricting the graph to a compute or memory budget can destroy closure under union, so the budget-feasible architectures may form only a partially ordered set. This prevents architecture search from silently assuming that every pair of feasible models has a feasible join.

## 5. Certified candidate lattices

For finite candidate domains \(D_s\), the public evidence state belongs to
\[
\mathcal L=\prod_{s\in K}\mathcal P(D_s),
\qquad a\preceq b\iff a_s\subseteq b_s\text{ for all }s.
\]
Meet is coordinatewise intersection, join is coordinatewise union, and bottom is represented by an empty candidate set in at least one required slot. This is the semantics of [`CandidateState`](../research_gym/core/hybrid.py); [`Box`](../research_gym/core/lattice.py) uses interval-hull join, which is a different abstraction domain.

A neural proposal \(q_t\) passes a membrane that checks its type, current constraints, provenance, and any required environment-owned verifier. An accepted hard refinement has the form
\[
a_{t+1}=a_t\wedge q_t.
\]

**Proposition P3 (finite descent).** Suppose the candidate domains are finite and every accepted *strict* update is an intersection. There can be at most \(\sum_s|a_{0,s}|\) strict decreases along a run.

*Proof.* Use the integer potential \(V(a)=\sum_s|a_s|\). A strict refinement removes at least one candidate, so \(V\) drops by at least one. It is nonnegative. \(\square\)

**Proposition P4 (conditional truth retention).** If the correct assignment \(x^\star\) is in \(a_t\), and the verifier establishes \(x^\star_s\in q_{t,s}\) for every slot under refinement, then \(x^\star\in a_t\wedge q_t\).

This follows from the definition of intersection. A model-generated provenance label does not establish the premise. The existing [hybrid contract](hybrid_contract.md) distinguishes claimed and verified soundness, guards bottom, and keeps uncertain proposals soft by default.

For **fixed, independently verified** \(q_1,q_2\), the terminal candidate meet is order independent:
\[
(a\wedge q_1)\wedge q_2=(a\wedge q_2)\wedge q_1.
\]
This identity concerns the final set. When proposals depend on the intermediate state, or when bottom rejection/abstention changes control flow, the executions may differ even when a mathematical meet is commutative.

## 6. Recurrent dynamics and dependency topology

**Proposition P5 (conditional convergence).** If an autonomous full-step operator \(T:X\to X\) maps a complete metric space to itself and is globally \(c\)-Lipschitz for \(0\le c<1\), then it has a unique fixed point and
\[
d(T^t(x),x^\star)\le c^t d(x,x^\star),\qquad
d(T^t(x),x^\star)\le \frac{c}{1-c}d(T^t(x),T^{t-1}(x)).
\]
This is the Banach contraction theorem plus a geometric tail bound. It applies to a tied block or a full multi-clock macro-update under the stated conditions. Standard residual attention blocks with arbitrary weights need not satisfy contraction. The existing LSA measured gradient alignment exponent \(\gamma\) is not a contraction certificate.

**Proposition P6 (causal-cone composition).** Let \(R_b\) be the dependency relation of a block visit, including identity dependencies for carried registers. The dependency relation of the composed architecture lies within the relational composition
\[
R_{w_L}\circ\cdots\circ R_{w_1}.
\]
This follows by induction over visits, since each output reads only its declared incoming registers. The bound concerns *possible influence* for fixed graph and weights; it does not certify that every dependency is causally used.

In the TypedLoop graph cell examined in [AF-17](research/AF17_RECEPTIVE_FIELD.md), each complete update crosses the bipartite atom-rule incidence graph twice. Thus a goal-only readout after \(L\) updates cannot depend on atom features outside incidence distance \(2L\). This is a concrete instantiation of the dependency theorem.

**Coordinate transport.** A change of basis \(G_i\) on state register \(i\) transports an operator to \(G_{\mathrm{out}}\circ U_b\circ G_{\mathrm{in}}^{-1}\). Along a composable closed path, the intervening coordinate changes telescope. Learned transports that fail to close define a measurable cycle discrepancy or holonomy diagnostic, subject to exact register and gauge definitions. The controller-mesh and RSI review documents already use related geometric tools. A measured cycle discrepancy is a diagnostic of transport compatibility and does not by itself give a behavioral or safety certificate.

## 7. Architecture identity and permitted transformations

Two architecture records should be considered identical only relative to an explicit declared equivalence:

| Operation | Mathematical status | Implication for comparisons |
| --- | --- | --- |
| Rename physical module labels consistently | structural isomorphism | canonicalize labels |
| Regroup parentheses in sequential composition | exact associativity | same visits and state interfaces |
| Swap adjacent blocks | conditional equivalence only when they commute on the relevant states | requires an order test |
| Tie two compatible parameter sets | moves upward in sharing partition lattice | usually changes function family and optimizer coupling |
| Split a tied class | moves downward in sharing partition lattice | increases independent parameters |
| Add a visit to the schedule | changes operator composition, cost and reachable causal cone | match visit/FLOP budget |
| Change truncation or detach mask | same forward map at fixed weights; different training gradient | match gradient policy |
| Add a certified meet | decreases public candidates | requires environment authority |
| Add a heuristic proposal | changes soft state or search policy | retains provenance type |
| Add a new port, memory or slow clock | changes the typed state-transition graph | record its data dependence and cost |

The resource vector should retain at least \((P_{\rm unique},P_{\rm applied},\text{FLOPs},\text{activation bytes},\text{state bytes},\text{verifier calls},\text{latency})\). There is no unconditional scalar order of "more powerful" architectures across these budgets. The existing [LSAD v1 protocol](loop_schedule_architecture_discovery_v1.md) already matches several of these axes in its finite search grammar.

## 8. Mathematical questions worth testing next

**LSA-02A, combinatorial gate.** Exhaust all sharing partitions through four visits. Verify canonical renaming, partition lattice universal properties, associativity, non-distributivity and type safety. This is implemented in [`tests/test_lsa_partition_lattice.py`](../tests/test_lsa_partition_lattice.py); no GPU is required.

**LSA-02B, order geometry.** On a *competent* pretrained or task-trained recurrent checkpoint, compare schedule permutations with fixed blocks, visit count, parameter identity, gradient policy and forward compute. Estimate commutator norm and task outcomes on independent graph/proof roots. Preregister matched control/sham variance and an outcome effect size using the repo's [FSI gate](evalmath_finite_sample_gate.md). The existing LSAD v1 chance-level training cells cannot establish this relation.

**LSA-02C, clock stability.** Measure a Jacobian norm or certified local Lipschitz upper bound for the full \(k\)-tick HRM macro-operator. Contrast with loop-count extension and stopping behavior, tracking input re-injection and normalization explicitly. A measured local derivative below one is local evidence; global fixed-point claims need a uniform bound on an invariant set.

**LSA-02D, information preservation.** Instrument candidate-set descent and truth retention on exact Horn and native Lean transitions. Separate hard environment-certified meets, model-only guidance and deliberate forged provenance. Use certified proof completion and exact mechanics as endpoints.

**LSA-02E, partial sharing.** Add an incidence-hypergraph record for shared core + per-visit adapters + gated pass memory. Check uniqueness and FLOP accounting against actual PyTorch parameter aliases before introducing it into the architecture planner. Keep old canonical hashes unchanged.

## 9. Prior art and evidence boundaries

The mathematical ingredients here have established antecedents: [abstract interpretation and lattice fixed points (Cousot & Cousot, 1977)](https://pcousot.github.io/publications.html); [Universal Transformers (Dehghani et al., 2018)](https://arxiv.org/abs/1807.03819); [Deep Equilibrium Models (Bai, Kolter & Koltun, 2019)](https://arxiv.org/abs/1909.01377); and [Lattice Deduction Transformers (Davis et al., 2026)](https://arxiv.org/abs/2605.08605). We are organizing these tools for the repo's mixed recurrent and certified architectures, without asserting a new general theorem about transformer expressivity.

A new architecture will need evidence of meaningful task improvement over matched controls, independent task roots, and a separate causal intervention for its proposed mechanism. The [PPT residual transfer at the pinned research commit](https://github.com/MoralityLabAI/LoopedTransformers/blob/3df3d5efaff77686bcc46190fe01c05dd053da2b/research/lean_cgrt_hrm400/PPT_RESIDUAL_TRANSFER.md) provides a scoped positive example on directed proof residuals; the negative LSAD and AF results constrain how far the present structural theory can be interpreted.
