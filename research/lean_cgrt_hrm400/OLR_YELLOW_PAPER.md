# Obligation Ledger Recurrence
## A verifier-coupled, addressed-memory architecture for compact looped transformers

**Yellow paper v0.1 | 24 November 2026 | MoralityLabAI research draft**

**Review status:** formal architecture specification and adversarial review brief.  
**Implementation anchor:** LoopedTransformers commit [bfdacac](https://github.com/MoralityLabAI/LoopedTransformers/tree/bfdacac96882ea81b9efe11eb85dec46fd68f817/research/lean_cgrt_hrm400).  
**Research branch:** [PR #318](https://github.com/MoralityLabAI/LoopedTransformers/pull/318).  
**Experimental status:** OLR-01B–04D and MeTTa bridges have completed native developmental evaluations. OLR-METTA-05 source certification and fitting completed its CI step; final recall scoring remained in progress at the initial paper snapshot. No general learned-reasoning or architecture-promotion claim.

> **Research decision, 2026-11-24:** The completed 05 native run and Fable's independent neural controls support **REDESIGN**. The explicit-map model closed every native evaluation proof; addressed attention lagged that control and simpler key-value storage. See [OLR-06 redesign and exact-ledger reference](OLR_06_REDESIGN.md). This v0.1 document is retained as the historical specification of the original architecture. The next proposed learned component is an incidence-directed policy over an exact verifier-owned receipt ledger; no 5M-model promotion has been authorized by the evidence.

## Abstract

Obligation Ledger Recurrence (OLR) is a family of recurrent neural controllers for reasoning tasks whose execution is independently checked. It proposes to organize the latent state by persistent obligations rather than by a single undifferentiated residual stream. A weight-tied attention block computes messages over evidence, obligations and current tool events. An address-conditioned write rule preserves memory for obligations that a new receipt does not directly concern. A separate deterministic execution boundary tracks proof states, accepted/rejected actions, budgets, source provenance and terminal certificates. Lean supplies native proof judgments; Hyperon MeTTa filters typed evidence and repair proposals.

The first implementation studies a deliberately partially observed Lean conjunction environment. Balanced paired worlds share their model-visible recall queries but require complementary branch choices. Earlier native guard receipts identify the branches. We formalize an exact memoryless accuracy bound, a sufficient explicit-memory baseline, the implemented selective recurrence, and a proposed generalization to changing AND-OR obligation graphs. Early experiments establish working native rollback and repair interfaces; the evidence for additional recurrent computation is mixed. The proposed scientific contribution is better learned retention and transport of relevant verifier evidence under finite model compute, especially when semantic obligations change across actions. A symbolic receipt table solves the first memory benchmark exactly. OLR must earn its cost on harder tasks with uncertain or changing bindings, not on retrieval alone.

## 1. Design problem and claim boundaries

Consider an agent with limited parameters, repeated internal computation, and access to an independent verifier. The agent can try an action, receive a typed receipt, restore a saved state, or continue with pending goals. An accepted action may create additional obligations rather than finish the task. A later action can depend on a fact discovered earlier, on which branch is still viable, or on whether a witness is in scope.

The architectural question is how to retain these distinctions without continually serializing an expanding history into the transformer context. OLR explores persistent per-obligation slots, a verifier-controlled status ledger, and a shared neural update rule. This is an inductive bias and a controller interface. It does not make the network a proof kernel.

Three separate statements require separate evidence:

1. **Engineering:** a native tool adapter implements typed actions, safe restoration, provenance and exact terminal certification.
2. **Learning:** a trained selective-memory model improves a specified endpoint relative to matched dense recurrence and transparent alternatives.
3. **Architecture:** the learned mechanism improves novel, materially harder tasks at equal resource budgets, with independently replicated evidence of the hypothesized causal pathway.

The present repository has engineering results and exploratory learning results. It has no established third-level result.

### 1.1 Related architectural families

The proposal belongs to an established literature. Neural Turing Machines (Graves, Wayne and Danihelka, 2014) use differentiable addressed memory for algorithmic tasks. Universal Transformers (Dehghani et al., 2018) share a self-attentive computation across recurrent steps. Recurrent Independent Mechanisms (Goyal et al., 2021) selectively update interacting recurrent modules. Slot Attention (Locatello et al., 2020) uses iterative attention to form object-centric slots. Neural Bellman-Ford Networks (Zhu et al., 2021) motivate learned graph propagation; exact AND-OR and min-plus search precede neural versions.

OLR's possible distinct research contribution is the **binding of persistent neural write domains to externally verified, revisable obligations and resource-constrained actions**. Hard addressed writes, tied attention, explicit memory, and symbolic rule matching individually are established ideas. In the first implementation, addresses are supplied by the environment, unlike learned object discovery. Any strong novelty claim requires a systematic comparison with existing modular memory, structured search and theorem-proving architectures.

## 2. Formal environment

Let a verifier environment be

$$
\mathcal E=(S,A,T,\mathcal O,\pi,C,\mathcal K).
$$

Here $S$ is the set of native proof states, $A(s)$ a bounded public action menu, $T(s,a)$ the verifier transition, $\mathcal O$ a typed observation alphabet, $\pi:S\to\mathcal O$ an observation projection, $C$ the action-cost function, and $\mathcal K(s)$ the terminal certificate predicate.

For a tactic $a_t$, the native response contains a verdict $v_t$, a successor state or a rejected proposal, and an audit receipt. A rejected tactic is evidence about that particular attempt in that proof state. It is not a logical refutation of the target proposition. A timeout is censored, not a negative logical label.

A policy sees a history

$$
H_t=(o_0,a_0,e_1,o_1,\ldots,e_t,o_t),
$$

where $o_t=\pi(s_t)$ and $e_t$ is a typed observation of the verifier response. It chooses an action subject to remaining budget $b_t$. The environment owns $s_t$, $b_t$, and the receipt store. A policy cannot set them through a latent prediction.

A typical evaluation objective is

$$
J(\theta)=\mathbb E[\mathbf1\{\mathcal K(s_T)\}
-\lambda\sum_{t<T} C(s_t,a_t)].
$$

Earlier OLR experiments use $\lambda=0.07$ per attempted Lean tactic. OLR-METTA-05 registers whole-proof completion and address accuracy as endpoints and records actual costs separately. Host focus, restore, symbolic queries, independent grading and model compute have distinct meters. They are not literally free operations even where the policy tactic budget excludes them.

### 2.1 Evidence authority and lifecycle

An environment receipt should carry an immutable proof-state identity, action identity, native verdict, source lineage, revision, and toolchain provenance. The current prototype records native transition state hashes and a smaller typed receipt (address, tested branch, Boolean verdict). Proposed extensions use an identity of the form

$$
\operatorname{id}(e)=H(\text{toolchain},\text{root},\text{state},
\text{action},\text{revision},\text{payload}).
$$

Let $\operatorname{Admit}(e,L)$ be a deterministic eligibility rule on the receipt and the authoritative ledger $L$. MeTTa provides a declarative implementation of such a rule. Its answers must be independently checked against the intended contract. It cannot promote an unverified semantic claim into a Lean proof.

The neural state $M_t$ is derived memory. It may predict that a tactic succeeds, but this prediction is never written as a native certificate. Only a fully closed state with an accepted kernel term and the registered axiom policy is eligible for a verified proof. In the present toy studies, the allowed axiom set is empty.

### 2.2 Atomic repair and rollback

A multi-action repair proposes $(a_1,\ldots,a_m)$ from a saved native state $s$. A host creates a speculative continuation $s^{(j)}$ using native transitions and charges each attempted action. The host commits the final state only if the registered terminal or transaction criterion is met. Otherwise it returns the original saved state, with costs retained. For an actual Lean closure macro the criterion includes $\mathcal K(s^{(m)})=1$.

An obligation identity is meaningful only within its root, branch, environment scope and revision. A restore must invalidate evidence that depended on discarded state changes or mark it as historical branch-local evidence. The current 05 environment only restores the same untouched conjunction state after its probes. General state-dependent invalidation remains future work.

## 3. The structural obligation graph (proposed full OLR)

The long-term design represents the proof state by a typed directed hypergraph $G_t=(V_t,E_t)$. Nodes represent scoped goals, hypotheses, candidate witnesses and computational guards. A hyperedge $r$ has a head $h(r)$, an ordered or multiset collection of premises $P(r)$, a positive action cost $c_r$, an applicability status, and a provenance key. Multiple rules with the same head are alternatives.

The following are exact structural reference recurrences for a fully observed, positive-cost Horn fragment. Let $F$ be source facts.

$$
D_0(v)=
\begin{cases}
0 & v\in F,\\
+\infty & \text{otherwise}.
\end{cases}
$$

$$
D_{k+1}(v)=\min\left(D_k(v),\min_{r:h(r)=v}
[c_r+\sum_{u\in P(r)}D_k(u)]\right).
$$

In an acyclic fully specified fragment after enough rounds, this gives the minimum derivation-tree cost under the stated accounting. It counts shared subproofs more than once and is not a general lower bound for Lean tactic calls or a proof-DAG optimum. For Boolean closure:

$$
E_{k+1}(v)=E_k(v)\lor
\bigvee_{r:h(r)=v}\bigwedge_{u\in P(r)}E_k(u).
$$

Unknown computational guards and unsupported dependent terms must remain unknown, not silently dropped from premise sets. Let $\mathcal R_t$ be the public backward-relevance frontier obtained from the active goal and candidate prerequisite edges. One proposed generalization is a forward evidence channel and a backward obligation channel that exchange messages only through identified rules. Certified facts and inferred availability remain different fields.

A possible learned difficulty adjustment is

$$
\widehat c_r=c_r+B\tanh f_\theta(z_r,b_t),
$$

with a positive lower floor for ranking where needed. $B$ is a registered finite correction bound. This is an estimated scoring feature rather than a certified admissible heuristic. A soft minimum can undercut a hard minimum. No correctness theorem is obtained by using a neural score inside the min-plus recurrence.

**Status:** the small structural oracle in [obligation_ledger.py](ppt_gym/obligation_ledger.py) executes exact restricted hypergraph propagation; the complete trainable bidirectional evidence/obligation graph above is a proposal. OLR-METTA-05 implements a smaller addressed receipt-memory cell described next.

## 4. Implemented addressed receipt recurrence

### 4.1 Tensor contract

In OLR-METTA-05, there are $N=5$ address slots, hidden width $d=24$ and four attention heads. At each event, a token

$$
x_t=[k_t,p_t,\beta_t,\nu_t]\in\mathbb R^{11}
$$

contains a four-way one-hot event type (receipt, veto, delay, query), a five-way one-hot address, and branch/verdict signs. A real receipt uses $\beta=2b-1$ and $\nu=2v-1$. Query tokens contain an address but zero branch/verdict fields. Veto and delay tokens contain neither an address nor a verdict. Root identifiers, theorem text and numerical guard operands are excluded.

Let $M_t\in\mathbb R^{N\times d}$ be a persistent latent slot matrix, initialized at zero. The fixed one-hot address vectors $p_i$ are embedded by $W_p$. The event embedding $E_\theta(x_t)$ is prepended to the position-augmented memory.

$$
X_t=[E_\theta(x_t);\,M_t+W_p I_N].
$$

The code uses a single shared attention and feed-forward block with post-residual layer normalization. With rowwise normalization,

$$
U_t=\operatorname{LN}_1(X_t+\tfrac12\operatorname{MHA}_\theta(X_t)),
$$

$$
V_t=\operatorname{LN}_2(U_t+\tfrac12\operatorname{FFN}_\theta(U_t)).
$$

Define $P_\theta(M_t,x_t)$ to be the last $N$ rows of $V_t$ (all candidate slot updates). The feed-forward network has intermediate width $2d$ and GELU nonlinearity. All parameters are shared across all history events and read calls.

### 4.2 Writes

For an admitted receipt at address $a_t$, define the binary addressed mask

$$
w_{t,i}=\mathbf1\{k_t=\mathrm{receipt}\}\mathbf1\{i=a_t\}.
$$

The update is

$$
M_{t+1,i}=(1-w_{t,i})M_{t,i}+w_{t,i}P_\theta(M_t,x_t)_i.
$$

The dense comparator changes only the mask:

$$
w^{\mathrm{dense}}_{t,i}=\mathbf1\{k_t=\mathrm{receipt}\}.
$$

Both variants compute every proposed slot update with identical tensor shapes and tied weights. They differ in where the proposal is committed. Neither writes on veto or delay tokens. A read query does not write. These are exact implementation facts from [olr_metta05_model.py](ppt_gym/olr_metta05_model.py).

### 4.3 Reads

For a recall query $q(a)$, compute a fresh proposal using the saved memory and query token. With a shared two-layer classifier $h_\theta$,

$$
z_a=h_\theta(P_\theta(M_T,q(a))_a),\qquad
\hat b_a=\arg\max_{b\in\{0,1\}} z_{a,b}.
$$

The recalled slot can attend to all other slots. Its value is not restricted to the addressed slot alone. The independent explicit-map MLP receives the relevant tested-branch sign and verdict sign plus an occupancy bit.

The first benchmark trains on three active obligations drawn from the five-address alphabet. Testing on five active obligations increases the number of populated slots and event visits, not the physical slot count or vocabulary. No new address extrapolation is demonstrated. All recalls are computed from the same frozen stored memory before final native execution; there is no learned feedback update after a recall decision.

### 4.4 Parameter and compute budget

For the registered dimensions, the implemented attention model has the following parameter count (by layer dimensions):

| Component | Parameters |
| --- | ---: |
| Event embedding, 11 to 24 | 288 |
| Address projection, 5 to 24 | 120 |
| Multihead self-attention | 2,400 |
| Feed-forward, 24 to 48 to 24 | 2,376 |
| Two layer norms | 96 |
| Classifier, 24 to 24 to 2 | 650 |
| Total | **5,930** |

The explicit-map MLP has 746 parameters by the same calculation. Runtime receipts should confirm these derived counts.

With $L=N+1$ tokens, one block has leading multiply-accumulate cost $O(Ld^2+L^2d)$. Persistent slot storage is $Nd$ scalars. For $K$ occupied obligations, the current schedule has $2K+2$ history events and $K$ reads, hence $3K+2$ block visits: 11 at $K=3$ and 17 at $K=5$. Addressed and dense arms have matched visit count and dominant tensor operations. The hard mask does not itself confer sparse FLOP savings because proposals for every slot are computed. Full profiling should include implementation overhead, memory traffic, verifier time and host-side operations.

For a scalable design, replacing $I_N$ with a compositional address embedding and using sparse graph-local proposals may reduce parameter and attention growth. That is a new experiment, not an achieved property.

## 5. Mathematical properties and limitations

**Proposition 1 (no direct overwrite).** In addressed mode, if $w_{t,i}=0$, then $M_{t+1,i}=M_{t,i}$. Consequently an interval with no admitted receipt addressed to $i$ preserves that stored row exactly in the mathematical transition.

*Proof.* Substitute $w_{t,i}=0$ in the update. Repeated application gives equality across the interval. This also applies to all rows on veto, delay, and non-mutating reads.

This proposition does not imply independent semantic storage. The proposal at another slot can depend on $M_{t,i}$, and an addressed write to $i$ may contain information from every other row. Reads also mix the slots through attention.

**Proposition 2 (paired alias bound).** Suppose two worlds $u,u'$ have identical current recall packets and complementary correct binary branch vectors $y$ and $\mathbf1-y$ over $K$ addresses. Any deterministic policy that reads only the current packets outputs an identical vector $z$ in both worlds. For every address,

$$
\mathbf1\{z_i=y_i\}+\mathbf1\{z_i=1-y_i\}=1.
$$

Therefore its address accuracy across the balanced pair is exactly $1/2$. For $K>0$ it cannot fully complete both worlds, and its pair-level whole-proof completion rate is at most $1/2$ (possibly zero). A randomized memoryless policy using randomness independent of the hidden world has the same expected address bound. This bound relies on the restricted policy projection and does not apply to an agent reading the full Lean contexts.

**Proposition 3 (receipt sufficiency).** For this grammar alone, each address has precisely one of two valid branches; a native guard test of branch $b_i$ returns $v_i=1$ iff it is the valid one. Its winner is

$$
y_i=b_i\oplus(1-v_i).
$$

Equivalently the signed winner is $\beta_i\nu_i$. A deterministic address map retaining one verified bit per obligation solves the recall exactly. This is independently checked by testing the complementary guard and compiling the complete proof; it is not a general inference rule for tactic rejection.

**Proposition 4 (conditional information requirement).** In an idealized version where all $2^K$ independent branch assignments are possible under a fixed query schedule, any deterministic lossless pre-query state must distinguish at least $2^K$ histories, requiring at least $K$ bits of distinguishable state. A probabilistic decoder with independent uniformly distributed labels and average bit error $\epsilon$ requires mutual information at least $K[1-h_2(\epsilon)]$ by the binary entropy bound. This does not establish a learned representation's bit capacity and is not a sample-complexity statement about the finite paired dataset. The current benchmark only guarantees complementary pairs, not exhaustive assignments for each identical schedule.

### 5.1 Causal sensitivity and routing

Write masks preserve untouched slots by construction. They do not guarantee causal locality of later queries. A stronger test holds the native world fixed and changes exactly one verdict in the *model input* history. Let $a^\star$ be the focal query, $j\ne a^\star$ the sham address. Define

$$
\Delta_{\rm target}=\mathbb E[Y(H)-Y(H^{a^\star})],\quad
\Delta_{\rm sham}=\mathbb E[Y(H)-Y(H^j)],
$$

where $Y$ measures focal correctness or full native proof completion with all other branch decisions supplied by the exact receipt map. The difference $\Delta_{\rm target}-\Delta_{\rm sham}$ estimates address-specific sensitivity under this intervention. Flipping a signed verdict changes exactly one input coordinate by absolute magnitude 2. It does not match downstream latent norms or logit margins. The hard route and fixed read already encode locality; the test measures whether the trained function uses that routing, not whether a new semantic feature emerged spontaneously.

A useful later intervention compares a matched local write, an address-permuted write, and a perturbation calibrated to the same final action margin. The EvalMath FSI-01 audit should set the prospective sham envelope and sample size before scoring.

## 6. Native Lean and MeTTa benchmark

For $K$ pending goals $R_i$, a Lean theorem binds

    P Q : Nat -> Prop
    h0 : forall n : Nat, P n
    h1 : forall n : Nat, P n -> Q n

and two rules per obligation,

$$
r_{i,j}:(a_i+b_i=c_{i,j})\to P(c_{i,j})\to Q(c_{i,j})\to R_i.
$$

The final goal is a conjunction of the $R_i$. Exactly one numerical equality per address is true in the registered grammar. The host splits the conjunction, saves the joint state, and performs a two-tactic speculative probe (apply rule, rfl) per address. It restores the untouched joint state after each probe. Hyperon 0.2.10 admits current unique native receipt facts and excludes marked stale synthetic aliases through a match query. The same receipt set is checked by independent Python code. Admission facts are generated by the host; the current MeTTa filter does not independently discover their provenance or solve the equality.

The final proof for a chosen branch uses four charged tactics: apply the rule, rfl, exact h0 _, and exact h1 _ (h0 _). The total successful policy cost is

$$
(K-1)+2K+4K=7K-1
$$

native tactics. Source/independent grading additionally tests the complementary guard at each address, so the grading trajectory uses $9K-1$ tactics. A wrong final branch terminates the policy attempt. The whole-vector decision is frozen before any final proof execution. Identical vectors at a root share a native execution and independently replayed receipt; individual policy rows are not independent native executions.

### 6.1 Frozen allocation and fits

| Split | Paired roots | Worlds | Occupied obligations |
| --- | ---: | ---: | ---: |
| TRAIN | 32 | 64 | 3 |
| Development | 12 | 24 | 3 |
| Load transfer | 12 | 24 | 5 |

Source namespace 20261208, operands 45–60. Paired worlds complement all winners while sharing addresses and schedule. Exact ordered numeric root specifications are disjoint across splits. Three seeds (17,29,43), 160 AdamW updates, batch 16, learning rate 0.002, weight decay 0.005, width 24. The loss averages cross-entropy over source recall positions. Addressed and dense arms have identical initialized state dictionaries, sampler indices and parameter counts. The policy cannot read query-side proof labels during fitting.

The controls are: dense memory; explicit-map MLP; exact receipt lookup; fixed-r0; memory-erased addressed checkpoint; focal receipt flip; nonfocal receipt flip. Exact lookup has the same admitted history information, no neural optimization, and a simpler sufficient representation. It is the correct ceiling for this task.

### 6.2 Acceptance endpoints

Primary developmental contrasts are whole-proof and address-level accuracy for addressed versus dense recurrence and focal correctness for natural versus target and nonfocal edits. A whole proof is successful only with native kernel closure and an independently compiled original theorem. Resample paired-root clusters with model seeds retained within each cluster. Report every seed, actual tool and model visits, failures and complete-pair rates. A 12-pair panel is too small for broad inference. The current registry is open development evidence; no sealed confirmation or FSI-01 pass follows.

## 7. Prior measured evidence

The full study reports linked below are authoritative for sample definitions, exact receipts, and limitations.

| Stage | Recorded result | Interpretation |
| --- | --- | --- |
| 01B | Hard Horn ledger 8/8, pooled evidence 4/8, edge swap 0/8 | Native graph-interface test, zero learned weights |
| 01C | 3-pass vs 1-pass complete pairs: 19/36 vs 23/36 DEV, 12/24 vs 6/24 larger operands; sham 11/24 | Mixed depth effects, small shared-root panels |
| 01D | Fixed-r0 early probe 16/16 vs late probe 8/16 under six-call cap | Deterministic scheduling and rollback benefit |
| MeTTa-01/02 | Native two-call STOP repair; canonical/hashed/full-set labels 21/24,24/24,24/24 | Native curation and action-set interface, no full-set gain over hashed control |
| MeTTa-03 | 3-pass utility 0.584/0.626 vs direct 0.560/0.650 | Paid-evidence selection, weak recurrence attribution |
| MeTTa-04 | 3-pass 43/48 and 38/48 vs direct 20/48 and 23/48 | Dependent proof control; direct openings were better, later decisions failed |
| MeTTa-04D | Natural/zero/sham 55/48/54 of 72 DEV, 63/53/61 of 72 larger operands | Carry erasure effect concentrated in seed 43; strong simple controls remain |

04D used 912 independently replayed native episodes and 48 axiom-free standalone declarations. The source-calibrated likelihood reader with scripted continuation exceeded the natural recurrent controller's utility on both panels (0.61333 vs 0.47806; 0.58042 vs 0.56875). The historic majority sign error was documented rather than silently replacing its old results. Larger operand ranges in 03–04D are hidden from the model, so those panels do not establish neural-input out-of-distribution generalization.

The 05 run is designed to resolve a sharper question: whether a hard addressed write bias improves a fixed-cost memory task over dense tied recurrence, where the prior stage observation alone cannot solve the task. It deliberately makes the information deficit explicit rather than asking whether a neural model can rediscover a deterministic schedule.

**Snapshot at writing:** source phase completed successfully in [workflow 45281855683](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45281855683). Query scores are not inferred from this status. See the eventual run artifact and results report before citing performance.

## 8. Beyond bit recall: proposed architecture extensions

A small addressed-memory benchmark can saturate even if the architecture has little utility for real proof search. The next experiment must create a reason to learn beyond exact storage. The following staged changes are proposed, subject to prospective registration.

### 8.1 Scoped, revisable obligations

Replace the fixed address alphabet with a registry assigning identities to scoped goal occurrences, with explicit parent/child incidence and branch/revision lifetimes. An action may split, discharge, create or invalidate goals. Store native facts, estimated applicability and unresolved obligations separately. Historical evidence after restore is admissible only for the same scope or a soundly justified transport relation.

For branch-dependent state updates, define a receipt validity predicate $V(e,s)$ supplied by the environment. A learned slot remains useful only where its evidence dependencies remain valid. Treat stale memory invalidation as a deterministic rule and compare it to learned invalidation with a safety membrane.

### 8.2 Partial evidence and unknown bindings

Make the target identity a learned retrieval problem: a receipt refers to a metavariable, source theorem or witness whose relation to a later goal must be inferred from public types and checked by Lean. Permit multiple plausible matches, inconsistent testimony, shared witnesses, and observations that are genuinely censored. Exact symbolic controls should receive every public fact. For example, a native tactic may unify a witness with an abstract goal only after one of several competing assignments. This creates a potential role for semantic attention and recurrent search beyond a fixed address table.

### 8.3 Bidirectional AND-OR messages and costed compute

Use rule tokens with explicit premise incidence rather than averaging all facts entering a head. Maintain separate status and value heads. A forward pass propagates candidate evidence, a backward pass marks obligations and alternate branches; tie the update across recurrent computation steps. Distinguish the internal refinement clock from tool events. Add a compute allocator only after comparing fixed visits at equal inference FLOPs. Measure whether a learned semantic heuristic reduces real proof search cost relative to exact structural and direct learned readers.

### 8.4 Action-set objectives and bounded repairs

The MeTTa bridge already includes the full optimal-action-set objective:

$$
\mathcal L_{\rm set}=-\log\frac{\sum_{a\in A^\star(s)}e^{z_a}}
{\sum_{a\in A_{\rm legal}(s)}e^{z_a}}.
$$

Here the source action set contains equally good feasible continuations according to independently checked total cost and completion. Use the canonical and hashed-singleton controls with exactly the same state presentations and source budgets. Avoid inventing positive labels from incomplete searches; missing evidence is censored. The 05 memory benchmark has unique winners and uses cross-entropy, so it does not test this loss.

### 8.5 Scale gate

A useful 5M-parameter prototype requires evidence on a harder native family where an explicit history table plus exact solver leaves meaningful headroom. Compare matched-size ordinary tied transformers, a recurrent memory architecture without addressing, content-addressed memory, explicit key-value lookup plus direct reader, and a strong solver with an amortized learned semantic scorer. Report active FLOPs, tool calls, peak memory, wall time and number of source tokens. No model scale-up is warranted solely by an associative-recall advantage.

## 9. Adversarial review checklist and required follow-up

1. **Mathematical correctness:** verify the tree-cost conditions, paired bound, sufficient statistic and conditional entropy statement. Look for misuse of rejection as refutation, proof DAG vs tree costs, and implicit oracle access.
2. **Actual architecture:** rederive dimensions and the number of physical and recurrent blocks from code. Distinguish direct write isolation from semantic independence. Is the hard addressed prior simply a key-value table with redundant attention?
3. **Counterfactual validity:** check whether query equality holds byte-for-byte, whether test edits change exactly one verdict, and whether using exact completions on nonfocal slots hides whole-policy failures.
4. **Baselines:** require exact map, explicit-map MLP, a direct sufficient-statistic readout, dense recurrence with equal compute, address-permuted mask, content-addressed and sparse alternatives. Include a direct root ranker plus optimal scripted continuation from 04D.
5. **Protocol:** check native event identity, source custody, paired root grouping, independent compilation, policy/grade meters, reproducibility and exact toolchain hashes. Audit whether the training bank duplicates effective input histories across roots and splits; distinct arithmetic identities alone cannot prevent equivalence in the deliberately masked representation.
6. **Difficulty:** require longer unknown-address or dependency shifts, repeated revisions, conflicting evidence, scope-sensitive restores and novel obligation compositions. The current delay/veto tokens never change stored memory; hence this is not a test of retention under intervening writes at arbitrary delay lengths.
7. **Decision:** identify the smallest prospective result that would justify a new architecture claim, and the strongest result that would rule it out. Do not recommend scaling until the simplest sufficient statistic and structural solver controls are defeated on a meaningful resource endpoint.

## Appendix A. Implementation and reproducibility anchors

Pinned 05 source commit bfdacac96882ea81b9efe11eb85dec46fd68f817:

- [Native environment and compiler](ppt_gym/olr_metta05_native.py)
- [Tied memory architecture](ppt_gym/olr_metta05_model.py)
- [Source/evaluation campaign](ppt_gym/olr_metta05_campaign.py)
- [Protocol tests](ppt_gym/tests/test_olr_metta05.py)
- [Registered design](OLR_METTA_05_REGISTRATION.md)
- [CI workflow](../../.github/workflows/olr-metta-05.yml)

From research/lean_cgrt_hrm400 with Lean 4.19.0 lgym, PyTorch 2.5.1 CPU and Hyperon 0.2.10 installed:

~~~sh
python -m unittest discover -s ppt_gym/tests -p test_olr_metta05.py -v
python -m ppt_gym.olr_metta05_campaign freeze --out runs/olr05-bank
python -m ppt_gym.olr_metta05_campaign fit --bank runs/olr05-bank --out runs/olr05-fit --steps 160
python -m ppt_gym.olr_metta05_campaign score --bank runs/olr05-bank --fit runs/olr05-fit --out runs/olr05-eval
~~~

Create-only directories. Inspect manifests before model promotion, and independently check the recorded native and compiler evidence.

## References

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401), Graves, Wayne, Danihelka (2014).
- [Universal Transformers](https://arxiv.org/abs/1807.03819), Dehghani et al. (2018).
- [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893), Goyal et al. (ICLR 2021).
- [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055), Locatello et al. (NeurIPS 2020).
- [Neural Bellman-Ford Networks](https://proceedings.nips.cc/paper_files/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html), Zhu et al. (NeurIPS 2021).
- OLR internal reports: [original architecture](OLR_01_OBLIGATION_LEDGER_RECURRENCE.md), [structural investigation](OLR_01_INVESTIGATION.md), [01B](OLR_01B_NATIVE_PREFLIGHT.md), [01C](OLR_01C_COMPUTE_ABLATION.md), [01D](OLR_01D_NATIVE_RESULTS.md), [MeTTa bridge](OLR_METTA_BRIDGE.md), [03](OLR_METTA_03_RESULTS.md), [04](OLR_METTA_04_RESULTS.md), [04D](OLR_METTA_04D_CARRY_RESULTS.md).
- DataGoblin: [native MeTTa curation study PR #46](https://github.com/MoralityLabAI/DataGoblin/pull/46), [action-set and macro reference protocol PR #57](https://github.com/MoralityLabAI/DataGoblin/pull/57).

## Appendix B. Fable reviewer assignment

You are the independent architecture and mathematical reviewer of OLR, not an implementation agent or advocate for the design. Read the paper and the pinned 05 source files first. Inspect the actual run receipts if complete. Treat all reported experiments as exploratory; do not infer success from a green CI wrapper alone.

Produce a referee report with: (i) exact mathematical errors or missing assumptions; (ii) a source-code audit mapping each claimed architecture component to actual implementation or proposal; (iii) prior-art nearest neighbors and the narrowest defensible novelty claim; (iv) the strongest cheap falsifying baseline; (v) a repaired formal architecture with explicit state, evidence and budget semantics; (vi) a prospective CPU experiment designed to distinguish learned semantic transport from hard addressing or a deterministic table. Include a minimal counterexample if you find a flaw. Prefer a decisive, inexpensive experiment over a larger model. Specify primary endpoints, independent sampling unit, seed plan, compute and verifier budgets, and intervention controls. Separate mathematical proof, observed engineering evidence, statistical evidence, and conjecture. Give a recommendation of STOP, REDESIGN, or PROCEED WITH A SPECIFIC GATE, with precise reasons and no new empirical claims absent receipts.

