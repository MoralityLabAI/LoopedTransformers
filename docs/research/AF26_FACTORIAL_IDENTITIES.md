# AF-26: exact native-gate policy factorization and its interaction limit

**Mathematical scope:** finite host-controlled Horn proof states with deterministic legal rule firing, two fixed action scorers and stable physical rule-slot tie breaking. This note proves two elementary policy identities and gives an explicit obstruction to interpreting one-factor swaps as additive causal mediation. It asserts **no Lean-kernel verification**, neural generalization, or statistically independent repeated graph sample.

## 1. Legal rule rank and native STOP gate

Let \(S\) be the host state set, \(A(s)\) its nonempty finite set of legal Horn rule slots, and \(\sigma\) the STOP slot. Let \(q_D(s,a)\) and \(q_V(s,a)\) denote the two original Direct and VI-2 score vectors, evaluated from **identical fixed model weights** at the same \(s\). Both scalars are native to their own source policy. Let ties break toward the least numbered physical slot, with \(\sigma\) last.

For source \(m\in\{D,V\}\), define

\[
H_m(s)=\arg\max_{a\in A(s)\cup\{\sigma\}}q_m(s,a),\qquad
R_m(s)=\arg\max_{a\in A(s)}q_m(s,a)
\]

and the Boolean native STOP gate \(G_m(s)=[H_m(s)=\sigma]\).

For independently selected gate source \(g\) and rule-ranking source \(r\),

\[
\pi_{g,r}(s)=
\begin{cases}
\sigma, & A(s)=\varnothing\ \text{or}\ G_g(s),\\
R_r(s), &\text{otherwise}.
\end{cases}
\]

The rule-rank source never contributes its STOP logit to the gate-source scores. Hence the policy makes **no cross-scale comparison** between, for example, Direct's raw policy logits and VI-2's cost-derived negative value scores.

### Proposition 1. Exact original-policy action recovery

With the same enabled mask, score vector and deterministic tie break,

\[
\pi_{D,D}(s)=H_D(s),\qquad \pi_{V,V}(s)=H_V(s).
\]

**Proof.** If the legal set is empty both policies return STOP. Otherwise, if the relevant native source selects STOP, the corresponding gate is true and the factored policy selects STOP. If it chooses a rule, that source's highest-scoring legal rule is precisely its full-vector selected action, because all other legal rules have lower priority under the same tie break. The factored policy selects that identical rule. \(\square\)

### Corollary 1. Full-trajectory identity

Assume an identical initial host state, the same fixed model and its exact score values at every state, and deterministic host successor transitions. Then \(\pi_{D,D}\) follows exactly the original Direct rollout and \(\pi_{V,V}\) follows exactly the original VI-2 rollout until termination or the common firing budget.

**Proof.** Proposition 1 gives identical first actions. Determinism gives identical successor states. Repeatedly applying Proposition 1 at the common successor yields the same next action. Induction continues to the first terminal state or budget exhaustion. \(\square\)

**Numerical qualification.** A CPU implementation can produce slightly different floating-point scores for the same model/state under different surrounding batch shapes. Such an implementation has violated the premise of *identical score values*. AF-26 therefore executes DD and VV through AF-25's original three-mode grouped batching path and checks exact per-seed proof-completion parity against the immutable prior aggregate. The blocked first hosted attempt is retained as evidence of this issue.

## 2. Factorial proof-outcome contrasts

Let \(Y_{gr}(c,\theta)\in\{0,1\}\) be whether the complete rollout of \(\pi_{g,r}\) finishes case \(c\) with fixed checkpoint \(\theta\) and original source-bound proof budget. It is a **closed-loop potential outcome**: changing the first action changes which neural states are visited and scored at later steps.

For a fixed preregistered set \(C\) of case-and-checkpoint pairs, write

\[
N_{gr}=\sum_{(c,\theta)\in C}Y_{gr}(c,\theta).
\]

The two single-factor gains relative to original VI-2 are

\[
\Delta_{\rm STOP}=N_{DV}-N_{VV},\qquad
\Delta_{\rm RULE}=N_{VD}-N_{VV}.
\]

The direct–VI-2 gap and factorial interaction are

\[
G=N_{DD}-N_{VV},\qquad
I=N_{DD}-N_{DV}-N_{VD}+N_{VV}.
\]

These quantities satisfy the identity

\[
G=\Delta_{\rm STOP}+\Delta_{\rm RULE}+I.
\]

That identity is exact algebra; it does **not** imply that the two swaps recover independent shares of a unique causal neural computation. The mixed policies visit different host states and create interaction.

## 3. A two-step counterexample to single-swap recovery

Let a finite deterministic Horn host begin in state \(s\), where actions \(a\) and \(b\) are legal and produce distinct states \(s_a,s_b\). Each successor has one further legal goal-producing rule \(g_a\) or \(g_b\), respectively. A goal is reached by firing the corresponding second rule.

Suppose the two original source policies have these choices:

| State | Direct native STOP | VI-2 native STOP | Direct rule preference | VI-2 rule preference |
|---|---|---|---|---|
| \(s\) | Continue | Continue | \(a\) | \(b\) |
| \(s_a\) | Continue | STOP | \(g_a\) | \(g_a\) |
| \(s_b\) | STOP | STOP | \(g_b\) | \(g_b\) |

The factorial complete-proof outcomes are

\[
Y_{DD}=1,\qquad Y_{DV}=Y_{VD}=Y_{VV}=0.
\]

Indeed DD selects \(a\), then \(g_a\), reaching the goal. DV selects \(b\) under the VI-2 rule ranking, then its Direct STOP gate terminates; VD selects \(a\), then its VI-2 gate terminates; VV selects \(b\), then terminates.

Consequently

\[
G=1,\quad\Delta_{\rm STOP}=0,\quad
\Delta_{\rm RULE}=0,\quad I=1.
\]

**Neither single-factor swap recovers even one proof**, although combining the two Direct decisions recovers a complete proof. The AF-26 preregistered \`NEITHER\` attribution outcome therefore does not prove that either neural head is irrelevant. The interaction and per-case paired transitions must also be reported.

The choice table can be realized by finite score vectors and positive Horn rules with different goal-reaching heads and a source-conditioned STOP action. It is an abstract finite-horizon existence witness, not an observation from AF-25 or AF-26.

## 4. Limits of forced continuation

Define \(\pi_{N,r}\) to ignore a model STOP preference while **at least one legal rule exists**, following \(R_r(s)\). For a reachable goal, this can remove premature neural STOP actions. For an unreachable goal with an enabled but irrelevant rule, it must initially fire that rule, preventing a correct immediate STOP. Therefore a higher proof-completion count under \(\pi_{N,r}\) is insufficient evidence of better reachability calibration.

AF-26 reports correct initial STOP on unprovable cases for **all six arms**, together with first-step and later premature STOP on provable cases. Its science conclusion is restricted to a registered discrete policy-level intervention on fixed synthetic finite Horn graphs.
