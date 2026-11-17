# AF-31: self-supported Horn cycles as a reachability obstruction

**Scope.** Elementary finite positive-Horn least-fixed-point theorem, applied to the prospectively registered AF-31 source families. The statements concern exact public incidence and constructive proofs. They do not assert new learned-model capacity or a performance improvement.

## 1. A source-free cyclic trapping invariant

Let \(V\) be a finite set of atoms and \(R\) a finite set of positive Horn rules \((P,h)\), with premises \(P\subseteq V\) and head \(h\in V\). Starting from seed facts \(S_0\), define the standard monotone operator

\[
T_R(S)=S\cup\{h : (P,h)\in R,\;P\subseteq S\}.
\]

The least forward Horn closure is \(S_\infty=\bigcup_{t\ge0}T_R^t(S_0)\).

**Proposition (cyclic support barrier).** Suppose \(C\subseteq V\) is nonempty and

1. \(C\cap S_0=\varnothing\);
2. every rule with head in \(C\) has at least one premise in \(C\):

\[
  (P,h)\in R,\ h\in C\quad\Longrightarrow\quad P\cap C\ne\varnothing.
\]

Then \(S_\infty\cap C=\varnothing\).

**Proof.** By assumption \(S_0\cap C=\varnothing\). Suppose the invariant \(S_t\cap C=\varnothing\) holds. For any rule whose head \(h\) is in \(C\), some premise \(p\in P\cap C\) is absent from \(S_t\). Therefore its full premise set is not contained in \(S_t\), and that rule cannot add \(h\) at step \(t+1\). The old \(S_t\) contains no \(C\)-atom, so \(S_{t+1}\cap C=\varnothing\). Induction on \(t\) proves the claim. \(\square\)

The condition is on **all** producing rules of atoms in \(C\), not merely on whether a cyclic subgraph exists. If an additional rule enters \(C\) entirely from already established facts, the invariant can fail. This is the exact distinction between unsupported mutual recursion and reachable recursive definitions.

## 2. A producer counterexample

Take initially known atoms \(\{0\}\) and two unknown atoms \(a,b\). Add rules \(a\to b\) and \(b\to a\). Every initially unknown atom has a producer; each has one incoming and one outgoing rule-incidence edge. Both remain outside the least fixed point because \(C=\{a,b\}\) satisfies the proposition.

Adding a goal rule \(b\to g\) produces an apparent backward path from \(a\) and \(b\) to \(g\), yet \(g\) remains unreachable. The unconditional producer predicate

\[
  D(v)=\mathbf{1}\{\exists (P,v)\in R\}
\]

is one for \(a,b,g\). A reverse goal-distance calculation ignoring supported premises gives all three finite distances. Consequently, **finite reverse ancestry and existence of a producer do not imply forward Horn reachability**.

This is a constructive logical obstruction to AF-30's local blocked-premise shortcut, which penalized atoms with no producing rule. AF-31 makes its decisive unsupported prerequisites producer-supported so the shortcut predicate becomes silent.

## 3. The two registered AF-31 traps

The source grammar has known roots \(S_0=\{0,1,2,3\}\), positive path \(4\to5\to9\to11\to14\) at depth 5 (with its initial two-root rule), or \(4\to5\to9\to11\to6\to14\) at depth 6. The apparently promising side route uses \((7,10)\to13\) followed by \(13\to14\).

### 2-cycle family (TRAIN, DEV and half of HELD)

The rules \(8\to10\) and \(10\to8\) generate each other's heads. The only source producers for 8 and 10 both depend on a member of \(C_2=\{8,10\}\); no rule has all premises outside this set and head inside. By the barrier proposition, \(C_2\) is absent from every reachable state.

The secondary route to 13 requires atom 10, so neither route to 13 can fire despite the root reaching atoms 7 and 12. The paired negative inserts atom 10 into one required productive premise, which therefore blocks the productive goal chain as well.

### 3-cycle family (unseen in TRAIN/DEV)

The rules \(8\to10\), \(10\to13\) and \(13\to8\) form the candidate trap \(C_3=\{8,10,13\}\). The additional rule \((7,10)\to13\) has a premise in \(C_3\) as required by the proposition. The root can derive 7 but cannot enter \(C_3\).

The alternative GOAL-producing rule \(13\to14\) remains blocked, and the paired negative's productive route depends on 10. The same exact invariant certifies the negative goal as unreachable.

In both cycle families every logical atom \(v\in\{4,\dots,14\}\) has at least one rule with head \(v\). The source generator and independent preflight verify this property for every encoded pair, together with \(C_2\) or \(C_3\) exclusion from exact public forward closure.

## 4. Matched pair logic

Each AF-31 positive and negative graph pair exchanges exactly two same-arity premise incidences. The positive's productive clause consumes a reachable predecessor \(x\in\{4,5,9\}\), and a decoy rule consumes the unreachable 10. In the negative, these two premises are exchanged. Because the cycle trap excludes 10 from every state reachable from the roots, the positive can derive the GOAL through its productive path, while the negative is stopped at its swapped middle clause.

The source-only preflight requires that the physical-ID-preserving directed radius-two incidence around all seven root/marked anchors, the complete per-node in/out degree profile and the legal root action mask be equal between the two members. Swapped premises are selected beyond these anchored induced radius-two views. Physical permutations are shared within a matched pair, so the exact local-null argument from AF-29 applies: every deterministic observer restricted to the registered matched local signature assigns a tied score to its two members, for paired AUC \(1/2\).

The full public Horn incidence provides a sound exact certificate via forward least-fixed-point closure. All neural STOP-only and action-ranker evaluations must compare against that available public computation, with the cost charged.

## 5. Proper inference scope

The trapping theorem allows immediate negative STOP on these synthetic graphs via public closure; it also explains why a reverse path scoring rule can be misled by producer-supported cycles. A trained model that learns to favor the productive path can be tested prospectively against the hand-coded PATH rule, a same-feature untrained linear controller, and the exact closure.

A theorem about the generator cannot establish that a model has learned the invariant. A strong claim requires source-locked heldout evidence of complete legal proofs, calibrated negative STOP for a certificate-free arm and sound control comparisons. AF-31 registers these as separate outcomes. Neither a passing source structural preflight nor complete nonlearned symbolic closure constitutes a neural science PASS.
