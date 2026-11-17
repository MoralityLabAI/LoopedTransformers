# AF-32: exact proof-budget sensitivity to goal-visible but nonprogressing Horn actions

**Source-only mathematical note, registered before AF-32 scored model evaluations.** The statements are about positive-Horn proof length in the explicitly stated action semantics, with physical-host and independent bitset-BFS tests on the AF-32 matched pairs.

## Horn action distance

Let \(V\) be a finite set of atoms, \(R\) a finite set of positive definite Horn rules \(P\Rightarrow h\), and \(g\in V\) the marked goal. A rule is *legal* in a known-state \(S\subseteq V\) precisely when \(P\subseteq S\) and \(h\notin S\). Its fired successor is \(S'=S\cup\{h\}\). One legal rule firing is one unit of proof work, irrespective of premise arity. STOP is a separate terminal decision and proves nothing.

Write \(d_R(S)\) for the minimum number of legal rule firings needed to reach any \(S'\ni g\), or \(+\infty\) if no such state exists. This is the host's exact shortest-firing distance. A controller's policy is scored with an external budget \(b\); the controller sees the public physical incidence and current known bits, but is never given \(d_R\) or \(b\).

### Proposition 1 (one-action proof-cost decomposition)

For any legal action \(a=(P,h)\) from a reachable-to-goal state \(S\) with \(d_R(S)=d>0\),

\[
d_R(S\cup\{h\})\in\{d-1,d\}.
\]

If \(d_R(S\cup\{h\})=d\), **every proof beginning with \(a\) requires at least \(d+1\) fired rules**. Thus a strict budget \(b=d\) necessarily rejects its completed proof, whereas the historical slack budget \(b=d+1\) can accommodate one such nonprogressing choice before a shortest remaining derivation.

**Proof.** The least-cost proof from \(S\cup\{h\}\) has length \(d'\le d\), because adding a positive Horn fact cannot remove a legal path to \(g\): along any previous proof one may omit redundant firings whose head is already known, and all other rule prerequisites remain satisfied. On the other hand, firing \(a\) and then following a shortest proof from its successor gives a proof from \(S\) of length \(1+d'\). Minimality gives \(d\le 1+d'\), or \(d'\ge d-1\). As both distances are nonnegative integers, \(d'=d-1\) or \(d\). In the second case every proof constrained to start with \(a\) takes at least \(1+d\) firings. \(\square\)

A strict-budget proof is therefore possible if and only if each chosen legal action decreases the exact remaining minimum distance by one, until the goal is proved. This is an *evaluation characterization*; it does not authorize using the distance oracle during model inference.

### Proposition 2 (independent distractor tax)

Let \(S_0\) have finite shortest goal proof length \(d\), and let \(a_1,\ldots,a_k\) be distinct currently legal actions whose fired heads remain irrelevant to every shortest productive derivation. More precisely, suppose for each \(j\) the facts produced by any subset of \(a_1,\ldots,a_j\) leave \(d_R\) equal to \(d\), and these actions remain legal in that order. Then every policy that fires all \(k\) actions before the productive derivation requires at least \(d+k\) fired rules.

**Proof.** After the \(k\) distractors, the exact distance remains \(d\) by hypothesis. The prefix has length \(k\); the suffix requires at least \(d\). \(\square\)

This condition is checkable using the exact independent host BFS for source-only validation. It is not asserted for arbitrary goal-visible edges: some alleged distractions can enable a shorter productive proof, in which case the hypothesis fails.

## AF-32: two independently goal-visible root distractions

Every registered reachable AF-32 case starts from \(S_0=\{0,1,2,3\}\) and has exactly one productive root rule concluding atom 4. The productive chain continues \(4\to5\to9\to11\to14\) in depth 5, or \(4\to5\to9\to11\to6\to14\) in depth 6. The source positive shortest length \(d\) is independently checked to be 5 or 6 as specified.

Two distinct *enabled at root* rules conclude atoms 7 and 12, with root premises drawn only from \(S_0\). Each is goal-visible under unconditional reverse incidence: there are clauses

\[
(7\wedge10)\Rightarrow13,\qquad(12\wedge10)\Rightarrow13,\qquad13\Rightarrow14.
\]

Hence both candidate heads have a two-clause apparent route toward 14. The obstacle is the required atom 10. It has a producing rule, yet belongs to an unsupported cycle: either \(8\to10\to8\), or \(8\to10\to13\to8\) where the middle edge is conditional on the root-derivable fact 7. Every producing rule for that cyclic source set has at least one premise inside it; no root fact belongs to it. By the cyclic barrier theorem of AF-31, atom 10 never enters the least forward fixed point.

Because neither 7 nor 12 can contribute to the productive proof path, both are independently legal **nonprogressing** actions on every reachable source case, and the exact oracle verifies

\[
d_R(S_0\cup\{7\})=d_R(S_0\cup\{12\})=d_R(S_0)=d.
\]

The host also verifies the two action slots are distinct physical rules, the productive root action is legal, and the two distractors are unique goal-visible branches with their own reverse incidences. Thus a policy that picks either of these first must fail strict completion, and a policy that first picks both requires at least \(d+2\), exceeding even the historical \(d+1\) budget.

All noninitial logical atoms 4..14 retain at least one producer. The root-two-diversion test therefore shifts difficulty from AF-30's producerless shortcut to real cyclic support and **exact proof-budget sensitivity**.

## Matched local observations and limits of inference

AF-32 positive/negative graph pairs swap exactly two unary premises between one productive clause and a decoy clause. Each pair shares the same initial known facts, root legal-action mask, physical atom/rule permutation, complete directed degree sequence, and *physical-ID-preserving induced directed radius-two neighborhoods* of four root facts plus marked atoms 6,13,14.

As in AF-29, every function restricted to this exact seven-anchor radius-two signature assigns identical scores to the two members of each pair, yielding matched score AUC \(1/2\) with ties counted half. The full public incidence remains available to all policy arms. Public Horn closure exactly solves positive and negative reachability, with its work charged.

The new strict score is a source-provable diagnostic of choosing proof-progressing actions; a model that achieves strict successes also must outperform the strong public PATH and ancestry baselines on the **same physical graph and budget** for the registered neural advantage criterion. Positive-only strict proof counts say nothing about learned negative STOP if a separate symbolic certificate supplies termination.
