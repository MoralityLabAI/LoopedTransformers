# AF-24B: Additive Horn proof-cost overcounting from shared subproofs

**Status:** source-locked mathematical analysis of the AF-24B graph family. This is a proof about an explicitly constructed finite Horn DAG, with empirical structural preflight receipts. It is **not a Lean-kernel certificate**, nor a claim that neural reasoning transfers.

## A shared-prefix family

Let two initially available facts jointly enable a unique chain of \(k\) legal rule firings to an intermediate fact \(P\). From \(P\), one rule produces \(U\) and another produces \(V\). The final Horn rule \((U\land V)\to G\) proves the goal. Suppose a separate path of \(L\) rule firings concludes \(G\) directly and has disjoint intermediate facts, so the alternative cannot share work with the productive prefix.

The exact shortest proof length is

\[
d^*(G)=\min(k+3,L).
\]

Indeed the shared derivation fires \(k\) prefix rules, then one rule for each needed subgoal, then the conjunction rule. The independent alternative is an \(L\)-firing chain. Positivity prevents an unenabled shortcut.

Consider an *additive derived-atom cost* recurrence, starting with zero cost for known facts:

\[
A(h)=\min_{r:\operatorname{head}(r)=h}
\left(1+\sum_{p\in\operatorname{prem}(r)}A(p)\right).
\]

Under the stated disjointness and unique-derivation assumptions,

\[
A(U)=A(V)=k+1,\quad
A(G\mid U\land V)=2k+3,\quad
A(G\mid \text{alternative})=L.
\]

The shared \(k\)-step prefix appears in **both subgoal costs**, although executing its rules once suffices to derive both subgoals. The additive recurrence therefore overcounts the shared route by exactly \(k\) rule firings:

\[
(2k+3)-(k+3)=k.
\]

### Misranking obstruction

When

\[
k+4<L<2k+3,
\]

the additive rule selects the alternative direct-goal derivation of length \(L\), although the true shortest proof requires only \(k+3\) actions and the experiment allows **at most \(k+4\)**. The chosen witness cannot finish within the original shortest-plus-one budget. Such integers \(L\) exist whenever \(k\ge4\).

For \(k=3\), the boundary case \(L=8\) already meets the failure condition with \(L=k+5<2k+3=9\). More generally, the sufficient condition is

\[
k+5\le L<2k+3
\]

for integer lengths. This interval is nonempty for \(k\ge3\).

For \(k=2\), the two additive scores **tie** when \(L=7\), whereas the true shortest proof costs \(5\) and the alternative costs \(7\). If the alternative's physical goal-rule slot is numerically smaller, the source-frozen tie-break selects the 7-step path, which exceeds the budget \(6\). This is the AF-24B depth-five counterexample.

### Exact AF-24B instantiations

| Frozen positive family | Shared prefix \(k\) | Exact best proof \(k+3\) | Alternative \(L\) | Additive shared | Additive alternative | Budget |
|---|---:|---:|---:|---:|---:|---:|
| Held depth 5 | 2 | 5 | 7 | 7 | 7, tie to alternative | 6 |
| Held depth 6 | 3 | 6 | 8 | 9 | 8, selects alternative | 7 |

Both structures use two goal-concluding Horn rules, with physical rule-slot tie-breaking controlled prospectively by the registered generator. [Source-only structural CI #43822626726](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43822626726) independently reconstructed shortest proof lengths and found **0/24** additive-closure completions in the shared-overcount family at each held-out depth.

## Why this matters for recurrent proof evaluation

The least-fixed-point **reachability** question stays easy: positive Horn closure determines whether \(G\) eventually follows, and the simple symbolic controller correctly recognizes all unprovable goals in this family. The **minimum number of rule firings** differs because distinct subgoals may reuse previously derived facts.

A learned value head, a graph-aware heuristic, or a bounded search controller must model **shared computation** to avoid the additive double-count. A recurrent network could conceivably learn a representation of reusable prefix states, but this source-only construction offers no evidence that it has done so.

The AF-24B frozen original-model test measures that question separately, using the identical AF-23 checkpoints and unchanged Bellman scoring. Its preregistered verdict is determined only by audited proof completion against stronger graph-aware controls across all eight seeds.

## Proof scope

The derivation assumes two source-independent goal routes, the specified shared prefix, one firing per Horn rule, and no other productive shortcut. The AF-24B generator's incidence is a particular exact finite instance, not a general complexity characterization of Horn proof search. Tactics such as symbolic forward BFS or dynamic set-valued cost estimation can account for reuse directly, and their computation should be measured.
