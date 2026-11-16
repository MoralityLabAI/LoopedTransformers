# AF-27: learned reachability can reverse the Bellman ranking of legal Horn rules

**Scope:** exact elementary mathematics for a fixed two-step min-plus backup and finite positive-Horn successor graph. The propositions below hold under their explicit formula and regularity assumptions. This document is not a Lean-kernel proof, empirical AF-27 result, or a claim of general neural reasoning capability.

## 1. The original source cost

For a nonterminal state \(s\), the frozen AF-23/AF-25 recurrent scorer produces a reachability estimate \(p(s)\in(0,1)\) and a clipped distance estimate \(\widehat d(s)\in[0,16]\). It defines

\[
V_0(s)=p(s)\widehat d(s)+\big(1-p(s)\big)16,\qquad
C_{\mathrm{STOP}}(s)=16p(s).
\]

For a legal first action \(a\), let \(x=T(s,a)\) and

\[
C(x)=1+\min_{b\in A(x)}V_0(T(x,b))
\]

when \(x\) has a legal continuation. The original two-step rule cost is

\[
B(x)=\min\big\{16p(x),C(x)\big\},\qquad
Q_2(s,a)=-1-B(x).
\]

The exact goal boundary is zero. If \(A(x)=\varnothing\), the source min is just \(16p(x)\). The source root STOP cost is \(-16p(s)\) in its own native value score scale.

The Direct policy supplies a separate action-logit vector. AF-26 demonstrated that choosing legal rules with those logits while retaining the VI-2 STOP gate can change completed-proof outcomes without changing any trained weight. AF-27 asks whether the same policy-level effect persists under new Horn dependency structures.

## 2. Proposition: piecewise reverse sign in the VI-2 reach head

Fix a successor \(x\), hold its distance estimate and its downstream continuation cost \(C(x)\) constant, and vary only its predicted reachability \(p\in(0,1)\). For nonterminal \(x\) with a legal continuation,

\[
B_x(p)=\min\{16p,C(x)\},\qquad
Q_x(p)=-1-B_x(p).
\]

Then

\[
Q_x(p)=
\begin{cases}
-1-16p,&p<C(x)/16,\\
-1-C(x),&p>C(x)/16.
\end{cases}
\]

Within the first branch \(dQ_x/dp=-16\); within the second \(dQ_x/dp=0\). The corner lies at \(p=C(x)/16\) when it belongs to \([0,1]\).

**Proof.** This is the definition of a minimum of one affine function and one fixed constant. The derivative follows on either open branch. \(\square\)

Under the same fixed-distance assumption, the *leaf* source \(V_0\) satisfies

\[
\frac{\partial V_0}{\partial p}
=\widehat d-16\le0.
\]

Thus a higher estimated reach probability **decreases** the leaf cost when \(\widehat d<16\), while an interior STOP-bound value \(16p\) **increases** when \(p\) rises. The numerical effect depends on which value wins the interior min. The second Bellman backup can therefore reward a successor's **spuriously low reachability estimate**, even on a host graph where reachability is invariant under every legal Horn rule.

## 3. Explicit two-rule misranking witness

Consider two legal first-action successors \(x\) and \(y\) that each have the same independently fixed continuation cost

\[
C(x)=C(y)=5,
\]

and neural reachability values \(p(x)=0.1\), \(p(y)=0.9\). Then

\[
B(x)=\min(1.6,5)=1.6,\qquad
B(y)=\min(14.4,5)=5,
\]

so

\[
Q_2(s,x)=-2.6>-6=Q_2(s,y).
\]

VI-2 prefers \(x\), solely because its learned reach estimate is lower and its **interior STOP boundary binds**. The true Horn closure can make **both successors provable**: any legal positive Horn rule adds facts monotonically, preserving eventual reachability when the goal is derivable from the original state.

This is a *numerical source-value* counterexample to the inference that a high VI-2 action score necessarily indicates high successor reachability. It is not an example of a false host-accepted proof, since the host continues to verify every physical rule.

## 4. Exact error decomposition on a provable state

Suppose \(x\) is provable and its true shortest remaining rule cost is \(d^*(x)\in[0,16]\). With its clipped learned distance \(\widehat d\), the source leaf error is

\[
V_0(x)-d^*(x)
=(1-p(x))\big(16-d^*(x)\big)
+p(x)\big(\widehat d(x)-d^*(x)\big).
\]

The identity is exact by expansion and regrouping. Therefore,

\[
\big|V_0(x)-d^*(x)\big|
\le (1-p(x))\big(16-d^*(x)\big)
+p(x)\big|\widehat d(x)-d^*(x)\big|.
\]

The interior STOP branch contributes an additional error when

\[
16p(x)<C(x).
\]

The source can thus misestimate a reachable successor's cost by two distinct mechanisms: distance error inside the reach-weighted leaf and a low reach logit that lets the interior STOP term undercut legitimate continuation. Since legal Horn firings preserve exact eventual reachability, internal changes in true reachability do not explain this piecewise source-value effect.

## 5. Action-gap robustness and why near ties matter

Let \(A(s)\) be the set of host-enabled legal rules, \(Q^*(a)\) their exact negative remaining action costs and \(\widehat Q(a)\) any learned scores on a *common comparable cost scale*. Suppose

\[
\max_{a\in A(s)}|\widehat Q(a)-Q^*(a)|\le\varepsilon
\]

and the exact gap \(\Delta\) between an optimal and any suboptimal legal action exceeds \(2\varepsilon\). Then every maximizer of \(\widehat Q\) is optimal.

**Proof.** For optimal \(a^*\) and suboptimal \(b\),

\[
\widehat Q(a^*)-\widehat Q(b)
\ge Q^*(a^*)-Q^*(b)-2\varepsilon
\ge\Delta-2\varepsilon>0.
\]

The inequality applies to each suboptimal \(b\), so none can maximize learned \(Q\). \(\square\)

This elementary sufficient condition does **not** authorize comparing raw Direct policy logits to VI-2 negative-cost values: they belong to different numerical scales. AF-27 records top-two *legal-rule* margins **within each native source policy**, which identify score sensitivity to finite precision but do not by themselves certify prediction correctness. The independent host BFS is allowed to label first-action optimality **after** the policy has already made its action.

The preceding AF-26 source-identity investigation found examples where action-margin differences below \(10^{-5}\) changed a frozen model's discrete tie-breaking under altered CPU batching. AF-27 therefore reports the exact margin and near-tie count alongside sampled checkpoint forward replays rather than assuming all scores are bitwise stable.

## 6. Computational scope

The registered AF-27 **BFS_NODE_MATCHED** controller expands at most as many distinct physical host states in one decision as the frozen neural two-step scorer evaluates on the same state. This offers a transparent *host-state examination count* comparison. A BFS expansion includes public Horn rule examinations, and a neural node evaluation includes four recurrent network loops, so **equal state counts do not imply equal FLOPs or latency**. The separate BFS64 arm tests whether larger symbolic search can resolve goals that the cheap node-matched controller misses.

The scientific interpretation must distinguish:

1. Whether a native STOP gate chooses correct termination on unprovable inputs.
2. Whether its legal-rule ranker selects a favorable action on solvable inputs.
3. Whether the entire closed-loop policy finishes the proof within a fixed action and inference budget.

AF-27's preregistered outcome addresses the first and third with **Direct's legal-rule ranking held constant**, while its independent margin and Bellman-branch receipts address the second descriptively.
