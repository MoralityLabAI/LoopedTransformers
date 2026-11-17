# AF-29: exact radius-two local indistinguishability with opposite global Horn entailment

**Proof scope:** finite positive-Horn rule systems with directed rule-premise incidence, fixed known facts, exact host reachability and an observation restricted to a registered local graph projection. The arguments below are mathematical theorems and finite source-verified witnesses. No Lean-kernel proof or new neural search capability is claimed.

## 1. An exact two-edge-swap witness

Fix an atom set containing initially true facts \(I=\{0,1,2,3\}\), a productive atom \(5\), unreachable decoy atom \(10\), a goal-path intermediate \(9\), a final marked subgoal \(6\), an irrelevant atom \(7\), and a goal \(14\).

Let a productive positive-Horn path derive \(5\) from \(I\) in exactly two rule firings, followed by a one-premise rule to \(9\), then one or two downstream rules from \(9\) to marked subgoal \(6\), and finally a goal-concluding rule from \(6\), possibly with an already known fact. All other clauses are chosen so that fact \(10\) is unreachable, and a second goal-concluding path also requires \(10\).

Consider two systems sharing the complete initial set, rule head map and all rules except these **two premises**:

\[
\begin{array}{c|cc}
 & r_{\mathrm{productive}}\to 9 & r_{\mathrm{decoy}}\to7\\ \hline
\mathcal H_+ & 5\to9 & 10\to7\\
\mathcal H_- & 10\to9 & 5\to7
\end{array}
\]

Both differences exchange the same two premise edges among two otherwise identical one-premise rules. Since \(5\) is derivable and \(10\) is not, \(\mathcal H_+\) can derive the goal, whereas \(\mathcal H_-\) cannot. The alternative goal route remains blocked by \(10\) in both cases.

The AF-29 source generator implements this swap for four-step, five-step and six-step proof families; it verifies the positive shortest proof length with two independent algorithms and checks the negative graph's full least-fixed-point closure.

## 2. Lemma: unchanged directed in/out degrees

**Lemma 1.** If two directed atom-to-rule premise edges \((x,r)\), \((y,q)\) are replaced by \((y,r)\), \((x,q)\), while every rule head and every other incidence stays fixed, then every atom's and every rule's premise/head in/out degree is unchanged.

**Proof.** Atom \(x\) loses one premise-use incidence at \(r\) and gains one at \(q\). Atom \(y\) undergoes the reverse exchange. Rules \(r\) and \(q\) each lose one premise input and gain one premise input. No other node or head incidence changes. The directed degree vector is therefore identical, node by node. \(\square\)

The argument does not assume that \(x\) and \(y\) have similar semantic roles or are exchangeable under a graph automorphism. Full directed graph information still reveals the swap.

## 3. Radius-two local observation and blindness

Represent the public Horn program as a bipartite incidence graph with atom nodes and rule nodes. A premise edge is directed atom-to-rule; a head edge is directed rule-to-atom. Graph distance for defining the local neighborhood ignores edge direction, while the observation keeps directed edge labels, actual physical node identifiers and which atoms are initially known.

Let \(A\) be the seven registered anchors: the four original known atoms and marked roles 6, 13 and 14. For each anchor \(a\in A\), write \(L_2(\mathcal H,a)\) for the **identity-preserving directed induced incidence graph** on all nodes at bipartite distance at most two from \(a\). Define the joint observed local signature

\[
L_2(\mathcal H) =
\big(L_2(\mathcal H,a)\big)_{a\in A}.
\]

The exact AF-29 source gate verifies for *every* registered positive/negative pair that

\[
\boxed{L_2(\mathcal H_+)=L_2(\mathcal H_-)}.
\]

It also verifies the exact equality of the root-enabled physical rule mask and the entire per-node directed degree sequence. Physical atom and rule permutations are shared within each pair, so the equality does not arise from discarding original physical identifiers.

The swapped premise edges are placed beyond the two-edge local radius around all seven anchors. During the initial source-only preflight, the four-firing templates originally exposed the productive input under marked subgoal 6. That **real structural leak failed the gate**. The generator was corrected before any model feature collection by placing the swapped productive head at an unmarked intermediate atom and retaining the same original exact proof depth. The failure and dated protocol amendment remain public.

## 4. Theorem: a local classifier cannot distinguish balanced pairs

Let \(\mathcal D=\{(\mathcal H^i_+,\mathcal H^i_-)\}_{i=1}^n\) be any finite set of opposite-labeled pairs satisfying \(L_2(\mathcal H^i_+)=L_2(\mathcal H^i_-)\) for every \(i\). Let \(f\) be any **deterministic classifier that receives only** \(L_2(\mathcal H)\), with no pair ID or global graph information.

**Theorem 2 (exact local classification ceiling).** On the balanced sample containing both members of each pair, \(f\)'s exact classification accuracy is \(1/2\).

**Proof.** The two members of each pair yield identical input \(L_2\), hence the deterministic classifier produces identical predicted labels. Since one member's true label is positive and the other's negative, the classifier predicts exactly one of the two labels correctly in each pair. Summing over \(n\) pairs gives \(n\) correct labels among \(2n\) total labels. \(\square\)

If \(f\) instead gives a deterministic scalar STOP score, its paired AUC, defined to award one win when the unreachable member's STOP score is strictly higher, zero when lower and half credit for ties, is also exactly \(1/2\). Both members of every pair have identical input and therefore identical score, producing exactly \(n\) ties.

This is a **local-information impossibility result**, not a limitation on the original recurrent model, which receives the entire public Horn rule-incidence graph and can in principle distinguish the two systems.

## 5. Perfect matched-pair ordering does not imply a useful global threshold

Suppose a STOP score \(s(\mathcal H)\in\mathbb R\) correctly orders all paired labels, meaning \(s(\mathcal H^i_-) > s(\mathcal H^i_+)\) for every \(i\). Its pairwise discrimination AUC is then exactly \(1\).

Define

\[
P_{\max}=\max_i s(\mathcal H^i_+),\qquad
N_{\min}=\min_i s(\mathcal H^i_-).
\]

**Proposition 3 (one common threshold).** A threshold \(t\) that chooses CONTINUE for **every reachable** pair member and STOP for **every unreachable** pair member exists if and only if

\[
\boxed{P_{\max}<N_{\min}.}
\]

**Proof.** All reachable scores satisfy \(s(\mathcal H^i_+)<t\) exactly when \(P_{\max}<t\). All unreachable scores satisfy \(s(\mathcal H^i_-)\ge t\) exactly when \(t\le N_{\min}\). Both conditions can hold precisely when the interval \((P_{\max},N_{\min}]\) is nonempty. \(\square\)

For example, two correctly ordered pairs can have scores \((s_+,s_-)=(0.1,0.2)\) and \((0.8,0.9)\). Paired AUC is \(1\), yet \(P_{\max}=0.8>N_{\min}=0.2\), so no single threshold can correctly classify all four roots.

A high or even perfect paired AUC is thus evidence of **within-pair global structural sensitivity**. Root-level STOP accuracy at one development-selected threshold additionally depends on the alignment of score ranges *between unrelated pairs and held-out grammars*. Complete proof success demands avoiding premature STOP at later reached states as well. AF-29 preregisters both the matched-pair ranking criterion and separate absolute STOP/proof gates to retain those distinctions.

## 6. Source-heldout experimental interpretation

AF-29 trains a light six-input logistic STOP gate using original frozen recurrent outputs and source-visible public features from **new source-labeled TRAIN grammars**, while keeping the original recurrent model and Direct legal-rule ranking unchanged. Its threshold is chosen only on distinct DEV grammars. It then evaluates 96 exact local-indistinguishable reachable/unreachable pairs at each held-out depth, repeated across eight independently source-trained original checkpoint initializations.

The primary paired test asks whether the resulting *full-graph-derived* STOP score ranks unreachable above reachable more often than the local-only \(1/2\) ceiling, and independently requires correct initial STOP and proof completion under the original shortest-plus-one action budget.

Even a successful paired classifier would remain inferior to a complete symbolic public Horn closure policy when that policy has acceptable computational cost. Since full positive Horn goal entailment is constructively decided by forward least-fixed-point iteration, the relevant scientific question is whether the frozen recurrent features plus a small train-only calibrator expose **transferable global structure**, and at what compute cost, relative to the nonlearned exact graph algorithm.
