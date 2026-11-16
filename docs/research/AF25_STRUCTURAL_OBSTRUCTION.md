# AF-25 structural obstruction: goal-ancestor traps and shared Horn subproofs

**Scope:** exact finite positive-Horn constructions with four initial facts and a shortest-proof-plus-one firing budget. The arguments are mathematical statements about the saved source grammar. The observed 80-case control outcomes come from independently replayed source-only CI, rather than a general formal verification of all possible Horn programs. No Lean proof has been claimed.

## 1. Two source-registered hard graphs

Both graphs begin with \(F_0=\{0,1,2,3\}\), have marked goal atom \(14\), and support two distinct goal-concluding rules. Premises are conjunctive, and every rule fires at most once because its head is added monotonically to the known set.

### Five-step productive route

The required productive rule chain is

\[
(0\land1)\to4,\quad(4\land2)\to5,\quad
5\to6,\quad5\to13,\quad(6\land13)\to14.
\]

This completes the goal in exactly **five** firings, including one shared productive prefix of two firings and two downstream subgoal rules.

The competing longer route uses

\[
3\to7,\;3\to8,\;2\to9,\;2\to10,\quad
(7\land8)\to11,\;(9\land10)\to12,\;(11\land12)\to14.
\]

Its seven nonredundant firings cannot complete in the registered budget \(5+1=6\).

### Six-step productive route

The productive rules are

\[
(0\land1)\to4,\;(4\land2)\to5,\;5\to9,\quad
9\to6,\;9\to13,\;(6\land13)\to14.
\]

They require **six** firings, reusing a three-firing prefix. The competing goal rule is \((11\land12)\to14\), whose prerequisites come from

\[
3\to7,\;3\to8,\;(7\land8)\to11,\quad
2\to10,\;(10\land9)\to12.
\]

It requires those five auxiliary firings, the three shared-prefix firings that derive \(9\), and the final goal rule: **nine** firings, exceeding the registered budget \(6+1=7\).

Both graph designs have only two-premise Horn rules and fit within twelve of the fourteen physical slots. The source generator separately permutes the physical atom and rule labels for each case; the longer goal rule receives a preregistered lower physical slot when the additive witness costs tie.

## 2. Additive recurrence double-counts shared work

For a known state \(F\), let an additive Horn derivation surrogate be

\[
A_F(x)=
\begin{cases}
0,&x\in F,\\
\displaystyle\min_{r:\operatorname{head}(r)=x}
\left(1+\sum_{p\in\operatorname{prem}(r)}A_F(p)\right),&x\notin F.
\end{cases}
\]

This recurrence charges a proof tree, while physical Horn proof search executes a **DAG** in which one produced fact can support several later rules.

For a unique shared prefix of \(k\) firings and two one-step subgoals \(U,V\), the exact shared goal proof costs

\[
d_{\rm shared}(G)=k+3.
\]

The additive recurrence assigns both subgoals the price \(k+1\), and therefore

\[
A_{\rm shared}(G)=2k+3,
\qquad
A_{\rm shared}(G)-d_{\rm shared}(G)=k.
\]

Thus an additive tree-cost approximation incurs exactly \(k\) surplus firings **under the stated unique shared-prefix grammar**. A separately constructed goal route with true and additive cost \(L=2k+3\) ties the additive shared route while taking \(L\) firings to execute.

The AF-25 depth-five instance has \(k=2\), \(d_{\rm shared}=5\), and both additive goal witnesses have score \(7\). The registered longer goal-rule physical tie breaker selects the seven-firing alternative at the initial structural witness. The depth-six instance has \(k=3\), \(d_{\rm shared}=6\), and both additive goal witnesses have score \(9\). Recursively rebuilding the witness after every new fact can sometimes repair the initial misranking; indeed the original source-only closure controller succeeds on **20/48** hard depth-six cases, while failing all 48 hard depth-five cases.

## 3. A two-action distraction exceeds the original budget

The goal-ancestor heuristic assigns each atom the minimum reverse rule-hop distance to the marked goal, ignoring whether conjunctive sibling premises can be obtained cheaply.

In the depth-five hard graph, productive first action \(4\) has reverse distance \(3\) to the goal through \(5\), one of \(6/13\), and \(14\). The decoy heads \(7,8,9,10\) each have reverse distance \(2\). On a provable root the heuristic therefore selects a decoy. After that first decoy, remaining enabled decoy heads still have reverse distance \(2\), so the heuristic spends at least a second action on the longer route before starting the productive route.

At depth six, productive first head \(4\) is reverse distance \(4\), while decoy root heads \(7,8,10\) have reverse distance \(2\). The same two-action distraction occurs.

**Lemma (shortest-plus-one obstruction).** Suppose a goal can be reached by a shortest proof of \(d\) rule firings. If a policy first applies two rules that are irrelevant to every \(d\)-firing completion, and the selected two rules do not enable an independent proof in at most \(d+1\) total firings, then the policy cannot complete the goal within \(d+1\) firings.

*Proof.* Every \(d\)-firing completion still requires its \(d\) productive firings after the two wasted moves. Such an execution takes at least \(d+2\) rule firings. By hypothesis the decoy-based alternative also exceeds \(d+1\). Either possible completion exceeds the budget. \(\square\)

The hard-family source instances satisfy both hypotheses. The independent structural receipt confirms **0/48** completed positives for ancestor greedy and BFS caps 4 and 16 at each depth. Those measured outcomes account for the actual search fallback and tie rules, whose exact behavior goes beyond this lemma.

## 4. Why the BFS expansion cap matters

A bounded public BFS controller tests legal monotone host states and, when it finds a goal, follows the first rule on a witnessed short path. A cap-exhausted controller falls back to the reverse-hop heuristic. The four or five initially available distinct successor states create many partial decoy/goal combinations under the frozen physical-rule ordering.

The preregistered source-only manifest records:

| Control | Hard positives completed, depth 5 /48 | Hard positives completed, depth 6 /48 |
|---|---:|---:|
| Ancestor greedy | 0 | 0 |
| Additive closure | 0 | 20 |
| BFS cap 4 | 0 | 0 |
| BFS cap 16 | 0 | 0 |
| **BFS cap 64** | **48** | **48** |

The larger BFS controller can explicitly discover goal paths before resorting to the inadequate reverse-hop fallback. This fact separates a **search-compute advantage** from any alleged learnability advantage: the original AF-25 graphs have a perfect symbolic solver once enough graph exploration is allowed.

## 5. Observable neural intervention

The officially scored AF-25 frozen neural results show that raw Direct logits complete **240/384 hard depth-five cases** pooled across eight original seeds, while the two-step Bellman VI-2 completes **131/384**. At depth six Direct completes **96/384** hard cases and VI-2 completes **95/384**. The difference between Direct and VI-2 uses exactly the same weights, providing a specific example of an inference-algorithm interaction under an unchanged host-legal state graph.

The original network was trained on a different rule grammar. No claim is made that the model learned the above additive-cost obstruction, nor that these finite graphs represent arbitrary mathematical theorem proving. A future mechanism study can ask which action logit or successor value changes cause the network to spend the extra budget on decoys or to STOP before completing a valid proof.
