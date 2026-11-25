# LBF-19 Track T: Tree-Automata Abstractions - Mathematical Specification

Companion files: [`LBF_19_TREE_AUTOMATA_REGISTRATION.md`](LBF_19_TREE_AUTOMATA_REGISTRATION.md) (registered first), [`lean/LBF19TreeAutomata.lean`](lean/LBF19TreeAutomata.lean) (zero-axiom Lean 4.19.0, no Mathlib), [`lbf19_tree_automata.py`](lbf19_tree_automata.py). Dev-only.

## 1. Terms, rules, rewriting
Ranked signature $\Sigma=\Sigma_0\cup\Sigma_1\cup\Sigma_2$ (leaf constants, unary, binary symbols). Terms $T_\Sigma$: `leaf f | un f t | bin f t u`. Patterns add variables $x\in\mathbb N$. A substitution $\sigma:\mathbb N\to T_\Sigma$ gives $p\sigma$.

A rule $\ell\to r$ is **admissible** iff $\ell$ is left-linear (every variable occurs at most once; Lean `Pat.Linear`) and $\mathrm{vars}(r)\subseteq\mathrm{vars}(\ell)$ (Lean `Rule.Ok`). Right-hand sides may drop or duplicate variables, so rules may change term size (`p(x,0)->x` shrinks; `x->s(x)` style rules grow; `p(x,y)->p(y,x)` is commutativity). One-step rewriting $\to_R$ (Lean `Step`) rewrites $\ell\sigma\to r\sigma$ at any position (context closure for `un`, `binL`, `binR`). $\mathrm{Reach}_R^{\le k}(S)$ (Lean `ReachN`) is the set reachable in at most $k$ steps from $S$.

## 2. Abstract state: NTA with epsilon edges
$A=(Q,\delta_0,\delta_1,\delta_2,\varepsilon,F)$ with relational transitions $\delta_0(f,q)$, $\delta_1(f,q_1,q)$, $\delta_2(f,q_1,q_2,q)$, $\varepsilon(q,q')$, finals $F$. $\mathrm{Run}_A(q,t)$ (bottom-up evaluation up to $\varepsilon^*$), $L_q(A)=\{t:\mathrm{Run}(q,t)\}$, $L(A)=\bigcup_{q\in F}L_q(A)$. **Support** of $A$ = its set of transitions; the support order $A\sqsubseteq B$ (Lean `Incl`) is transition-set inclusion and implies $L(A)\subseteq L(B)$ (`Run.mono`).

## 3. Completion transfer operator
**Matching.** For a substitution of states $\rho:\mathrm{Var}\to Q$:
- $\mathrm{MatchL}_A(\rho,\ell,q)$ ($\ell$ read bottom-up; a variable $x$ matches exactly at state $\rho(x)=q$),
- $\mathrm{MatchR}_A(\rho,r,q)$ ($r$ is built; a bare variable needs $\rho(x)\to_\varepsilon^* q$).

**One round (`Complete1 R A A'`).** $A'\sqsupseteq A$ and for every rule in $R$, every $\rho$, $q$: $\mathrm{MatchL}_A(\rho,\ell,q)\Rightarrow \mathrm{MatchR}_{A'}(\rho,r,q)$. Concretely $T_R(A)$ adds, for every match, the rhs spine transitions into $q$ with fresh (memoized) intermediate states keyed by $(\text{rule},\text{rhs position},\rho|_{\text{vars below}})$, and an $\varepsilon$-edge $\rho(x)\to q$ when $r=x$. $T_R^k=(T_R)^k$.

**Widening.** For any map $h:Q\to Q'$, $A/h$ (Lean `NTA.map`) has transitions pushed forward by $h$. Implementation: $h$ merges states with equal depth-$j$ head-symbol signature ($j=1,2$; coarser $j$ = fewer merges). A *round* is $A\mapsto (T_R(A))/h$.

### Theorems (all proved in Lean, zero axioms; see `#print axioms` log)
1. **Lhs completeness** (`matchL_complete`): for linear $\ell$, $\mathrm{Run}(q,\ell\sigma)\Rightarrow\exists\rho.\ \mathrm{MatchL}(\rho,\ell,q)\wedge\forall x\in\mathrm{vars}(\ell).\ \mathrm{Run}(\rho x,\sigma x)$. (Linearity is exactly what lets the $\rho$'s of sibling subpatterns be combined.)
2. **Rhs soundness** (`matchR_run`): $\mathrm{MatchR}(\rho,r,q)$ and $\mathrm{Run}(\rho x,\sigma x)$ for all vars of $r$ imply $\mathrm{Run}(q,r\sigma)$.
3. **Completion step soundness** (`complete1_run_step`, `complete1_accepts_step`): if `Complete1 R A A'` and $t\to_R t'$ then $\mathrm{Run}_A(q,t)\Rightarrow\mathrm{Run}_{A'}(q,t')$ (induction on the context).
4. **Widening soundness** (`Run.map`, `widen_accepts`): $\mathrm{Run}_A(q,t)\Rightarrow\mathrm{Run}_{A/h}(h q,t)$, so $L(A)\subseteq L(A/h)$ (merging only enlarges).
5. **k-step soundness** (`chain_sound`, `completion_chain_sound`, `completion_widening_chain_sound`): for $A_0$ and any sequence $A_{i+1}$ with $A_i\sqsubseteq A_{i+1}$ language-wise and $\to_R(L(A_i))\subseteq L(A_{i+1})$ (in particular $A_{i+1}=T_R(A_i)$ or $T_R(A_i)/h_i$):
$$\mathrm{Reach}_R^{\le k}(L(A_0))\ \subseteq\ L(A_k).$$
6. **String embedding** (`string_step_is_tree_step`, `stringRule_ok`): see §4.

**Remark (what is over-approximated).** $T_R$ rewrites *all* matched redexes in parallel per round, so $L(T_R^k(A_0))$ contains $k$ parallel rounds, a strict superset of $k$ sequential steps in general (e.g. two independent redexes need 2 sequential steps but 1 round). This is the intended precision/soundness trade; the converse inclusion is false. Fresh-state sharing across targets $q$ and $\varepsilon$-merging add further (sound) over-approximation. No claim of termination to a fixpoint is made: only $k$ rounds are executed, so no termination widening is needed; the cap `limit` in the code is a resource guard that reports `blowup`, not an unsound shortcut.

## 4. Embedding of the LBF-18 fixed-length string domain
Strings $w=c_1\dots c_n$ are unary chains $\mathrm{enc}(w)=c_1(c_2(\cdots c_n(\$)))$ (Lean `encode`/`wrap`). A string rule $u\to v$ becomes the tree rule $u_1(\cdots u_m(x))\to v_1(\cdots v_{m'}(x))$ with tail variable $x$ (Lean `stringRule`; admissible, linear). Lemma `string_step_is_tree_step`: $\mathrm{enc}(\mathrm{pre}\,u\,\mathrm{suf})\to_R\mathrm{enc}(\mathrm{pre}\,v\,\mathrm{suf})$, i.e. string rewriting at any position is exactly tree rewriting on chains (also checked by the engine on all test systems: `embedding_ok`).

The LBF-18 positional-bigram envelope $\alpha_2(S)=(\text{first letters},\{(i,c_i,c_{i+1})\})$ is the NTA with states $(i,c)$ (position, letter), a transition $c:(i{+}1,d)\to(i,c)$ for each allowed $(i,c,d)$, final states $(0,c)$, and the end marker $(n,\$)$. Its language is exactly $\gamma(\alpha_2(S))$. Running $T_R$ on it with **positional canonical naming** of fresh states (the child of state $(i,\cdot)$ at rhs depth 1 is $(i{+}1,\text{its head letter})$) is a particular *state-identification* of the generic completion, hence sound, and keeps the state space inside $\{(i,c)\}$, i.e. a positional-bigram support set of size $\le n\cdot|\Sigma|^2$. So the unary-chain case yields a bigram-closed abstraction whose "support" is precisely the LBF-18 bitset; the tree-automaton domain strictly generalizes it (with branching, the transition arity is 2 and positions are replaced by tree-contexts of the state names).

Where tree automata *differ* for chains: the generic unnamed completion on the DAG automaton of the seeds is positional-free-and-exact (it never conflates positions beyond what completion merges), so it is typically *more precise* than the bigram envelope but has unbounded size growth; the positional naming bounds the size at the cost of exactly the bigram over-approximation. Empirical comparison in the results report.

## 5. Boolean support vs continuous brightness
Soundness depends only on the **support** (the transition set $\Delta=\delta_0\cup\delta_1\cup\delta_2\cup\varepsilon$ and $F$): `Incl`, `Run.mono`, `Complete1`, and `NTA.map` are statements about Boolean transition presence. A *brightness* $b:\Delta\to\mathbb R_{\ge0}$ (a learned ranking weight per transition, the analogue of LBF-18 cell brightness) is metadata used only to **order** candidate terms/branches (e.g. weighted run score $=$ product or sum of $b$ along the run); it never removes a transition. Requirements for a learned transfer to stay inside the theorem: (i) the output support must satisfy `Complete1` against its input support (or be a state-merging $/h$ of such), (ii) brightness may be zeroed for ranking but a transition with $b=0$ is kept in $\Delta$ ("zero-preserving gate", as in LBF-18 positivity). Under widening $A/h$, brightness of merged transitions is combined by $\max$ or sum (ranking choice; support is the union).

## 6. Experiment interface (implemented)
`lbf19_tree_automata.py`: exact `succs`/`ball_layers` (k-ball), `dag_automaton` (hash-consed A0, $L(A_0)$ = seed set), `completion_round` ($T_R$), `signature_classes` (widening map $h$), `run_ta` (rounds with checkpoints and blow-up guards), `tree_alpha` (size-agnostic positional-free n-gram baseline = root symbols, (symbol, depth-bucket) nodes, parent/child-index edges), `chain_system` (embedding experiment). The Python is validated against the Lean-proved specification only by testing (0 false negatives), not by extraction; the Lean model abstracts the fresh-state bookkeeping into the existential in `Complete1`.

## 7. Limits of the specification
- Left-linearity is required for completeness of lhs matching (theorem 1); non-left-linear rules need an over-approximating treatment (e.g. match with equal-state constraints still sound only for the "all-$\rho$ equal-at-both-vars" case; not formalized).
- Conditional/AC rewriting, binders (real Lean terms), and rewrite *strategies* are out of scope. Real Lean goals have binders, dependent types, and metavariables; this track only models first-order ranked trees.
- The formal statement is the $\le k$-round language inclusion, which is weaker than any claim about specific Lean-certificate soundness.
