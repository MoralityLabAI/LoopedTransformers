# LBF-19 Track T: Tree-Automata Abstractions - Results (dev-only, no confirmatory claims)

Registration: [LBF_19_TREE_AUTOMATA_REGISTRATION.md](LBF_19_TREE_AUTOMATA_REGISTRATION.md) (committed `11581588` before any result). Spec: [LBF_19_TREE_AUTOMATA_SPECIFICATION.md](LBF_19_TREE_AUTOMATA_SPECIFICATION.md). Lean: [lean/LBF19TreeAutomata.lean](lean/LBF19TreeAutomata.lean) (log: [lean_compile.log](results/lbf19/tree_automata/lean_compile.log)). Code: [lbf19_tree_automata.py](lbf19_tree_automata.py), tests `tests/test_lbf19_tree_automata.py` (7 pass).

## 1. Lean
Lean 4.19.0 core only, compiles with exit 0, no `sorry`; **18/18 `#print axioms` report "does not depend on any axioms"** (incl. `matchL_complete`, `complete1_run_step`, `chain_sound`, `completion_chain_sound`, `Run.map`/`widen_accepts` (widening), `completion_widening_chain_sound`, `string_step_is_tree_step`). Main theorem (`chain_sound`): for any sequence of rounds with `L(A_i) subset L(A_{i+1})` and `Step(L(A_i)) subset L(A_{i+1})` (complete or complete-then-merge), `ReachN R L(A0) k subset L(A_k)`. Limits: left-linear rules, first-order ranked terms, relational NTA with epsilon; Python fresh-state bookkeeping is abstracted away in `Complete1`.

## 2. Tree experiments (16 systems x k=1..4 x {exact, sig2, sig1})
Two runs (**amendment disclosed**): *Run A* ([tree_systems.json](results/lbf19/tree_automata/tree_systems.json)) exactly as registered (random seed terms). After seeing that many random seed terms contain no redex (balls of 6-9 terms for 10/16 systems, i.e. trivial), I added *Run B* ([planted/](results/lbf19/tree_automata/planted/tree_systems.json)): same systems, seeds = random terms with 1-2 subterms replaced by lhs instances. This amendment was made after Run A results were seen; both are reported. Balls never truncated (max 914 terms at k=4); FN checked on the full ball.

**Soundness (H1): 0 false negatives in all 192 (system,k,variant) cells of both runs, 0 blow-ups; chain case 0 FN (Section 3). PASS.**

Run B means over 16 systems (Run A is within a few points of these; see JSON):

| variant | k | mean states | mean transitions | FPR N1 (k+1..k+2 steps) | FPR N2 (mutations) | baseline FPR N1 / N2 | baseline size |
|---|---|---|---|---|---|---|---|
| exact | 1 | 90.1 | 103.9 | 0.943 | 0.004 | 0.937 / 0.500 | 115.4 |
| exact | 2 | 90.5 | 105.8 | 1.000 | 0.002 | 0.997 / 0.507 | 116.6 |
| exact | 4 | 90.5 | 105.8 | 1.000 | 0.001 | 1.000 / 0.512 | 116.7 |
| sig2 | 4 | 86.9 | 103.1 | 1.000 | 0.002 | | |
| sig1 | 4 | 68.9 | 107.7 | 1.000 | 0.012 | | |

N3 (random terms) FPR 0.000 everywhere (baseline <= 0.001): uninformative.

Findings (honest):
- **Structural precision (N2): the NTA is far more precise than the size-agnostic n-gram baseline** (0.1-1% vs ~50% false positives; the baseline here is the oracle-best `alpha(ball)`), at roughly the same or smaller size (~90 states vs ~116 abstract atoms). H2 supported on N2; **not** supported on N1 (both ~1.0).
- **Step-bound discrimination fails:** the abstraction saturates after k=1-2 (states constant 90.1 -> 90.5; N1 FPR 1.0). Because fresh states are memoized by (rule, position, state-substitution) the construction reaches a finite closure that approximates the full reachability set R*, not the k-ball; it cannot distinguish k from k+2 steps. This is sound but means T_R^k does not give a k-sensitive envelope on these small systems. A k-sensitive abstraction needs step counters in states (not done).
- **Widening (H3):** sig2 removes ~4% of states, sig1 ~24%, with FPR N2 rising 0.001 -> 0.012 (monotone, sound), but transitions do not shrink (sig1 actually 107.7 vs 105.8): state merging reduced states, not support size. Partially supported.
- **No blow-up observed** (max 133 states, <= 5 s per system). This is a negative for H4's blow-up prediction in this regime: seeds are 6 terms, rule systems tiny, and the DAG automaton plus saturation keep size bounded. Blow-up for large, commutative-heavy, AC-style or growing rule sets is untested.
- Limits: 6 seeds/system, 2-4 rules, N1 negatives empty for systems with trivial balls (FPR None excluded), sampled membership only for negatives, no real Lean goals (no binders, types).

## 3. Unary-chain (string) case, 5 systems x k=1..4, length 8 over 4 letters ([chain_embedding.json](results/lbf19/tree_automata/chain_embedding.json))
- Embedding check: tree engine on `enc(w)` reproduces the string k-balls exactly (`embedding_ok` 5/5; also Lean `string_step_is_tree_step`).
- 0 FN for TA-positional, TA-DAG and the LBF-18-style positional-bigram envelope iterate `E_k`.
- TA-positional language is a subset of `gamma(E_k)` in 18/20 cells, equal in 1/20, and `E_k` subset TA in 1/20; **2/20 cells (k=2 for 2-3 systems) are incomparable, so the registered H5 criterion "subset in all cells" FAILS**: parallel rounds on positional states can admit double rewrites that the sequential envelope iterate excludes.
- Mean language size / ball size: TA-positional ~64-69x, `E_k` ~78-91x (TA is ~15-25% tighter on average); TA-DAG (unnamed, exact completion) 1.0-1.2x ball but ~52-55 states vs 29.6 for positional naming. So the unnamed tree automaton is nearly exact on chains but ~2x larger; positional canonical naming caps size at the LBF-18 bigram support at the cost of ~60x language inflation.

## 4. Figures (plain-code SVG)
[states vs k](results/lbf19/tree_automata/fig_states_vs_k.svg), [FPR vs k](results/lbf19/tree_automata/fig_fpr_vs_k.svg), [chain language sizes](results/lbf19/tree_automata/fig_chain_language_sizes.svg); Run B figures in `planted/`.

## 5. Bottom line
Soundness holds in Lean (zero-axiom) and empirically (0 FN). The tree NTA beats positional n-grams by ~100x on structural-mutation negatives, but collapses to a k-insensitive closure on step-bound negatives; the chain-case "recovers the LBF-18 envelope" claim holds only as "sound and usually tighter", not "equal".
