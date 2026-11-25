# LBF-19 Track T: Tree-Automata Abstractions for Variable-Size Expression Trees - Registration

**Status: dev-only track. No confirmatory claims.** Registered BEFORE any result of this track is computed. Branch `research/lbf19-expensive-verifier-20261125` (based on merged LBF-18, PR #349). No sealed cohorts are touched; all data are synthetic and generated from the fixed seeds below.

Motivation: LBF-18 positional u64 bigram bitsets are sound only for fixed-length strings. Lean goals are variable-size trees. We study nondeterministic bottom-up tree automata (NTA) as abstract states with a completion transfer operator `T_R` (+ optional state-merging widening) and test soundness and precision empirically, with the soundness theorem proved in zero-axiom Lean (`lean/LBF19TreeAutomata.lean`, spec in `LBF_19_TREE_AUTOMATA_SPECIFICATION.md`).

## 1. Hypotheses
- **H1 (soundness, hard):** for every system, seed set, k in {1,2,3,4} and widening variant, `Reach_R^{<=k}(L(A0)) subset L(T_R^k(A0))`. Predicted FN = 0. Any FN = implementation bug or specification bug; the track fails until resolved.
- **H2 (precision vs. size-agnostic baseline):** unwidened `T_R^k` has lower false-positive rate (FPR) on near-ball negatives than the best-possible size-agnostic positional n-gram envelope (`alpha(exact k-ball)`, which upper-bounds the precision of ANY sound transfer in that domain), at the price of larger size. Predicted but not guaranteed; negative outcome will be reported.
- **H3 (widening trade-off):** merging states by head-symbol signature (`sig1`, `sig2`) reduces states/transitions monotonically with coarser signature and increases FPR (never decreasing the language). Soundness is unaffected (H1).
- **H4 (over-approximation source):** `T_R^k` performs k *parallel* completion rounds, so it over-approximates the k-sequential-step ball; FPR on "k+1..k+2 step" negatives is expected > 0 and growth of states in k is expected to be super-linear for length-increasing / commutative systems (blow-up expected and to be reported).
- **H5 (string embedding):** on unary chains (fixed-length strings, same-length rules) with positional canonical state naming, `L(T_R^k(A0))` restricted to length n is sound and is no coarser than the LBF-18-style positional-bigram envelope iterate `E_k = alpha(gamma(E_{k-1}) union Step gamma(E_{k-1}))`. Equality or strict improvement will be reported per system; failure of "no coarser" is reported honestly.

## 2. Setup (fixed before running)
- Signature: leaves {a,b,z}, unary {s,g,h}, binary {p,m}. Terms 5..40 nodes.
- Seed set per system: 6 random terms with node counts drawn uniformly in [5,40]; `A0` = hash-consed DAG automaton (state per distinct subterm), `L(A0)` = seeds exactly.
- Rule systems (left-linear lhs, rhs variables subset of lhs variables, 2..4 rules):
  - **C1 named arithmetic-like** (1 system): `p(x,z)->x`, `s(g(x))->h(x)`, `p(x,y)->p(y,x)`, `g(h(x))->x`.
  - **C2 size-preserving random** (5 systems), **C3 shrinking random** (5), **C4 length-changing mixed (growing and shrinking, rhs may duplicate or drop variables)** (5). Rule-system seeds: 19100+i (C2), 19200+i (C3), 19300+i (C4), i=0..4. Seed-term seeds: 19400 + index.
  - **C5 unary chain** (strings over {a,b,c,d}, `$` end marker, length n=8, same-length rules of width 2..3, 5 systems seeds 19500+i, 6 seed words each), A0 = positional-bigram automaton of the seed words.
- Variants: `exact` (no widening), `sig2`, `sig1` (merge states with equal depth-j head-symbol signature, applied after each round). Per-run caps: 30,000 states or 120 s -> recorded as `blowup` (not hidden).
- Baseline: positional-free ("size-agnostic") n-gram envelope = root-symbol set plus parent/child-index/child-symbol edge set (depth bucket min(depth,3)), computed as `alpha(exact k-ball)` (oracle-best in domain; not a transfer).
- Exact k-ball: BFS over single rewrites at any position, <=k steps, cap 400,000 terms; if truncated, record `ball_truncated` and soundness is checked on 2,000 random k-step walks instead (flagged as sampled).
- Negative cohorts for precision (sample sizes <=2,000 each, not in exact ball): **N1** terms from k+1..k+2 further rewrite steps outside the ball; **N2** ball terms with one random subterm replaced (relabeled symbol or a random small tree); **N3** random terms of size matching seeds.

## 3. Metrics
FN count/rate vs exact k-ball (per system,k,variant); FPR per negative cohort (automaton and baseline); states, transitions, runtime; ball size; chain case: equality/inclusion vs `gamma(E_k)` over all 4^8 strings.

## 4. Pass/fail criteria
- **Soundness PASS:** FN = 0 in all (system,k,variant) not `blowup`/truncated-sampled; truncated cases need 0 FN on sampled walks and are labeled as such. Any FN -> FAIL.
- **Lean PASS:** `lean/LBF19TreeAutomata.lean` compiles with Lean 4.19.0, no `sorry`, `#print axioms` of every theorem shows none.
- **Precision/size comparisons** (descriptive, no pass/fail beyond reporting): H2 counted as supported if exact-variant mean FPR on N1 and N2 is lower than baseline FPR in a majority of non-blowup (system,k) cells; H3 supported if mean states decrease exact>sig2>sig1 and mean FPR does not decrease; H5 supported if in all C5 cells the TA language is a subset of `gamma(E_k)` (sound by H1 supersetting the ball).
- Tests: `pytest tests/test_lbf19_tree_automata.py` passes.
- Honest negative results (blow-up, FPR near 1, baseline wins) are reported verbatim.
