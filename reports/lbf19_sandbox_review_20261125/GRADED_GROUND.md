# Min-plus bounded completion for ground rewriting

Let R be a finite set of ground rules l → r over a ranked signature, S a finite set of initial ground terms, and H a budget. Let Q contain a named state q_u for every distinct subterm u of S and of both sides of R. Structural transitions f(q_u1,...,q_un) → q_f(u1,...,un) have weight zero. Weighted epsilon transitions represent replacement of an earlier ground term by a reachable ground term. Define d(t,q) as the minimum cost of a bottom-up run accepting t into q; cap all costs above H at H+1.

Initialize epsilon distances by the identity. Saturate using the following rule until stable:
for each l → r and each q, add/update q_r → q of weight d(l,q)+1. Compute min-plus epsilon closure. Each update is a nonincreasing integer bounded below, so the capped finite matrix terminates.

## Proposition (paper proof sketch, not yet mechanized)

For the ground-rule fragment, d(t,q_u) equals the minimum concrete rewrite distance from u to t, truncated at H+1.

Soundness: each zero-weight constructor run corresponds to independent child derivations, whose rewrite counts add. An epsilon edge q_r → q_u inserted with weight d(l,q_u)+1 corresponds to a derivation u →* l → r, and an accepting run for r's state extends this derivation. Min-plus closure concatenates witnesses. Induct on the derivations of finite saturation weights and the run structure.

Completeness: by induction on the concrete derivation length. A redex l occurs in an accepting run of the previous concrete term. At that position take the state q and the cost of its subtree run (including its epsilon transitions). Saturation makes r acceptable in q at no more than the previous subtree cost plus one. Replace that run, preserve the surrounding run, and obtain cost no greater than the new path length. The argument uses ground rules; no variable substitution costs are duplicated. Together the inequalities yield equality for distances at most H; all larger distances are mapped to H+1.

The cost of a parent combines child costs by addition, not max. The rule a → b from p(a,a) reaches p(b,b) after two steps; max would incorrectly admit it at one step.

## Why this does not yet extend to arbitrary left-linear rules

Initial f(a), rules a → b and f(x) → g(x,x). There is a length-2 path f(a) → f(b) → g(b,b). A naive construction that charges each copy of b for the pre-duplication rewrite and additionally charges the duplication reports cost 3, a false negative at H=2. Erasure and non-linear matching also require care. Use explicit derivation sharing or a safe graded overapproximation; do not assert general exactness.

Space for the reference weighted epsilon matrix is O(N²), with N distinct ground subterms. The proof-of-concept uses Floyd–Warshall at each saturation round, O(N³) plus rule recognition costs; a worklist and sparse epsilon graph are natural optimizations. A fully expanded budget-state representation would have O(N(H+1)) states, but the minimum-cost representation avoids that factor. For general variable rules, bound growth and track sharing before proposing Lean formalization.

## Measured pilot

100 deterministic synthetic systems, K=5, 4676 terms compared against concrete BFS to K+2 and random negatives. 434 terms first reachable beyond K; all truncated distances matched. Up to 19 states and 3 rounds. To avoid unbounded fixture explosion, random ground rules were size-nonincreasing, with an additional controlled growth rule every fifth system. This fixture restriction was chosen after an initial unbounded random generation exceeded a 50,000-term cap. The results are exploratory.

Run `python graded_ground_audit.py`. The code is a stand-alone reference, not yet wired into LBF-19 Track T or formally checked in Lean.
