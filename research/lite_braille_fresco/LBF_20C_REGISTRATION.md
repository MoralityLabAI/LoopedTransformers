# LBF-20C registration: premise-set Pareto completion

Development only. Parent ed749440 (LBF-20B). Freeze the code, analysis, and fixtures before execution. No new fits. The original 129-parameter guard ranker is copied byte-for-byte from LBF-20B artifact 15003891508, expected SHA-256 0471fe790ba1045ba991af10929be9a110a8bce9ccb4864a0f2f39f40de0c22e. No development labels update it.

## Algebra

A concrete ground rewrite path carries a label (M,k): the set of distinct premise IDs used and its sequential step count. Product is (union,+); alternative paths are joined by nondominated union. (M,k) dominates (N,j) if M⊆N and k≤j. For fixed endpoints domination is preserved by prefix/suffix multiplication. Store one shortest witness per mask, discard dominated pairs. Epsilon entries carry concrete witnesses and saturated closure. With G guard IDs and horizon H, each state-pair upward closure grows in a finite universe of 2^G(H+1) elements. This is a worst-case exponential factor in G, not a polynomial theorem in rule count. No beam or top-k label truncation.

A proof sketch for ground exactness: structural runs lift child paths in context and add rewrite counts while unioning guards. Each epsilon update appends a concrete ground rule; closure concatenates witnesses. For completeness, simulate each concrete rewrite in a bottom-up run using the saturation edge for its redex. A discarded label has a dominating witness with the same endpoints; replacing any subderivation preserves or improves both the budget and required guard set. Thus every path within H has a dominating retained witness, while every retained witness is concrete. Filtering by a set of admissible premises gives exactly the truncated feasible distance. The Lean file formalizes the abstract domination, context-pruning and query-preservation lemmas, not the implementation or full ground saturation refinement.

## Independent fixture audit

96 fresh deterministic systems, seed 2026112506, H=4, five possible guard IDs. Independent labeled-state BFS, all targets plus 16 random negative probes/system. Exact Pareto comparison, replay every epsilon and target witness, check saturation and triangle coverage, all 32 allowed masks at every target. Any cap breach or mismatch fails. Include a strict 1-premise/4-step vs 2-premise/2-step tradeoff.

## Live native development study

18 original LBF-20B dev roots (retrospective) and 18 new deliberately planted tradeoff roots (seed 2026112505, six per sharing/misleading/mixed template). New fixtures are exploratory, shared templates, not a lineage-disjoint confirmation. All arms share exact guard-text grouping within each root, one native probe per unknown guard group and the same final explicit proof. No semantic guard equivalence is assumed. A portfolio rejection excludes that guard for the rest of the episode; this is a search restriction, not a theorem of falsehood, and the evolving context can in principle change solver success.

- lazy_steps: recompute shortest-step WeightedWitness after each failed probe, deduplicated guard cache (stronger than the original per-rule LBF-20B controller).
- pareto_steps: precompute exact nondominated plans once; minimize (steps, newly required groups).
- pareto_guards: minimize (newly required groups, steps).
- pareto_risk: minimize (sum of reciprocal clipped frozen probabilities over unknown groups, steps). A heuristic, not a calibrated expected-cost identity.
- probe_all: probe each distinct guard once before deferred shortest proof.

All select the first unknown guard in the chosen witness; ties deterministic. Same 24 native tactic request cap, four-second soft wall budget and rewrite horizon H=20. Rotate arm order by root index. Actual Lean 4.19.0 verdicts and final kernel proof; standalone recompile all successful proofs, standard Lean axioms permitted explicitly as in LBF-20B. Archive planning, verifier and end-to-end timings, closure by request budgets, group counts, rejection counts, all root outcomes, paired call wins/ties/losses conditional on both arms solving. Distinguish fewer tactic requests from better actual latency. No binary superiority claim, no extrapolation to state-dependent guards or general Lean proof search. Do not tune using these results.
