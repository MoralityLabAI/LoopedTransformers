# LBF-20A registration: witness-carrying bounded ground-tree completion

Development-only CPU experiment, after merged LBF-19 audit d023c453. Freeze this design and code before the first CI execution. No model fitting, old cohort access or fresh confirmatory claim. Ground rewriting over ranked nullary, unary and binary constructors.

## Objective

Recover exact truncated ground rewrite distance and a replayable witness using a finite named-subterm automaton. Distinguish concrete sequential steps from unweighted completion rounds. Every finite epsilon entry is accompanied by a concrete path; every accepted target must yield a replayable path. A finite path constitutes a soundness witness, independently of learned ranking.

Let E(q,r) be minimum known cost of rewriting state label q to state label r. Structural transitions for named subterms have zero local cost and add child costs. Initialize identities. For each ground rule l→r and state q, evaluate l from q; insert E(q,r) at cost d(l,q)+1. Apply min-plus closure to fixed point, rejecting paths above H. E stored as path tuples; tie on equal length preserves old witness.

Termination: each of |Q|² entries decreases through H+2 possibilities (infinity and 0..H). A stable table satisfies bounded triangle and rule-update inequalities. The Lean theorem capped_exact states equivalence of bounded concrete walks and weighted runs under explicit table soundness, subterm-closure, identity, closure and saturation premises. It does not claim to extract the Python implementation or prove its indexing. The executable audit checks these premises on the finite fixtures and independently replays all stored witnesses.

## Registered development checks

- 128 random ground systems, seed 2026112502, H=5. Independent concrete BFS through H+2, plus 30 randomly generated negative probes per system (deduplicated). Rules include cycles, non-increasing random rewrites, and a controlled growth rule every eighth system. Any BFS cap exceedance (50,000 terms) or distance mismatch fails the run; do not silently skip systems.
- Independently replay every finite epsilon witness and every finite target witness, check fixed-point inequalities. Zero mismatches.
- Structured scaling: complete binary trees with 4, 8, 16, 32 leaves, cyclic a→b→c→a. Exact distance from all-a to all-c is 2n. Reject at H=2n−1, accept with path length 2n. State space contains exactly 3^n leaf assignments with the fixed skeleton. Enumerate BFS for n=4,8; use analytic count and witness for n=16,32, not a claimed exhaustive traversal.
- Separate shrinking and growing rule examples: f(a)→b under g, and a→f(a) applied three times. Check exact distance and reject below it.
- 22 concrete equality certificates (16 random systems, 4 scale cases, 2 adversarial). Explicit congruence under a one-hole context. Compile in Lean 4.19.0 and check zero axioms.
- 14 theorem declarations for relational weighted-run semantics and conditional completeness/soundness, compile axiom-free.

Record actual runtime and counts. This is a proof-of-concept, not an efficiency benchmark against all classical search, and the random fixtures are not independent training seeds. No unrestricted left-linear variable-copying claim: pre-duplication histories require sharing, as explained in the LBF-19 audit.

Next gate after this: native tactic success/branching that is unknown before elaboration, with matched call/time budgets and deferred proof replay where valid.
