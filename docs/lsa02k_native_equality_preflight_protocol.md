# LSA-02K: Native Lean equality-transport benchmark preflight

**Registered November 24, 2026.** CPU-only benchmark engineering after [LSA-02J](../reports/lsa02j_branch_utility_cpu_20261124.md). The public Horn fixed-point solver attained 32/32 there; the learned viability ranker gained only 1.5625 points over shuffled-label supervision. The next question is whether a verified equality problem family offers meaningful search-compute headroom against a strong classical planner.

## Public theorem grammar

Use opaque functions a,b,c,d : Nat → Nat, a point x : Nat, and an arbitrary indexed predicate P : Nat → Prop. Six public equations quantify over the suffix z : Nat. Each equates two distinct two-letter function compositions. The theorem transports a proof of P(source x) to P(target x).

A rewrite action instantiates the equation at the exact public suffix and rewrites the target, in either direction. The last action is exact w once the target index matches the public witness. Native Lean 4.19 checks every successor goal, each final proof term and an independent standalone proof compilation with no extra axioms. The finite word model supplies candidate proposals and public planning only. The recorded shortest distance is shortest **within the registered reversible word-rewrite grammar**, not a claim about every possible Lean tactic or general dependent type theory.

## Frozen source

Generator seed 20261204, with 64 training, 32 calibration, 32 ordinary test and 32 length-OOD roots. The generator selects distinct endpoints from exact reachable rewrite-distance shells 4–5, 7–8, 10–11 before evaluation. Lengths 7–9 are crossed with these shells; the OOD shard uses length 10. Public rows contain only theorem inputs; no path, chosen action, hidden distance or solution labels. Whole equation systems are separated across shards modulo all 24 symbol renamings, rule order and rule orientation. This is internal synthetic lineage separation, not a claim of external corpus independence.

Compile 64 train-only reference proof scripts, preserve their actual public candidate menus and chosen native-certified reference actions. Other candidates remain unlabeled, since a generated move is not a native negative receipt. No model training during this preflight. The ordinary and OOD shards are authenticated by hash but not parsed during evaluation.

## Registered controls and resource accounting

All live proofs have 16 units maximum. Each attempted rewrite, terminal exact witness, and explicit rollback costs one unit. Local Hamming-ordered DFS and five independent root-seeded uniform DFS streams (17,31,47,61,83) use identical public menus and native verification. A rollback can only follow a repeated word on the current proof path or an exhausted local menu. No learned checkpoint or native reference oracle participates in online selection.

A deterministic bidirectional BFS over public equations is the strong symbolic competitor. Its expansion caps are 16,64,256,1024, alongside a 100,000-expansion reference limit. Every discovered reference path is independently replayed and compiled in Lean. A planning expansion enumerates one word's public neighbors; report enumerated candidate edges as well. Re-run every cap, checking the bounded planner reproduces the exact planner's deterministic prefix. Report both:
- proof closures under each planning expansion cap, with 16 available native proof units;
- proof closures under each illustrative joint cap on planning expansions plus native tactic count.

These counters are separate resource axes: equal counts are not evidence of equal runtime or compute. Record actual planner wall time for later cost-sensitive work. No restriction on native proof cost is relaxed to reward a learned policy.

## Gates

Report 32 independent calibration roots, exact public-planner saturation, Hamming and uniform DFS full proof closure, distance-band breakdown, and the number of actual native replays/standalone proof certificates. Treat five uniform streams as repeated observations on the same theorem roots. The predeclared headroom diagnostic is at least 20% unsolved (no more than 80% closed) at the illustrative joint cap of 64. It is a screening threshold for planning-compute research only. The stronger public solver may saturate the family; that would rule out a claim of learned superiority on unrestricted proof closure.

If there is headroom, register a subsequent learned ordering study with source-only verified targets, same-worker duplicate CPU training, shuffled sham, a strong public symbolic comparator, matched compute ledgers and independently frozen evaluation. No model training, architecture promotion, external Mathlib claim or sealed test opening occurs in LSA-02K.
