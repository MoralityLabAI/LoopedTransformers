# LSA-02K: Native equality transport benchmark and symbolic search-compute frontier

**Date:** 2026-11-24. **Engineering and native proof certification: PASS. Registered bounded-search headroom: PASS.** [PR #334](https://github.com/MoralityLabAI/LoopedTransformers/pull/334) merged at a72496b77754cf4e63904ff8595b9729fbec1c22. [Completed CPU run 45281056858](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45281056858) and PPT CPU, G03 native Lean and LBF integrated proof-contract regressions all passed. This is a benchmark preflight without learned-policy fitting.

## Research question and public grammar

[LSA-02J](lsa02j_branch_utility_cpu_20261124.md) found that a public Horn fixed-point solver solved every calibration theorem and exactly matched native proof costs, while the learned viability head nearly tied the shuffled-label sham. LSA-02K instead asks whether a bounded classical equality planner leaves useful search-compute headroom.

Each theorem quantifies opaque unary functions a,b,c,d : Nat → Nat, an index x : Nat, an arbitrary predicate P : Nat → Prop, six equations equating distinct two-letter compositions for every suffix, and a witness w : P(source x). The goal is P(target x). An action proposes a concrete forward or reverse equality rewrite instantiated at the exact suffix, or exact w after reaching its index. Native Lean checks every successor target and closing proof. This is indexed-predicate equality transport within a restricted reversible word-rewrite grammar, not general dependent type theory or Mathlib extraction.

The [registered protocol](../docs/lsa02k_native_equality_preflight_protocol.md) froze generator seed 20261204: 64 train roots, 32 calibration roots, 32 ordinary-test and 32 length-OOD roots. Equation systems are disjoint across all shards modulo symbol renaming, rule order and orientation. Generation uses reachable rewrite-distance shells 4–5, 7–8 and 10–11; rows expose the public equations and endpoints without paths, distances, or chosen actions. Ordinary/OOD shards are hash-authenticated but their contents were never parsed during evaluation.

Frozen suite manifest SHA-256: d928d8b5ffbbd9760e02fc9af903704bdac0cea7adc0b6e5be9103e7f2153874. Native train source manifest SHA-256: 1324e9a1c8d6e18f155fe3da5104204818aafe09c337087b75d82ebfd6163341.

## Native source and controls

All 64 training reference proofs passed live Lean 4.19.0 kernel verification, replay and independent axiom-free standalone compilation. Their selected trajectories yielded **546 training reference-action frames**. Other candidate proposals remain unlabeled. No native negative labels are fabricated from the symbolic planner.

The calibration cohort contains 32 independent theorem systems. Every live search has a maximum of 16 charged units, one per rewrite, terminal witness action or stack rollback. Local Hamming-guided DFS and model-free uniform DFS (five streams: 17,31,47,61,83) use identical public rewrite menus. The separate public bidirectional BFS planner derives paths entirely from visible equations and endpoints. It reports expansions, enumerated edges, planning time and native proof tactic cost separately.

| Policy | Native closure | Effort |
| --- | ---: | ---: |
| Hamming-guided DFS | 5/32 = 15.625% | 481 total charged units |
| Uniform DFS, five streams | 2/160 = 1.25% | 2,551 total charged units; 387 rollbacks |
| Exact public bidirectional BFS | 32/32 = 100% | Mean 110.125 expansions; max 428 |

The 32 roots are the independent statistical unit. The five random streams repeat those roots. The descriptive 95% root bootstrap interval for Hamming minus uniform closure is +2.5 to +27.5 percentage points.

All 224 calibration episodes (32 Hamming, 160 uniform, 32 exact-reference) passed native replay. All 39 successful scripts passed separate standalone compilation with no extra axioms. This includes all 32 reference paths. The reported minimum path lengths are exact within the reversible length-preserving public word grammar only.

## Classical planning frontier

The exact bidirectional solver is complete for this finite public task grammar on the observed cohort. To establish a bounded-search research problem, rerun the public planner with caps of 16, 64, 256 and 1,024 expanded words and verify successful plans match the reference planner. Every cap's successful script must fit the independent 16-tactic native proof limit.

| Expansion cap | Certified closures | Closures with expansions + tactics ≤ cap | Expanded words, total | Enumerated edges, total |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 5/32 | 3/32 | 477 | 2,680 |
| 64 | 16/32 | 15/32 | 1,408 | 8,131 |
| 256 | 27/32 | 27/32 | 3,111 | 18,851 |
| 1,024 | 32/32 | 32/32 | 3,524 | 21,567 |

A planning expansion enumerates one public word's rewrite neighbors. The joint count is an explicitly illustrative accounting convention; one symbolic expansion and one Lean tactic call have different wall-clock and compute costs. It does not by itself establish equal-compute superiority.

| Exact grammar rewrite distance | Roots | Hamming closures | Uniform average closure | Planner ≤64 expansions | Mean exact expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4–5 | 11 | 5 | 3.636% | 11 | 16.09 |
| 7–8 | 11 | 0 | 0% | 5 | 62.09 |
| 10–11 | 10 | 0 | 0% | 0 | 266.4 |

The registered joint-64 headroom threshold requires at most 80% certified closure. Observed closure is 15/32 = 46.875%, leaving 17 unsolved roots at that illustrative cap. **Headroom passes.** The public exact solver still saturates unrestricted closure, so the scientific target is improved search efficiency under measured resource limits, rather than superiority over the unrestricted exact solver.

## Next experiment: learned search ordering under measured budget

Build a train-only verified distance/next-rewrite teacher from the 64 frozen training roots, with every positive action backed by a concrete Lean-compiled continuation. Keep alternative actions unlabeled unless their distances and evidence are explicitly audited. Use the existing 546 source reference frames and predeclare a compact recurrent policy with a same-worker duplicated CPU training protocol and within-state shuffled sham.

The primary comparison should be learned heuristic ordering versus public bidirectional BFS and stronger inexpensive public heuristics such as Hamming/edit-distance guided best-first search. Fix the evaluation root roster prospectively before fitting; use a new development cohort for scientific evaluation. Compare certified closure at explicit expansion/edge caps and measure Lean calls, CPU wall time, model calls and native tactic costs separately. Include exact public-planner results as a ceiling and characterize all seed-paired uncertainty by theorem lineage. A positive developmental result would motivate independent sealed confirmation, not immediate architecture promotion.

No model was fitted in this study. No ordinary/OOD task contents were opened by the evaluators. The existing learned-advantage gate remains unpassed.

## Reproduction

- [PR #334](https://github.com/MoralityLabAI/LoopedTransformers/pull/334)
- [Protocol](../docs/lsa02k_native_equality_preflight_protocol.md)
- [Equality suite](../research/lean_cgrt_hrm400/ppt_gym/equality_suite.py)
- [Public search and native replay](../research/lean_cgrt_hrm400/ppt_gym/equality_search.py)
- [Native source and evaluation](../research/lean_cgrt_hrm400/ppt_gym/equality_preflight.py)
- [Completed CI run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45281056858)
- [Raw native preflight artifact 14870455825](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45281056858/artifacts/14870455825)

The artifact stores the frozen suite, training references, independent Lean proofs, native action journals, exact solver and capped per-root receipts. Its configured retention is 14 days. This permanent report preserves the verified numerical results and gate.
