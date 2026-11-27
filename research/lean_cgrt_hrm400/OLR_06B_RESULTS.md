# OLR-06B.1: paid-relation allocation results

**Status:** exploratory synthetic gate completed; no learned-incidence promotion.  
**Registration:** [06B protocol](OLR_06B_REGISTRATION.md), committed before source fit.  
**Verified run:** [45317655282](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45317655282), commit b91445cd54dac4b62d4f0ea2e401efea62c2f69f, success.  
**Receipts:** [artifact 14885956451](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45317655282/artifacts/14885956451).  
**Parent:** [06A reproduction](OLR_06A_REPRODUCTION.md).

## Execution and custody

The exact signed ledger and paid-query environment passed all 11 CPU tests, including 50 independent Fraction-valued signed-constraint Bellman enumerations, 400 random signed-graph exhaustive comparisons, and 150 static budget checks. Two preflight failures concerned only the delayed-gain fixture and its relabeling test. The first fixture admitted a tied two-step opening that could recover the full optimum. The corrected four-query path strictly requires a choice beyond two-step immediate planning. The six-component relabeling test was updated accordingly. These corrections preceded source model fitting; no query score informed them.

The actual campaign froze 160 source roots and 40+40 evaluation roots with disjoint canonical weighted-graph signatures. Source supervision covered 4,363 reachable states. Three seeds (0,1,2) each fit all five arms for 240 AdamW steps. The evaluator froze 1,200 complete neural state-action tables before final scoring; 1,520 policy/root trials included 1,200 learned and 320 reference trials. Every policy was scored by exact expectation over registered independent Bernoulli query outcomes, and a sampled common-outcome trajectory checked signed closure separately. No native Lean certificates are claimed.

Terminal utility = resolved-weight fraction minus 0.01 per paid cost unit. The exact Bellman reference optimizes that utility, not full-resolution probability. All arms use public probabilities and costs, exact closure, immediate expected gain, the same candidate masks and budget. Model-visible incidence is an already known public structure; unknown semantic binding is outside this pilot.

## Expected coverage and utility

Reference policies run once per root, learned policies over three seeds on the same roots. Root is the independent sampling unit (40 per panel). Transfer combines more components, longer/diamond layouts and larger budgets.

| Policy | Parameters | DEV coverage | DEV utility | Transfer coverage | Transfer utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact Bellman | 0 | .50616 | .47639 | .53587 | .48937 |
| Two-step receding horizon | 0 | .49790 | .46826 | .52275 | .47578 |
| Bidirectional incidence | 33,265 | .47119 | .44337 | .49616 | .44901 |
| Query-to-component only | 33,265 | .46932 | .44143 | .49315 | .44617 |
| Component-to-query only | 33,265 | .46825 | .44042 | .48981 | .44272 |
| Dense tied attention | 30,385 | .46753 | .43953 | .49218 | .44499 |
| One-step greedy | 0 | .46560 | .43609 | .48155 | .43491 |
| Full-packet direct MLP | 44,073 | .38260 | .35329 | .38687 | .33877 |

Analytical leading MACs per model decision: 1,809,648 for all incidence variants, 1,554,768 for dense attention, 43,920 for direct MLP. These exclude activation/memory traffic and exact closure. They are not measured CPU FLOPs. The one-direction arms compute both projection paths and mask one output, matching the bidirectional arm's tensor schedule and initial parameters. Dense/direct comparison is not equal active compute. Exact planning runtime was not separately profiled and cannot be inferred from the zero neural MAC field.

Average expected queries for bidirectional: 2.628 DEV, 4.230 transfer; average paid cost: 2.782, 4.715. Exact reference: 2.577/3.847 queries, 2.977/4.650 cost. Remaining utility regret is .03302/.04036 for the bidirectional policy. Two-step regret is .00813/.01359.

Training/target construction elapsed 33.10 seconds and complete frozen scoring 17.25 seconds on one CPU thread (Python 3.11.18, Torch 2.5.1+cpu, deterministic algorithms, MKLDNN off). These aggregate run times include substantial offline evaluation and are not deployment latency comparisons.

## Root-clustered contrasts

Coverage difference in percentage points, descriptive 95% bootstrap intervals (10,000 resamples, model seeds averaged inside root):

| Bidirectional minus | DEV | Transfer |
| --- | ---: | ---: |
| Component-to-query | +0.29 [-0.17, +0.92] | +0.64 [-0.08, +1.33] |
| Query-to-component | +0.19 [-0.41, +0.94] | +0.30 [-0.88, +1.40] |
| Dense tied | +0.37 [-0.31, +1.08] | +0.40 [-0.66, +1.37] |
| Direct MLP | +8.86 [+6.03, +11.93] | +10.93 [+7.77, +14.21] |
| Greedy | +0.56 [-1.50, +2.93] | +1.46 [-0.71, +3.77] |
| Two-step planner | -2.67 [-4.05, -1.45] | -2.66 [-4.36, -1.17] |

Per-seed bidirectional utility DEV .44803/.44521/.43687; transfer .45215/.44814/.44675. The observed incidence advantage over dense or either matched direction is small and its intervals include zero. The two-step planner is stronger on both panels. The direct MLP is substantially weaker but uses roughly 1/41 of the analytical MACs per decision.

## Decision

Do not scale the 33K cell based on this gate. Fable's synthetic parity result replicates at short refinement depths, but exact closure handles that algebraic task. Here known independent success probabilities and simple graph structure make short exact lookahead strong; learning largely approximates greedy selection. The explicit exact ledger remains useful.

The next meaningful gate should introduce **semantic applicability uncertainty** rather than only more synthetic graph nodes: the model must estimate whether a native relation query is likely to resolve a scoped obligation, while all policies retain the exact ledger and legal menu. Compare a direct calibrated semantic scorer plus depth-two search against an incidence recurrent scorer plus the identical search, with matched source examples, probe budgets, and measured compute. A candidate query with no proof is an unresolved/censored outcome, never a refutation. Start with native Lean receipts for a small family of dependent equalities, revision validity, and legitimate competing bindings. Exact normalization and symbolic proof search remain mandatory. Advance only with a useful improvement over the direct-score/search combination on a prospective native cohort.

No general reasoning, native Lean or 5M architecture claim follows from 06B.
