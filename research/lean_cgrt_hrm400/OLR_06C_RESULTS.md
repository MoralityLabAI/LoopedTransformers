# OLR-06C.1 results: scoped native Boolean applicability

**Status:** native receipt/transport engineering gate passed; learned recurrence gate unresolved/negative on transfer.  
**Registration:** [OLR_06C_REGISTRATION.md](OLR_06C_REGISTRATION.md), bed5974.  
**Executed code:** 8e7009e46d36c69c8468c0642e33f11f94dd7bd2.  
**Verified workflow:** [45318621772](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45318621772), native-semantic job 138191440777, success.  
**Artifact:** [14886303117](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45318621772/artifacts/14886303117).

## Native protocol and execution

The registered family consists of four or five public constant Boolean functions of three scoped variables. Exact prefix syntax is disjoint across roots; transfer increases expression depth and changes query topology and obligation count. This is syntactic/load transfer within a two-valued semantic family, not semantic disjointness. Positive anchors and signed relations (both orientations) were proved using actual Lean 4.19 case-split/reflexivity macros, checked by the native kernel and replayed from reset state. Each attempted macro costs one registered query unit. A compound relation macro contains more elementary work than an anchor macro; the query budget is not a claim of equal elementary tactic counts or FLOPs.

All native macro verdicts agreed with the independent eight-assignment evaluator. Rejection restricts predictive support in this specifically exhaustive grammar but supplies no positive ledger receipt. Each learned model produces a factorized prior over node signs from public syntax and, for graph arms, public candidate incidence. Source labels are native query acceptance bits. All fitted priors are frozen before evaluation-native checks. The planner enumerates sign assignments under that prior, conditions on observed verdicts, and ranks queries by the identical two-step expected-utility search. Posterior knowledge never substitutes for a positive proof receipt.

The native outcome cache is constructed once per root and reused at selected paid indices during adaptive policy evaluation. Thus repeated policy trials are not independent native executions. Independent Lean source compilation first proves accepted receipt lemmas, then builds explicit existential/AND/equality transport terms from only the positive receipt subset selected by a policy. No final Boolean search is used for the composed witness terms. Every nonempty selected subset has a compiled certificate; partial certificates are not full original roots.

| Meter | TRAIN | DEV + transfer |
| --- | ---: | ---: |
| Distinct roots | 64 | 48 |
| Distinct native query macros | 558 | 465 |
| Fresh reset-state replays | 558 | 465 |
| Positive standalone receipt lemmas | 281 | 232 |
| Full-root transported declarations | 64 | 54 |
| Partial transported declarations | 0 | 128 |
| Axiom-free standalone declarations | 345 | 414 |

Full-root declaration count can exceed root count because different selected receipt masks may close the same root. The evaluator scored 912 policy/root trials (720 learned and 192 deterministic reference). The source and score steps took 55.65 and 44.13 seconds respectively in the hosted run. These timings include native queries, compilation, fitting and/or offline scoring; they are not deployment latency measurements.

## Held-out certified outcomes

Each learned arm uses seeds 0,1,2, 200 AdamW updates, batch 16, and width 48. Bi and one-way share initialization, parameter count, sampler and the same tensor operation schedule with one direction masked. Dense and direct variants have different compute. All learned arms use the same planner and full registered query budget; the local MLP reads individual expression syntax, while the full-packet MLP receives encoded expressions and incidence. Both symbolic reference variants use the same complete public syntax. Their finite evaluator is not privileged.

| Policy | Parameters | DEV full | DEV coverage | DEV utility | Transfer full | Transfer coverage | Transfer utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact Boolean evaluator + depth 2 | 0 | 24/24 | 1.0000 | 1.2100 | 24/24 | 1.0000 | 1.2000 |
| Exact Boolean evaluator + full horizon | 0 | 24/24 | 1.0000 | 1.2100 | 24/24 | 1.0000 | 1.2000 |
| Bidirectional incidence | 81,865 | 51/72 | .9167 | .8940 | 23/72 | .8250 | .4714 |
| One-direction incidence | 81,865 | 46/72 | .9028 | .8215 | 34/72 | .8444 | .6297 |
| Dense tied attention | 79,225 | 48/72 | .9028 | .8490 | 28/72 | .8444 | .5456 |
| Local syntax MLP | 53,977 | 49/72 | .9063 | .8639 | 27/72 | .8278 | .5276 |
| Full-packet MLP | 158,237 | 34/72 | .8507 | .6403 | 19/72 | .8000 | .4093 |
| Uniform prior + depth 2 | 0 | 5/24 | .7604 | .3530 | 4/24 | .6833 | .2825 |
| Uniform prior + full horizon | 0 | 5/24 | .7500 | .3504 | 4/24 | .7083 | .2888 |

Utility is indicator(full original root) + .25*resolved fraction - .01*attempted queries. Learned trials repeat three seeds on the same 24 roots per panel. The exact evaluator closes every root with four or five positive queries, respectively. Within the registered receipt language, n queries are necessary to resolve n initially unanchored independent sign variables: each anchor or relative equation adds at most one independent binary constraint. A connected spanning-tree receipt set plus one anchor attains that bound. This is a query-language lower bound, not a general theorem about arbitrary Lean proof search.

Bi full-root results by seed: DEV 17,19,15 of 24; transfer 7,8,8. One-way: DEV 16,13,17; transfer 12,10,12. Average bi query expenditure 4.347 DEV and 5.431 transfer; one-way 4.306 and 5.361. Exact evaluator 4 and 5.

### Semantic accuracy and calibration

| Reader | DEV query accuracy | DEV query NLL | Transfer query accuracy | Transfer query NLL |
| --- | ---: | ---: | ---: | ---: |
| Bi | .7380 | 1.3465 | .5947 | 2.0545 |
| One-way | .7319 | 1.3324 | .6654 | 1.5405 |
| Dense | .6848 | 1.4016 | .6359 | 1.6522 |
| Local | .7357 | 1.1775 | .6149 | 1.6885 |
| Full | .5984 | 1.2008 | .5641 | 1.1121 |
| Uniform | about .50 | .6931 | about .49 | .6931 |

Training query accuracies lie approximately .96-.99, with low source NLL. Every learned prior has higher held-out query NLL than uniform. This is severe miscalibration/overconfident errors; source fit does not justify interpreting the scores as calibrated probabilities. The learned prior improves some proof allocation relative to uniform, but this does not establish a well-calibrated semantic model. Temperature scaling or regularization needs a fresh source-only calibration split, not tuning on these evaluation panels.

### Root-clustered contrasts

Descriptive 95% bootstrap intervals resample 24 independent roots within each panel, retaining all seeds within each root. Full completion differences (percentage points):

| Bi minus | DEV | Transfer |
| --- | ---: | ---: |
| Matched one-way | +6.94 [-1.39,+15.28] | -15.28 [-23.61,-8.33] |
| Dense | +4.17 [-5.56,+13.89] | -6.94 [-16.67,+2.78] |
| Local syntax reader | +2.78 [-4.17,+9.72] | -5.56 [-11.11,-1.39] |
| Full-packet reader | +23.61 [+6.94,+40.28] | +5.56 [-11.11,+22.22] |
| Uniform prior depth 2 | +50.00 [+31.94,+68.06] | +15.28 [+1.39,+30.56] |

No powered mechanism claim follows. The direction-specific contrast reverses under the registered transfer distribution. The simplest complete Boolean evaluator is exact and cheaper in algorithmic work than running a learned inference model plus search; no comparative measured latency claim is made.

## Decision

Retain the verifier-owned ledger and positive-only proof transport. The native semantic query interface is now independently kernel-backed. Do not promote the learned bidirectional prior: its small development advantage does not transfer, and the public finite evaluator saturates. This family is closed as an engineering preflight.

Next investigate opaque, scoped rewrite witnesses with a bounded native semantic search. A query proposes a relation or witness binding between public terms, and actual Lean verifies a bounded proof attempt. Search failure must be censored except where an independently established exhaustive alternative licenses a negative fact. Use complete public terms and source hypotheses. Compare exact normalization/congruence where applicable, bounded bidirectional symbolic search, a direct calibrated candidate scorer with that same search, and the incidence recurrence under matched active compute. The native task must leave meaningful headroom under a common finite solver/tactic budget, rather than merely lengthening a Boolean truth table. Use source-only calibration and fresh normalized expression-system splits. No 5M scaling or FSI-01 confirmation based on this run.
