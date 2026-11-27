# OLR-06 implementation work order

2026-11-24. Derived from the supplied Fable pilot and review logs, and the native OLR-05 results. This is our independent specification; the inaccessible reviewer artifact is not required.

## Decisions

1. The ledger is exact and verifier-owned. Store immutable scoped receipts, revisions and incidence. Compute known constraint closure outside the network. Unknown, contradiction and rejected tactic are distinct statuses.
2. Start from the supplied 33,123-parameter Structured bidirectional cell (width 48). Separate parent-to-child and child-to-parent projections, scatter aggregation, immutable input reinjection, tied residual update. Use only admitted known edges for exact closure; candidate unknown edges remain explicitly typed for learned query scoring. No generic attention over stored facts.
3. Retain width 48 and 5M scaling embargo until a resource-matched useful native result. Pure parity accuracy is a preflight.

## OLR-06A: reproduction

Reproduce supplied ledger_pilot.py with original seeds 0,1,2, 2000 updates, source 20000 graphs, dev/deep 3000 each. Archive per-seed metrics at visits 8,16,32. Preserve original generator and sampler before modifications. Record actual compute, unknown accuracy, anchor distance, exact solver comparison. Add permutation and irrelevant-edge tests. Do not claim reproduction until run.

## OLR-06B: paid relation discovery

Build a finite-horizon synthetic gym. Public state consists of signed known edges, anchors, candidate query endpoints, costs and declared success probabilities. Each query has a hidden outcome sampled from a registered public distribution; only the environment reveals it after payment. Anchor probes reveal a sign; relation probes reveal a signed edge or an inconclusive outcome. Start with consistent hidden assignments. Recompute exact closure after every query. Unknown components remain unknown. Initially omit revision events to make exact dynamic programming tractable.

Primary utility: number of resolved obligations at budget B. Secondary cost and calibration. Compare exact closure + greedy expected gain, exact closure + depth-two lookahead, exhaustive belief-state planner on small graphs, direct candidate MLP, tied dense, one-direction and bidirectional incidence models. Exact static anchor knapsack remains mandatory. Give every policy identical public metadata. No hidden outcomes in policy features.

First construct paired cases where a zero-immediate-gain relation query enables later anchor coverage. Verify reference planner by exhaustive enumeration before fitting. Generate independent graph roots, with splits by topology and query-layout signatures, not numerical names. Source-only action values from the exact small-instance planner; no held-out fitting. Freeze seeds and model dimensions before scoring. Measure actual compute and query count; do not call equal updates equal FLOPs.

## OLR-06C: native transfer

Replace synthetic query outcomes with actual Lean-checked equality or dependent witness receipts. Rejection is censored unless an independently established exhaustive alternative licenses a negative fact. Preserve exact scope and revision validity. Require independent replay and axiom audit. Include normalization, unification and bounded symbolic proof search controls with full legitimate public types. Primary endpoint native certified root closure at fixed tactic budget.

## Decision

Before confirmatory evaluation use EvalMath FSI-01 to register power, paired-root sample size, effect floor and shams. A proposed minimum useful effect is five percentage points beyond strongest same-information control, positive paired-root lower confidence bound and consistent direction across seeds. Require no unsafe commits. Examine compute-matched edge-direction ablations and unrelated-edge perturbations calibrated to action margins. If exact or simple direct methods saturate the task, redesign the task rather than weaken controls. No 5M promotion on synthetic propagation accuracy.
