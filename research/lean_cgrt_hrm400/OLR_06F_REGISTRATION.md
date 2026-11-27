# OLR-06F: anytime abstraction and concrete-search allocation

**Registration:** 2026-11-24, before generating the new cohort or scoring.
**Parent:** [06E results](OLR_06E_RESULTS.md). Its concrete-expansion headroom gate failed. No learned model is fitted in 06F. This is a separate total-computation endpoint, with actual CPU timing reported.

## New algorithm and proof contract

A pattern table is an incrementally computed reverse Dijkstra search on a projected rewrite window, in half-cost units. Let S be the settled states and q the minimum valid tentative distance among unsettled frontier entries. For projected pattern p use its exact distance if p is settled, otherwise q. If no frontier entry remains, an unsettled pattern is unreachable in the abstraction. This is a valid lower bound because all states with smaller distance have been settled; as work proceeds the bound is nondecreasing. If p is settled later it receives its exact value. Apply the registered boundary cost partition: inside-window rewrite cost 2, cross-boundary affected symbol cost 1. Sum over disjoint windows, ceil(sum/2), then maximize across partitions and Hamming/2. Every concrete rewrite costs at most two half-units across a partition. This lower bound remains valid even when some tables are partially computed. Weighted best-first search uses it as a ranking heuristic, without an optimality guarantee.

Reuse tables keyed by target-window pattern and left/right boundary flags within the same equation system. Share exact concrete adjacency across target searches. Each target has its own priority queue and best known g. On extracting a queue entry, recompute the current bound and reinsert if the priority increased; all stale entries are lower estimates of current priority. A common read-only abstraction state is updated between concrete visits, not during one. No neural state or hidden reference path is used. Every discovered leaf remains a proposal until fresh native Lean replay and independent compilation; full conjunction certificates reuse the certified leaf lemmas.

## Allocation

New seed 20261228. 12 DEV roots (3 targets, lengths 7–9), 12 transfer roots (5 targets, length 10). Generate distinct public equation systems and targets in the same three distance bands as 06E. Authenticate the 06E bank from artifact 14888520606 (registry manifest dd50ca18548b334c8b943b0122ece1305c3d223f7ff4f5f96b79c1f8d74e83e8), exclude its 24 roots and 252 older alpha lineages. No old sealed LSA roots are opened. This is unsealed development evidence.

Charge one unit for each fresh concrete neighbor enumeration and each settled abstract state enumeration. Both costs, enumerated edges, cache hits, logical visits, queue rekeys and wall time are reported separately. These equal-unit counts are an illustrative total-work metric, not equal FLOPs or measured runtime. No free abstraction work: heap initialization, stale-entry removal and other overhead appear in CPU timings. Total budget caps 128,256,512. Stop immediately at the cap. Cache-hit discovery requires current charged work +1 as its minimum permitted cap; fresh discovery uses actual charged work. Freeze all paths before reference and native grading. Maximum 15 rewrites per leaf plus exact witness. Native calls and independent grading costs are separate.

Arms:
- cached independent bidirectional BFS and shared demand ledger from 06E, no abstraction;
- cached round-robin Hamming weighted search, same dynamic queue machinery;
- eager width <=2 and eager width <=3: complete all corresponding pattern tables in deterministic width/key order before concrete search; if budget ends during setup, no free search;
- anytime cyclic quarter: one abstract expansion per four scheduled operations (then three concrete visits); deterministic cyclic table selection;
- anytime demand quarter and anytime demand half: same respective one-in-four/one-in-two schedules; prioritize a table by the number of current frontier-window occurrences whose pattern has not settled, divided by 4^window width. Ties deterministic. If no eligible abstraction exists, do a concrete visit. Concrete visits use round-robin target frontiers. No teacher labels.
- unrestricted independent bidirectional BFS after freeze, ceiling only.

Schedule counts logical visits plus abstract expansions, while charged budget counts fresh concrete and abstract expansions. A cached visit consumes a logical schedule slot and no additional charged work; report both counters. Use identical queue priority g+1.5h and native budget for weighted search arms.

## Gate

Primary endpoint full native conjunction completion at 512 total work units. Secondary 128/256 completion, native leaf coverage, planning time, abstract and concrete edges/expansions, actual native tactics. Compare paired roots. Require at least 20% unsolved for the strongest fixed same-information policy in each panel combined (best per-panel totals), and a hindsight portfolio gap of at least 10 percentage points over that fixed-per-panel baseline before registering a learned compute allocator. Hindsight portfolio is explicitly privileged and cannot be a deployable policy. This is only a headroom screen; if it fails, close this finite rewrite family for learned scheduling rather than weaken controls.

Verify partial bound monotonicity, admissibility against exhaustive small graphs, final equality with full PDB, no free setup, budget prefixes including cached visits, and native proof certificate composition. Zero model training, no FSI-01 confirmation, no 5M promotion.
