# OLR-06E: shared frontier and proof-path ledger preflight

**Registered 2026-11-24 before generation/scoring.** Development engineering only.
Parent: [06D results](OLR_06D_RESULTS.md). The learned residual selected scale zero
in every fit, and PDB setup dominated single-root timings.

## Native task and source separation

Each root binds opaque unary functions a,b,c,d, six public reversible pair
equations, x, predicate P and a single witness w:P(source x). The complete goal
is the conjunction of K distinct target predicates P(target_i x). Every target
has a public grammar path of length <=11 by generator construction. No target
path, distance, or hidden solution is exposed in the frozen root row.

Freeze seed 20261224, 12 DEV systems (K=3, lengths 7–9), 12 load-transfer systems
(K=5, length 10). Cycle the existing reachable distance bands 4–5,7–8,10–11.
Exclude all 160 LSA-02K and all 92 OLR-06D alpha lineages by authenticating the
archived 06D bank (manifest ee03f60eaf6f5b0cfc7930ac79d3fcc2215615853740816a4c0cba7fd7969179).
The 06D roots are all unsealed. Do not parse old sealed LSA roots. No training
or calibration in this preflight; no future learned benefit claimed.

## Shared resource contract

Global concrete neighbor-enumeration caps 128,256,512 per conjunction. Every
leaf path has <=15 rewrites plus terminal witness (16 native tactics per leaf).
All policies run to 512 or terminal, preserving deterministic prefixes at
lower caps. Count first discovery of each feasible leaf path, logical frontier
visits, actual neighbor enumerations, actual enumerated candidate edges,
cache hits, path length, PDB setup and elapsed planning seconds. PDB construction
is metered separately; also report illustrative combined abstract+concrete
counts at each cap. A cache hit is not a fresh neighbor enumeration, but does
incur logical search work/time. Report both. No claim that these counts equate
to FLOPs. Stop immediately when the fresh cap is reached. Capped misses remain
censored. Per-root native certification costs are separate from planner cost.

Registered arms:
- independent round-robin bidirectional BFS: a separate bidirectional search
  for each target, global shared cap, no adjacency cache;
- cached independent: same scheduling and search, shared exact neighbor cache
  within the root (not shared visited frontiers);
- reverse BFS from the common witness, shared by all targets;
- shared proposal ledger with exact undirected discovered rewrite incidence
  and union-find frontier components. Each expansion enumerates all neighbors,
  adds actual public grammar edges and merges components. Recover shortest
  paths in the discovered subgraph by BFS from the witness. Balanced scheduler
  selects the active component with smallest unexpanded frontier, then the
  smallest discovery-level boundary word. Demand scheduler divides frontier
  size by pending target count (witness component uses total unresolved count).
  These are deterministic heuristics over exact state, not learned recurrence;
- PDB round-robin weighted best-first: shared database with exact source goal,
  shared adjacency cache, independent target frontiers, same 1.5 heuristic
  weight as 06D. Include the abstraction setup count/time;
- unrestricted independent bidirectional references, evaluated after all
  bounded paths freeze, as a ceiling only.

Deduplicate successful leaf paths for native grading; each actual successful
path must pass fresh Lean native replay and independent compilation against
the original EqCase. Compile full original conjunctions using only those
independently checked leaf lemmas and explicit And.intro terms. No solver in
the conjunction proof. Count distinct actual native leaf replays and standalone
leaf and conjunction declarations. A root is credited at a cap iff every
required target has a certified path found by that cap and its conjunction
certificate passes. Partial coverage is certified leaves / K.

## Decision

Primary endpoint complete native conjunctions at 256 concrete expansions;
secondary at 128/512, certified leaf coverage, unique and logical expansions,
actual elapsed time, and illustrative total including PDB setup. Report
paired-root descriptive intervals. Require at least 20% of roots unsolved
by the best registered same-information shared structural controller at
256 as a *headroom diagnostic* before investing in a learned frontier
scheduler. Include cached independent and exact shared graph controls in any
later comparison. An unrestricted exact solver is expected to saturate.
No architecture promotion, FSI-01 or native learning claim.
