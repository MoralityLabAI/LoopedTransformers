# OLR-06D: bounded opaque rewrite search with an exact obligation ledger

Date: 2026-11-24. Prospective development registration. Parent: OLR_06C_RESULTS.md.
This gate studies amortized ranking of public rewrite states. A bounded search
failure is censored; it is not proof of non-equivalence. No architecture promotion.

## Native grammar and custody

Reuse the exact LSA-02K EqCase, neighbor, native replay and standalone compiler
from main commit 3230f1c50d65fdfa47c30fab789bb730c2f8c0f5. Their opaque unary function equations rewrite
two adjacent letters of a fixed-length word. The proof transports a witness
P(source x) to P(target x), with at most 16 native tactics. Native Lean 4.19
certifies each successful path. A symbolic neighbor or shortest-distance
label is a proposal within this finite grammar, not a native negative label.

Freeze new equation-system-disjoint cohorts using seed 20261220: 48 TRAIN,
12 source-only calibration, 16 development, 16 length-transfer roots.
Use lengths 7–9 and exact generation distance shells 4–5,7–8,10–11;
transfer length 10. Exclude all 160 prior LSA-02K alpha lineages from its
original manifest (artifact 14870455825, run 45281056858, manifest SHA256
d928d8b5ffbbd9760e02fc9af903704bdac0cea7adc0b6e5be9103e7f2153874).
Only the old manifest is parsed; no old sealed theorem rows are opened.
All new cohorts are explicitly unsealed development evidence.

## Exact and learned heuristics

Compute an admissible cost-partitioned pattern database (PDB) from the public
equations. Partition positions into disjoint contiguous windows of maximum
length 3, using all three offsets, also partitions of widths 1 and 2.
Internal pair rewrites cost 2 half-units; crossing a window boundary costs
1 half-unit for the affected position. Project all possible public rules,
including existentially abstracted boundary symbols. Dijkstra from the goal
pattern gives exact abstract distances. A partition sum divided by 2 lower
bounds the full rewrite distance, since each concrete rewrite contributes
at most 2 half-units across windows. Take the ceiling of the maximum across
partitions, and max with ceil(Hamming/2). Unreachable abstraction is an
explicit infinite lower bound. Retain actual PDB setup work.

All learned arms receive the same complete public word-pair encoding,
canonical equation roster, legal direction mask at each adjacent factor,
position/pair incidence and PDB/Hamming features. Predict residual
(d_exact - h_PDB) / 15, not semantic truth. A position/factor graph uses
one shared cell with immutable input reinjection, separate position→factor
and factor→position messages, width 32 and three visits. The one-direction
control masks position→factor but executes both computations. Controls:
bidirectional, one-direction, local pooled MLP, dense tied attention and
full-packet MLP. Equal parameters and operation schedule only for matched
direction arms. Report actual parameter counts and CPU timings.

Source labels from exact reverse BFS within each TRAIN/calibration system,
12 selected nontrivial frames per root (include root target, sample across
reachable depths). All sampled positive continuations compile independently
in Lean; replay one reference per source root live. The symbolic BFS distance
is exact within grammar, not necessarily minimum arbitrary Lean tactic count.
Train 3 seeds 17,29,43, 160 AdamW updates, batch 32, lr .002.
Select nonnegative residual scale from 0,.25,.5,1,1.5,2 by minimum
calibration absolute distance error, using only source calibration labels.

## Search and reporting

Weighted best-first graph search with priority g + 1.5*h. All learned
variants use h = h_PDB + calibrated nonnegative predicted residual.
Classical controls use PDB or Hamming (half Hamming rounded up). A learned
residual need not be admissible; no optimality claim for weighted search.
Use public bidirectional BFS and unrestricted exact solver as additional
references. Caps: 64 and 128 expanded words, one enumeration per expansion,
15 rewrites + terminal exact w. Search must not open a hidden solution
or native reference path before policy choice. Reopen only at improved g.
Certify discovered paths by actual Lean replay and independent compiler.
Record model batches/rows, neighbor edges, expansion caps, planner and native
wall time separately. PDB construction is charged as separate symbolic work.
Primary endpoint native certified completion at 64 expansions. Secondary:
128-expansion closure, native tactic count, time, and source distance error.
Paired root-clustered descriptive intervals, three seeds retained per root.
Exact unrestricted reference saturates the grammar by construction.
Fitting and evaluation are exploratory; no FSI-01 or 5M promotion.
If the PDB or bidirectional classical planner remains stronger, retire
learned distance ranking for this family rather than weaken the control.
