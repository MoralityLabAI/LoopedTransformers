# LOGRANK-BPD-22: resident and evidence-aware separator memory

Development only. Protocol registered before the main run. 18 fresh worlds (12/96 dimensions, three independent roots per dimension and family: chain, two-hub chain, global triples), 36 paired noise conditions (.15/.35). 24 structured conditions admitted; 12 dense conditions refused by scope >6 before table allocation. Four epochs of 16 pair queries, eight carried across consecutive epochs, with one new noisy unary observation between epochs. Factorial stable/shifted source popularity crossed with local/uniform update sites; query streams match across update regimes, update streams match across popularity regimes. Local site is public, uniform actual draws are independent of the four sampled forecast scenarios. 11 arms per admitted regime, 1,056 trajectories. No future query answers, native training, protected data or confirmation. Independent unit is world.

## Cost model

Fix a dependency DAG, a pinned target cache C, valid resident set R and direct demand d. In topological order from requesters to dependencies, D_e=d_e+sum of parent misses. A retained resident in C has zero misses; a cold selected message has min(1,D_e); an unselected message has D_e. Work W(C,R;d)=sum_e w_e M_e. A selected message is filled by the end of the epoch iff it was already resident or D_e>0. Proof: each miss requests every dependency once; a selected cold message fills on its first miss and cannot be evicted during the epoch. Induction in dependency order proves the counts, independent of request order. This accounts for cold fills exactly in the counted unsigned model; signed query contractions and policy overhead are separate. It does not imply global optimality or the warm-cache submodularity theorem for a changing target cache.

Resident greedy selects the best positive predicted reduction in current work per charged byte. Retention greedy evaluates current work plus the average predicted next-epoch work under the same target and current demand forecast, after removing messages whose source-side component contains a sampled changed owner. Final epoch uses current-only objective. Known public local site uses one scenario; uniform uses four iid public scenarios drawn independently of realized future updates. Forecast is not the full uniform expectation or future query knowledge. Both planners evaluate all feasible single additions and compare exact integer ratios; no global knapsack or dynamic-policy optimum is claimed. Lazy fills, selective invalidation, fixed conservative byte charges from 21, same caps 16/64 KiB. Charges exclude temporary arrays, policy objects, factor tables and actual allocator/RSS. Direct recomputation and LRU, planned cold and warm marginal are controls.

## Results

Mean counted message arithmetic, 96D hub chains (six paired noise conditions, three roots per stratum):

| Regime | KiB | LRU | Planned cold | Warm marginal | Resident | Retention |
|---|---:|---:|---:|---:|---:|---:|
| stable/local | 16 | 88524 | 69591 | 46028 | 43324 | 43208 |
| stable/uniform | 16 | 88524 | 69899 | 45966 | 43012 | 43454 |
| shift/local | 16 | 114975 | 93171 | 64773 | 61930 | 61796 |
| shift/uniform | 16 | 115223 | 93220 | 64284 | 61148 | 61360 |
| stable/local | 64 | 62972 | 38538 | 38538 | 37421 | 35198 |
| stable/uniform | 64 | 65933 | 37901 | 37901 | 36850 | 35846 |
| shift/local | 64 | 86296 | 55485 | 55485 | 54044 | 52189 |
| shift/uniform | 64 | 89047 | 54707 | 54707 | 54110 | 53526 |

At 64 KiB retention reduces mean arithmetic vs planned cold by 8.7%, 5.4%, 5.9%, 2.2% respectively. At 16 KiB resident improves on warm marginal in every aggregate regime, while retention is nearly tied with resident, and slightly worse under uniform updates. The new planners lose badly on elapsed time: at 64 KiB stable/local, planned cold 26.49ms, resident 156.59ms, retention 297.02ms. Timing is descriptive; main 96D run overlapped part of the 12D audit, arm order rotated. All predictions exact; no promotion of expensive policy on runtime evidence.

## 22B: sparse-delta implementation diagnostic

After main results, replace full vector candidate evaluation with exact delta propagation. The directed message dependency graph of a tree has unique nonbacktracking dependency paths from any candidate edge to its descendants. Adding a target changes its local miss count; propagate the difference downstream. For each reached dependency update its current incoming demand, miss count and final fill flag, then compute future-scenario misses from the updated fill flag and survival indicator. Recurse whenever current or any future miss count changes. Unreached branches have unchanged inputs, cache membership and outputs. Induction along the unique paths proves equality to full candidate evaluation. This argument is specific to the tree dependency graph, not arbitrary reconvergent DAGs.

Paired isolated rerun (alternating implementation order) on all 96 resident/retention hub trajectories at 96D reproduces all non-timing records from the original run, including greedy plans, exact outputs, events and work. At 64 KiB retention vector vs sparse mean runtime in ms: stable/local 327.50 vs 99.85; stable/uniform 449.40 vs 146.94; shift/local 354.72 vs 105.91; shift/uniform 477.64 vs 164.57. A 2.90-3.35x implementation speedup, still slower than planned cold's main-run times. Exposed-root engineering diagnostic, not prospective evidence or a new optimality result.

## Verification

Independent replay PASS: 36 graph replays, 96 factorial-stream checks, 384 exact partition references, 6,144 independent query references, 786,432 enumerated states, 67,584 exact arm-query comparisons, 4,224 arithmetic ledgers, 385,937 cache transitions, 2,880 invalidation replays, 1,536 predicted unsigned-work and fill-state matches (recursive request replay and actual execution), 1,536 warm-plan and 1,536 resident/retention-plan replays. All 22,535 resident/retention selected steps independently checked for cost, feasibility and positive gain; full candidate maxima at 9,955 steps and all 768 stopping points at 12D. The 96D audit checks selected steps and final work, not independent maximization of every candidate. No independent allocator or timing measurement, CI, Lean or native integration. Full raw receipts, root-level summaries and sparse diagnostic in bundle outside Git.

## Reproduce

Python 3.13.5, NumPy 2.3.5; pinned copies of the earlier solver, cache and independent reference included. Run sequentially for less timing interference.

```
for n in 12 96; do OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot22.py "$n"; python audit22.py "$n"; done
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sparse22.py 96
```

Next: exploit the exact cold-fill objective as a cheap local improvement to a warm plan, using only a bounded shortlist of additions/swaps and a time-budgeted gate. Derive cost-aware survival scores from expected repeated demand, compare with direct marginal and planned cold on fresh worlds, and account for policy metadata, workspace and update forecast uncertainty. Retain exact reference and dense refusal. No deployment recommendation for the expensive retention planner.
