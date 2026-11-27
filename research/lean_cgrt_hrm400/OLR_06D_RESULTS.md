# OLR-06D: completed bounded opaque rewrite experiment

**Date:** 2026-11-24. Native engineering PASS; learned residual gate negative.
[Registration](OLR_06D_REGISTRATION.md) · [CI 45319896547](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45319896547) · [artifact 14886919196](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45319896547/artifacts/14886919196).
Executed code 96fc796acf8510dc5950cbb9b438a2e23204d3f8. The three protocol tests, including exhaustive small-graph PDB admissibility and consistency, passed.

The source contains 576 training and 144 separate calibration frames from 60 equation systems. All 720 selected source continuations compiled independently without axioms; 60 source root episodes were replayed natively. All 15 learned fits completed. The frozen evaluation has 32 new equation systems, 608 policy trials, 60 distinct successful native path replays and 60 independent axiom-free certificates. No old sealed LSA theorem contents were parsed.

## Outcome

| Policy | DEV at 64 | Transfer at 64 | DEV at 128 | Transfer at 128 |
| --- | ---: | ---: | ---: | ---: |
| Cost-partitioned PDB + weighted search | 14/16 | 12/16 | 15/16 | 15/16 |
| Bidirectional BFS | 9/16 | 6/16 | 11/16 | 9/16 |
| Hamming/2 + weighted search | 6/16 | 7/16 | 9/16 | 8/16 |
| All learned PDB residual arms, per seed | 14/16 | 12/16 | 15/16 | 15/16 |

Unrestricted public bidirectional search certified 16/16 on each panel. Every fitted checkpoint selected residual scale 0 by the registered source-only calibration grid. Source PDB MAE .685764; calibration PDB MAE .847222. All calibrated models have those same errors. Thus all learned policies execute identical PDB ordering with additional model work; differences in their architecture cannot be interpreted as effects on the selected proof path. Stored parameters: bi/one-way 19,873; dense 19,233; local 7,425; full 185,793.

PDB mean concrete expansions 28.44 DEV and 41.81 transfer versus unrestricted bidirectional 152.94 and 177.63. Mean concrete edges 193.44 and 295.75. This is substantial concrete-search headroom improvement over the old control.

## End-to-end accounting audit

Downloaded archive and independently verified all 18 manifest-listed files. Mean hosted planning timings (ms), excluding native certification:

| Policy | DEV | Transfer |
| --- | ---: | ---: |
| Unrestricted bidirectional BFS | 2.717 | 3.503 |
| PDB setup + bounded weighted search | 6.464 | 9.175 |
| Same PDB + bi reader | 31.442 | 47.382 |

PDB construction across 32 roots enumerated 17,494 abstract states and 97,922 abstract edges. The abstract-state counts and concrete expansions are distinct work units. If counted together under an illustrative equal-unit cap, none of the PDB paths fits 256; at 512 the counts are DEV 9/16, transfer 0/16, at 1024 15/16 both. These are offline accounting of the recorded completed paths, not independently rerun cap policies. The native proof budget is separate. The PDB is useful for reducing concrete search effort, but these timings do not demonstrate end-to-end speedup on single roots. No matched-FLOP neural benefit.

Registry manifest SHA256 ee03f60eaf6f5b0cfc7930ac79d3fcc2215615853740816a4c0cba7fd7969179. Fit manifest 420114835339a8ddfca5e6b5f3c6dcc42d6817c59898113512de789949e97501. Eval manifest b638265a7748672a42d77f70933945654aad4c72f9075b8d5fc8ca2583d07416.

## Decision

Retire this learned distance residual family. Do not turn calibration off after observing the result. Retain the exact admissible pattern bound as a baseline. Test multi-obligation reuse next: one public equation system, a common witness, several required targets, a global budget. Cache explored transitions and connect pending goals through exact shared path incidence. The graph exploration ledger must remain proposal-only until Lean certifies the selected paths. A learned scheduler has a plausible role only if strong shared exact search leaves resource-bounded headroom. See [06E registration](OLR_06E_REGISTRATION.md). No 5M promotion.
