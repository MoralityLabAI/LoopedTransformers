| Card | Rows | Baseline correct | Policy correct | Delta | Best control delta |
| --- | --- | --- | --- | --- | --- |
| 4sample | 4 | 2/4 | 4/4 | 0.5000 | 0.0000 |
| 8sample | 8 | 4/8 | 8/8 | 0.5000 | 0.0000 |
| 32_seed23 | 32 | 11/32 | 17/32 | 0.1875 | 0.0312 |
| 32_seed37 | 32 | 9/32 | 17/32 | 0.2500 | 0.0000 |

The mean card delta is 0.359375. Weighting scored appearances by card size gives 0.263158 across 76 rows. The four-example card is contained in the eight-example card; the two later cards are disjoint from those discovery cards and from each other by recorded trajectory ID. The audit's stated identity rule yields 72 retained records with 24 baseline hits and 42 policy hits. The full audit preserves both identity definitions and repeated-record diagnostics. Identity rule: `env:trajectory_id:step`. These events span 71 recorded episodes; one episode appears at two different steps. All four cards replay their stored per-row decisions exactly.
