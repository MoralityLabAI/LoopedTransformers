| Historical experiment | Arm | Eval loss | Bucket accuracy |
| --- | --- | --- | --- |
| SVD static | base_frozen | 1.001876 | 0.515625 |
| SVD static | vpd_static_selected | 0.972768 | 0.531250 |
| SVD static | random_trunk_control | 1.002200 | 0.515625 |
| Learned static | base_frozen | 1.001876 | 0.515625 |
| Learned static | vpd_selected | 1.016567 | 0.500000 |
| Learned static | random_trunk_control | 0.991228 | 0.515625 |
| Learned validated | base_frozen | 1.001876 | 0.515625 |
| Learned validated | vpd_validated | 1.004116 | 0.515625 |
| Learned validated | random_trunk_control | 1.004754 | 0.515625 |
| Learned validated (0 steps) | base_frozen | 1.001876 | 0.515625 |
| Learned validated (0 steps) | vpd_validated | 0.999113 | 0.515625 |
| Learned validated (0 steps) | random_trunk_control | 1.001876 | 0.515625 |

| Identity used for dataset comparison | Shared distinct identities | Evaluation rows sharing a training identity |
| --- | --- | --- |
| exact text | 637 | 666 / 3,270 |
| row id | 0 | 0 / 3,270 |
| task prefix before MODEL ACTION | 2591 | 3,117 / 3,270 |

The archived dataset has disjoint row identifiers. Exact text overlaps remain, and task context before the literal `MODEL_ACTION:` delimiter is shared by most evaluation rows. This source-task identity check exposes a different level of dependence from a row-ID split. It also leaves the historical 64-row evaluation subset unauthenticated because the exact invocation and subset IDs are absent.
