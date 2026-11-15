# AF-19: graph-ancestry STOP readout, scored local results

**Registered science: FAIL. Independent local engineering: PASS.** Frozen scientific source: commit 6ccae99a2336f725b87889229d72e04acca95116. The preregistered PROTOCOL.json SHA-256 is 2a92d76a6f71a09c3ce674755dc819f93c1a9e9a0957f1111178c804889937d7. This report documents local Python 3.13 and Torch 2.10 CPU scoring. Hosted pinned Python 3.12/Torch 2.5.1 and Lean 4.19.0 verification remains outstanding.

## Experiment and fairness

AF-19 uses the unchanged native TypedLoop(16) graph cell and the AF-18 Bellman/STOP training objective. Both candidate policies have **6,036 trainable parameters**, identical initial tensors per seed, identical 240 AdamW updates and sampled training states. Both compute the backward goal-ancestry graph mask and identical learned query/key/value attention. Only the selection of rule tokens differs: the matched local control reads direct goal-concluding rules, while the ancestor policy reads every rule in the goal's backward Horn dependency closure. At a fixed loop count, native cell calls and attention arithmetic are matched. Comparisons between two and four loops represent different recurrence compute budgets.

Training and development inputs are the original AF-17 928/352 cases. The 416 prospective heldout cases use the original AND/OR Horn templates on new physical graph layouts 62–66, with source-feature hashes checked for disjointness from prior AF-16/17/18 evaluations. All proposed proof steps remain checked by the independent host-owned Horn legality gate.

## Frozen heldout proof success

Percent of provable Horn goals solved within the exact shortest proof budget plus one action:

| Seed | Eligible | Depth | Local L2 | Ancestor L2 | Local L4 | Ancestor L4 |
|---:|---|---|---:|---:|---:|---:|
| 17 | No | 4 | 62.16% | 0% | 0% | 56.76% |
| 17 | No | 5 | 37.84% | 0% | 0% | 37.84% |
| 29 | Yes | 4 | 64.86% | 0% | 21.62% | 72.97% |
| 29 | Yes | 5 | 60.81% | 0% | 8.11% | 62.16% |
| 43 | Yes | 4 | 0% | 13.51% | 56.76% | 21.62% |
| 43 | Yes | 5 | 0% | 5.41% | 51.35% | 22.97% |
| 61 | Yes | 4 | 56.76% | 21.62% | 18.92% | 54.05% |
| 61 | Yes | 5 | 56.76% | 24.32% | 21.62% | 45.95% |

Eligibility requires all four models for a seed to reach 65% proof success separately on train and development data. Seed 17 is ineligible because its local four-loop dev success is 48%. Eligible seeds 29, 43 and 61 each fail one or more preregistered proof-success improvement and absolute-success gates. **The registered aggregate is FAIL.**

Seed 29 exhibits an especially informative reversal. At four calls, ancestor pooling gains **51.35 percentage points** at depth 4 and **54.05 points** at depth 5 against local four-loop attention. At two calls, its ancestor policy nevertheless STOPs on all provable heldout cases, despite 92.67% dev proof success. Seed 43's four-loop ancestor model performs worse than the matched local control. These are seed-specific effects rather than reliable architecture-wide proof improvements.

## Mechanistic causal contrast

The registered paired probes change a distant initial fact while holding the Horn graph fixed and flipping the true goal from unreachable to reachable. Fact-to-goal undirected incidence distance is 8, 10 or 12. For every seed, local L2 STOP/reachability/distance outputs are identical across all three pairs, consistent with the 2L goal-local causal cone. Every trained ancestor L2 STOP output has a difference greater than 0.0001 on each pair. The independent verifier reconstructs the physical backward-rule masks without consulting the model's mask function.

All 48 paired causal and irrelevant-rule replays passed. Altering metadata of a zero-incidence padded rule changes the STOP, distance and reachability heads by precisely zero on the registered probes. Tests also confirm stability under disconnected-atom changes. Such results are restricted to disconnected and unused components. A connected neural path may transmit a logically irrelevant feature.

## Evidence and limits

The independent auditor checked **16 frozen Torch checkpoints against their named NumPy exports**, **11,648 host-executed episodes** including fixed/uniform/oracle baselines, **225 sampled neural forward replays**, and **328 physical-incidence BFS Bellman-cost labels**. Zero false accepted Horn proofs occurred. Ten regular and ten optimized local tests passed with a tamper-detection control.

The local aggregate results.json SHA-256 is dfd16656e6fa0d367ab296c0e4472bb74662d945ae34fbb32da394510de7f19d. The 79-file binary evidence archive AF19_local_evidence_20261115.zip is **1,708,244 bytes**, SHA-256 d057fd834920b9b54a524d5d26b84092d5ce2a71d87a95258ff3faf8e44c3581. A per-member SHA-256 manifest is included inside the ZIP. The ZIP is a conversation attachment rather than a GitHub Release.

AF-19's source is frozen at 6ccae99; PR #100 stays draft and unmerged, stacked on AF-18 PR #93. Hosted verification and the separate Lean check remain unexecuted. AF-01 through AF-18 sources and scientific outcomes remain unchanged.

The next research question is whether paired reachable/unreachable STOP calibration can convert expanded causal access into reliable rule selection, including when logically irrelevant graph branches remain neurally connected.
