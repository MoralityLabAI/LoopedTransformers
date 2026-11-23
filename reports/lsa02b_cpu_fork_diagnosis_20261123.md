# LSA-02B CPU fork and block-order diagnostic

**Date:** 2026-11-23. **Evidence class:** development-only, post-hoc analysis of already-frozen CPU checkpoints and real-Lean calibration transcripts. [CPU-training run 45167489504](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167489504) supplied the source artifacts. [Read-only diagnostic run 45169357742](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45169357742) reloaded the source, four checkpoints and four native proof receipts, verified their hashes, and recomputed the measurements on GitHub-hosted CPU. No new Lean executions, training updates or sealed test reads occurred.

## The native training source has a large forced-choice component

| Source property | Verified count |
| --- | ---: |
| Distinct theorem training lineages | 24 |
| State/action frames | 124 |
| Singleton public candidate menus | **73** |
| Menus with at least two tactics | **51** |
| Forks containing both shortest and nonshortest actions | **51** |
| Forks with multiple equally shortest choices | 0 |
| Independent roots with discriminating forks | 23 |
| Mean discriminating choice points per root | 2.125 |

The earlier train-only optimal-label accuracy of roughly 74–80% included 73 states on which the public menu forced a single action. It is therefore an inflated estimate of what the learner acquired about selecting a proof continuation.

## Strict fork-only check

At **the 51 genuinely discriminating training states**, the exact uniform-action baseline from the source candidate menus, aggregated over independent theorem roots, is **39.40%**. The train-order score is:

| Seed | Trained dispatch | Root-weighted optimal continuation among forks | Uniform-menu baseline | Frozen AB/BA choices changed on training forks |
| --- | --- | ---: | ---: | ---: |
| 101 | AB | **45.29%** | 39.40% | 10 / 51 |
| 101 | BA | **43.12%** | 39.40% | 2 / 51 |
| 103 | AB | **40.58%** | 39.40% | 3 / 51 |
| 103 | BA | **51.45%** | 39.40% | 2 / 51 |

One root contained only singletons. The fork-only accuracy groups correlated steps at the theorem level over the 23 eligible roots. These are in-sample training diagnostics, and individual checkpoints repeatedly reuse the same 51 source frames. Four score rows therefore do **not** represent independent statistical experiments or proof improvements.

The trained weights exhibit small but measurable block-order-dependent continuous logit displacement on these frames. Frozen AB/BA policy choices diverged at a subset of training forks, demonstrating that the recurrent blocks are noncommuting in behavior at some source inputs.

## Native calibration on eight distinct roots

Both training orders at seed 101 completed 1/8 full proofs, and both at seed 103 completed 2/8. The independent Lean reference ceiling is 6/8, with the same real kernel and standalone source-certificate authority across all four cells. The public-order reference closed 1/8 at either seed.

Across the 8-root native proof traces:

| Checkpoint | AB vs BA shared public decision packets | Changed actions on shared packets | First-action changes | Full-proof AB/BA delta |
| --- | ---: | ---: | ---: | ---: |
| 101 / trained AB | 26 | **2** | 0 | 0 |
| 101 / trained BA | 28 | 0 | 0 | 0 |
| 103 / trained AB | 34 | 0 | 0 | 0 |
| 103 / trained BA | 29 | 0 | 0 | 0 |

The shared-packet comparison requires the identical publicly printed input, remaining tactic budget and candidate menu, so a divergent trajectory is never matched merely because it came from the same theorem. Across this small pilot, the internal order sometimes changes a later action while leaving final proof completion unchanged. Every recorded native action trace and closure was independently replayed; the checkpoint data and manifests were read-only.

## Research decision

**`BLOCKED_INCOMPETENT_CHECKPOINT`** remains the formal LSA-02B competence disposition. The training models have limited genuine proof-policy ability, and the current root set does not support an AB/BA performance benefit or a general architectural mechanism claim.

A more useful CPU follow-up should test a **compact, typed public-state serialization** against the existing 384-byte truncated raw JSON, on newly generated train/calibration lineages. Hold the physical blocks, schedule, parameter budget, optimization exposures and native proof authority fixed. Make *fork-only* shortest-continuation prediction and full native proof completion separately reported gates. The current source contains adequate branch alternatives to study action selection, though the text representation and the small corpus limit what this pilot establishes.

That study belongs ahead of a full gym. A future architecture search or dynamic recursion scheduler can use a stronger teacher/model pair once a clear competence gate passes. Ordinary and OOD locked splits remain unopened.

## Files and receipts

- [Native training protocol](../docs/lsa02b_cpu_training_pilot.md)
- [Read-only diagnostic contract](../docs/lsa02b_cpu_diagnostics_protocol.md)
- [Native Lean AB/BA reference ceiling](lsa02b_native_paired_smoke_20261123.md)
- [CPU fit and first native results](lsa02b_cloud_cpu_pilot_20261123.md)
- Source artifact run: [45167489504](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167489504)
- Frozen diagnostic run: [45169357742](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45169357742)

Artifacts are retained by the GitHub workflow for fourteen days; this report preserves the descriptive decision and source run identifiers for subsequent registration.
