# NAN-PRIME-06 local model progress

Updated 2026-11-20T09:46:02.546912+00:00

Real pinned Nanbeige replies pass through the official Prime vf-run / subprocess / null-harness / trace / task-reward path on this machine. No cloud runtime or hosted inference is used. The fixed mock engineering episodes are excluded.

Partial source-order counts are not final accuracy estimates. This condition scores complete final replies through EOS or the registered schema cap; its results are separate from native first-complete-prefix scores.

| Backend | Official episodes recorded | Fresh requests recorded | Worker/reply bindings | Failed/incomplete episodes |
|---|---:|---:|---:|---:|
| NF4 | 162/162 | 162 | 162 | 0 |
| BF16 | 92/162 | 93 | 92 | 0 |

Bindings require the saved raw assistant message to match the worker reply exactly. The pinned SDK `Trace.last_reply` accessor separately strips outer whitespace; that behavior is counted explicitly in the JSON. The registered anchored grammar already strips outer whitespace, so this representation check changes neither model output nor task grades.

Correctness tables include successful official episodes only. Failed or incomplete episodes remain explicit missing graded coverage, with per-suite counts in the JSON; they are neither fabricated zero-reward answers nor new independent roots. Final completeness also requires all 162 raw reply bindings.

| Backend | DEV suite | Condition | Correct | Format valid | Source roots represented |
|---|---|---|---:|---:|---:|
| NF4 | arc1 | copy | 3/3 | 3/3 | 3 |
| NF4 | arc1 | main | 1/3 | 3/3 | 3 |
| NF4 | arc2 | copy | 3/3 | 3/3 | 3 |
| NF4 | arc2 | main | 1/3 | 2/3 | 3 |
| NF4 | pointer | copy | 16/16 | 16/16 | 16 |
| NF4 | pointer | hop1 | 15/16 | 15/16 | 16 |
| NF4 | pointer | hop2 | 1/16 | 14/16 | 16 |
| NF4 | pointer | hop3 | 0/16 | 7/16 | 16 |
| NF4 | pointer | hop4 | 0/16 | 8/16 | 16 |
| NF4 | routing | copy | 14/16 | 14/16 | 16 |
| NF4 | routing | main | 4/16 | 11/16 | 16 |
| NF4 | storyworld | copy | 16/16 | 16/16 | 16 |
| NF4 | storyworld | main | 0/16 | 16/16 | 16 |
| NF4 | sudoku | copy | 3/3 | 3/3 | 3 |
| NF4 | sudoku | main | 0/3 | 0/3 | 3 |
| BF16 | arc1 | copy | 3/3 | 3/3 | 3 |
| BF16 | arc1 | main | 1/3 | 3/3 | 3 |
| BF16 | arc2 | copy | 3/3 | 3/3 | 3 |
| BF16 | arc2 | main | 0/3 | 2/3 | 3 |
| BF16 | pointer | copy | 5/16 | 5/16 | 16 |
| BF16 | pointer | hop1 | 10/16 | 10/16 | 16 |
| BF16 | pointer | hop2 | 1/16 | 8/16 | 16 |
| BF16 | pointer | hop3 | 1/16 | 6/16 | 16 |
| BF16 | pointer | hop4 | 0/16 | 6/16 | 16 |

NF4 attempt-aware audit: 162/162 unique tasks have successful, independently graded SDK replies; 0 remain missing graded coverage. Primary worker produced 162 neural replies and bound 162 to distinct successful SDK occurrences. Primary runner exit code: 0; tail status: not_required. These attempts stay separate; orphan neural replies never rescue an unsuccessful SDK episode.

NF4 independent audit passed: 162 episodes, 162 fresh replies and 6480 physical vectors. Final-reply main correctness 22/105; copy 55/57. Native generated-prefix tokens agree in 162/162 contexts. These are endpoint diagnostics, not new independent model replications.

NF4 prefix-state check: 6480/6480 physical vectors match native inference byte for byte; maximum absolute change 0. The JSON retains all source-root-weighted site/loop groups and loop-displacement comparisons. Residuals are measured before the final loop RMSNorm. This checks transport/endpoint consistency within a backend, not BF16/NF4 equivalence.

For NF4, native prefix-stop scores were 26/105 main and 57/57 copy; complete-reply scores are 22/105 and 55/57. All 162 prefixes agree exactly, so these six lost correct answers follow the shared prefixes. Whole-output format validity falls from 160/162 to 131/162; no parser repair or inference change was introduced.

BF16 attempt-aware audit: 92/162 unique tasks have successful, independently graded SDK replies; 70 remain missing graded coverage. Primary worker produced 93 neural replies and bound 92 to distinct successful SDK occurrences. Primary runner exit code: 1; tail status: passed. These attempts stay separate; orphan neural replies never rescue an unsuccessful SDK episode.

The bounded BF16 tail runner exited 127 with 0 neural requests and 0 successful SDK episodes. Missing cases are not counted as wrong answers. Fixed-cohort missing-outcome bounds and all task groups are in the JSON; they are not confidence intervals.

[Frozen protocol](../../experiments/nan_prime06/PROTOCOL.md); registration commit `f3859240d5a7e1aa77c650f431fd8d5b2a6a099c`. The final independent audit checks SDK/worker/source bindings, tokenizer decoding, scores, full replies and all captured physical-vector dimensions. The one-checkpoint, reused-development design supports descriptive findings only. No VPD/BPD transfer is run and the original 168-case test stays sealed.
