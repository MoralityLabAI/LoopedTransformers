# NAN-PRIME-06 local model progress

Updated 2026-11-20T08:51:36.247498+00:00

Real pinned Nanbeige replies pass through the official Prime vf-run / subprocess / null-harness / trace / task-reward path on this machine. No cloud runtime or hosted inference is used. The fixed mock engineering episodes are excluded.

Partial source-order counts are not final accuracy estimates. This condition scores complete final replies through EOS or the registered schema cap; its results are separate from native first-complete-prefix scores.

| Backend | Official episodes complete | Fresh requests recorded | Worker/reply bindings | Recorded errors |
|---|---:|---:|---:|---:|
| NF4 | 162/162 | 162 | 162 | 0 |
| BF16 | 79/162 | 79 | 79 | 0 |

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
| BF16 | pointer | copy | 5/15 | 5/15 | 15 |
| BF16 | pointer | hop1 | 10/16 | 10/16 | 16 |
| BF16 | pointer | hop2 | 1/16 | 8/16 | 16 |
| BF16 | pointer | hop3 | 1/16 | 6/16 | 16 |
| BF16 | pointer | hop4 | 0/16 | 6/16 | 16 |

NF4 independent audit passed: 162 episodes, 162 fresh replies and 6480 physical vectors. Final-reply main correctness 22/105; copy 55/57. Native generated-prefix tokens agree in 162/162 contexts. These are endpoint diagnostics, not new independent model replications.

For NF4, native prefix-stop scores were 26/105 main and 57/57 copy; complete-reply scores are 22/105 and 55/57. All 162 prefixes agree exactly, so these six lost correct answers follow the shared prefixes. Whole-output format validity falls from 160/162 to 131/162; no parser repair or inference change was introduced.

[Frozen protocol](../../experiments/nan_prime06/PROTOCOL.md); registration commit `f3859240d5a7e1aa77c650f431fd8d5b2a6a099c`. The final independent audit checks SDK/worker/source bindings, tokenizer decoding, scores, full replies and all captured physical-vector dimensions. The one-checkpoint, reused-development design supports descriptive findings only. No VPD/BPD transfer is run and the original 168-case test stays sealed.
