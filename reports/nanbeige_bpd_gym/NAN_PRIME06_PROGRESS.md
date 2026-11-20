# NAN-PRIME-06 local model progress

Updated 2026-11-20T07:04:39.401730+00:00

Real pinned Nanbeige replies pass through the official Prime vf-run / subprocess / null-harness / trace / task-reward path on this machine. No cloud runtime or hosted inference is used. The fixed mock engineering episodes are excluded.

Partial source-order counts are not final accuracy estimates. This condition scores complete final replies through EOS or the registered schema cap; its results are separate from native first-complete-prefix scores.

| Backend | Official episodes complete | Fresh requests recorded | Worker/reply bindings | Recorded errors |
|---|---:|---:|---:|---:|
| NF4 | 42/162 | 42 | 42 | 0 |
| BF16 | 0/162 | 0 | 0 | 0 |

| Backend | DEV suite | Condition | Correct | Format valid | Source roots represented |
|---|---|---|---:|---:|---:|
| NF4 | pointer | copy | 8/8 | 8/8 | 8 |
| NF4 | pointer | hop1 | 8/9 | 8/9 | 9 |
| NF4 | pointer | hop2 | 1/9 | 9/9 | 9 |
| NF4 | pointer | hop3 | 0/8 | 3/8 | 8 |
| NF4 | pointer | hop4 | 0/8 | 4/8 | 8 |

[Frozen protocol](../../experiments/nan_prime06/PROTOCOL.md); registration commit `f3859240d5a7e1aa77c650f431fd8d5b2a6a099c`. The final independent audit checks SDK/worker/source bindings, tokenizer decoding, scores, full replies and all captured physical-vector dimensions. The one-checkpoint, reused-development design supports descriptive findings only. No VPD/BPD transfer is run and the original 168-case test stays sealed.
