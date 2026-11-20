# NAN-PRIME-06 local model progress

Updated 2026-11-20T07:18:45.830918+00:00

Real pinned Nanbeige replies pass through the official Prime vf-run / subprocess / null-harness / trace / task-reward path on this machine. No cloud runtime or hosted inference is used. The fixed mock engineering episodes are excluded.

Partial source-order counts are not final accuracy estimates. This condition scores complete final replies through EOS or the registered schema cap; its results are separate from native first-complete-prefix scores.

| Backend | Official episodes complete | Fresh requests recorded | Worker/reply bindings | Recorded errors |
|---|---:|---:|---:|---:|
| NF4 | 92/162 | 92 | 92 | 0 |
| BF16 | 0/162 | 0 | 0 | 0 |

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

[Frozen protocol](../../experiments/nan_prime06/PROTOCOL.md); registration commit `f3859240d5a7e1aa77c650f431fd8d5b2a6a099c`. The final independent audit checks SDK/worker/source bindings, tokenizer decoding, scores, full replies and all captured physical-vector dimensions. The one-checkpoint, reused-development design supports descriptive findings only. No VPD/BPD transfer is run and the original 168-case test stays sealed.
