# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.46554232804232804**. Defined for **45/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2919 |
| pair_any_disagreement_count | 78 |
| per_step_mismatch_count | 182 |
| per_step_comparisons | 11676 |
| oracle_correct_count | 22134 |
| oracle_comparisons | 23352 |
| both_oracle_correct_count | 10997 |
| both_oracle_comparisons | 11676 |
| endpoint_gate_coverage | 0.578936929789766 |
| pair_any_disagreement_rate | 0.02672147995889003 |
| per_step_mismatch_rate | 0.015587529976019185 |
| oracle_continuation_accuracy | 0.947841726618705 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.016786570743405275 | 0.9321685508735869 | 0.9218910585817061 | 2919 |
| 2 | 0.020897567660157587 | 0.987667009249743 | 0.9760191846522782 | 2919 |
| 3 | 0.015416238437821172 | 0.9167523124357657 | 0.9095580678314491 | 2919 |
| 4 | 0.009249743062692703 | 0.9698526892771497 | 0.9688249400479616 | 2919 |

Both complete futures are correct for 2646/2919 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2646 |
| agree_shared_error | 195 |
| disagree_one_side_correct | 3 |
| disagree_both_sides_error | 75 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 8 | 64 | 0.125 |
| 2 | 9 | 64 | 0.140625 |
| 3 | 7 | 64 | 0.109375 |
| 4 | 9 | 64 | 0.140625 |
| 5 | 7 | 64 | 0.109375 |
| 6 | 30 | 64 | 0.46875 |
| 7 | 62 | 64 | 0.96875 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 36 | 64 | 0.5625 |
| 11 | 8 | 64 | 0.125 |
| 12 | 9 | 64 | 0.140625 |
| 13 | 7 | 64 | 0.109375 |
| 14 | 9 | 64 | 0.140625 |
| 15 | 7 | 64 | 0.109375 |
| 16 | 9 | 64 | 0.140625 |
| 17 | 7 | 64 | 0.109375 |
| 18 | 10 | 64 | 0.15625 |
| 19 | 12 | 64 | 0.1875 |
| 20 | 26 | 64 | 0.40625 |
| 21 | 43 | 64 | 0.671875 |
| 22 | 36 | 64 | 0.5625 |
| 23 | 15 | 64 | 0.234375 |
| 24 | 10 | 64 | 0.15625 |
| 25 | 7 | 64 | 0.109375 |
| 26 | 9 | 64 | 0.140625 |
| 27 | 7 | 64 | 0.109375 |
| 28 | 9 | 64 | 0.140625 |
| 29 | 8 | 64 | 0.125 |
| 30 | 11 | 64 | 0.171875 |
| 31 | 12 | 64 | 0.1875 |
| 32 | 15 | 64 | 0.234375 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
