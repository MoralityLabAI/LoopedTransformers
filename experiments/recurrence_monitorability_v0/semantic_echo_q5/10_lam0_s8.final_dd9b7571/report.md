# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.3789086552267386**. Defined for **60/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 3580 |
| pair_any_disagreement_count | 593 |
| per_step_mismatch_count | 1335 |
| per_step_comparisons | 14320 |
| oracle_correct_count | 27226 |
| oracle_comparisons | 28640 |
| both_oracle_correct_count | 12967 |
| both_oracle_comparisons | 14320 |
| endpoint_gate_coverage | 0.7100357001190004 |
| pair_any_disagreement_rate | 0.16564245810055866 |
| per_step_mismatch_rate | 0.09322625698324022 |
| oracle_continuation_accuracy | 0.9506284916201118 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.05139664804469274 | 0.9935754189944134 | 0.9530726256983241 | 3580 |
| 2 | 0.08659217877094973 | 0.9871508379888269 | 0.9206703910614525 | 3580 |
| 3 | 0.10726256983240223 | 0.9857541899441341 | 0.9011173184357542 | 3580 |
| 4 | 0.1276536312849162 | 0.9824022346368715 | 0.8812849162011173 | 3580 |

Both complete futures are correct for 2987/3580 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2987 |
| agree_shared_error | 0 |
| disagree_one_side_correct | 550 |
| disagree_both_sides_error | 43 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 9 | 64 | 0.140625 |
| 2 | 11 | 64 | 0.171875 |
| 3 | 17 | 64 | 0.265625 |
| 4 | 51 | 64 | 0.796875 |
| 5 | 52 | 64 | 0.8125 |
| 6 | 64 | 64 | 1.0 |
| 7 | 64 | 64 | 1.0 |
| 8 | 64 | 64 | 1.0 |
| 9 | 64 | 64 | 1.0 |
| 10 | 64 | 64 | 1.0 |
| 11 | 64 | 64 | 1.0 |
| 12 | 64 | 64 | 1.0 |
| 13 | 63 | 64 | 0.984375 |
| 14 | 62 | 64 | 0.96875 |
| 15 | 61 | 64 | 0.953125 |
| 16 | 53 | 64 | 0.828125 |
| 17 | 49 | 64 | 0.765625 |
| 18 | 48 | 64 | 0.75 |
| 19 | 44 | 64 | 0.6875 |
| 20 | 43 | 64 | 0.671875 |
| 21 | 39 | 64 | 0.609375 |
| 22 | 34 | 64 | 0.53125 |
| 23 | 32 | 64 | 0.5 |
| 24 | 25 | 64 | 0.390625 |
| 25 | 23 | 64 | 0.359375 |
| 26 | 25 | 64 | 0.390625 |
| 27 | 24 | 64 | 0.375 |
| 28 | 20 | 64 | 0.3125 |
| 29 | 21 | 64 | 0.328125 |
| 30 | 21 | 64 | 0.328125 |
| 31 | 20 | 64 | 0.3125 |
| 32 | 19 | 64 | 0.296875 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
