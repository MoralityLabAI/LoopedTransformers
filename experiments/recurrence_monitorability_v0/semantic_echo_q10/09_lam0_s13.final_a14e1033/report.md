# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.340136968766001**. Defined for **31/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2905 |
| pair_any_disagreement_count | 62 |
| per_step_mismatch_count | 127 |
| per_step_comparisons | 11620 |
| oracle_correct_count | 21999 |
| oracle_comparisons | 23240 |
| both_oracle_correct_count | 10978 |
| both_oracle_comparisons | 11620 |
| endpoint_gate_coverage | 0.5761602538675129 |
| pair_any_disagreement_rate | 0.02134251290877797 |
| per_step_mismatch_rate | 0.010929432013769363 |
| oracle_continuation_accuracy | 0.9466006884681584 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.013769363166953529 | 0.923580034423408 | 0.9266781411359725 | 2905 |
| 2 | 0.010327022375215147 | 0.9776247848537005 | 0.9762478485370052 | 2905 |
| 3 | 0.010327022375215147 | 0.9108433734939759 | 0.9108433734939759 | 2905 |
| 4 | 0.00929432013769363 | 0.9734939759036144 | 0.9734939759036144 | 2905 |

Both complete futures are correct for 2646/2905 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2646 |
| agree_shared_error | 197 |
| disagree_one_side_correct | 0 |
| disagree_both_sides_error | 62 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 10 | 64 | 0.15625 |
| 3 | 7 | 64 | 0.109375 |
| 4 | 9 | 64 | 0.140625 |
| 5 | 7 | 64 | 0.109375 |
| 6 | 9 | 64 | 0.140625 |
| 7 | 40 | 64 | 0.625 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 9 | 64 | 0.140625 |
| 11 | 7 | 64 | 0.109375 |
| 12 | 9 | 64 | 0.140625 |
| 13 | 7 | 64 | 0.109375 |
| 14 | 9 | 64 | 0.140625 |
| 15 | 8 | 64 | 0.125 |
| 16 | 16 | 64 | 0.25 |
| 17 | 33 | 64 | 0.515625 |
| 18 | 39 | 64 | 0.609375 |
| 19 | 19 | 64 | 0.296875 |
| 20 | 11 | 64 | 0.171875 |
| 21 | 9 | 64 | 0.140625 |
| 22 | 9 | 64 | 0.140625 |
| 23 | 7 | 64 | 0.109375 |
| 24 | 10 | 64 | 0.15625 |
| 25 | 14 | 64 | 0.21875 |
| 26 | 34 | 64 | 0.53125 |
| 27 | 31 | 64 | 0.484375 |
| 28 | 22 | 64 | 0.34375 |
| 29 | 13 | 64 | 0.203125 |
| 30 | 9 | 64 | 0.140625 |
| 31 | 7 | 64 | 0.109375 |
| 32 | 11 | 64 | 0.171875 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
