# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.28995109986091894**. Defined for **53/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 3971 |
| pair_any_disagreement_count | 374 |
| per_step_mismatch_count | 893 |
| per_step_comparisons | 15884 |
| oracle_correct_count | 30748 |
| oracle_comparisons | 31768 |
| both_oracle_correct_count | 14957 |
| both_oracle_comparisons | 15884 |
| endpoint_gate_coverage | 0.7875842919476398 |
| pair_any_disagreement_rate | 0.09418282548476455 |
| per_step_mismatch_rate | 0.056220095693779906 |
| oracle_continuation_accuracy | 0.9678922185847394 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.03248552002014606 | 0.9974817426340972 | 0.968521782926215 | 3971 |
| 2 | 0.05137245026441702 | 0.9929488793754722 | 0.9488793754721733 | 3971 |
| 3 | 0.06547469151347268 | 0.9904306220095693 | 0.933266179803576 | 3971 |
| 4 | 0.07554772097708386 | 0.9894233190632082 | 0.9221858473936037 | 3971 |

Both complete futures are correct for 3595/3971 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 3595 |
| agree_shared_error | 2 |
| disagree_one_side_correct | 332 |
| disagree_both_sides_error | 42 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 64 | 64 | 1.0 |
| 2 | 64 | 64 | 1.0 |
| 3 | 64 | 64 | 1.0 |
| 4 | 64 | 64 | 1.0 |
| 5 | 64 | 64 | 1.0 |
| 6 | 64 | 64 | 1.0 |
| 7 | 64 | 64 | 1.0 |
| 8 | 64 | 64 | 1.0 |
| 9 | 64 | 64 | 1.0 |
| 10 | 64 | 64 | 1.0 |
| 11 | 62 | 64 | 0.96875 |
| 12 | 59 | 64 | 0.921875 |
| 13 | 54 | 64 | 0.84375 |
| 14 | 52 | 64 | 0.8125 |
| 15 | 48 | 64 | 0.75 |
| 16 | 42 | 64 | 0.65625 |
| 17 | 39 | 64 | 0.609375 |
| 18 | 34 | 64 | 0.53125 |
| 19 | 35 | 64 | 0.546875 |
| 20 | 32 | 64 | 0.5 |
| 21 | 34 | 64 | 0.53125 |
| 22 | 31 | 64 | 0.484375 |
| 23 | 34 | 64 | 0.53125 |
| 24 | 27 | 64 | 0.421875 |
| 25 | 27 | 64 | 0.421875 |
| 26 | 25 | 64 | 0.390625 |
| 27 | 27 | 64 | 0.421875 |
| 28 | 28 | 64 | 0.4375 |
| 29 | 25 | 64 | 0.390625 |
| 30 | 22 | 64 | 0.34375 |
| 31 | 22 | 64 | 0.34375 |
| 32 | 23 | 64 | 0.359375 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
