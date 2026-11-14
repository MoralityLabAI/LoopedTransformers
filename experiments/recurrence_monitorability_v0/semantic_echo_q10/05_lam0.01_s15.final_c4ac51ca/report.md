# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.01171875**. Defined for **64/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 5031 |
| pair_any_disagreement_count | 3 |
| per_step_mismatch_count | 9 |
| per_step_comparisons | 20124 |
| oracle_correct_count | 40239 |
| oracle_comparisons | 40248 |
| both_oracle_correct_count | 20115 |
| both_oracle_comparisons | 20124 |
| endpoint_gate_coverage | 0.9978183260610869 |
| pair_any_disagreement_rate | 0.0005963029218843172 |
| per_step_mismatch_rate | 0.0004472271914132379 |
| oracle_continuation_accuracy | 0.9997763864042933 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.00019876764062810574 | 1.0 | 0.9998012323593719 | 5031 |
| 2 | 0.0003975352812562115 | 1.0 | 0.9996024647187438 | 5031 |
| 3 | 0.0005963029218843172 | 1.0 | 0.9994036970781157 | 5031 |
| 4 | 0.0005963029218843172 | 1.0 | 0.9994036970781157 | 5031 |

Both complete futures are correct for 5028/5031 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 5028 |
| agree_shared_error | 0 |
| disagree_one_side_correct | 3 |
| disagree_both_sides_error | 0 |

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
| 11 | 64 | 64 | 1.0 |
| 12 | 64 | 64 | 1.0 |
| 13 | 64 | 64 | 1.0 |
| 14 | 64 | 64 | 1.0 |
| 15 | 64 | 64 | 1.0 |
| 16 | 64 | 64 | 1.0 |
| 17 | 64 | 64 | 1.0 |
| 18 | 63 | 64 | 0.984375 |
| 19 | 63 | 64 | 0.984375 |
| 20 | 63 | 64 | 0.984375 |
| 21 | 63 | 64 | 0.984375 |
| 22 | 63 | 64 | 0.984375 |
| 23 | 63 | 64 | 0.984375 |
| 24 | 63 | 64 | 0.984375 |
| 25 | 63 | 64 | 0.984375 |
| 26 | 63 | 64 | 0.984375 |
| 27 | 63 | 64 | 0.984375 |
| 28 | 63 | 64 | 0.984375 |
| 29 | 63 | 64 | 0.984375 |
| 30 | 64 | 64 | 1.0 |
| 31 | 63 | 64 | 0.984375 |
| 32 | 63 | 64 | 0.984375 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
