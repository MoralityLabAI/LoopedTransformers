# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.00027901785714285713**. Defined for **64/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 5042 |
| pair_any_disagreement_count | 27 |
| per_step_mismatch_count | 27 |
| per_step_comparisons | 20168 |
| oracle_correct_count | 40309 |
| oracle_comparisons | 40336 |
| both_oracle_correct_count | 20141 |
| both_oracle_comparisons | 20168 |
| endpoint_gate_coverage | 1.0 |
| pair_any_disagreement_rate | 0.0053550178500595 |
| per_step_mismatch_rate | 0.001338754462514875 |
| oracle_continuation_accuracy | 0.9993306227687425 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.0 | 1.0 | 1.0 | 5042 |
| 2 | 0.0 | 1.0 | 1.0 | 5042 |
| 3 | 0.0 | 1.0 | 1.0 | 5042 |
| 4 | 0.0053550178500595 | 1.0 | 0.9946449821499405 | 5042 |

Both complete futures are correct for 5015/5042 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 5015 |
| agree_shared_error | 0 |
| disagree_one_side_correct | 27 |
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
| 18 | 64 | 64 | 1.0 |
| 19 | 64 | 64 | 1.0 |
| 20 | 64 | 64 | 1.0 |
| 21 | 64 | 64 | 1.0 |
| 22 | 64 | 64 | 1.0 |
| 23 | 64 | 64 | 1.0 |
| 24 | 64 | 64 | 1.0 |
| 25 | 64 | 64 | 1.0 |
| 26 | 64 | 64 | 1.0 |
| 27 | 64 | 64 | 1.0 |
| 28 | 64 | 64 | 1.0 |
| 29 | 64 | 64 | 1.0 |
| 30 | 64 | 64 | 1.0 |
| 31 | 64 | 64 | 1.0 |
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
