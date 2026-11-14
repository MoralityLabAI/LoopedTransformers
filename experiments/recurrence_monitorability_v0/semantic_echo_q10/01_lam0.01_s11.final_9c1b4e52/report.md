# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.35781858301881186**. Defined for **46/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 3023 |
| pair_any_disagreement_count | 157 |
| per_step_mismatch_count | 390 |
| per_step_comparisons | 12092 |
| oracle_correct_count | 22233 |
| oracle_comparisons | 24184 |
| both_oracle_correct_count | 11006 |
| both_oracle_comparisons | 12092 |
| endpoint_gate_coverage | 0.5995636652122174 |
| pair_any_disagreement_rate | 0.05193516374462454 |
| per_step_mismatch_rate | 0.03225272907707575 |
| oracle_continuation_accuracy | 0.9193268276546477 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.027786966589480648 | 0.8881905391994708 | 0.8815745947734039 | 3023 |
| 2 | 0.03638769434336751 | 0.9526959973536222 | 0.9374793251736685 | 3023 |
| 3 | 0.03241812768772742 | 0.8967912669533576 | 0.8862057558716506 | 3023 |
| 4 | 0.03241812768772742 | 0.9599735362222958 | 0.9517036056897122 | 3023 |

Both complete futures are correct for 2646/3023 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2646 |
| agree_shared_error | 220 |
| disagree_one_side_correct | 4 |
| disagree_both_sides_error | 153 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 22 | 64 | 0.34375 |
| 2 | 51 | 64 | 0.796875 |
| 3 | 46 | 64 | 0.71875 |
| 4 | 55 | 64 | 0.859375 |
| 5 | 50 | 64 | 0.78125 |
| 6 | 57 | 64 | 0.890625 |
| 7 | 50 | 64 | 0.78125 |
| 8 | 64 | 64 | 1.0 |
| 9 | 18 | 64 | 0.28125 |
| 10 | 14 | 64 | 0.21875 |
| 11 | 10 | 64 | 0.15625 |
| 12 | 23 | 64 | 0.359375 |
| 13 | 12 | 64 | 0.1875 |
| 14 | 21 | 64 | 0.328125 |
| 15 | 18 | 64 | 0.28125 |
| 16 | 20 | 64 | 0.3125 |
| 17 | 8 | 64 | 0.125 |
| 18 | 15 | 64 | 0.234375 |
| 19 | 9 | 64 | 0.140625 |
| 20 | 17 | 64 | 0.265625 |
| 21 | 13 | 64 | 0.203125 |
| 22 | 12 | 64 | 0.1875 |
| 23 | 9 | 64 | 0.140625 |
| 24 | 20 | 64 | 0.3125 |
| 25 | 10 | 64 | 0.15625 |
| 26 | 17 | 64 | 0.265625 |
| 27 | 15 | 64 | 0.234375 |
| 28 | 22 | 64 | 0.34375 |
| 29 | 9 | 64 | 0.140625 |
| 30 | 21 | 64 | 0.328125 |
| 31 | 11 | 64 | 0.171875 |
| 32 | 20 | 64 | 0.3125 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
