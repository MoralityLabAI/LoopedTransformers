# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.20990470177970177**. Defined for **40/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2956 |
| pair_any_disagreement_count | 189 |
| per_step_mismatch_count | 284 |
| per_step_comparisons | 11824 |
| oracle_correct_count | 21788 |
| oracle_comparisons | 23648 |
| both_oracle_correct_count | 10788 |
| both_oracle_comparisons | 11824 |
| endpoint_gate_coverage | 0.5862752875842919 |
| pair_any_disagreement_rate | 0.06393775372124492 |
| per_step_mismatch_rate | 0.02401894451962111 |
| oracle_continuation_accuracy | 0.9213464140730717 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.02368064952638701 | 0.8900541271989174 | 0.8863328822733424 | 2956 |
| 2 | 0.02165087956698241 | 0.9397834912043301 | 0.9411366711772666 | 2956 |
| 3 | 0.02740189445196211 | 0.9073071718538566 | 0.9042625169147497 | 2956 |
| 4 | 0.02334235453315291 | 0.9509472259810555 | 0.9509472259810555 | 2956 |

Both complete futures are correct for 2521/2956 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2521 |
| agree_shared_error | 246 |
| disagree_one_side_correct | 92 |
| disagree_both_sides_error | 97 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 9 | 64 | 0.140625 |
| 2 | 10 | 64 | 0.15625 |
| 3 | 8 | 64 | 0.125 |
| 4 | 12 | 64 | 0.1875 |
| 5 | 11 | 64 | 0.171875 |
| 6 | 12 | 64 | 0.1875 |
| 7 | 11 | 64 | 0.171875 |
| 8 | 64 | 64 | 1.0 |
| 9 | 9 | 64 | 0.140625 |
| 10 | 10 | 64 | 0.15625 |
| 11 | 10 | 64 | 0.15625 |
| 12 | 12 | 64 | 0.1875 |
| 13 | 10 | 64 | 0.15625 |
| 14 | 13 | 64 | 0.203125 |
| 15 | 14 | 64 | 0.21875 |
| 16 | 15 | 64 | 0.234375 |
| 17 | 14 | 64 | 0.21875 |
| 18 | 14 | 64 | 0.21875 |
| 19 | 11 | 64 | 0.171875 |
| 20 | 13 | 64 | 0.203125 |
| 21 | 12 | 64 | 0.1875 |
| 22 | 19 | 64 | 0.296875 |
| 23 | 18 | 64 | 0.28125 |
| 24 | 22 | 64 | 0.34375 |
| 25 | 11 | 64 | 0.171875 |
| 26 | 15 | 64 | 0.234375 |
| 27 | 10 | 64 | 0.15625 |
| 28 | 13 | 64 | 0.203125 |
| 29 | 14 | 64 | 0.21875 |
| 30 | 10 | 64 | 0.15625 |
| 31 | 10 | 64 | 0.15625 |
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
