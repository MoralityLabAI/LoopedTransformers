# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.46880952380952384**. Defined for **40/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2881 |
| pair_any_disagreement_count | 54 |
| per_step_mismatch_count | 130 |
| per_step_comparisons | 11524 |
| oracle_correct_count | 21997 |
| oracle_comparisons | 23048 |
| both_oracle_correct_count | 10954 |
| both_oracle_comparisons | 11524 |
| endpoint_gate_coverage | 0.5714002380007933 |
| pair_any_disagreement_rate | 0.01874349184311003 |
| per_step_mismatch_rate | 0.011280805275945853 |
| oracle_continuation_accuracy | 0.9543995140576189 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.013189864630336688 | 0.9347448802499132 | 0.9319680666435266 | 2881 |
| 2 | 0.013884068031933356 | 0.9871572370704617 | 0.9857688302672684 | 2881 |
| 3 | 0.007289135716765012 | 0.9212079139187782 | 0.9201666088163832 | 2881 |
| 4 | 0.010760152724748352 | 0.9770912877473099 | 0.9770912877473099 | 2881 |

Both complete futures are correct for 2646/2881 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2646 |
| agree_shared_error | 181 |
| disagree_one_side_correct | 0 |
| disagree_both_sides_error | 54 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 8 | 64 | 0.125 |
| 3 | 8 | 64 | 0.125 |
| 4 | 9 | 64 | 0.140625 |
| 5 | 7 | 64 | 0.109375 |
| 6 | 10 | 64 | 0.15625 |
| 7 | 61 | 64 | 0.953125 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 13 | 64 | 0.203125 |
| 11 | 7 | 64 | 0.109375 |
| 12 | 9 | 64 | 0.140625 |
| 13 | 7 | 64 | 0.109375 |
| 14 | 9 | 64 | 0.140625 |
| 15 | 7 | 64 | 0.109375 |
| 16 | 9 | 64 | 0.140625 |
| 17 | 10 | 64 | 0.15625 |
| 18 | 61 | 64 | 0.953125 |
| 19 | 61 | 64 | 0.953125 |
| 20 | 58 | 64 | 0.90625 |
| 21 | 16 | 64 | 0.25 |
| 22 | 9 | 64 | 0.140625 |
| 23 | 7 | 64 | 0.109375 |
| 24 | 9 | 64 | 0.140625 |
| 25 | 7 | 64 | 0.109375 |
| 26 | 9 | 64 | 0.140625 |
| 27 | 7 | 64 | 0.109375 |
| 28 | 17 | 64 | 0.265625 |
| 29 | 50 | 64 | 0.78125 |
| 30 | 62 | 64 | 0.96875 |
| 31 | 44 | 64 | 0.6875 |
| 32 | 16 | 64 | 0.25 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
