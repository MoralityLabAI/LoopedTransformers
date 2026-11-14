# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.5923197457812842**. Defined for **39/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2690 |
| pair_any_disagreement_count | 259 |
| per_step_mismatch_count | 465 |
| per_step_comparisons | 10760 |
| oracle_correct_count | 20661 |
| oracle_comparisons | 21520 |
| both_oracle_correct_count | 10200 |
| both_oracle_comparisons | 10760 |
| endpoint_gate_coverage | 0.5335184450614835 |
| pair_any_disagreement_rate | 0.09628252788104089 |
| per_step_mismatch_rate | 0.04321561338289963 |
| oracle_continuation_accuracy | 0.9600836431226766 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.0379182156133829 | 0.958364312267658 | 0.9654275092936803 | 2690 |
| 2 | 0.04498141263940521 | 0.9594795539033457 | 0.9661710037174721 | 2690 |
| 3 | 0.04460966542750929 | 0.949814126394052 | 0.9609665427509294 | 2690 |
| 4 | 0.04535315985130112 | 0.9561338289962825 | 0.9643122676579926 | 2690 |

Both complete futures are correct for 2421/2690 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2421 |
| agree_shared_error | 10 |
| disagree_one_side_correct | 144 |
| disagree_both_sides_error | 115 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 53 | 64 | 0.828125 |
| 3 | 7 | 64 | 0.109375 |
| 4 | 9 | 64 | 0.140625 |
| 5 | 7 | 64 | 0.109375 |
| 6 | 9 | 64 | 0.140625 |
| 7 | 7 | 64 | 0.109375 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 26 | 64 | 0.40625 |
| 11 | 7 | 64 | 0.109375 |
| 12 | 8 | 64 | 0.125 |
| 13 | 7 | 64 | 0.109375 |
| 14 | 7 | 64 | 0.109375 |
| 15 | 7 | 64 | 0.109375 |
| 16 | 7 | 64 | 0.109375 |
| 17 | 37 | 64 | 0.578125 |
| 18 | 28 | 64 | 0.4375 |
| 19 | 7 | 64 | 0.109375 |
| 20 | 7 | 64 | 0.109375 |
| 21 | 7 | 64 | 0.109375 |
| 22 | 7 | 64 | 0.109375 |
| 23 | 7 | 64 | 0.109375 |
| 24 | 9 | 64 | 0.140625 |
| 25 | 13 | 64 | 0.203125 |
| 26 | 22 | 64 | 0.34375 |
| 27 | 15 | 64 | 0.234375 |
| 28 | 9 | 64 | 0.140625 |
| 29 | 7 | 64 | 0.109375 |
| 30 | 7 | 64 | 0.109375 |
| 31 | 8 | 64 | 0.125 |
| 32 | 7 | 64 | 0.109375 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
