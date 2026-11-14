# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.4732440476190476**. Defined for **40/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2834 |
| pair_any_disagreement_count | 139 |
| per_step_mismatch_count | 310 |
| per_step_comparisons | 11336 |
| oracle_correct_count | 21579 |
| oracle_comparisons | 22672 |
| both_oracle_correct_count | 10704 |
| both_oracle_comparisons | 11336 |
| endpoint_gate_coverage | 0.5620785402618008 |
| pair_any_disagreement_rate | 0.04904728299223712 |
| per_step_mismatch_rate | 0.02734650670430487 |
| oracle_continuation_accuracy | 0.9517907551164432 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.027170077628793227 | 0.9760056457304164 | 0.9735356386732533 | 2834 |
| 2 | 0.034227240649259 | 0.9502470007057163 | 0.952364149611856 | 2834 |
| 3 | 0.030345800988002825 | 0.9435426958362738 | 0.9428369795342273 | 2834 |
| 4 | 0.017642907551164433 | 0.9371912491178547 | 0.9386026817219478 | 2834 |

Both complete futures are correct for 2648/2834 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2648 |
| agree_shared_error | 47 |
| disagree_one_side_correct | 20 |
| disagree_both_sides_error | 119 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 7 | 64 | 0.109375 |
| 3 | 7 | 64 | 0.109375 |
| 4 | 7 | 64 | 0.109375 |
| 5 | 7 | 64 | 0.109375 |
| 6 | 9 | 64 | 0.140625 |
| 7 | 9 | 64 | 0.140625 |
| 8 | 64 | 64 | 1.0 |
| 9 | 59 | 64 | 0.921875 |
| 10 | 7 | 64 | 0.109375 |
| 11 | 7 | 64 | 0.109375 |
| 12 | 7 | 64 | 0.109375 |
| 13 | 8 | 64 | 0.125 |
| 14 | 61 | 64 | 0.953125 |
| 15 | 61 | 64 | 0.953125 |
| 16 | 14 | 64 | 0.21875 |
| 17 | 9 | 64 | 0.140625 |
| 18 | 9 | 64 | 0.140625 |
| 19 | 11 | 64 | 0.171875 |
| 20 | 44 | 64 | 0.6875 |
| 21 | 50 | 64 | 0.78125 |
| 22 | 17 | 64 | 0.265625 |
| 23 | 8 | 64 | 0.125 |
| 24 | 7 | 64 | 0.109375 |
| 25 | 11 | 64 | 0.171875 |
| 26 | 21 | 64 | 0.328125 |
| 27 | 37 | 64 | 0.578125 |
| 28 | 16 | 64 | 0.25 |
| 29 | 10 | 64 | 0.15625 |
| 30 | 7 | 64 | 0.109375 |
| 31 | 9 | 64 | 0.140625 |
| 32 | 12 | 64 | 0.1875 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
