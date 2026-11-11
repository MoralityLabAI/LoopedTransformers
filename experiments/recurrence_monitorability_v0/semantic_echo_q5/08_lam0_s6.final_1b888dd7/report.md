# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.1126264273057594**. Defined for **62/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 3986 |
| pair_any_disagreement_count | 364 |
| per_step_mismatch_count | 611 |
| per_step_comparisons | 15944 |
| oracle_correct_count | 30479 |
| oracle_comparisons | 31888 |
| both_oracle_correct_count | 14937 |
| both_oracle_comparisons | 15944 |
| endpoint_gate_coverage | 0.7905593018643395 |
| pair_any_disagreement_rate | 0.09131961866532864 |
| per_step_mismatch_rate | 0.0383216256899147 |
| oracle_continuation_accuracy | 0.955814099347717 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.03361766181635725 | 0.9352734570998494 | 0.9365278474661315 | 3986 |
| 2 | 0.03938785750125439 | 0.9844455594581034 | 0.9724034119417962 | 3986 |
| 3 | 0.03637732062217762 | 0.948820873055695 | 0.9194681384846964 | 3986 |
| 4 | 0.04390366281986954 | 0.9927245358755645 | 0.9568489713998997 | 3986 |

Both complete futures are correct for 3421/3986 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 3421 |
| agree_shared_error | 201 |
| disagree_one_side_correct | 341 |
| disagree_both_sides_error | 23 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 10 | 64 | 0.15625 |
| 2 | 31 | 64 | 0.484375 |
| 3 | 28 | 64 | 0.4375 |
| 4 | 50 | 64 | 0.78125 |
| 5 | 59 | 64 | 0.921875 |
| 6 | 64 | 64 | 1.0 |
| 7 | 62 | 64 | 0.96875 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 64 | 64 | 1.0 |
| 11 | 62 | 64 | 0.96875 |
| 12 | 63 | 64 | 0.984375 |
| 13 | 60 | 64 | 0.9375 |
| 14 | 59 | 64 | 0.921875 |
| 15 | 54 | 64 | 0.84375 |
| 16 | 57 | 64 | 0.890625 |
| 17 | 56 | 64 | 0.875 |
| 18 | 59 | 64 | 0.921875 |
| 19 | 56 | 64 | 0.875 |
| 20 | 56 | 64 | 0.875 |
| 21 | 53 | 64 | 0.828125 |
| 22 | 56 | 64 | 0.875 |
| 23 | 53 | 64 | 0.828125 |
| 24 | 56 | 64 | 0.875 |
| 25 | 52 | 64 | 0.8125 |
| 26 | 51 | 64 | 0.796875 |
| 27 | 48 | 64 | 0.75 |
| 28 | 48 | 64 | 0.75 |
| 29 | 47 | 64 | 0.734375 |
| 30 | 50 | 64 | 0.78125 |
| 31 | 46 | 64 | 0.71875 |
| 32 | 50 | 64 | 0.78125 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
