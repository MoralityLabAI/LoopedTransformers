# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.556600529100529**. Defined for **45/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 3035 |
| pair_any_disagreement_count | 261 |
| per_step_mismatch_count | 692 |
| per_step_comparisons | 12140 |
| oracle_correct_count | 23258 |
| oracle_comparisons | 24280 |
| both_oracle_correct_count | 11325 |
| both_oracle_comparisons | 12140 |
| endpoint_gate_coverage | 0.6019436731455772 |
| pair_any_disagreement_rate | 0.08599670510708401 |
| per_step_mismatch_rate | 0.05700164744645799 |
| oracle_continuation_accuracy | 0.9579077429983526 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.041515650741350905 | 0.9841845140032949 | 0.9512355848434926 | 3035 |
| 2 | 0.055683690280065895 | 0.9835255354200988 | 0.9393739703459637 | 3035 |
| 3 | 0.06359143327841846 | 0.9746293245469522 | 0.9298187808896211 | 3035 |
| 4 | 0.0672158154859967 | 0.9752883031301482 | 0.9252059308072488 | 3035 |

Both complete futures are correct for 2762/3035 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2762 |
| agree_shared_error | 12 |
| disagree_one_side_correct | 170 |
| disagree_both_sides_error | 91 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 64 | 64 | 1.0 |
| 2 | 63 | 64 | 0.984375 |
| 3 | 63 | 64 | 0.984375 |
| 4 | 63 | 64 | 0.984375 |
| 5 | 63 | 64 | 0.984375 |
| 6 | 64 | 64 | 1.0 |
| 7 | 64 | 64 | 1.0 |
| 8 | 64 | 64 | 1.0 |
| 9 | 55 | 64 | 0.859375 |
| 10 | 37 | 64 | 0.578125 |
| 11 | 17 | 64 | 0.265625 |
| 12 | 17 | 64 | 0.265625 |
| 13 | 14 | 64 | 0.21875 |
| 14 | 11 | 64 | 0.171875 |
| 15 | 11 | 64 | 0.171875 |
| 16 | 10 | 64 | 0.15625 |
| 17 | 8 | 64 | 0.125 |
| 18 | 11 | 64 | 0.171875 |
| 19 | 10 | 64 | 0.15625 |
| 20 | 14 | 64 | 0.21875 |
| 21 | 10 | 64 | 0.15625 |
| 22 | 14 | 64 | 0.21875 |
| 23 | 20 | 64 | 0.3125 |
| 24 | 22 | 64 | 0.34375 |
| 25 | 9 | 64 | 0.140625 |
| 26 | 11 | 64 | 0.171875 |
| 27 | 11 | 64 | 0.171875 |
| 28 | 9 | 64 | 0.140625 |
| 29 | 12 | 64 | 0.1875 |
| 30 | 14 | 64 | 0.21875 |
| 31 | 14 | 64 | 0.21875 |
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
