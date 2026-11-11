# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.5020453855955512**. Defined for **29/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2633 |
| pair_any_disagreement_count | 485 |
| per_step_mismatch_count | 1247 |
| per_step_comparisons | 10532 |
| oracle_correct_count | 19224 |
| oracle_comparisons | 21064 |
| both_oracle_correct_count | 9059 |
| both_oracle_comparisons | 10532 |
| endpoint_gate_coverage | 0.5222134073780246 |
| pair_any_disagreement_rate | 0.18420053171287504 |
| per_step_mismatch_rate | 0.11840106342575009 |
| oracle_continuation_accuracy | 0.9126471705279149 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.05962780098746677 | 0.9434105582985188 | 0.9039118875807064 | 2633 |
| 2 | 0.1033042157235093 | 0.9878465628560578 | 0.8982149639194835 | 2633 |
| 3 | 0.1405241169768325 | 0.9380934295480441 | 0.8241549563235853 | 2633 |
| 4 | 0.1701481200151918 | 0.9772123053551083 | 0.8283327003418154 | 2633 |

Both complete futures are correct for 2053/2633 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2053 |
| agree_shared_error | 95 |
| disagree_one_side_correct | 394 |
| disagree_both_sides_error | 91 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 12 | 64 | 0.1875 |
| 3 | 10 | 64 | 0.15625 |
| 4 | 11 | 64 | 0.171875 |
| 5 | 12 | 64 | 0.1875 |
| 6 | 64 | 64 | 1.0 |
| 7 | 62 | 64 | 0.96875 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 64 | 64 | 1.0 |
| 11 | 44 | 64 | 0.6875 |
| 12 | 17 | 64 | 0.265625 |
| 13 | 13 | 64 | 0.203125 |
| 14 | 15 | 64 | 0.234375 |
| 15 | 13 | 64 | 0.203125 |
| 16 | 13 | 64 | 0.203125 |
| 17 | 11 | 64 | 0.171875 |
| 18 | 12 | 64 | 0.1875 |
| 19 | 10 | 64 | 0.15625 |
| 20 | 11 | 64 | 0.171875 |
| 21 | 10 | 64 | 0.15625 |
| 22 | 14 | 64 | 0.21875 |
| 23 | 13 | 64 | 0.203125 |
| 24 | 12 | 64 | 0.1875 |
| 25 | 11 | 64 | 0.171875 |
| 26 | 11 | 64 | 0.171875 |
| 27 | 11 | 64 | 0.171875 |
| 28 | 13 | 64 | 0.203125 |
| 29 | 11 | 64 | 0.171875 |
| 30 | 10 | 64 | 0.15625 |
| 31 | 13 | 64 | 0.203125 |
| 32 | 14 | 64 | 0.21875 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
