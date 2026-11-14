# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.003700657894736842**. Defined for **64/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 4854 |
| pair_any_disagreement_count | 70 |
| per_step_mismatch_count | 180 |
| per_step_comparisons | 19416 |
| oracle_correct_count | 38642 |
| oracle_comparisons | 38832 |
| both_oracle_correct_count | 19236 |
| both_oracle_comparisons | 19416 |
| endpoint_gate_coverage | 0.9627132090440301 |
| pair_any_disagreement_rate | 0.01442109600329625 |
| per_step_mismatch_rate | 0.009270704573547589 |
| oracle_continuation_accuracy | 0.9951071281417387 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.003914297486608982 | 1.0 | 0.996085702513391 | 4854 |
| 2 | 0.007622579316028018 | 0.9997939843428101 | 0.992377420683972 | 4854 |
| 3 | 0.011124845488257108 | 0.9993819530284301 | 0.9888751545117429 | 4854 |
| 4 | 0.01442109600329625 | 0.9987639060568603 | 0.9855789039967038 | 4854 |

Both complete futures are correct for 4784/4854 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 4784 |
| agree_shared_error | 0 |
| disagree_one_side_correct | 64 |
| disagree_both_sides_error | 6 |

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
| 21 | 63 | 64 | 0.984375 |
| 22 | 63 | 64 | 0.984375 |
| 23 | 63 | 64 | 0.984375 |
| 24 | 63 | 64 | 0.984375 |
| 25 | 63 | 64 | 0.984375 |
| 26 | 63 | 64 | 0.984375 |
| 27 | 63 | 64 | 0.984375 |
| 28 | 63 | 64 | 0.984375 |
| 29 | 63 | 64 | 0.984375 |
| 30 | 63 | 64 | 0.984375 |
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
