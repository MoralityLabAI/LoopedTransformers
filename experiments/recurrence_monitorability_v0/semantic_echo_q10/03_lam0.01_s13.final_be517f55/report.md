# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.3054303861040465**. Defined for **64/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 4524 |
| pair_any_disagreement_count | 555 |
| per_step_mismatch_count | 1133 |
| per_step_comparisons | 18096 |
| oracle_correct_count | 34876 |
| oracle_comparisons | 36192 |
| both_oracle_correct_count | 16895 |
| both_oracle_comparisons | 18096 |
| endpoint_gate_coverage | 0.8972629908766363 |
| pair_any_disagreement_rate | 0.1226790450928382 |
| per_step_mismatch_rate | 0.0626105216622458 |
| oracle_continuation_accuracy | 0.9636383731211318 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.03779840848806366 | 0.9964633068081344 | 0.9615384615384616 | 4524 |
| 2 | 0.05857648099027409 | 0.9935897435897436 | 0.9394341290893015 | 4524 |
| 3 | 0.06786030061892132 | 0.9913793103448276 | 0.9294871794871795 | 4524 |
| 4 | 0.08620689655172414 | 0.9876215738284704 | 0.9095932802829354 | 4524 |

Both complete futures are correct for 3952/4524 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 3952 |
| agree_shared_error | 17 |
| disagree_one_side_correct | 503 |
| disagree_both_sides_error | 52 |

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
| 14 | 62 | 64 | 0.96875 |
| 15 | 62 | 64 | 0.96875 |
| 16 | 58 | 64 | 0.90625 |
| 17 | 56 | 64 | 0.875 |
| 18 | 47 | 64 | 0.734375 |
| 19 | 43 | 64 | 0.671875 |
| 20 | 45 | 64 | 0.703125 |
| 21 | 39 | 64 | 0.609375 |
| 22 | 41 | 64 | 0.640625 |
| 23 | 35 | 64 | 0.546875 |
| 24 | 41 | 64 | 0.640625 |
| 25 | 40 | 64 | 0.625 |
| 26 | 34 | 64 | 0.53125 |
| 27 | 34 | 64 | 0.53125 |
| 28 | 35 | 64 | 0.546875 |
| 29 | 30 | 64 | 0.46875 |
| 30 | 33 | 64 | 0.515625 |
| 31 | 34 | 64 | 0.53125 |
| 32 | 28 | 64 | 0.4375 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
