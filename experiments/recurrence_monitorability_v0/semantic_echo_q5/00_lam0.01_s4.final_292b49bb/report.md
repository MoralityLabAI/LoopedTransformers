# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.23581861019400538**. Defined for **64/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 4314 |
| pair_any_disagreement_count | 361 |
| per_step_mismatch_count | 903 |
| per_step_comparisons | 17256 |
| oracle_correct_count | 33599 |
| oracle_comparisons | 34512 |
| both_oracle_correct_count | 16348 |
| both_oracle_comparisons | 17256 |
| endpoint_gate_coverage | 0.8556128520428401 |
| pair_any_disagreement_rate | 0.08368103847936949 |
| per_step_mismatch_rate | 0.052329624478442284 |
| oracle_continuation_accuracy | 0.9735454334724154 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.023875753361149745 | 1.0 | 0.9761242466388502 | 4314 |
| 2 | 0.04357904496986555 | 0.9997681965693093 | 0.9561891515994436 | 4314 |
| 3 | 0.06235512285581827 | 0.9995363931386184 | 0.9371812702828002 | 4314 |
| 4 | 0.07950857672693556 | 0.9995363931386184 | 0.9200278164116829 | 4314 |

Both complete futures are correct for 3953/4314 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 3953 |
| agree_shared_error | 0 |
| disagree_one_side_correct | 358 |
| disagree_both_sides_error | 3 |

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
| 15 | 63 | 64 | 0.984375 |
| 16 | 62 | 64 | 0.96875 |
| 17 | 62 | 64 | 0.96875 |
| 18 | 59 | 64 | 0.921875 |
| 19 | 56 | 64 | 0.875 |
| 20 | 55 | 64 | 0.859375 |
| 21 | 49 | 64 | 0.765625 |
| 22 | 42 | 64 | 0.65625 |
| 23 | 38 | 64 | 0.59375 |
| 24 | 32 | 64 | 0.5 |
| 25 | 27 | 64 | 0.421875 |
| 26 | 28 | 64 | 0.4375 |
| 27 | 21 | 64 | 0.328125 |
| 28 | 21 | 64 | 0.328125 |
| 29 | 20 | 64 | 0.3125 |
| 30 | 22 | 64 | 0.34375 |
| 31 | 19 | 64 | 0.296875 |
| 32 | 18 | 64 | 0.28125 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
