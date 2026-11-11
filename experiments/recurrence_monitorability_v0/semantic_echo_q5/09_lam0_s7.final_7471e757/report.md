# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.6506972349934534**. Defined for **42/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2950 |
| pair_any_disagreement_count | 168 |
| per_step_mismatch_count | 460 |
| per_step_comparisons | 11800 |
| oracle_correct_count | 22289 |
| oracle_comparisons | 23600 |
| both_oracle_correct_count | 10974 |
| both_oracle_comparisons | 11800 |
| endpoint_gate_coverage | 0.585085283617612 |
| pair_any_disagreement_rate | 0.05694915254237288 |
| per_step_mismatch_rate | 0.03898305084745763 |
| oracle_continuation_accuracy | 0.9444491525423728 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.029830508474576273 | 0.9294915254237288 | 0.915593220338983 | 2950 |
| 2 | 0.04338983050847458 | 0.98 | 0.9603389830508474 | 2950 |
| 3 | 0.03762711864406779 | 0.9274576271186441 | 0.9128813559322034 | 2950 |
| 4 | 0.04508474576271186 | 0.976271186440678 | 0.9535593220338983 | 2950 |

Both complete futures are correct for 2649/2950 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2649 |
| agree_shared_error | 133 |
| disagree_one_side_correct | 52 |
| disagree_both_sides_error | 116 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 14 | 64 | 0.21875 |
| 3 | 10 | 64 | 0.15625 |
| 4 | 18 | 64 | 0.28125 |
| 5 | 49 | 64 | 0.765625 |
| 6 | 64 | 64 | 1.0 |
| 7 | 62 | 64 | 0.96875 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 51 | 64 | 0.796875 |
| 11 | 10 | 64 | 0.15625 |
| 12 | 10 | 64 | 0.15625 |
| 13 | 8 | 64 | 0.125 |
| 14 | 11 | 64 | 0.171875 |
| 15 | 9 | 64 | 0.140625 |
| 16 | 9 | 64 | 0.140625 |
| 17 | 8 | 64 | 0.125 |
| 18 | 10 | 64 | 0.15625 |
| 19 | 8 | 64 | 0.125 |
| 20 | 11 | 64 | 0.171875 |
| 21 | 12 | 64 | 0.1875 |
| 22 | 21 | 64 | 0.328125 |
| 23 | 27 | 64 | 0.421875 |
| 24 | 23 | 64 | 0.359375 |
| 25 | 12 | 64 | 0.1875 |
| 26 | 14 | 64 | 0.21875 |
| 27 | 11 | 64 | 0.171875 |
| 28 | 12 | 64 | 0.1875 |
| 29 | 13 | 64 | 0.203125 |
| 30 | 15 | 64 | 0.234375 |
| 31 | 11 | 64 | 0.171875 |
| 32 | 11 | 64 | 0.171875 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
