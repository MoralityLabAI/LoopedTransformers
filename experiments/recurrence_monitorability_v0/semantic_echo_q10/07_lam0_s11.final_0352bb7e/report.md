# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.40385161332292485**. Defined for **50/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 3104 |
| pair_any_disagreement_count | 183 |
| per_step_mismatch_count | 348 |
| per_step_comparisons | 12416 |
| oracle_correct_count | 22804 |
| oracle_comparisons | 24832 |
| both_oracle_correct_count | 11312 |
| both_oracle_comparisons | 12416 |
| endpoint_gate_coverage | 0.6156287187623959 |
| pair_any_disagreement_rate | 0.05895618556701031 |
| per_step_mismatch_rate | 0.02802835051546392 |
| oracle_continuation_accuracy | 0.9183311855670103 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.02802835051546392 | 0.9033505154639175 | 0.9017396907216495 | 3104 |
| 2 | 0.02029639175257732 | 0.9407216494845361 | 0.9410438144329897 | 3104 |
| 3 | 0.03157216494845361 | 0.8930412371134021 | 0.8888530927835051 | 3104 |
| 4 | 0.03221649484536082 | 0.9420103092783505 | 0.935889175257732 | 3104 |

Both complete futures are correct for 2682/3104 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2682 |
| agree_shared_error | 239 |
| disagree_one_side_correct | 51 |
| disagree_both_sides_error | 132 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 11 | 64 | 0.171875 |
| 2 | 63 | 64 | 0.984375 |
| 3 | 22 | 64 | 0.34375 |
| 4 | 15 | 64 | 0.234375 |
| 5 | 15 | 64 | 0.234375 |
| 6 | 17 | 64 | 0.265625 |
| 7 | 19 | 64 | 0.296875 |
| 8 | 64 | 64 | 1.0 |
| 9 | 59 | 64 | 0.921875 |
| 10 | 17 | 64 | 0.265625 |
| 11 | 15 | 64 | 0.234375 |
| 12 | 16 | 64 | 0.25 |
| 13 | 12 | 64 | 0.1875 |
| 14 | 25 | 64 | 0.390625 |
| 15 | 51 | 64 | 0.796875 |
| 16 | 46 | 64 | 0.71875 |
| 17 | 16 | 64 | 0.25 |
| 18 | 15 | 64 | 0.234375 |
| 19 | 11 | 64 | 0.171875 |
| 20 | 15 | 64 | 0.234375 |
| 21 | 15 | 64 | 0.234375 |
| 22 | 37 | 64 | 0.578125 |
| 23 | 32 | 64 | 0.5 |
| 24 | 18 | 64 | 0.28125 |
| 25 | 13 | 64 | 0.203125 |
| 26 | 15 | 64 | 0.234375 |
| 27 | 10 | 64 | 0.15625 |
| 28 | 15 | 64 | 0.234375 |
| 29 | 24 | 64 | 0.375 |
| 30 | 28 | 64 | 0.4375 |
| 31 | 15 | 64 | 0.234375 |
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
