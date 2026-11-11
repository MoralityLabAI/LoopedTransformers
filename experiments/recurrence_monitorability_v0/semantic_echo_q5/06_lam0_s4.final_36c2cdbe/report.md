# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.24931273275593763**. Defined for **17/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2817 |
| pair_any_disagreement_count | 206 |
| per_step_mismatch_count | 358 |
| per_step_comparisons | 11268 |
| oracle_correct_count | 21862 |
| oracle_comparisons | 22536 |
| both_oracle_correct_count | 10753 |
| both_oracle_comparisons | 11268 |
| endpoint_gate_coverage | 0.5587068623562078 |
| pair_any_disagreement_rate | 0.07312744053958112 |
| per_step_mismatch_rate | 0.03177138800141995 |
| oracle_continuation_accuracy | 0.9700922967696131 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.02875399361022364 | 0.969826056088037 | 0.9708910188143415 | 2817 |
| 2 | 0.02023429179978701 | 0.9897053603123891 | 0.9836705715299965 | 2817 |
| 3 | 0.04117855875044373 | 0.9648562300319489 | 0.9534966276180333 | 2817 |
| 4 | 0.03691870784522542 | 0.9751508697195598 | 0.9531416400425985 | 2817 |

Both complete futures are correct for 2597/2817 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 2597 |
| agree_shared_error | 14 |
| disagree_one_side_correct | 108 |
| disagree_both_sides_error | 98 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 7 | 64 | 0.109375 |
| 3 | 7 | 64 | 0.109375 |
| 4 | 9 | 64 | 0.140625 |
| 5 | 8 | 64 | 0.125 |
| 6 | 56 | 64 | 0.875 |
| 7 | 62 | 64 | 0.96875 |
| 8 | 64 | 64 | 1.0 |
| 9 | 64 | 64 | 1.0 |
| 10 | 52 | 64 | 0.8125 |
| 11 | 10 | 64 | 0.15625 |
| 12 | 9 | 64 | 0.140625 |
| 13 | 7 | 64 | 0.109375 |
| 14 | 9 | 64 | 0.140625 |
| 15 | 7 | 64 | 0.109375 |
| 16 | 9 | 64 | 0.140625 |
| 17 | 7 | 64 | 0.109375 |
| 18 | 7 | 64 | 0.109375 |
| 19 | 8 | 64 | 0.125 |
| 20 | 9 | 64 | 0.140625 |
| 21 | 11 | 64 | 0.171875 |
| 22 | 14 | 64 | 0.21875 |
| 23 | 14 | 64 | 0.21875 |
| 24 | 16 | 64 | 0.25 |
| 25 | 10 | 64 | 0.15625 |
| 26 | 12 | 64 | 0.1875 |
| 27 | 10 | 64 | 0.15625 |
| 28 | 9 | 64 | 0.140625 |
| 29 | 10 | 64 | 0.15625 |
| 30 | 10 | 64 | 0.15625 |
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
