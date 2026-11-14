# Semantic Echo: MEASURED_CHECKPOINT

Scope: `own_head_only`. Device `cuda`, dtype `float32`.

## Primary: mean conditional continuation disagreement

Unweighted item mean: **0.45541663412033784**. Defined for **27/64** generated items.
Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.
The run contains 64 unique permutation values across 64 item draws.

Native cursor/head parity passed on 2048 visit/item comparisons.
Retained 33 complete token states for each of 64 items, including the initial state.

## Pooled continuation counts

Pooled pair rates are descriptive companions to the item mean.

| Quantity | Value |
|---|---:|
| eligible_pairs | 5042 |
| endpoint_gated_pairs | 2381 |
| pair_any_disagreement_count | 393 |
| per_step_mismatch_count | 897 |
| per_step_comparisons | 9524 |
| oracle_correct_count | 17293 |
| oracle_comparisons | 19048 |
| both_oracle_correct_count | 8237 |
| both_oracle_comparisons | 9524 |
| endpoint_gate_coverage | 0.47223324077746925 |
| pair_any_disagreement_rate | 0.16505669886602267 |
| per_step_mismatch_rate | 0.09418311633767325 |
| oracle_continuation_accuracy | 0.9078643427131458 |

## Future offsets

Donor is the earlier visit t; receiver is the later visit u.

| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |
|---:|---:|---:|---:|---:|
| 1 | 0.04493910121797564 | 0.9290214195716086 | 0.8966820663586729 | 2381 |
| 2 | 0.08399832003359933 | 0.9819403611927762 | 0.9109617807643847 | 2381 |
| 3 | 0.10625787484250315 | 0.9126417471650567 | 0.8227635447291054 | 2381 |
| 4 | 0.14153716925661486 | 0.9659806803863923 | 0.8429231415371693 | 2381 |

Both complete futures are correct for 1847/2381 gated pairs.

| Complete-future comparison | Pair count |
|---|---:|
| agree_both_correct | 1847 |
| agree_shared_error | 141 |
| disagree_one_side_correct | 301 |
| disagree_both_sides_error | 92 |

## Accuracy by trace depth

| Loop | Correct items | Items | Accuracy |
|---:|---:|---:|---:|
| 1 | 7 | 64 | 0.109375 |
| 2 | 9 | 64 | 0.140625 |
| 3 | 7 | 64 | 0.109375 |
| 4 | 9 | 64 | 0.140625 |
| 5 | 7 | 64 | 0.109375 |
| 6 | 11 | 64 | 0.171875 |
| 7 | 28 | 64 | 0.4375 |
| 8 | 64 | 64 | 1.0 |
| 9 | 62 | 64 | 0.96875 |
| 10 | 10 | 64 | 0.15625 |
| 11 | 7 | 64 | 0.109375 |
| 12 | 9 | 64 | 0.140625 |
| 13 | 7 | 64 | 0.109375 |
| 14 | 9 | 64 | 0.140625 |
| 15 | 7 | 64 | 0.109375 |
| 16 | 10 | 64 | 0.15625 |
| 17 | 11 | 64 | 0.171875 |
| 18 | 27 | 64 | 0.421875 |
| 19 | 35 | 64 | 0.546875 |
| 20 | 18 | 64 | 0.28125 |
| 21 | 9 | 64 | 0.140625 |
| 22 | 10 | 64 | 0.15625 |
| 23 | 6 | 64 | 0.09375 |
| 24 | 6 | 64 | 0.09375 |
| 25 | 5 | 64 | 0.078125 |
| 26 | 9 | 64 | 0.140625 |
| 27 | 11 | 64 | 0.171875 |
| 28 | 17 | 64 | 0.265625 |
| 29 | 26 | 64 | 0.40625 |
| 30 | 17 | 64 | 0.265625 |
| 31 | 8 | 64 | 0.125 |
| 32 | 7 | 64 | 0.109375 |

Selected 16 endpoint-gated pairs for finite interventions; the cap is 16 per checkpoint.
Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.

Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.
The fixed input embedding remains unchanged during every replay.

## Files

- `metrics.json`: configuration, hashes, coverage and per-item metrics.
- `pairs.jsonl`: all structural pairs and selected finite intervention results.
- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.

Pair counts overlap within graphs. These descriptive counts carry no independence assumption.
