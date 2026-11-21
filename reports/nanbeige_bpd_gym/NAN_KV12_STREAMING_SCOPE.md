# NAN-KV-12: reading partial-report denominators

This is a post-registration audit of two already published aggregate reports.
It changes no model, roster, scoring rule, frozen source or existing outcome.
The original 168-case test is not accessed. No new inference or causal,
accuracy, significance or equivalence comparison is introduced.

The frozen reporter first lists closed case artifacts for its within-policy
cache/full-prefix audit. Later, its historical-prefix routine independently
lists the closed artifacts once for each historical backend. The worker can
finish additional cases between those scans. Each archived case and numerical
array is immutable, but a live report is not one atomic snapshot of the whole
run. Its different endpoints can therefore have different closed coverage.

These counts were recomputed from the immutable committed JSON reports by
summing the historical groups' `contexts` field at loop 2 separately for each
`reference_backend`. Loop 1 is the same set of prefixes and is not added as
another set of cases.

| Published report commit | Within-policy contexts | Initial-prefix comparison with default NF4 | Initial-prefix comparison with BF16 |
|---|---:|---:|---:|
| [018c6e8](https://github.com/MoralityLabAI/LoopedTransformers/blob/018c6e8864c711400b6218dda5c7084796945795/reports/nanbeige_bpd_gym/nan_kv12_progress.json) | 32 | 32 | 32 |
| [e98d8f4](https://github.com/MoralityLabAI/LoopedTransformers/blob/e98d8f4cb5b2860e3d10f1aa32a3579464679a9b/reports/nanbeige_bpd_gym/nan_kv12_progress.json) | 64 | 66 | 67 |
| [673ebcb, terminal](https://github.com/MoralityLabAI/LoopedTransformers/blob/673ebcb39eda450ee912b8aafbe6495316e48fb4/reports/nanbeige_bpd_gym/nan_kv12_progress.json) | 130 | 130 | 130 |

Use the coverage stored with each endpoint and group. Do not describe all
historical groups in the second report as a matched 64-context BF16/NF4 set,
subtract their group summaries as though the coverage were identical, or pool
loops into extra cases. The benchmark index's closed-context count refers to
the within-policy endpoint. The complete registered scope remains 130 public
contexts from 41 source families; variants, tokens and loops are repeated
measures, not new independent units.

The terminal report checks the worker receipt and registered coverage. Both
historical comparisons and the current condition now have all 130 closed
cases. This particular streaming-coverage difference has disappeared; it does not
make later generated histories matched across numerical policies. Historical
comparisons remain confined to their common initial full-prefix histories.

Exact source: [frozen reporter](https://github.com/MoralityLabAI/LoopedTransformers/blob/6d2df495d29cf8cf0ba7b7f8dac910bada2a61d3/experiments/nan_kv12/report.py),
SHA-256 `2b946dabb14d23294ad2934c6c2c9e4edd72222fed8830be6d26a28082b0ffe5`.
Source seal: `5f7d135a48aafc4e1f2afd0c2e3fbe34998e722d05d584b42ee11743b7475315`.
All original numerical differences, failures and partial publications remain
preserved. See the [current report](NAN_KV12_PROGRESS.md) and its
[aggregate JSON](nan_kv12_progress.json) for subsequent evidence.
