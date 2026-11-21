# FULL26: actual BPD dispatch engineering

The frozen FULL26 A/B row dispatch passed an additional synthetic CPU check
using the actual pinned BPD component algebra, pass-mean/sphere ablation,
state-only CI network, masked-component context and pointwise chunk adapter.
The frozen model runner, sources, parser, criteria and queue were unchanged.
This check used no Nanbeige weights or forwards, no CUDA allocation and no
held-out test access. It is engineering evidence, not a competence or causal result.

| Check | Result |
|---|---:|
| Physical calls checked | 24 |
| Nonzero reduced-route writes | 18/18 |
| All-one native closure | Bit exact |
| Clean A row | Bit exact native |
| Right-padding positions | Bit exact native |
| Physical pass counter | One increment per call, including chunked inputs |
| Maximum independent controller-mask absolute error | 5.649187717615689e-08 |
| Maximum independent BF16 local-write relative L2 error | 0 |

The three configurations cover A longer than B, B longer than A, and the
remaining edited B1 trajectory after a clean reply ends. Every configuration
uses two recurrent physical invocations under all-one, component-only,
fixed CAL-mean-mask and reached-state routes. Observations select each row's
last valid token and preserve its actual native output before the edit.
The independent replay uses float64 equations followed by BF16 rounding.
The original tolerances remain mask rtol=2e-5/atol=1e-5 and write relative
L2<=0.03; none was loosened to obtain a pass.

[Machine receipt](nan_full26_actual_bpd_check.json) records all24 invocations
and source hashes. [Reproduction code](../../mechinterp/nanbeige_bpd_gym/full26_actual_bpd_check.py)
extracts the exact frozen row-dispatch closure without instantiating a
Nanbeige model. Source seal:
`0b86ed3319c680460027e8445febb234208cd51f4bc05a4bb4b77cd6f18085f6`.
The original full1549-context/216-pair-per-precision run remains registered
in the [full-gym report](NAN_FULL26_PROGRESS.md). Its CUDA capacity and native
closure checks still must pass on the actual checkpoint before scored inference.

Numerical work used one CPU thread and Below Normal priority while the
existing NF4 context worker continued. The continuous goal remains active,
and the original168-case test stays sealed.

## Closed-artifact host path

The frozen host, pairing and report functions also passed an isolated
synthetic-artifact check using the actual fixed SAE dictionaries and
qualified T2 CAL-fit factor files. This created no neural response or model
grade. Every fixture source is CAL-only; no original test was opened.

| Host/report check | Result |
|---|---:|
| Synthetic closed episodes audited | 16 |
| Physical vectors checked | 880 |
| Correct, wrong and malformed packet grades | Expected results |
| Component/static/reached-state mask and write replays | Passed |
| Identical-history/shape prefix pairs | 9 admitted |
| Deliberate history or shape mismatches | 2 excluded |
| Truncated append-only journal records | Ignored |
| Unequal-length root weighting | 2.0, as expected |
| Incorrect token-weighted alternative | 1.5, rejected |
| Frozen report rendered outside real report directory | Passed |

The weighting fixture assigns three prefixes to one root and one to another.
Its expected value is calculated independently. Generated-token counts are
not treated as independent roots. All 12 expected report groups and zero
unaccounted fixture prefixes were checked, including the final geometry
export. The real FULL26 progress report was untouched.

The first fixture attempt completed all 16 host audits and geometry checks,
then failed because its isolated folder omitted the `reports` parent that
exists in the real repository. That failure is preserved in the
[machine receipt](nan_full26_host_check.json). Only the fixture environment
setup was corrected; the complete synthetic check then passed. No frozen
source, tolerance, model output or grade was repaired/replayed.
[Reproduction code](../../mechinterp/nanbeige_bpd_gym/full26_host_check.py)
records source hashes. Numerical work used one CPU thread, Below Normal
priority and zero GPU allocations while the actual context model continued.
