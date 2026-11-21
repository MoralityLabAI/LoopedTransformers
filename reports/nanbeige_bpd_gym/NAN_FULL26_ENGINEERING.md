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
