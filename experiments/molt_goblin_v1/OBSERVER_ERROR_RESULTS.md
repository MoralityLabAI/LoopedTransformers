# MOLT-08 measured development results

[CPU run 45852042755](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45852042755)
passed at source head 97f82e09c25d371b4f8482b1d09dc6bd22d76e70. Four model
initializations, 16 shared development worlds, 960 candidate rows and 1,536
method decisions across three doses. All directional interval enclosure and
nonlinear gate parity checks passed. The model/world design is crossed.

Full-dose slice (64 model/world rows per method):

| Method | Accepted | Fallback | False accept | Correct | Mean record bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear only | 64 | 0 | 7 | 57 | 31.375 |
| Global curvature | 0 | 64 | 0 | 64 | 128 |
| Interval 1 scalar | 7 | 57 | 0 | 64 | 118.625 |
| Interval 1 signed | 7 | 57 | 0 | 64 | 126.375 |
| Interval 4 scalar | 41 | 23 | 0 | 64 | 70.875 |
| Interval 4 signed | 57 | 7 | 0 | 64 | 112.625 |
| Interval 16 scalar | 60 | 4 | 0 | 64 | 39.875 |
| Interval 16 signed | 63 | 1 | 0 | 64 | 102.25 |

The linear-only diagnostic false-accepted 12 candidates, including seven
first-selected ranks. Sixteen-segment scalar ranks 2/3/4/5 were selected
5/38/14/3 times, with four full ambient fallbacks. Signed rank-1/2/3/4
counts were 2/15/44/2, with one fallback. Bound metadata costs eight bytes
for scalar and 80 bytes for five signed intervals. Mean record bytes
exclude the map. The scalar method has lower payload at this slice despite
admitting fewer compressed records. Shared-map amortization remains necessary.

Summed estimated candidate-search work at full dose was 0.0934 s for
linear-only, 0.1612 s for global curvature, 0.6986 s for interval4 scalar
and 1.5384 s for interval16 scalar. These are within-panel timing sums
excluding map construction and full fallback; not measured deployment costs.

Verdict: useful conditional nonlinear error discrimination in the strict
chain fixture. The global bound has poor practical resolution here.
The interval method needs the original displacement at ingestion and does
not provide outward-rounded floating-point guarantees. No formal certificate,
native BPD mechanism, population-level claim, or promotion. Full raw
development receipts are stored in the run's temporary artifact.

Next: [DataGoblin #92](https://github.com/MoralityLabAI/DataGoblin/pull/92)
accounts for full map overhead and metadata over actual distinct records.
Measure reuse of genuinely different displacements within the same frozen
binding before claiming payload savings. An outward-rounded implementation
and a production binding audit would be required for hard certification.
