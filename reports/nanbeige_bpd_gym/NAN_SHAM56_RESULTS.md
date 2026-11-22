# SHAM56 fine-coordinate controls for the NF4 route contrast

All four fine-coordinate sham recipes matched the 0.75% residual L2 dose and
cleared the 0.10 leakage limit in every fit world. The selected screened BPD
recipe did the same in all eight disjoint CAL check worlds. The active edits
reproduced the FRONTIER55 finite geometry and full-head receipts exactly.

| Panel | Sham type | Fine-spacing cap | Physical passes / 8 worlds | Full-head passes / 8 worlds | Median peak absolute logit shift across signed shams |
| --- | --- | ---: | ---: | ---: | ---: |
| Fit | Screened BPD span | 0.25 | 8 | 0 | 0.30859 |
| Fit | Screened BPD span | 0.50 | 8 | 0 | 0.25781 |
| Fit | Ambient | 0.25 | 8 | 0 | 0.14063 |
| Fit | Ambient | 0.50 | 8 | 0 | 0.14063 |
| Disjoint CAL check | Screened BPD span | 0.25 | 8 | 0 | 0.19202 |

Every signed sham in the table exceeded the 0.125 peak limit. All retained
the clean greedy winner and passed the bare-label margin limit. The fit
tiebreak fixed the screened BPD 0.25 recipe before the check. The initial
control family therefore clears the physical comparison and fails the
full-head comparison.

The reusable native adapter checks all four predicted full heads, then
requires exact live pass-1 head and companion-row closure before pass 2.
The staged CPU fixture completed four signed edited continuations and
observed different BPD reads for active and sham. A refused head and a
refused physical dose issued zero edited continuations. These are
synthetic software checks. The archive scored 128 unique first-pass
states and made zero new pretrained decoder calls. Both four-root CAL
panels were previously exposed. [Archive protocol](../../experiments/nan_sham56/PROTOCOL.md)
· [Archive aggregate](nan_sham56_results.json)
· [CPU fixture](nan_sham56_native_verification.json)

## Peak-aware follow-up

A 24-dimensional ambient design used the analytical full-head directional
derivative to reduce the predicted peak inside the fine BF16 support.
One fixed recipe was applied to both CAL panels. The actual rounded
sham cleared all gates in **16/16 worlds**, covering both worlds of all
eight roots. No current-head result was used to choose the direction
after rounding.

| Panel | Full four-arm passes | Median predicted peak, start to finish | Median rounded sham peak | Max absolute rounded contrast leakage |
| --- | ---: | ---: | ---: | ---: |
| Fit | 8/8 | 0.13764 to 0.11560 | 0.11719 | 0.0002831 |
| Disjoint CAL check | 8/8 | 0.14285 to 0.11409 | 0.12109 | 0.0001649 |

Peak absolute logit drift was at most 0.125 for all passing shams in
both panels. In the disjoint check, the median signed sham RMS drift
was 0.02914 and the median active RMS was 0.01801. The median signed
peak difference was 0.0625. The rounded sham retained about 0.18% of
its squared norm in the original BPD write span. Those measurements
remain part of the interpretation of a later pass-2 contrast. All
results are descriptive on previously exposed CAL data. [Peak-aware protocol](../../experiments/nan_sham56/MINIMAX.md)
· [Peak-aware aggregate](nan_sham56_minimax_results.json)

The peak-aware live adapter also passed the staged CPU fixture. It completed
four edited continuations and observed distinct active and sham pass-2 read
vectors. Refused head and physical plans produced zero edited continuations.
The CPU fixture does not establish the native pretrained effects or GPU
memory feasibility. [Peak-aware live fixture](nan_sham56_minimax_native_verification.json)
