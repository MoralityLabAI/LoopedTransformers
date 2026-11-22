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
