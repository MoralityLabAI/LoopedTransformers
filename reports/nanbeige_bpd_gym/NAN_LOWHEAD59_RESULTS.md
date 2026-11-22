# LOWHEAD59 low-sensitivity shams on exposed NF4 CAL

A vocabulary-informed atlas lowered the median minimum continuous head RMS
to **0.01410** on fit and **0.01409** on the disjoint CAL check. The earlier
random 24-dimensional DEV atlas had a median continuous minimum of 0.02813.
Each value applies to its own local atlas and root panel.

Both recipes used the fixed 0.75% residual dose and the same active
learned-minus-static edits. All 16 worlds passed the physical dose,
alignment, leakage and separation checks under both recipes. The complete
head ruled out every minimum-RMS sham. The targeted chord recovered nine
full-head passes and none with both signed RMS gaps at most 0.005.

| Panel | Recipe | Full four-arm gate | RMS-gap criterion among full passes | Median rounded sham RMS across physical passes | Median rounded sham peak across physical passes |
| --- | --- | ---: | ---: | ---: | ---: |
| Fit | Lowest RMS | 0/8 | 0/8 | 0.01446 | 0.85938 |
| Fit | Targeted chord | 4/8 | 0/8 | 0.02890 | 0.12500 |
| Disjoint check | Lowest RMS | 0/8 | 0/8 | 0.01447 | 0.85938 |
| Disjoint check | Targeted chord | 5/8 | 0/8 | 0.02881 | 0.12500 |

Without the peak gate, the minimum-RMS recipe met the RMS-gap threshold
in five worlds of each panel. Its sham peak was about 59 times its RMS,
compared with about 4.4 times for the targeted chord. No sham changed the
unrestricted greedy token or exceeded the bare-label margin limit. The
blocking head condition was the maximum across vocabulary logits.

The global proposal came from 41,536 deterministic vocabulary rows. The
analytic local derivative and actual BF16 logits each used all 166,144
rows. Head scoring consumed 112 unique clean or edited states. Each of
the eight exposed CAL roots contributed both worlds. The signed RMS
and peak criteria refer to the current head. There were zero new decoder
forwards or formal tests, and the ARG-03 seal was preserved.
[Protocol](../../experiments/nan_lowhead59/PROTOCOL.md) · [Aggregate](nan_lowhead59_results.json)
