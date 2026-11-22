# DEGREE48 exact fixed-degree wiring audit

At the 0.50 rank cutoff, every degree-preserving wiring already
contained a complete source-to-sink path in all 16 archived
focused DEV roots per precision. The observed backbone averaged
7.375 nodes in each precision. The exact null means were 7.462
for NF4 and 7.037 for BF16. These graphs use the GAUGE42
observational transport fit and the native IO18 coordinate
system. The measured DELAY39 interventions supply a separate
test of wiring and future effects.

| Backend | Rank cutoff | Observed spanning roots | Mean null spanning probability | Observed backbone size | Mean null backbone size |
|---|---:|---:|---:|---:|---:|
| NF4 | 0.25 | 13/16 | 0.801 | 2.125 | 2.004 |
| BF16 | 0.25 | 6/16 | 0.375 | 1.188 | 1.094 |
| NF4 | 0.50 | 16/16 | 1.000 | 7.375 | 7.462 |
| BF16 | 0.50 | 16/16 | 1.000 | 7.375 | 7.037 |

The primary cutoff conditioned on 10,071 labelled middle-graph
wirings per NF4 root and 7,058 per BF16 root. At that cutoff the
exact median conditional upper tail for backbone size was 1.000
for NF4 and 0.825 for BF16. The tails describe individual finite
graph nulls over the registered margins. They supply no
population inference across the reused source roots. Model
transport fit quality also limits the scope of this comparison.

The null fixes every retained source and sink edge and every
labelled middle-node degree. It counts all feasible simple
labelled bipartite wirings, so the expected backbone size and
spanning probability have no Monte Carlo error. This extends the
degree-conditioned control discussed in the PercInterp utility
audit. A finite graph can shift which individual occurrences lie
on a path even when the existence of at least one path is forced.
[PercInterp E09](https://github.com/MoralityLabAI/percolation-interp/blob/ae871eac4a95d787b980dad0abd5a60d9140b4e4/docs/e09_utility_decision.md)

The CPU fixture matched exhaustive enumeration of all 64 labelled
2×3 graphs at 1,728 distinct degree-and-endpoint conditions. An
independent 4×4 balanced brute force counted 90 labelled wirings,
and a 7×8 case with the real DELAY39 dimensions counted
1,670,911,375 with exact transpose agreement. The archive replay
matched all 64 native-gauge path summaries and the signed sums
recorded by GAUGE42. Maximum sampled private commitment was 106.1
MiB and working set was 87.8 MiB. There were zero decoder forwards,
zero inferential tests and no access to the sealed ARG-03 test.

## Apply to measured DELAY39 graphs

The measured adapter reuses SELECT47’s terminal and journal custody
check. It reads only complete signed graph records, then reports
world coverage and the observed-minus-null backbone size on roots
with both worlds. Full, partial and refused synthetic device receipts
passed the adapter test. Live DELAY39 donor results remain pending.

```powershell
python -B -m experiments.nan_degree48.measured --backend nf4 --source-dir runs/nan_delay39/source --output runs/nan_degree48/measured_nf4
```

[Protocol](../../experiments/nan_degree48/PROTOCOL.md) ·
[Aggregate results](nan_degree48_results.json)
