# DARK44 finite-dose preservation of the current readout

The same phase-1 edit can preserve Nanbeige’s bare-label margin and change
its full vocabulary readout. At the registered 1% residual-L2 dose, all
32 margin-null donor edits passed the 0.125-logit margin gate. Every edit
exceeded 0.125 logits somewhere in the vocabulary. Across these edits the
largest individual logit changes ranged from 0.189 to 0.297.

## Intervention and matched controls

Use four deterministically selected COMP14 CAL roots, both worlds, both
signs and the separately fitted NF4 and BF16 IO18 banks. The original
donor direction uses the twin’s difference in reached BPD reads. The
margin-null donor removes the projection of that direction along the
continuous donor-minus-recipient margin gradient in the seven-write span.
The visible control follows that projected gradient. The original donor
and both matched controls use the same post-BF16 physical dose calibration.
All 96 signed edits met the physical radius and direction gates.

| Backend | Phase-1 edit | Margin gate | Full-head gate | Unrestricted winner retained | Mean absolute margin change | Mean full-head RMS change | Mean clean-to-edit KL |
|---|---|---:|---:|---:|---:|---:|---:|
| NF4 | Original donor | 16/16 | 0/16 | 16/16 | 0.0303 | 0.0663 | 0.00149 |
| NF4 | Margin-null donor | 16/16 | 0/16 | 16/16 | 0.0088 | 0.0687 | 0.00134 |
| NF4 | Visible span | 15/16 | 0/16 | 16/16 | 0.0762 | 0.0420 | 0.00089 |
| BF16 | Original donor | 16/16 | 0/16 | 14/16 | 0.0313 | 0.0677 | 0.00125 |
| BF16 | Margin-null donor | 16/16 | 0/16 | 14/16 | 0.0059 | 0.0692 | 0.00118 |
| BF16 | Visible span | 16/16 | 0/16 | 14/16 | 0.0771 | 0.0419 | 0.00093 |

The margin-null donor had a smaller absolute margin change and a larger
full-head RMS change in every one of its 32 dose-matched comparisons to
the visible control. The null projection retained an average 89.8% of
the original donor norm in NF4 and 90.6% in BF16 before physical rescaling.

## Verification and interpretation

The head audit streamed all 166,144 logits for 112 reached and clean
states. The clean unrestricted winner and its logit matched the archived
winner and score in every world. The largest log-normalizer replay error
was 4.1e-7. A dense fixture checked the streaming KL, RMS, maximum error,
bare-label retrieval and deterministic winner selection. A second
calculation reconciled the gates against the private measured receipts.
Maximum sampled private commitment was 138.9 MiB and working set was
119.5 MiB. The audit ran alongside RAM28 with zero pretrained decoder
forwards and zero formal comparisons.

For the TRM-style current-versus-future comparison, a margin-null edit can
qualify for a **margin-specific** question. A full-readout preservation
claim needs a separately qualified direction or radius and the same
finite BF16 vocabulary check. An unrestricted token can stay fixed even
when the surrounding logits move, as it did in all 16 NF4 margin-null
cases. DELAY39 will measure the future response in native trajectories
with a matched physical sham. DARK44 only measures the immediate head on
four exposed CAL roots, and the original ARG-03 test remains sealed.

[Protocol](../../experiments/nan_dark44/PROTOCOL.md) ·
[Aggregate results](nan_dark44_results.json)
