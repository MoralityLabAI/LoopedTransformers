# SPECTRAL45 readout-preserving dose calibration

At **0.25% residual L2**, the margin-null donor preserved both the
bare-label margin and all 166,144 vocabulary logits within 0.125 logits,
with the unrestricted winner unchanged, on **32 of 32** signed edits
across NF4 and staged BF16. At 0.5% it passed 28 of 32 and at 1% it
passed zero. SPECTRAL45 used the four COMP14 focused CAL roots outside
DARK44’s four-root cohort. Both cohorts come from exposed COMP14 data.

## Dose, donor alignment and choice of objective

The continuous RMS-normalized head Gram was evaluated in the null
space of the current donor-minus-recipient bare-label gradient on the
seven eligible IO18 writes. The reference direction is DARK44’s
margin-null donor. The minimum-eigenvalue direction minimizes uniform
RMS logit sensitivity. The regularized direction combines lower
sensitivity with alignment to the donor.

Each condition has 16 signed edits per precision. All met the
post-rounding physical dose and direction tolerances.

| Dose | Direction | NF4 both readout gates | BF16 both readout gates | NF4 / BF16 both gates plus ≥0.8 donor cosine |
|---:|---|---:|---:|---:|
| 1% | Margin-null donor | 0/16 | 0/16 | 0/16 · 0/16 |
| 1% | Minimum RMS | 0/16 | 0/16 | 0/16 · 0/16 |
| 1% | Regularized donor | 0/16 | 0/16 | 0/16 · 0/16 |
| 0.5% | Margin-null donor | 16/16 | 12/16 | 16/16 · 11/16 |
| 0.5% | Minimum RMS | 0/16 | 1/16 | 0/16 · 0/16 |
| 0.5% | Regularized donor | 16/16 | 12/16 | 12/16 · 8/16 |
| 0.25% | Margin-null donor | 16/16 | 16/16 | 16/16 · 12/16 |
| 0.25% | Minimum RMS | 16/16 | 16/16 | 0/16 · 0/16 |
| 0.25% | Regularized donor | 16/16 | 16/16 | 12/16 · 8/16 |

The minimum-RMS direction reduced finite RMS logit drift in all 32
matched comparisons at each dose. At 0.5% it reduced the peak logit
drift in one of 32, and it met the donor-alignment gate in zero of
16 worlds. The regularized donor also reduced RMS drift in all 32
matched comparisons at each dose, and retained the alignment gate
in 10 of 16 worlds. The reference margin-null direction retained
alignment in 14 of 16 worlds.

Use 0.25% as the tested common dose for a prospective full-head
preservation check in native continuation. Check the actual reached
BF16 head before the next pass, keep alignment-qualified coverage
explicit, and use a physically dose-matched sham. For a larger-dose
full-head claim, tune the worst-token or top-order objective, and
confirm it against the finite head before scoring future effects.
The matched continuation, intervention response and sham floor
remain a separate causal study.

## Verification

The 30-column streamed vocabulary Gram covered both physical write
spans and the native residual term from RMSNorm. The constrained
physical Gram had positive minimum eigenvalues in all 16 states.
A dense reference independently verified the Gram and the head
derivative. Sixteen synthetic orthogonal basis changes preserved
all three physical directions within 2.6e-15 L2. The current-head
audit covered 304 clean and edited states, with exact archived
unrestricted winners and top logits, and maximum clean log-normalizer
error below 9.4e-8. Another calculation reconciled the finite gates
and verified that DARK44 and SPECTRAL45 used disjoint CAL roots.

The audit peaked at 141.7 MiB sampled private commitment and 125.1
MiB working set while RAM28 continued. There were zero new decoder
forwards, zero formal tests, and no access to the sealed ARG-03 test.

[Protocol](../../experiments/nan_spectral45/PROTOCOL.md) ·
[Aggregate results](nan_spectral45_results.json)
