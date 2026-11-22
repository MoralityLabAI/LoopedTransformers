# NAN-CARRY-63 native endpoint attribution

Which measured part of the last-layer residual already contains the difference
between two counterfactual answer margins? Use closed COMP14 traces for the
same 8 CAL and 16 exposed DEV roots in both precisions. Keep focused, copy,
and erased arms. The earlier GAUGE42 linear transport missed the direct-head
tangent on DEV. This study observes the actual MLP output and measures the
contribution of its rank-eight projections at the native endpoints.

Layer 21 has an additive MLP residual. Capture sites provide the post-layer
state h, the MLP down-projection write y, and its input x. The difference h-y
is an effective carry that includes rounding from the residual add. The
capture does not directly provide the post-attention residual. Bound the
carry error in the continuous linear numerator from the adjacent BF16 gaps
of the two measured h vectors, allowing an FP32 intermediate. The bound
applies to that term at the native endpoint scales.

For each phase, sign the readout as token A minus token B using the two gold
labels. Measure its A-world minus B-world contrast. Split the measured
difference in y into its orthogonal projections inside and outside the
rank-eight write span. The effective carry supplies the remaining state
difference. Compare the fitted T2 span, its SVD initializer, and eight
prespecified Gaussian spans of the same rank, fixed across roots and phases.
Apply the same T2 span at both passes. Rotate and rescale the fitted basis,
checking the whole physical projection and resulting rounded coalition scores.
For the initializer and trained bank also measure how well x V U predicts
the observed write difference. A probe that projects the actual output and
a factor operator that predicts the output have distinct inputs.

In the continuous head, split the margin swing into the three linear
numerator terms at the mean inverse RMS plus the numerator-averaged change
in inverse RMS. Use the difference from the replayed BF16 head as the
rounding remainder. The three finite attribution terms are exact Shapley
allocations of the eight rounded coalition margins. Each coalition adds
its selected measured differences to the B-world state then rounds to BF16.
The empty and full coalitions use the captured BF16 bytes. The first attempt
stopped before any backend summary when an FP64 cancellation gave a native
zero coordinate a tiny nonzero value. The amended calculation anchors
algebraically equivalent coalitions to the native endpoints.
Exact closure tests the accounting and does not establish any component's
behavior under a native intervention.

Summarize the phase 2 minus phase 1 change inside each root. Primary cells
are focused DEV by precision. CAL, copy and erased are descriptive checks.
Report absolute and signed changes, contribution sizes relative to the sum
of absolute native gains, and per-root comparisons with initializer and
random controls. Undefined shares for zero denominators are retained.
The formal test count is zero. There is no physical intervention dose or
estimated causal sham floor for this endpoint audit. Grade it as descriptive
observational evidence, with native matched-sham tests belonging to the
queued interventions. Keep the original ARG-03 test sealed.

Use the pinned NumPy and tokenizer in an admitted below-normal single-thread
CPU lease. The minimum start RAM is 5 GiB, reserve 4 GiB, private cap 0.75 GiB.
No Torch, pretrained forwards, or GPU work. Keep individual rows local.
