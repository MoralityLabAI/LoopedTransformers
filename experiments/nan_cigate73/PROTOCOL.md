# NAN-CIGATE-73 pointwise BPD controller readout replay

ATTNSPAN67 compared the linear VU map and physical write span at
layer 21. CIGATE73 uses the independently audited rank-eight BF16
T1 and T2 fits to replay the trained pointwise controllers on
the original clean COMP14 attention projection inputs. Both
compared recipes retain the native kernel residual. T1 uses
zero centering and no sphere rescaling. Apply T1 at pass 2 as
an explicitly out-of-fit-depth control. T2 uses the checkpoint's
pass-mean centering and sphere-rescale contract. Compare the
learned pointwise gates with the frozen original FIT07 CAL mean
masks used in the RANKSITE13 diagnoses. Neither PI/CI mask is
fit to COMP14.

For each A/B pair at each physical pass, set a = X V - mu
and c = mask - 1. With a native residual gate of one, the
adjustment before sphere rescaling is (c * a) U. Partition
the signed A-minus-B write contrast into
mean(c) * delta(a) and delta(c) * mean(a), both projected by U.
Retain the sphere rescale change and the final BF16 rounding
change separately. Freeze the observed residual and its mean
inverse RMS for each paired readout. This yields an exact
additive accounting for the local replay on the supplied clean
states. Report the analogous retained-native all-one no-op and
pure linear SVD/learned VU readouts for context.

The primary descriptive comparison is the dynamic minus
original-static change in pass-2 versus pass-1 answer-aligned
attention numerator at layer 21, averaged over 16 exposed
focused DEV roots. Retain all eight CAL and 16 DEV roots,
their two worlds, focused/copy/erased arms, and both visits.
Report physical edit L2 against the native write, gate
saturation, and the four projected effects. Erased twins
must give zero contrast, all-one must reproduce native bytes,
and positive gauge rescaling and permutation must preserve
physical results. The archived CI replay must match within
its original threshold. Replay arithmetic uses FP64 and
one final BF16 rounding, with no edited downstream MLP or
model pass. Fit recipe, depth, and ablation policy differ
between T1 and T2. Effects apply to this registered clean
input intervention policy only.

The independent unit is the graph root. This is descriptive
development evidence with zero formal tests and uncalibrated
causal effect and sham floors. No sealed test access or
new pretrained model forwards. The fitted controller is only
available here at BF16 layer 21. Native reached-state behavior
and correctness mediation require separate experiments.
