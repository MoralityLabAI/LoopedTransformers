# INVERSE65 residual bounds across recurrent depth

The largest sampled answer contrast appears at layer 17 in pass 2.
Layer indices are zero-based. The contrast is already positive before
that layer's attention on all 16 focused
DEV roots in both precisions. Layer 17 attention increases it on all
16 roots in both precisions. The MLP reduces it on average.
After layer 21 the contrast is still positive on average and is about
one-third of the layer-17 value.

| Focused DEV, mean contrast | NF4 | BF16 |
|---|---:|---:|
| Pass 2 layer 5, outgoing finite bare-label margin | −0.007 | +0.018 |
| Pass 2 layer 11, outgoing finite bare-label margin | −0.027 | −0.005 |
| Pass 2 layer 17, outgoing finite bare-label margin | +31.556 | +44.581 |
| Pass 2 layer 21, outgoing finite bare-label margin | +10.578 | +13.195 |
| Pass 2 layer 17, incoming numerator | [18.485, 19.205] | [25.690, 26.442] |
| Pass 2 layer 17, measured attention write numerator | +17.570 | +26.647 |
| Pass 2 layer 17, pre-MLP numerator | [36.252, 36.578] | [52.542, 52.883] |
| Pass 2 layer 17, measured MLP write numerator | −4.832 | −8.082 |

The signed contrast compares the two gold label logits across each
counterfactual pair. Intermediate positions use the frozen final head as
a lens. For the numerator terms, both worlds use the mean inverse RMS
of their measured outgoing states at that site. The incoming bound is
therefore for a linear component of the outgoing readout. The exact
finite scores in the first four rows include the head's normalization
and rounding. Both labels beat the other world's label at layer 17 in
16/16 NF4 and 16/16 BF16 roots. This restricted pair comparison uses
two labels and provides a distinct metric from unrestricted native
greedy correctness.

As a depth comparison, the mean incoming numerator increases by
[18.190, 19.514] NF4 and [25.375, 26.740] BF16 from pass-2 layer 11 to
layer 17, using each site's outgoing RMS scale. At layer 17 the incoming
and pre-MLP numerator gains from pass 1 to pass 2 are also positive on
every focused root. The mean layer-17 attention-add rounding
contributions are bounded by [−0.523, +0.523] NF4 and [−0.547, +0.546]
BF16 logits, and the MLP-add terms by [−0.164, +0.162] NF4 and
[−0.171, +0.170] BF16 logits. These conservative bounds follow by
subtracting operand intervals at the same scale. Rounding is small
relative to the measured writes at this site.

The last-layer BPD write result from CARRY63 describes a site downstream
of the largest sampled bare-label contrast. A delayed PercInterp test
should instrument the interval between sampled layers 11 and 17,
including the incoming residual and the measured layer-17 attention
write. Retain the last-layer readout to test whether the change survives
to the output. These archived twins identify a window and a readout
contrast for that intervention. Identifying a donor-specific causal path
still requires matched live edits.

## Evidence and checks

The inverse searches all 65,279 finite numerical BF16 values for each
coordinate, collapsing signed zero. Monotonicity gives the endpoints
of the compatible input set under the registered FP32-to-BF16
addition rule. At the focused layer-17 pass-2 site, 44.6% of incoming
coordinates and 73.8–73.9% of pre-MLP coordinates were unique. Some
input coordinates admitted more than 30,000 alternatives. The
sign conclusion follows from projecting the entire compatible set
through a fixed head direction. No single inferred hidden state is
required.

Across both precisions, 2,304 sampled states from 8 CAL and 16 DEV roots
had compatible intervals. The stored logits replayed exactly at all 576 native end-of-pass states. Erased twins had zero contrasts and zero paired-width
intervals. An independent reduction reconciled 2,304 root-site intervals.
Ten synthetic stress cases matched exhaustive enumeration. A separate
random-weight CPU fixture checked both intervals against 256 directly
captured inputs and pre-MLP residuals from the snapshot's decoder code.
All captures were inside their intervals. All 197 unique incoming and
245 unique pre-MLP predictions matched. The archive replay
peaked at 142 MiB private commitment. It made zero new pretrained
forwards, ran no formal tests and kept the sealed ARG-03 set untouched.

[Protocol](../../experiments/nan_inverse65/PROTOCOL.md) ·
[Direct-capture verification](../../experiments/nan_inverse65/CPU_VERIFICATION.json) ·
[Aggregate results](nan_inverse65_results.json)
