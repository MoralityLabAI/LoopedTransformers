# NAN-ATTNSPAN-67 attention write spans and the answer readout

INVERSE65 measured a strong layer-17 attention contribution to the growth
of the answer contrast in pass 2. Compare fixed low-rank physical write
spans at that site and at layer 21. Use the independently audited COMP14
archive at both precisions. Keep the eight CAL and 16 exposed DEV roots,
focused/copy/erased arms, and both phases. No pretrained forward, edit,
formal comparison or sealed ARG-03 access.

## Frozen comparisons

Measure the output span of the saved randomized weight SVD at ranks 4,
8 and 16, using the first four factors of its rank-eight bank. Compare
to eight deterministic Gaussian spans at rank four and eight. Include
the same backend's fitted rank-eight attention BPD read/write pairs at
layer 21 for BF16, both T1 and T2. Those comparisons use the raw factor
operators and physical write spans. The attention BPD controls use their
own audited fitted parameters and original SVD initializer. No trained
attention cell exists at layer 17 in this comparison.

Fit three rank-four controls from focused CAL roots only. Uncentered
PCA uses each root's output-write contrast from the second phase minus
the first, with each phase weighted by the observed endpoint's mean
inverse RMS. A supervised span uses the left singular vectors of the
cross-covariance of these gain vectors and unit bare-label head
directions. The shuffled baseline rolls the CAL label directions forward
by one source root before doing that decomposition. A fourth rank-four
control uses only the CAL label directions. Fit the same controls for each
precision and layer. DEV is used for descriptive evaluation.

## Endpoints

Use the signed A-world minus B-world contrast under the fixed bare-label
A-minus-B head. Report the pass-2 physical write-contrast energy captured
by each span. Report agreement with the full attention numerator gain,
including errors relative to predicting zero, signed retention and the
number of roots where a projection retains the full write gain's sign.
Logit numerator terms use the actual outgoing states to fix the inverse
RMS scale. For each span also re-read the eight BF16 endpoint coalitions
to recover the in-span, off-span and remaining-state allocations. The
finite calculation treats the downstream measured state as held fixed
and cannot substitute for replay through the intervening native MLP.

For saved SVD and trained BPD operators, score a separate reconstruction
from the measured attention output-projection input contrast multiplied
by the saved VU. This prediction uses no measured output vector.
Contrast reconstruction quality and task-readout error, including the
result of using no prediction. In-span allocations condition on the
measured native output. Layer-17 focused DEV is primary. Layer 21, CAL
and the two controls retain their original denominators. Zero-denominator
shares stay undefined. Run only the archive's CPU resource lease.
