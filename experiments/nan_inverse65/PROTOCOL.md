# NAN-INVERSE-65 archive residual bounds

CARRY63 measured the effective carry at the final layer. PREMLP64 captures
the actual operands during a native run. INVERSE65 uses existing archives
to bound the unknown operands without another pretrained forward. Each
sampled layer retains its own attention and MLP projection writes and
outgoing layer residual. The sampled sites are layers 5, 11, 17, and 21
in both physical passes.

For each archived coordinate let stored attention and MLP writes be o and
y, and the outgoing residual h. Invert h = add(add(a,o),y) on the ordered
finite BF16 lattice, where add widens to FP32 and rounds the sum to BF16.
The unknown incoming residual is a. The two rounded sums are consecutive
native attention and MLP residual adds under the pinned configuration.
Treat signed zeros as a single numerical value. Each sum is monotone for
fixed finite writes, so two binary searches give all compatible input
codes. Forward evaluations check both endpoints and their neighboring
codes. The images of the two endpoints under the first sum are the tight
minimum and maximum possible pre-MLP numerical values for that coordinate.
Some values between those endpoints may be unattainable.

Use the exact observed outgoing states for the common inverse-RMS
weighting. Measure the signed A-world minus B-world contrast on the
bare A-gold versus B-gold token margin. For each phase and root, report
intervals on the incoming and pre-MLP continuous linear numerator.
Use per-coordinate extrema over the independent inverse sets. Correlations
in the true decoder can narrow those bounds. Also report the measured
attention and MLP numerator terms, effective incoming, add-rounding
bounds, RMS change, and the exact finite outgoing contrast. Report depth
and phase differences as intervals and retain copy and graph-erased
controls. Identical erased twins share their unknown source value by
determinism, so their paired interval collapses to zero.

Report all eight sampled readout positions. Intermediate positions use the
same final head as a lens. Primary results are the 16 exposed focused
DEV roots per precision, with 8 CAL roots as a descriptive check.
There are no formal tests, native interventions, calibrated sham
floors or claims about which specific computational path carried the
signal. No Torch, pretrained forwards, or sealed ARG-03 access.
