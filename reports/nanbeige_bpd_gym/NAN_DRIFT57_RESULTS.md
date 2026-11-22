# DRIFT57 full-head profile of the NF4 sham on exposed DEV

The frozen peak-aware SHAM56 recipe and its prespecified RMS refinement
matched the signed physical dose and contrast controls in all 32 worlds
from 16 previously exposed focused COMP14 DEV roots. All four current
heads passed in **30/32 worlds**, covering both worlds of **15/16 roots**,
for each recipe. The same two worlds were refused by the active control
under both recipes. In those worlds each active sign changed the
unrestricted greedy token. Absolute peak and bare-margin shifts stayed
below 0.125.

| Full-head passes | Peak-aware sham | RMS-refined sham |
| --- | ---: | ---: |
| Worlds | 30/32 | 30/32 |
| Both-world roots | 15/16 | 15/16 |
| Worlds with absolute RMS gap ≤0.005 in both signs | 0/32 | 0/32 |
| Median active RMS shift | 0.01825 | 0.01825 |
| Median sham RMS shift | 0.02924 | 0.02924 |
| Median signed RMS gap | +0.01083 | +0.01083 |
| Median cosine between logit-shift vectors | 0.00190 | 0.00190 |
| Median RMS distance between logit-shift vectors | 0.03452 | 0.03444 |

The median worst-sign RMS gap among the 30 worlds common to both recipes
was 0.012264 peak-aware and 0.012246 refined. The refinement improved that
gap in seven worlds and preserved all original gate decisions. The median
lowest continuous RMS in the 24-dimensional derivative atlas was **0.02813**.
The original and refined continuous medians were approximately 0.02904.
These minima concern unit directions in the registered atlas at the nominal
dose. The finite rounded head and more general ambient spaces have different
optimization problems.

The two execution blocks streamed 174 unique states over each full
166,144-row head. The clean winner and top logit reproduced exactly, with
maximum clean log-normalizer disagreement below 9.8 × 10⁻⁸. Peak sampled
private memory was about 410 MiB and peak working set about 393 MiB. The
unit is the source root. This is an archive audit of the first pass on
exposed DEV roots, with zero formal tests and no new decoder calls. The
original ARG-03 seal was preserved. [Protocol](../../experiments/nan_drift57/PROTOCOL.md)
· [Aggregate](nan_drift57_results.json)
