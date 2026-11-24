# DOSE51 low-dose physical qualification of adaptive and CAL-static BPD routes

Using the phase-1 COMP14 archive, we applied both signs of each route's write
displacement at a target of 0.25% of the clean residual's L2 norm. Each learned
mask and CAL-static mask had the same soft mass at its recipient. Qualification
required realized dose within 3% of target, direction cosine at least 0.98,
and realized-dose gaps across the four arms within 3% of target. The current
head gate required the same greedy token, maximum absolute vocabulary-logit
shift at most 0.125, and absolute bare-label margin shift at most 0.125.

| Precision | Eligible signed edits / 128 | Both routes and signs eligible / 32 worlds | Four doses and full head passed / 32 worlds | Both worlds qualified / 16 roots |
| --- | ---: | ---: | ---: | ---: |
| NF4 | 0 | 0 | 0 | 0 |
| BF16 | 6 | 1 | 1 | 0 |

The 0.98 cosine floor excluded 250 of 256 signed edits. The median
direction cosine after rounding was about 0.693 learned and 0.827 static
in NF4, and 0.853 learned and 0.947 static in BF16. All six angularly
eligible edits also passed the physical-dose and full-head rules. The
clean replay matched all selected and maximum logits exactly. The
largest clean log-normalizer error was (1.20 × 10⁻⁷) across 166,144
vocabulary rows.

In the one matched BF16 world, the learned and static edits remained
physically distinct in both signs. Their displacement difference was
about 39.5% of the nominal dose and their cosine was about 0.922 at
both signs. Before residual scaling and rounding, the median direction
cosine across all worlds was 0.963 in NF4 and 0.976 in BF16.
Equal soft gate mass alone did not produce equal physical interventions.
The four-arm comparison lacks both-world coverage in both precisions.

A separate CPU fixture on real BF16 tensors exercised the staged tiny
decoder and streamed head. Four distinct edits entered pass 2, exact
same-sign aliases used two continuations, and refused dose or head
plans entered no edited pass 2. Recovered unedited output matched exactly.
The first archive attempt stopped before scoring because the reused
reader returned the projection input and residual only. The final
freeze loads the missing write directly from the same hash-verified
archive and preserves the scientific gates.

The independent unit is the source root. The 16 exposed DEV roots,
their repeated worlds, both signs and the two precision lanes are
reported descriptively. The experiment calibrated directional admission
and observed no downstream pretrained intervention outcomes. The ARG-03
seal was preserved. [Protocol](../../experiments/nan_dose51/PROTOCOL.md)
· [Aggregate and CPU verification](nan_dose51_results.json)
