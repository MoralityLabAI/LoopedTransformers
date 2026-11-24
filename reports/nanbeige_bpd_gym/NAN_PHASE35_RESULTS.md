# PHASE35 Nanbeige story residual geometry

PHASE35 rechecked the archive hashes for all **72 native NF4 episodes and
3,990 prefixes** in the previously verified STORYAUDIT30 block. It read
15,960 residual vectors at layers 17 and 21, two physical passes at each
prefix. The phase labels come from the execution metadata. CAL has three
roots and DEV has nine. Statistics average within episode, then root.
Worlds, arms, prefixes and coordinates are repeated measurements.

## Layer 21, DEV main

| View | Phase-paired prefixes | Full cosine | Cosine with channel 2279 omitted | Share of phase-difference energy at 2279 | CAL threshold balanced phase accuracy |
|---|---:|---:|---:|---:|---:|
| First prefix | 18 | 0.05388 | 0.12080 | 90.80% | 100% |
| All prefixes | 884 | 0.29017 | 0.06615 | 83.70% | 100% |

The coordinate contributes **96.42%** of phase-1 first-prefix residual
energy and **0.129%** of the corresponding phase-2 energy. The strongest
of the ten coordinate controls accounts for **0.452%** of the
first-prefix phase-difference energy and **0.216%** across all prefixes.
At layer 17, channel 2279 accounts for **2.34%** and **5.11%** in
those respective views. The layer-21 phase classifier stays at 100%
balanced accuracy across all prefixes in each of DEV main, copy and
erased. Other coordinates can also separate phases on the first prefix.

Omitting 2279 **raises** the first-prefix mean cosine and **lowers** the
all-prefix mean cosine. Cosine changes reflect both the dot product
and vector norms, so the direction depends on the sampled token
positions. Full-residual cosine alone cannot localize the cause.

## Counterfactual worlds

DEV main A/B first-prefix whole-vector cosines were **0.99999583**
in phase 1 and **0.99975222** in phase 2. Channel 2279 accounts for
**69.88%** and **10.51%** of the respective squared A/B differences.
These shares describe differences of very different total sizes.
The 36 A/B pairs had matching semantic lengths but different batch rows
and total padding. Padded length also differed in 24 of the 36 pairs. Erased controls also show a large phase-1
share at this coordinate. A same-canvas rerun with a world-specific
readout is necessary to assign the difference to world content.
No intervention or task improvement was measured by this audit.

For the BPD experiment, model phase routing explicitly. Carry 2279
as a phase-control coordinate alongside a predeclared matched-dose
test of donor-specific future correctness. Check the signed read
effect after normalization before ranking cross-pass pathways.
Report full-vector and coordinate-omitted geometry together.
[PercInterp and BPD experiment bridge](NAN_PERCINTERP_BPD_BRIDGE.md)

## Custody and resources

Source units are the 12 audited story roots. Scope is retrospective
development on one NF4 checkpoint. Physical intervention dose and
calibrated sham floor are unavailable, with zero formal comparisons
and verdict `REDESIGN_OR_PILOT_ONLY`. The canonical RAM28 final
parameter receipt and matched BF16 geometry are pending. The original
168-case test remains sealed.

The audit ran in one Below Normal CPU process alongside existing
inference. Maximum sampled private RAM was **36.5 MiB** and minimum
sampled available RAM was **7.55 GiB**. The audit made zero neural forwards.
[Full sanitized metrics](nan_phase35_results.json) ·
[Source and input bindings](../../experiments/nan_phase35/SOURCE_MANIFEST.json) ·
[Closed story audit](NAN_STORYAUDIT30_RESULTS.md)
