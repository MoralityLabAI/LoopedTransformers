# PEF-06H | Posthoc trained-head h8 control

This extra control was added after an independent local sandbox replay
of the original PEF-05 checkpoints. The first PEF-06 specification
committed in PRE_AUDIT.md identified the natural **h10 with 01
continuation** as the primary audit stage. The output head was
originally trained only on the eight-bit h8 task; its predictions
at h10 are *post-training extrapolations*, not the original
trained-task accuracy. This document preserves that distinction.

The companion frozen_coalitions.py now additionally measures
the original **trained h8 task head** on the same complete
256-point cube for all twelve original checkpoint models.
All old-input singleton and 2-bit interventions preserve
the newest two bits that define this trained task.

An independent local sandbox replay, using the original
source frozen checkpoint artifact SHA256 digests and a
compatible PEF-03 block implementation, gave the following
descriptive full-cube measures at **h8**. The local replay
is distinct from hosted CI and is not a new heldout result.

| Seed | Arm | Original trained h8 task correct /256 | Single-old-bit class changes (summed over six flips) | Pair-only class changes (summed over 15 pairs) |
|---:|---|---:|---:|---:|
|101|baseline|244|96|90|
|101|protect|248|60|52|
|101|sham|247|68|57|
|109|baseline|248|48|24|
|109|protect|252|32|24|
|109|sham|252|32|24|
|127|baseline|250|50|41|
|127|protect|249|56|40|
|127|sham|249|54|40|
|157|baseline|256|0|0|
|157|protect|256|0|0|
|157|sham|256|0|0|

The same sandbox computed h10 continued-head original-label
agreement counts of 117/122/123 for seed101
(baseline/protect/sham), 124/119/122 for seed109,
111/102/110 for seed127 and 128/128/128 for seed157.
These h10 values are *not* h8 supervised-task scores.

A pair-only class change means the original class prediction
survives each singleton flip separately but changes under the
double flip. It can occur because a nonlinear hidden-state
interaction moves the logits, or because two individually
smaller logit changes jointly cross a fixed class boundary.
Therefore it is accompanied by exact four-corner hidden-state
and logit finite differences. It cannot by itself identify
a learned AND-pair circuit.

The original PEF-05 protected-margin rescue criterion was
negative. These posthoc full-source-domain intervention
counts do not change it, and the 256 inputs include
previously seen training/probe/test histories.

The source code and complete original model checkpoint
manifest are the reproducible authorities. Local CPU
results may differ numerically from another CPU host,
as the merged PEF-05R audit already demonstrated.
