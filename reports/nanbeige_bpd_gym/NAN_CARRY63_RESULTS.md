# CARRY63 native answer-margin attribution

The effective carry has the largest signed component of the pass-2 answer
margin on all 16 focused DEV roots in each precision. The measured last-layer
MLP write partially offsets that gain. The fitted rank-eight BPD span
captures 45–48% of the pass-2 MLP write-contrast energy and contributes
less than 1% of the observed gain in absolute finite margin. These are
paired endpoint measurements on exposed roots.

| Focused DEV, 16 roots per precision | NF4 | BF16 |
|---|---:|---:|
| Native A/B margin contrast, mean absolute, pass 1 → pass 2 | 0.012 → 10.578 | 0.015 → 13.195 |
| Native signed margin gain | +10.572 | +13.188 |
| Effective carry, signed finite attribution | +14.562 | +17.789 |
| MLP outside fitted span, signed finite attribution | −3.991 | −4.562 |
| MLP inside fitted span, signed finite attribution | +0.002 | −0.040 |
| Fitted span, absolute finite attribution / absolute native gain | 0.57% | 0.74% |
| SVD initializer, absolute finite attribution / absolute native gain | 0.67% | 0.83% |
| Fitted span, pass-2 write-contrast energy | 44.68% | 48.48% |

The fitted span had larger absolute in-span margin attribution than the
per-root median of eight equal-rank Gaussian spans on 13/16 roots in
each precision. It exceeded the SVD initializer on 1/16 in each. A
separate prediction test used the archived MLP inputs and the fitted
factor operator VU. It obtained mean absolute numerator gains of 0.023
and 0.068 logits, against 3.994 and 4.602 logits in the complete measured
MLP write. The squared errors relative to predicting zero were 1.001 and
0.980. A physical write span can capture substantial activation energy
while giving little leverage on this particular answer contrast.

## What the measurement identifies

For each phase, the three finite attribution terms average all orders of
adding the in-span write difference, the off-span write difference and the
effective carry to the B-world state under the same BF16 head. Effective
carry is post-layer residual minus measured MLP output, so it includes
rounding and any computation already present before the MLP residual add.
The origin of the signal and its causal path require separate measurements.

An independent continuous decomposition of the head separates the linear
numerator, RMS change, and BF16 remainder. The effective carry's continuous
numerator gain exceeded the residual-add rounding bound on 16/16 focused
DEV roots in each backend. The largest per-root bounds were 0.532 NF4
and 0.547 BF16 logits. The corresponding mean absolute carry numerator
gains were 14.586 and 17.809. The bound applies to the continuous
numerator only.

This constrains how to use BPD, VPD and PercInterp on this gym. Compare
selector outputs in physical coordinates and include the initializer.
Test the selected directions against a task-relevant readout as well as
activation energy. The carry result gives a
concrete site for the next instrumentation pass: capture the native
pre-MLP residual beside the MLP input and write through both phases, then
use the registered delayed and dose-matched controls to test candidate
cross-pass links. CARRY63 itself applied no native intervention.

## Verification

The replay used 8 CAL and 16 DEV COMP14 roots at both precisions, with
focused, copy and erased arms. All 576 archived states replayed their
stored selected and top-token logits exactly. The erased contrasts were
all zero. Changing the trained basis by a rotation and rescaling kept
the physical projection within 1.8e-13 maximum absolute error and preserved
every rounded coalition score. The continuous decomposition and BF16
remainder closed within 7.2e-15 logits. An independent reduction reconciled
2,880 finite allocations and all in-span gain aggregates.

The first attempt stopped before producing a backend summary when floating
point cancellation changed an exact zero at a coalition endpoint. The
rerun anchored those endpoints to the captured bytes and rechecked closure.
It peaked at 147 MiB sampled private commitment and retained at least 5.86
GiB available RAM. No Torch, new pretrained forwards, formal inference or
sealed-test access. Evidence class is descriptive observational.

[Protocol](../../experiments/nan_carry63/PROTOCOL.md) ·
[Aggregate results](nan_carry63_results.json)
