# ATTNSPAN67 attention write spans and the answer margin

At layer 17, the best fixed weight-SVD control captured under 0.5% of the
observed attention contribution to the answer-margin gain in either
precision. A rank-four PCA span fit to the measured CAL write contrasts
captured 59.4% in NF4 and 43.3% in BF16 on DEV. These percentages are computed by projecting the measured
native attention outputs. The separate input-only prediction is reported
below.

| Layer 17, focused DEV | NF4 | BF16 |
|---|---:|---:|
| Full measured attention numerator gain, logits | 17.575 | 26.643 |
| Rank-16 weight SVD, pass-2 write energy captured | 1.18% | 1.05% |
| Rank-16 weight SVD, numerator gain captured | 0.39% | 0.49% |
| Rank-4 CAL write PCA, pass-2 write energy captured | 51.11% | 44.25% |
| Rank-4 CAL write PCA, numerator gain captured | 59.4% | 43.3% |
| Rank-4 CAL label directions, pass-2 write energy captured | 15.73% | 19.16% |
| Rank-4 CAL label directions, numerator gain captured | 51.9% | 51.2% |

The rank-16 SVD predicted mean gains of 0.099 and 0.170 logits when
applied to the measured attention inputs. Its squared errors relative to
predicting zero were 0.988 and 0.987. Rank-four readout-aware calibration
and its shuffled-label control were close to plain CAL write PCA.
The CAL label-only control had the lowest numerator squared error among
the fixed rank-four controls in both precisions. This exposes dependence
on task label geometry even in the absence of training on activation
writes.

At layer 21, the trained BF16 attention BPD spans and their rank-eight
initializer each captured about 3.1% of the measured pass-2 write
energy and 0.042 logits of the 3.086-logit attention gain. Using their
read/write factors to predict from attention inputs gave mean gains of
0.052–0.053 logits and squared errors of 0.962–0.963 relative to
predicting zero. This compares the native physical spans and raw VU
operators. Controller-dependent nonlinear behavior requires a separate
test. This archive does not supply a trained layer-17 attention bank.

## Held-out answer labels

All 16 focused DEV roots use an unordered answer-label pair that appears
in CAL. The eight CAL pairs partition the 16 bare labels. After observing
the primary result, we registered a secondary replay that leaves out
both labels of each target pair, fits on the remaining seven CAL roots,
and scores the same 16 DEV roots.

| Layer-17 numerator gain retained | NF4 | BF16 |
|---|---:|---:|
| CAL write PCA, shared label inventory | 59.4% | 43.3% |
| CAL write PCA, both target labels held out | 0.066% | 0.185% |
| CAL label directions, both target labels held out | −0.022% | 0.019% |
| CAL readout-aligned span, both target labels held out | 0.062% | 0.168% |

Holding out both labels also cut the PCA span's pass-2 energy capture to
0.94% and 0.73%, respectively. Its normalized numerator error rose to
0.998 NF4 and 0.996 BF16, near the zero-prediction baseline. These spans
capture label-conditioned structure across independent prompts sharing
an answer inventory. The label-disjoint replay gives little evidence of
a shared output direction for the task. A causal BPD or PercInterp
protocol should hold out label identities as well as source roots when
testing transfer of directions fit to activation data. WINDOW66 can
still score the delayed effect of directions defined independently of
these held-out label pairs.

## Verification and interpretation

The primary replay used 8 CAL and 16 DEV roots in both backends, both
phases, two physical layers and the focused, copy and erased arms.
The stored native logits replayed exactly at all 576 end-of-pass
states. An independent reduction reconciled 13,536 finite allocations,
the projected energy fractions and the numerator error ratios. Erased
twins had zero physical and readout contrasts. The CPU replay peaked
at 163 MiB private commitment. No new pretrained forwards, formal tests
or sealed ARG-03 access.

The finite calculation holds other observed state contributions fixed
and recomputes the same BF16 head for each coalition. It gives an
endpoint readout allocation. A live edit through attention, the native
MLP and the remainder of pass 2 is needed to measure an actual pathway.

[Protocol](../../experiments/nan_attnspan67/PROTOCOL.md) ·
[Primary aggregate](nan_attnspan67_results.json) ·
[Label-disjoint aggregate](nan_attnspan67_label_holdout_results.json)
