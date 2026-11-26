# MOLT-10: exact response-target factorization for MMS

## MOLT-09 evidence

Native engineering [run 45854775751](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45854775751)
passed at b30cc1d495ad228af3d6a1b5bf23b8eef7cd8ee7. Two anchors and
sixteen distinct TRAIN cross-item donors each; native swap/dense parity
1.669e-6 maximum, clean parity zero, two-step FD relative disagreement
8.524e-11 maximum. Both stacked and terminal rank-four approximations
accepted 0/32 at the prespecified ten-percent relative-error and all-visit
argmax gate. The restricted eight-dimensional observer has a fairly flat
spectrum. All five continuation arms scored 1/17 on validation, equal to
the constant baseline. The teacher had eight training updates and the
native factor fit had two; no competence, factor-quality or training
advantage is established. All compressed training records used full
fallback. The exact native-coordinate payload was 2,048 B; the shared
native chart 8,192 B; dense teacher response targets 1,310,720 B in fp32.
A local map is rejected by the storage planner. This is a useful failure.

## Identity

For the native decoder, let N be its final normalization, W the tied
unembedding with vocabulary rows, and C=I-11^T/V the vocabulary centering
projector. The centered response at visit j is

    r_j = C W [N(h_j^patched)-N(h_j^clean)].

Store the normalized-state difference b_j and frozen centered teacher
unembedding W_t rather than a full vocabulary response. This is an exact
real-arithmetic factorization of the declared linear readout *after*
normalization. It does not require a low-rank approximation or freeze
the student embedding. For a trainable student W_s and its normalized
difference a_j, the complete centered-logit squared error is

    ||W_s a_j - W_t b_j||²
      = a_j^T (W_s^T W_s) a_j
      - 2 a_j^T (W_s^T W_t) b_j
      + b_j^T (W_t^T W_t) b_j.

All W symbols here are centered. Keep all derivatives through the student
head and a_j; the teacher and teacher Gram are frozen. A single frozen
teacher Gram cannot replace both student-dependent matrices if the student
unembedding is updated. The scalar loss may have roundoff cancellation;
no clamping is used. Compare against direct squared errors and gradients.

## Engineering panel

Reuse the validated native BPD factor loader and exact native output-span
write coordinates. No projected donor is needed; we compare response
storage representations instead. Fresh engineering teacher/factor fit
with the same tiny MOLT-09 recipe. Convert model and target calculations
to float64. Two anchors and sixteen distinct donors, all TRAIN, visit 4
through 8. No new native intervention API claim beyond MOLT-09's parity.
Compare four continuation arms with identical initialization, sample
schedule, optimizer and full auxiliary MSE: dense stored teacher logits,
decode normalized teacher difference, Gram objective, and fresh frozen
teacher recomputation. Check full native readout reconstruction, gradients
on the first and last updates, losses, final parameters and ordinary
validation predictions. Tests include a student with changed unembedding
and nonzero response loss so gradient parity is not merely a zero-loss
initialization artifact. Task/trace loss retains the existing fp32
cross-entropy implementation. No intervention during evaluation.
No distinct task population or mechanism claim; this is numerical parity.

## Accounting

With 32 records, 5 visits, d=128, V=2048, fp64 response storage falls
from 2,621,440 to 163,840 B (16x, excluding the head). Charge the
2,097,152 B frozen centered head if it is not already resident, and
131,072 B teacher Gram if separately retained. Charge native coordinates
and their common chart equally. Recompute needs no stored response
targets but requires the complete teacher resident and two teacher
forward trajectories per batch; its teacher weight/runtime costs are
outside the target-payload number. The development program deliberately
retains all representations for comparison, along with gradient/state
parity references, so it does not measure a reduction in total resident
RAM. Timings include the actual student/teacher calls and parity snapshot
overhead; do not claim exact FLOP equivalence or a measured training
speedup. Gram work can be expensive when vocab is small or batch is small.
Only the real-arithmetic identity is exact; native float64 gradients and
state trajectories have tolerance gates.

    python -m pytest -q tests/test_aux_response_factor.py
    python -m research_gym.interp.aux_gym_v2_response_factor --data /tmp/molt10/data --run /tmp/molt10/teacher --factors /tmp/molt10/factors --out /tmp/molt10/replay

No protected readout, formal certification, confirmation, or promotion.
A larger competent-teacher experiment with real resource accounting,
operator hard caps and prospective EvalMath design remains necessary.
