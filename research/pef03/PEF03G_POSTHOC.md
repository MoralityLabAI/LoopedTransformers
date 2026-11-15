# PEF-03G | Natural counterfactual versus hybrid-subspace mediation

**Status:** Explicitly posthoc, downstream of PEF-03, PEF-03N and
PEF-03F. No new training, checkpoint selection, or parameter tuning.

The first pinned future-reactivation audit
([PEF-03F run 43681766062](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43681766062))
found that replacing an intermediate head-nullspace component with
that of an oldest-bit-flipped donor could change the predicted task
class for up to 14 of 52 pairs in seed 17. The completely natural
oldest-bit counterfactual changed the predicted task class on only
2 of 52 pairs in the original PEF-03. This difference motivates
an explicit finite counterfactual decomposition.

## Pinned weights, fixed intervention and exact identity

Use the *original* frozen models archived by run 43680455308:

- seed 17 SHA256:
  c174fbfd8bef4fa026806e62e66ad48a38827935c1121a487514de9948f61432
- seed 29 SHA256:
  76dfcf7c401a8d60ae4570cf281b0aa788362bb76bde7f0c040438333a5a6e01

Let s denote a memory state after visit j on one of the 52
original heldout eight-bit histories. Let s' be its corresponding
naturally generated memory state with **only the oldest input bit**
flipped. This counterfactual is a genuine model trajectory. Newer
inputs are held fixed and the supervised last-two-bit target is
unchanged.

Let P_R and P_N be the orthogonal projectors onto the rowspace and
nullspace of the final frozen 4x16 classifier head W.

Form two hybrid states:

    s_R = P_R s' + P_N s
    s_N = P_R s  + P_N s'

For a fixed suffix of the original newer inputs, denote the future
task-logit map from visit-j memory by G_j. Define

    Y0 = G_j(s),
    YD = G_j(s'),
    YR = G_j(s_R),
    YN = G_j(s_N).

Then the exact *finite-difference decomposition* is

    Δnatural = YD-Y0
    Δrow     = YR-Y0
    Δnull    = YN-Y0
    I        = YD-YR-YN+Y0

    Δnatural = Δrow + Δnull + I.

This identity holds for every nonlinear G_j. The interaction I
captures nonadditivity in the hybrid interventions. It should be
reported alongside raw effects; a large Δnull need not mean the
natural counterfactual changes the task class.

At the final visit G_(L-1) is precisely the affine head Wz+b,
so Δnull=I=0 exactly over real arithmetic. The FP32 implementation
requires both quantities to remain below 1e-4.

## Independent local derivative

For the first heldout pair, project the naturally occurring hidden
difference δ=s'-s into the final head kernel: δ_N=P_Nδ. Differentiate
the **actual** future logit map along that input-specific direction:

    v_j = D G_j(s)[δ_N].

This compares a model-trajectory tangent with the finite off-manifold
transplant. A nonzero v_j proves local future-logit sensitivity to
the naturally aligned nullspace direction. It does not establish a
large finite intervention or causally independent information channel.

## Integrity and evidence

The code audits every heldout pair at all eight visits, verifies the
two natural arms reproduce their frozen full-trajectory final logits,
decomposes row/null/interaction effects, and cross-checks the null-only
patch against the independently written PEF-03F runner on a
small scratch-trained model. Full model weights and the source
checkpoint SHA256 remain unchanged.

The workflow retrieves the exact original checkpoint artifact from
run 43680455308 and writes only new posthoc JSON receipts. It runs
on CPU with Torch 2.10.0. A green CI result certifies the stated
numerical checks; it does not make a nonlinear hybrid state
in-distribution.

The conclusion of this experiment concerns synthetic eight-bit input
histories in a single 16D tied attention model. The next scientific
test is a prospectively registered masked-activation mechanism on
several independently trained checkpoints with multiple tasks and a
head-state independence control.
