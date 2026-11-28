# MOLT-15: amortized finite repair

Development continuation of [MOLT-14](CLOCK_REPAIR_RESULTS.md). The raw
native swap recovered 17/32 changed-target suffixes, compared with 10/32
for zero-start native margin regression. The frozen teacher is competent
on the ordinary sequence task. We therefore investigate learning by a
small *repair controller*, with explicit counterfactual headroom. No
ordinary teacher learning, autonomous reasoning, or promotion claim.

## Objective and controls

Use the fixed addressed seed 157, its native fitted 16-component U span,
and the same physical residual dose as the raw native swap. The target
is the clean donor's actual cut legal margins; the teacher's legal
argmax is the target class for CE. This is privileged observed donor
information. Gold state labels are used only for final evaluation.
No access to patched suffix outcomes is used in fitting or selection.

A raw swap may be a good initialization for the nonlinear finite
optimization. Compare the raw swap, warm-start native margin L2, zero-
and warm-start native CE, and warm-start random and observer-chart CE.
The analytical CE solver uses damped Fisher scoring in the legal
contrast chart, with eight Jacobian evaluations and four projected
line-search trials per step. L2 uses the same budget and Gauss-Newton
curvature. Every solver retains its start if no candidate lowers its
own objective. Actual physical dose can differ within the common cap.

For amortization, train identically sized one-hidden-layer controllers
(width 64) with zero output layer initialization. Inputs are the reached
recipient h, raw swap, donor-minus-anchor legal margins, and the dose
cap, standardized with TRAIN statistics only. The output is a bounded
coefficient correction to the projected raw swap:

    c0 = Q^T d_raw
    c = project_ball(c0 + R tanh(f_theta(x)), R)
    d = Q c.

Use an orthonormal Q. Compare learned native CE, learned native L2,
learned random CE and learned observer CE. The random chart is seeded.
The observer chart is the top eigenspace of the average TRAIN margin
Gramian; it is a common fixed chart, not the per-case SVD of MOLT-14.
All controllers start from the same parameter seed and share the
minibatch index stream. Freeze teacher and native factors; optimize
controller parameters only, using differentiable actual LayerNorm and
tied unembedding. Report training steps, parameter count, time, and
loss endpoints. The analytical solver and controller are separate
amortization comparators with different one-time fitting costs, not
matched training-FLOP arms.

## Fixed development roster

Generate 64 TRAIN roots each at depths 4 and 8, and 16 DEV roots each
at the same depths, excluding prior root rosters and their donor edits.
Ensure disjoint exact world tuples across both new groups, including
both donor variants. Each root has a changed-target addition edit and
a compensated same-target edit, preserving the future operand word.
This is a source-world split, not an exclusion of individual primitive
transitions from the already trained teacher's corpus.

Fit four controllers for 128 updates, batch 32, seed 15715. Evaluate
every arm through the actual dense teacher with an additive physical
write and verify dose and margin parity. Keep both anchor and donor
competence, complete suffix and terminal accuracy, same-target
preservation, teacher winner match, CE, margin error and dose. No
filtering to successful interventions. The primary descriptive compare
is changed-target complete suffix recovery against raw swap and
warm-start native analytical CE, with same-target preservation and
work as guardrails. Equal-rank random and observer controls share the
same source inputs and training budget. No independent confirmation
or claim of native specificity based solely on this small panel.

No protected readout; one fixed teacher and 32 development source
roots with paired edits, not independent logits or 64 independent
worlds. Controller gains do not establish teacher training gains.
All changes remain development-only. The existing child-free CPU
supervisor enforces address-space and CPU time caps and samples RSS.
It does not certify a hard process-tree physical RAM limit.

    python -m pytest -q tests/test_molt_amortized_repair.py
    python -m research_gym.interp.molt_amortized_repair --clock /tmp/molt15/clock \
      --factors /tmp/molt15/bpd --bpd-checkout .native-bpd \
      --bpd-source-receipt /tmp/molt15/bpd-source.json --out /tmp/molt15/repair
