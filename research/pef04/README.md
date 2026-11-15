# PEF-04 | Prospective independent-seed old-history transfer

See the immutable [preregistration](PRE_REGISTRATION.md), committed before
the first PEF-04 training. This campaign tests whether a new downstream
head can recover an oldest input bit after two additional, **natural**
iterations of independently trained frozen recurrent attention models.

## Pair-closed split

The 128 possible seven-bit *recent suffixes* are ordered by the frozen
SHA256 rule. Both counterfactual members \`(0,suffix)\` and
\`(1,suffix)\` share a split. There are 80/24/24 suffix pairs allocated
to trunk training, downstream head fitting, and heldout evaluation,
respectively. The heldout cohort is 48 histories forming 24 exact
counterfactual pairs. Both members remain entirely unseen by the
trunk and probe-fitting stages.

Any decoder using only the seven-bit suffix has exactly **24/48**
oldest-bit accuracy and **0/24** pairs correct on both members.
That includes the supervised last-two-bit task label.

## Model and transfer

For six **new** seeds 101,109,127,131,149,157 the original PEF-03
attention trunk is trained for 250 fixed AdamW steps on four-way
last-two-bit prediction, then frozen with SHA256 receipts. All six
checkpoints are frozen before any downstream probe or heldout
evaluation.

After eight visits, run the same frozen recurrent block for two
additional visits with identical continuation bits for both
counterfactual members. Primary continuation is \`00\`; \`11\`
and direct h8 readout are fixed secondary diagnostics.

A ridge head trained on the independent 48 probe-fitting histories
then predicts the oldest original bit from h10. No gradients from
this head are propagated into the frozen recurrent model.

For each seed, report heldout correct histories (out of 48), pairs
whose two oldest-bit predictions are both correct (out of 24),
paired prediction flips, causal memory L2 distances, original
last-two-bit supervised-task accuracy and an identically fitted
random-initialized attention-trunk baseline.

The preregistered criterion requires four of six new models with
oldest-bit accuracy ≥36/48 **and** pairwise both-correct ≥12/24;
four or more supervised-trunk models must also obtain ≥41/48
on the original task. The output is recorded even on failure.

## Limits

A fitted readout is a **new downstream task**, while the frozen trunk
performs a natural history-to-state computation. This does not
demonstrate spontaneous task routing by the old trained task head.
We do not use intermediate off-manifold row/nullspace transplantation
in the registered scoring. A random-initialized control tests whether
oldest-bit decoding is a generic property of the architecture.

The small synthetic attention model and six seeds provide a bounded
reproducibility campaign. The exact affine PEF-01/02 formulas are
calibrations and do not automatically hold for learned attention.

Run the registered default experiment:

    python -m research.pef04.pef04 --out /tmp/pef04

Run the isolated smoke tests:

    python -m pytest -q research/pef04/test_pef04.py

Dependencies are the pinned CPU PyTorch 2.10.0 and pytest in hosted
CI. Checkpoint bytes and sealed receipts are stored as workflow
artifacts; no binary weights are committed.
