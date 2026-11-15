# PEF-05: protecting a training-only old-history direction

The complete protocol is in PRE_REGISTRATION.md. Its first commit
754aa207554340cc8beecd1038c29f6663fdcdfb predates the experiment.

The prior PEF-04 experiment missed its registered oldest-bit recovery
criterion. Its retrospective optimizer replay saw substantial losses
of linear decodability near updates 125–150 in seeds 101, 127 and 157,
while seed 109 was a strong-retention control.

This new four-seed experiment tests the causal effect of first-order
gradient protection on a train-only old-history separation direction.

All three arms within a seed start from the same weights, optimizer
moments and supervised batch stream at the end of update 125. During
updates 126–150, the protect arm removes the component of the
clipped task gradient that is predicted to decrease a fixed old-bit
separation score. The sham arm removes a *norm-matched* random
component. Baseline uses the unmodified task gradient. After update
150 every arm resumes vanilla updates through 250.

The protection score is computed only on source **training pairs**.
A seed-specific anchor is fixed using all 80 paired training suffixes
at step 125. The first 16 training suffix pairs supply the
differentiable score during intervention. Neither the downstream
probe split nor heldout split contributes to gradient surgery.

A new salt assigns the 128 seven-bit suffix pairs 80/24/24 across
training, probe fitting and heldout. The score uses naturally
generated recurrent states after eight original bits and two extra
input bits (0,1), a new full ten-bit continuation configuration.

All three checkpoints are written and SHA256-sealed before any
downstream head is fitted. The new probe head uses 48 probe-only
histories; heldout contains 48 histories constituting 24 complete
oldest-bit counterfactual pairs.

The source-only predictor has exactly 24/48 correct oldest-bit
responses and zero fully correct oldest-bit pairs. Three previously
weak seeds are the primary test; the previously strong seed 109
is a noninferiority control. See PRE_REGISTRATION.md for the
frozen numerical success criterion.

The original PEF-04 result and optimizer traces are retained.
This experiment can establish intervention-associated changes in
a tiny tied-attention recurrent model. The first-order gradient
projection does not guarantee a monotone old-bit score under
finite AdamW updates or restore an older bit for every seed.

Reproduction:

    python -m pip install pytest
    python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pytest research/pef05/test_pef05.py -q
    python -m research.pef05.pef05 --out /tmp/pef05_registered

All binary checkpoints and audited receipts are collected by the
PEF-05 GitHub Actions workflow. No GPU or paid compute is required.
