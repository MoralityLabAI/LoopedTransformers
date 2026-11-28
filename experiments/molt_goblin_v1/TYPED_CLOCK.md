# MOLT-12: typed primitive-operation clock

Continuation of the MOLT-11 failed teacher competence gate. The original
AUX-V2 LOOP-T teacher obtained 7.35% validation exact after 256 updates,
versus a 5.88% constant. The following is a distinct structured teacher
interface, not a reported gain on the original input format.

For an affine recurrence x <- a*x+b mod 17, expose the legal state register
and public stage operand, using the word (multiply.add)^L. The host generates
the sequence and exact labels. The neural stage sees the previous *predicted*
17-way probability vector and the current operation token. It does not see
the prior gold state, future operands, or modular arithmetic at inference.
The first state is the public initial value.

Both arms reuse the actual native pre-norm shared attention/MLP block,
tied embedding and final normalization, on three positions per stage.
The same physical block is called at each primitive visit. The recurrent
carry is the legal-state softmax, differentiable across visits. The full
vocabulary loss and unrestricted predictions are also scored to prevent
a legal-mask-only competence claim. In the separable arm the three
positions are a mixture of state embeddings, an operand embedding and a
fixed query embedding. The addressed arm adds a probability-weighted
mixture of learned operation/state-address embeddings to the query.
Address rows are part of the same tied embedding table in both arms.
This is a strong explicit inductive bias and finite primitive lookup
curriculum, not discovery of modular arithmetic from unstructured text.

Train on all 561 legal primitive input pairs, sampled with replacement,
then 64 fixed-depth closed-loop continuation updates. Training sequences
are 512 distinct depth-four worlds. Same initialization, batch stream,
optimizer and forward stage-example budget across arms; address gather
cost and actual FLOPs can differ. Fixed 512 primitive updates, batch 64,
rollout batch 16, seeds 157 and 163. The held-out development sequences
are fresh exact world tuples (depths 1,2,4,8,12, 128 per depth), excluding
the training tuples. All primitive transitions are in the training
corpus; this is compositional sequence transfer, not novel-operation
generalization. Shared model/world roots are crossed.

Before native factor fitting, require >=98% unrestricted primitive
accuracy, >=90% unrestricted terminal exact at depth 4 and >=80% at
depth 8. Retain complete traces, unrestricted outputs and all failed
arms. No data-dependent retries or selection of a favorable seed. The
native BPD follow-up uses the predeclared addressed seed 157 only if it
passes. BPD will fit the actual decoder.blocks.0.mlp.2 shared site;
competence does not establish factor fidelity or a causal mechanism.
This is a development screen with new thresholds, not confirmation.

    python -m pytest -q tests/test_molt_typed_clock.py
    python -m research_gym.interp.molt_typed_clock --out /tmp/molt12/clock

Use the existing MOLT-11 per-process limit wrapper; no paid accelerator,
protected split or promotion. The original MOLT-11 memory parity is
unchanged. The next question is whether useful response memory improves
ordinary evaluation of a competent teacher, with an ordinary task-only
and reinvested control.
