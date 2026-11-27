# MOLT-13 measured carry-quotient results

[CPU run 45920579894](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45920579894)
passed at source head 2bcf3f623967e9341dabd05e6483c15032441512.
Both focused tests passed. The addressed teacher competence and native
BPD engineering fit replayed under the unchanged limits. The fit has
16 components, 8 steps, clean parity zero, relative weight residual
0.447408617, final reconstruction KL 0.0002686254.

Sixteen fresh development worlds, 8 each at affine depths 4 and 8:
all anchor and donor suffixes correct, additive fitted-factor patch
correct on 8/16 complete suffixes and 8/16 terminal readouts.
At a fixed absolute numerical threshold 1e-8, every rank tuple
(future observer, future observer restricted to native span, observer
including the cut) was (16,16,63). The native span has dimension 16.
Maximum independent Jacobian factorization error 5.05374e-17,
maximum suffix replay error 4.44089e-15. The carry phase elapsed
5.87363 s and worker kernel lifetime RSS was 497,627,136 bytes.
The exact suffix quotient does not preserve the current unrestricted
readout. The result does not establish donor-specific superiority or
a reduction below the native span dimension: it is full rank in that span.

The replay uses additive write displacement from fitted native factors,
not a new native swap API parity test. All comparisons are development
only at one frozen checkpoint; no confirmation, statistical equivalence,
or promotion. No hard process-tree physical RAM guarantee.

Next: use the legal-carry boundary to target finite donor recovery within
a declared native write span and fixed dose, with an equally optimized
random-span control. The observed full rank does not ensure good
conditioning or global reachability of the donor carry. The existing
teacher is saturated on ordinary task competence; a learning-effect
study needs a distinct prespecified headroom condition.
