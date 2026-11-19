# LGA-10E first implementation freeze (prospective 61068)

This file is included in the immutable implementation commit whose direct
parent is the locked protocol at
\`8de00d66080cc3550efaf962da9387603a2d2eed\`.

Before seed 61068 opens, the source runner verifies that its exact checkout
SHA matches the frozen implementation, that the direct parent is the protocol
commit, and that all imported source/protocol/test/workflow files exactly
match the committed bytes. It runs only on the **first** Actions push attempt
of \`research/loopscope13e-61068-native-fp32-20261119\`. PRs, manual workflow
dispatch and any rerun execute only the tests.

This study compares a frozen baseline, an FP32-HVP candidate filter layered
on FP64 geometry, and an explicit mixed-precision donor/sham oracle-assisted
search. Native FP32 refers exclusively to the **execution** endpoint, with
the original \`native_precision_check\` and all nine physical applied edits.
No FP32-only *construction* claim is made for the two new methods.

The registered primary acceptance is >=61/64 for \`mixed_donor_repair\`,
where a valid case has one source-preserving donor within 0.20 original dose
distance, eight complete shams within the 7×32×8 search budget, and the
frozen 1e-8/LGA-10C physical/native numerical gates. Failures always remain
in the original 64-case denominator. One untrained seed, eight independent
input bases and eight correlated goals per base. Source seed 61068 is reserved
until the implementation commit is frozen. Science seeds remain closed.

The result parser records original and repaired source HVPs, used native
precision oracles, physically realized edits, selection rejections, resource
measurements, and the independent NumPy gate reconstruction. Any observed
engineering threshold failure remains scientifically reportable.
