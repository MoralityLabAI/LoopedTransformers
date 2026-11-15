# PEF-05R | Cross-run numerical stability audit

**Status: posthoc forensic comparison**, selected after two successful PEF-05
CPU GitHub Actions runs returned slightly different results. The
underlying registered PEF-05 decision remains *criterion_not_met*.

Original hosted training runs:
- [43687506414](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43687506414), artifact ID **14150782670**.
- [43688425647](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43688425647), artifact ID **14151262267**.

GitHub file `research/pef05/pef05.py` had the same content blob SHA
`87c414464a2d2e0a73f6329d9dbdc43ce8561832` on the two input run
commits. The input branch's other changes concerned audit sources and
documentation.

Preliminary independent container audit verified that initial tensor
hashes, the pair-closed split and supervised batch index digests match,
while the step125 common checkpoint tensor hashes already differ.
The relative L2 drift in common step125 parameters ranges from
approximately **5.7e-6** to **2.4e-3** across the four seeds and grows
during further training. The original decision remains negative in both
runs.

## Frozen audit protocol

Retrieve exactly the two original GitHub artifacts through Actions
without training a model. Check source and checkpoint SHA256, pair split
manifest, initial RNG/parameter digests, per-seed optimizer fork
identity and supervised minibatch-index stream hashes. Compare all
common, step150 and step250 state tensors numerically and inspect the
optimizer moment tensors; log relative L2, maximum absolute parameter
change, and cosines at each stage.

Compare every per-arm final heldout oldest-bit and original-task count,
active gradient intervention step, and signed-margin receipt.
Attribute no causal mechanism solely from identical seeds.

A separate **same-host deterministic training smoke** repeats a
125-step unmodified prefix twice on the same CPU worker with identical
source and RNG inputs, recording model/optimizer hashes at steps
0,1,10,50,125. This smoke is an investigation of implementation-level
repeatability, not a new PEF-05 registered trial or a replacement
checkpoint. The original two-run comparison is the primary result.

### Conclusions permitted

A differing step125 tensor digest with an identical initial tensor
digest and batch stream demonstrates that the two executions did
not produce bitwise identical optimizer trajectories. Source-level
equality and different GitHub runner CPU implementations are plausible
contributors; the audit cannot infer the exact cause without hardware
and per-step traces. A large later parameter difference is an
observed sensitivity; its growth is consistent with perturbation
amplification and not proof of a positive Lyapunov exponent or chaos.

The original PEF-05 score and negative decision must be preserved
and linked to each **exact archived checkpoint SHA256**. This campaign
does not select between the two original executions as a winner.
