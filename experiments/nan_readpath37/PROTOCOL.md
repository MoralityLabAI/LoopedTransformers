# READPATH37 archived direct-readout pilot

Run a bounded CPU calculation on the audited NF4 and BF16 COMP14 pointer
states and the target-matched IO18 T2 layer-21 banks. The pilot retains every
focused, copy, and focused-erased context, both worlds, both passes, and all
eight fitted write directions. Use 8 CAL roots and 16 DEV roots per precision.
The task readout is the pinned bare one-token label versus the other world's
label for the same root. Compare with the independent archived unrestricted
native first-token grade, which is a separate endpoint.

Verify both immutable audits, archive hashes, snapshot revision and head
shard, and factor-bank hashes. Stream the head shard digest in one MiB
blocks, then seek only selected head rows and the final RMSNorm weight.
Replay the original FP32 RMS, BF16 cast, BF16 scale multiplication, and
BF16-rounded head output at the stored operator aliases and native top token.
The maximum tolerated absolute discrepancy is 0.125 logit. Audit every phase
and retain failed closures. Raw states, logits and per-case data stay local.

At the last-token layer-21 residual boundary, add and subtract unit vectors
at 1% and 2% of the original residual L2, and round the reached residual
to BF16. Compare all eight BPD writes, eight fixed orthonormal controls, and
coordinates 2278, 2279 and 2280. Orient each learned write by the sign of its
native fitted read coefficient. The same physical control vectors repeat
across worlds and layers of this comparison. Report observed two-sided
margin response per L2 relative to the original state and keep its sign.
Require each side to match the requested L2 within 3%. Report valid counts
at each dose and never replace an unqualified effect with a zero.
The recorded shift is at the output lens only, with the archived state, final
norm and head fixed. It does not replay a modified decoder or KV state.

## PercInterp direct-path control

For each retained state, within one physical pass build the three-stage
graph source → eight BPD occurrences → target-versus-foil margin.
The magnitude of the first edge is the unmasked component write L2
`abs(x @ V_i) * norm(U_i)`. Rank the second by the magnitude of the signed
2%-dose central response; the signed values remain local. The first-edge
input is gauge invariant under reciprocal component scaling. Zero edges
and dose failures are absent. Exclude incomplete eight-edge conditions from
the complete-graph count and report the failures separately.

Use PercInterp's within-group rank-entry rule with ties entering together;
here the groups are the two edge stages. Compare the forward/backward
bottleneck nodes against the kept-in/kept-out local degree rule at every
breakpoint. In this graph each component's parents are sources and its
children are sinks, so the memberships must be exactly equal. This is a
negative control for the selector, using real direct readout data. A
nontrivial test requires measured cross-pass edges and matched budgets on
fresh roots.

Report episode and root denominators, requested and reached dose coverage,
CAL/DEV and phase separately. The screen is exposed retrospective evidence
on one pinned model, using two quantization conditions. A matched sham
threshold for delayed correctness and a fresh causal confirmation set
remain separate gates. Zero formal tests and zero new neural forwards.
The original 168-case test stays sealed. Use one Below Normal CPU process,
5 GiB available at admission, 4 GiB reserve, 0.75 GiB private cap and
600 seconds. Stop only the audit process when a budget is exceeded.

[PercInterp rank and bottleneck definitions](https://github.com/MoralityLabAI/percolation-interp/blob/282184519f7bbd05a81ae76c3faa5d655945ee04/src/percinterp/measure.py) ·
[COMP14 protocol](../nan_comp14/PROTOCOL.md)
