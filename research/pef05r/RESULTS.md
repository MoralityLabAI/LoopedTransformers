# PEF-05R results | The hidden CPU-math variable

**Audit protocol:** AUDIT_PROTOCOL.md, committed before new audit scripts.
The comparison uses exactly two previously completed and archived PEF-05
registered runs. Both preserve the original negative memory-rescue verdict.

Original runs:
- [A: 43687506414](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43687506414), archived artifact 14150782670.
- [B: 43688425647](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43688425647), archived artifact 14151262267.

The original intervention-source file was identical in A and B,
Git blob SHA 87c414464a2d2e0a73f6329d9dbdc43ce8561832.
The PEF-05 pair-closed source manifest, four exact original
parameter-initialization hashes, and all supervised minibatch
sample-stream hashes were identical. The first archived common
optimizer checkpoint at **step 125** already had different weight
and AdamW moment tensors, before any gradient surgery began.

## 1. Quantified amplification of execution-path discrepancies

**Independent archived-checkpoint audit:** [GitHub CI 43689354646](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689354646),
artifact ID 14151777311.

| Seed | Relative weight difference at step125 | Relative difference at baseline step250 | Later/earlier drift ratio |
| --- | ---: | ---: | ---: |
| 101 | 0.000006541 | 0.0234192 | 3580.1 |
| 109 | 0.000056968 | 0.0384680 | 675.3 |
| 127 | 0.00243486 | 0.0486755 | 20.0 |
| 157 | 0.000005734 | 0.0162730 | 2838.0 |

This describes **two particular execution paths**. The drift
ratios are neither measured universal Lipschitz constants nor
estimates of a positive Lyapunov exponent.

Heldout accuracy differed across nominally identical-seed runs:
- seed109 baseline oldbit **44/48 versus 40/48**;
- seed127 baseline **44/48 versus 46/48**;
- seed157 baseline **40/48 in both**;
- seed101 baseline **48/48 in both**.

Protection and sham also varied within some seeds. Both complete
registered runs reported **0/3 vulnerable rescue seeds**, leaving
the original negative conclusion unchanged.

## 2. Direct processor-dispatch crossover on one AVX-512 machine

The same local AVX-512 CPU, Python/Torch 2.10.0+cpu, training
implementation, single-thread setting, source split and minibatch
sequence was used to reconstruct each original model through the
unchanged first 125 updates.

**Only DNNL_MAX_CPU_ISA was switched between independent processes,
before importing PyTorch:**

| oneDNN maximum ISA | seed101 step125 | seed109 | seed127 | seed157 | Source matched |
| --- | --- | --- | --- | --- | --- |
| AVX512_CORE | 1338478d2ef7 | 4eb37523ce14 | 329168605519 | 72105788516b | Run A, all four |
| AVX2 | e45b05e1777c | 568c1b9561d7 | 1f0889a98a5a | 5aa31d6aa696 | Run B, all four |

These strings are the leading 12 hex digits of the **exact full
64-digit checkpoint parameter SHA256 values**; the complete
reference table is in isa_replay.py.

The CPU was unchanged, and PyTorch continued reporting
ATen CPU capability AVX512 in both configurations. The native
oneDNN AVX2 selection is therefore sufficient to switch this
processor's optimizer history between the exact two original
archived step125 parameter states.

On the same AVX-512 processor,
ATEN_CPU_CAPABILITY=avx2 **without** DNNL_MAX_CPU_ISA=AVX2 still
reproduced the AVX512_CORE checkpoints. Setting only
DNNL_MAX_CPU_ISA=AVX2 reproduced the AVX2 originals.
This isolates a oneDNN dispatch-sensitive computation
on the controlled processor and model.

**Hosted independent support:** [ISA replay CI 43689914382](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689914382)
ran on an AVX-512-capable GitHub worker and, with
DNNL_MAX_CPU_ISA=AVX2, reproduced all four original B
**model and AdamW optimizer** SHA256 values exactly. It
also reproduced the A references when forcing AVX512_CORE.
The original source blob SHA was checked before execution.


A later [AVX512-capable hosted ISA crossover run 43690202840](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43690202840)
confirmed the two exact model **and optimizer** reference paths
again on an **AMD EPYC 9V74 80-Core Processor**. The host reported
AVX512 capability in both subprocesses. With DNNL_MAX_CPU_ISA=AVX2
it reproduced original run B; changing the same variable to
AVX512_CORE reproduced original run A.

## 3. Cross-host caveat and additional observed execution path

A *different* hosted worker in [CI 43689910254](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689910254)
was given the same DNNL_MAX_CPU_ISA=AVX2 setting but produced a
**third** set of parameter and optimizer SHA256 values. Its
initial model hashes matched the original reference, and the
divergence began during the source training. The worker's
processor details were absent from that first failure receipt,
so the exact processor or other CPU kernel contributing to
the third trajectory remains unverified.

This restricts the portability claim: pinning the oneDNN
ISA is demonstrably causal on the tested AVX512 processor.
It is **not sufficient evidence of bitwise identical training
across every GitHub-hosted CPU**.

Subsequent isa_replay.py revisions therefore preserve an
additional_cpu_execution_path record with the actual model
and optimizer SHA256, CPU capability, processor model,
Torch version and oneDNN maximum-ISA settings. An unmatched
host is always labeled as such. The pipeline's analysis
passes when it has recorded and audited a genuine alternative
path, with an optional strict mode that fails on any mismatch.

## 4. Repeatability on one hosted runner

[Same-runner CI 43689354646](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689354646)
repeated unmodified 125-step source training **twice**
for seeds101 and127 on the same AVX2-capable worker.
The exact model *and optimizer* SHA256 were bitwise identical
at steps 0,1,10,50,125 within each seed. Both step125 states
also matched the original archived B run.

That distinguishes stable within-worker execution from
cross-worker numerical-path variation in this specific
experiment. It does not establish universal determinism
outside the tested environment.

## 5. Consequences for future PEF training

Any next prospective training study should freeze the full
execution provenance before evaluation: model and source
code commits, CPU architecture/model and ISA dispatch,
oneDNN maximum ISA, PyTorch build, single-thread and
deterministic settings, optimizer state, seeded batch
sequence, and source split manifest.

For multi-machine bitwise replay, either require a common
tested CPU backend or explicitly accept numeric tolerance
and treat each checkpoint SHA as a separate scored
execution. To distinguish intrinsic optimizer sensitivity
from hardware dispatch, compare one treatment's naturally
evolved checkpoints from the *same pinned common prefix*.

The discrete scalar bound in
[MathCan PEF-05R PR #51](https://github.com/patrickdugan/MathCan/pull/51)
formalizes how bounded local update discrepancies can
accumulate under a Lipschitz update-map assumption. It
does not assert the learned optimizer satisfies any
particular Lipschitz constant.

This is a **posthoc numerical causality and reproducibility
audit**. The original PEF-05 negative intervention result,
registered success threshold, and all source artifacts
remain unchanged.
