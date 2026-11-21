# Native NF4/BF16 gradient preflight: completed results

Both backends passed all eight native-checkpoint gradient cases and independent terminal audits. Each used both worlds of the first two COMP14 CAL roots in source order, at depths1/2. No optimizer steps occurred and no factors were trained. This establishes the registered differentiation/resource prerequisite for FIT16, not optimizer feasibility or trained VPD/BPD.

| Backend | Depth1 passed | Depth2 passed | CUDA allocated peak GiB | CUDA reserved peak GiB | Added saved-tensor CPU peak GiB |
|---|---:|---:|---:|---:|---:|
| bf16 | 4/4 | 4/4 | 2.575499 | 2.691406 | 0.158185 |
| nf4 | 4/4 | 4/4 | 2.758388 | 2.974609 | 0.158185 |

All-one full-vocabulary logits and normalized boundaries were bit-exact against clean native execution. V/U gradients were finite and strictly nonzero under the fixed component=.35/residual=.7 masks. The host audit independently recomputed full-vocabulary trajectory KL from archived logits, verified gradient shapes/norms/finiteness, archive/prefix hashes and saved-tensor limits. It did not replay a second full-model backward or finite-difference gradient. The target parameters and factor values remained unchanged.

NF4 retained actual native NF4 kernels,22 GPU decoder layers and the unchanged BF16 head; BF16 used four resident layers plus staging and the same head. CPU embeddings preserve context capacity. The added CPU saved-tensor peak is not total process RAM; allocator peaks are not driver residency or WDDM paging measurements. One numerical CPU thread and Below Normal priority protect the separate math process. No package/weight changes or paid compute.

The four prefixes arise from two reused CAL roots; depths, worlds and gradients are correlated engineering measurements. Fixed masks are unmatched-dose probes, with no calibrated causal floor/sham/power and zero formal comparisons. No mechanism specificity, backend equivalence, donor transfer or original failed-gate repair is established. The168-case test stayed sealed.

Freeze commit `0616c3dac53dd5de96ac8d2cd9bf70b83a558e3e`, source seal `61f37ab12d17ab1d3569b9a6ebeff59e25deac1cc9c8f00e6240e43d4cfe23c5`, preceded all GRAD15 inference. This presentation uses only the immutable terminal aggregate at `f71e5d514696bd7f067353792722d28fbe8f6003`, SHA256 `51344c72d7146a6239d92004e35038e3add4fc3dac1649f0dd7171dd0e18ddd3`, with no new inference, grading, fitting or analysis of raw runs.

The continuous goal remains active. [FIT16](../../experiments/nan_fit16/PROTOCOL.md) is separately frozen actual factor/controller fitting with matched BF16 controls; its outcomes remain separate.

[Immutable aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/f71e5d514696bd7f067353792722d28fbe8f6003/reports/nanbeige_bpd_gym/nan_grad15_progress.json) · [Frozen protocol](../../experiments/nan_grad15/PROTOCOL.md) · [Complete preflight stream](NAN_GRAD15_PROGRESS.md) · [Actual fitting stream](NAN_FIT16_PROGRESS.md)
