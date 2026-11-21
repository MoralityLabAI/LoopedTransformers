# CTX21: BF16 reached the 8K block

At **2026-11-21 12:42:07 UTC**, the authenticated BF16 worker was still live
and its append-only journal recorded all **312 registered 4K contexts closed**
plus its **first closed 8K-budget context**. The first 8K case is a CAL case.
The [metadata receipt](nan_ctx21_bf16_8k_observation.json) binds the closed
case, worker manifest, source seal and journal prefix through that completion.
Only execution metadata was extracted; this observation adds no model call
or grade and does not read live atomic progress files.

This is a concrete capacity milestone, **not a terminal correctness audit**
or an all-shape capacity claim. BF16 remains subject to the original 85%
CUDA allocation fraction, four resident decoder layers plus reusable stage,
GPU head, one CPU thread, Below Normal priority and four-hour backend budget.
No loading policy, precision, prompt or frozen source was changed.
NF4 keeps all 22 quantized decoder layers on the GPU. The capacity contrast
therefore includes different decoder placement and does not isolate precision.

The separate [terminal NF4 audit](NAN_CTX21_PROGRESS.md) retains all 312
completed 4K contexts and the CUDA OOM on its first 8K forward. Its 312
unobserved 8K outcomes remain unknown. BF16's completed forward does not
supply NF4 outcomes or establish BF16/NF4 correctness or geometry equivalence.
Later BF16 contexts may still fail. Final accuracy, unchanged-target
verification, paired recurrence geometry and allocator peaks require the
registered terminal audit and report.

The automatic queues continue unchanged. The full storyworld trace and
batched clean/intervention A/B conditions remain frozen and queued; this
milestone is not their result. The continuous goal remains active and the
original 168-case ARG-03 test stays sealed.
