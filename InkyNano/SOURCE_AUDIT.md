# INK-NANO-00: pinned source audit (2026-11-24)

Checkpoint: thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7.
Reference implementation: huggingface/transformers@d8f10a82213a3a9db48385aba3a4b57ab535de50.

Sources:
- https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7
- https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/config.json
- https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/model.safetensors.index.json
- https://github.com/huggingface/transformers/blob/d8f10a82213a3a9db48385aba3a4b57ab535de50/src/transformers/conversion_mapping.py
- https://github.com/huggingface/transformers/blob/d8f10a82213a3a9db48385aba3a4b57ab535de50/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py

## Verified by source inspection

The main weight file is approximately 19.5 GB; the separately listed MTP file is 636 MB. Physical layers 0–23; loop core [6,18) runs three passes. d=2304, m=1152, E=128, top-k=8. Native routed scale is 8; native bias affects ranking, but selected weights are gathered from the unbiased sigmoid scores and normalized before scale. Keep both physical and visit indices.

The layer-0 and layer-1 checkpoint index exposes model.llm.layers.{l}.mlp.experts.w13_weight and w2_weight plus scale, amax and outer_scale_denominator entries (presence across all layers must be checked mechanically). The native conversion_mapping.py defines Inkling2LoopDequantizeExperts followed by Interleave(dim=1) for w13 and dequantization for w2. Native tensors are gate_up_proj [128,2304,2304] and down_proj [128,2304,1152]. Split gate_up_proj along its output dimension only AFTER the upstream conversion. The native gate/up softcaps use c*tanh(x/c), with c=4 and c=25 respectively. The sigmoid acts on the raw gate, in FP32. Native expert accumulation is FP32 with output cast back to input dtype.

This refines the generic SwiGLU equation in SPEC.md. The exact reference is implemented in native_expert.py. The source audit does not establish byte-level parity or measured quantization quality.

## Executable metadata pass

Run inventory.py --remote --output inventory.json for the pinned metadata/index and bounded safetensors header requests. The program refuses a full-file response to a range request; it checks complete key coverage, expected layer IDs, byte ranges, per-tensor shapes and dtypes, and records source hashes. If range delivery fails, download the checkpoint explicitly in Silico and run inventory.py --local /path/to/snapshot. Do not allocate GPU time until this metadata pass succeeds. All output artifacts belong outside Git unless small and reviewed.

## Native numerical gate (pending)

1. Install Transformers at the exact commit above in an isolated Silico environment. Record torch/CUDA versions and upstream conversion implementation.
2. Use pinned conversion mapping to extract each selected expert with correct scales and Interleave; compare extracted gate/up/down with the corresponding loaded upstream parameter slice. Confirm scale grouping and original quantization formats using header/quant config. For the pilot only, if using full upstream loading, account for full BF16 expert bank (~49 GB) plus backbone and working allocations; use CPU RAM or an adequately provisioned GPU. Do not attempt full load on RTX 3050.
3. Run native_expert.audit_native(module, expert_id, inputs) against the loaded upstream expert module with native dtype and deterministic FP32-generated input probes, plus real development activations when available. Check whole-expert parity with maximum relative L2 and absolute tolerances calibrated on repeat execution. Use a single expert weight of one to isolate expert math; separately test multi-expert weighted accumulation. Never certify numerical parity from synthetic matrices.
4. Feed the native dequantized, rearranged matrices to spectral.nested_svd. Record exact and randomized residuals. The actual pilot remains blocked until native dequantization parity is recorded.

## Additional deployment control
At r=128, INT4 factors for all experts cost ~2.038 GB decimal. The nonexpert backbone could dominate the 4 GiB cap, particularly separate embeddings and output head; use inventory byte counts to derive text-only backbone footprint and verify actual tying. Test full compressed bank residency as a control, since it may eliminate transfers at sufficiently low rank.
