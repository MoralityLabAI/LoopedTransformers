# Silico work order: spectral atlas

Status: pending. Start with one GPU and a short bounded pilot; require a cost/runtime estimate and approval before expanding. No credentials in repository, logs or manifests. Do not upload private trace text.

## 00: inventory and exact reconstruction
Resolve and record HF revision for thinkingmachines/Inkling-2-Mini and Transformers/runtime commit; optionally repeat later for the separate varloop research checkpoint. Inventory actual tensors, quantization metadata, fused gate/up layout, orientation, expert order, all non-expert tensors and vision/MTP dependencies. Do not hard-code guessed tensor keys. Read the index and safetensors header, resolve quant scales through upstream code, and stream a layer/expert at a time. Verify a reconstructed expert against upstream for seeded inputs; record max absolute/relative L2 error, dtype and tolerance established from native repeated runs. Stop on ambiguous packing or failed parity. Source shard is roughly 20 GB, and dequantizing all weights at once may exceed GPU/RAM capacity.

## 01A: bounded pilot
Sample physical layers 0, 6, 17, 23 and four expert IDs per layer selected by a fixed seed before inspecting spectra. Cover both source quant formats. Compute nested ranks 32/64/96/128/192/256 (512 only if justified). Record exact full-SVD reference on at least one expert per layer; compare randomized rank-256 with fixed seed and residual check. Use FP32 decomposition; evaluate BF16 and candidate quantized factors separately. Save singular energy, Frobenius residual, randomized error versus exact, activation-weighted projection error and whole SwiGLU expert error on captured development inputs. Pilot quality and cost determine the next run. No claims of 4-bit runtime until actual packing and supported kernels exist.

## 01B: atlas
If pilot accepted, process experts by physical layer with resumable per-layer shards. Prefer nested rank 256 or lower as chosen by pilot; never create three copies for the recurrent passes. Save nonexpert tensors unchanged with source references and explicit license. Store per-matrix U*S and Vh, singular spectrum separately, orientation, factor dtype, quant scales, actual bytes, source shard/hash, extractor version, seed, error metrics, SHA256, and completed expert IDs. Use atomic writes and verify hashes after resuming. Record decomposition wall time, peak memory, transfer volume, cloud GPU seconds and total cost. Do not push weights to Git.

Proposed HF repo: user-selected namespace / InkyNano-SpectralAtlas. Initial upload should be a compact pilot shard and model card only, after reviewing license and cost. Pin the exact derivative revision when consuming it. Separate raw higher-rank research factors from packed deployment factors. Avoid unbounded uploads: source checkpoint plus staging and derivative storage must fit allocated disk.

## 02: trace capture
Instrument the pinned native router after expert selection; preserve post-loop index and actual layer/pass. Export per root/turn/token/layer/pass selected IDs and pre/post-normalization weights. Capture prompt and generation phases separately; split roots before training. Store provenance, tokenization revision, sampling seeds and teacher scores; omit private prompts. Capture activation samples only in the development split, and retain their provenance. Use the strict JSONL analysis contract in routing.py; native parity and coverage precede traces.

## Deliverables
Inventory manifest; per-format reconstruction receipt; pilot table and cost projection; hash-addressed atlas manifest; teacher-trace adapter; sampled trace validation; and an explicit GO/REDESIGN decision for the next GPU allocation. Use local CPU synthetic tests first.
