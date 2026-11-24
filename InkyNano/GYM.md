# INK-NANO-GYM: Silico → RTX 3050

Status: development implementation, pending native export. This work order is deliberately split: the cloud runs the teacher and factorizes weights; local training uses compact demand targets. The 3050 does not need to load the teacher for controller training.

## Cloud job

Checkout the pinned PR branch commit, use the teacher and Transformers revisions in SOURCE_AUDIT.md, record versions and hardware. Start with one GPU, cap the pilot at 45 GPU minutes and report actual cost; stop sooner on inventory/parity failure. Full-bank allocation requires pilot receipts and a separately approved estimate. Use only public/synthetic task roots and retain license/provenance. Install pinned Transformers, torch, safetensors, accelerate, huggingface_hub, and use a compatible CUDA environment. Native eager dequantized load requires substantial host RAM or a GPU with enough capacity for the ~49 GB expert bank plus backbone, logits, activation and allocator overhead; measure before expanding. Do not use device_map auto if it breaks per-module hooks or parity.

1. Run inventory.py against the pinned checkpoint. Confirm numerical extraction parity using native conversion and SOURCE_AUDIT.md. Set up the teacher through its native loader, text-only path where supported, eager experts. Run gym_export.py on a supplied JSONL roster of public prompts and continuations. For each independent root, provide root, family, split (train/dev), prompt and continuation. The roster must be grouped and split by independent family, with no shared families across splits. Include code, math/Lean, dialogue and tool planning; start 16 roots for smoke, then target 128 train and 64 dev independent roots. Confirm teacher behavior separately from reference continuation behavior; forced continuations are a demand-data proxy. Keep natural generated continuations as a separate roster/arm.
2. Capture all native visits, including native selected weights (normalize when building demand arrays), prompt length and tokens. Export causal prefix/segment demand samples for whole-turn and 32/64/128-token horizons. Check prefix invariance on a development root before using batched traces; stop if routing at earlier tokens depends on a later suffix. No future continuation IDs or full-segment demand enter controller features. Export exact routing mass oracle ceilings by horizon/capacity. Do not open a sealed split.
3. Run the 16-expert spectral pilot using native expert parameter slices. Record nonlinear expert errors and projection residuals at nested ranks 32/64/96/128/192/256. Export pilot factors as safetensors, native source IDs and rank metrics. After approval, use --atlas to export one shard per physical layer for all 128 experts; default export rank 128, FP16 or INT4 block-packed factors as selected by pilot. FP16 rank 128 full bank is 8.154 GB decimal; INT4 ideal 2.038 GB plus scales and padding. Never upload unbounded duplicate ranks. Record actual byte count and source revision. For INT4 use the supplied reference packing with 64-element blocks, signed symmetric codes and FP16 scales; this is a storage format, not a fast GPU kernel.
4. Seal the handoff with SHA-256 manifest, include receipts, native parity data, cost, source license, source revision and root roster hash. Publish to the user's selected Hugging Face namespace as a research dataset or model repository with model card; keep private until license and data provenance are reviewed. Do not put weights in Git. Download only routing artifacts to the 3050 for post-training; defer atlas download until local inference.

## Handoff contract

- inventory.json: validated checkpoint metadata.
- roster.jsonl: public root metadata and token IDs, prompt length, source family and train/dev split.
- trace.jsonl: native top-8 visits indexed by root, turn, absolute token, layer and pass, weights plus raw native weights.
- demand_H{0,32,64,128}.npz: arrays x[N,24,256] (prompt and completed-prefix demand), y[N,24,128] (normalized future segment demand), visit_count[N,24], root/family/split/start/end. H0 is whole continuation; H>0 uses consecutive windows. No teacher embeddings required for v1.
- spectral/layer_00.safetensors, layer_06, layer_17, layer_23, pilot_metrics.json and atlas_index.json for the pilot; 24 layer shards and complete=true for the full atlas, plus factor format, shape and rank.
- manifest.json: file paths, byte counts and SHA256; metadata identifies pinned source, roster hash and exporter commit; native parity/cost/teacher correctness receipts supplied separately. Publish manifest last.

## Local training (no paid GPU required)

Use [Stage 1](STAGE1.md) for the full controller comparison, per-root receipts,
archive verification, and sharded reruns. The commands below run the original
single-horizon trainer.

```sh
python InkyNano/gym_data.py --roster export/roster.jsonl --trace export/trace.jsonl --out export
python InkyNano/gym_train.py --data export/demand_H64.npz --out local/mesh64 --device cuda --resident 8 --max-swaps 2
python InkyNano/gym_train.py --data export/demand_H0.npz --out local/turn --device cuda --resident 8 --max-swaps 0
```

Train small predictor on train roots only. The explicit LDT state is a product of per-layer candidate-set lattices (powersets), with hard membership/byte/slot constraints. Learned demand is soft evidence. At each boundary, choose a feasible cache R' by scoring residents and additions and enforcing k slots and at most b admissions, using exact uniform-cost constrained maximization. Reset the lattice at a new allocation epoch, since swapping is not a monotone deduction over an old epoch. Never exclude an expert permanently because of model evidence. Within an epoch a loaded set is fixed; native restricted routing handles mixture. Four six-layer mesh blocks exchange pooled summaries; compare independent local predictor and classical global/prompt/history rules. This is a first LDT-inspired controller, not a reproduction of all existing LDT architectures.

Training uses cross entropy against normalized future demand. Evaluate sequentially on dev roots under equal admission caps, cold-start costs and actual expert bytes. Report mass per root, transfer bytes, oracle headroom and checkpoint parameter count; the included trainer uses unit admission costs (convert with manifest bytes for uniform packed layouts). The learned controller's future demand predictions do not need to be exact top-8 labels. Prompt-demand features assume a full-bank prompt prefill; account for its time and transfers separately. Later windows in this offline gym consume past teacher traces. Deployment must replace them with the student's available router observations, and test distribution shift. A cheap token-only prefill predictor is a separate control. Avoid claiming quality until on-policy compressed runtime executes restricted routing. Next stage: local runtime adapter, actual task scores and measured PCIe transfer.

## Local acceptance

Verify manifest hashes, run CPU tests, then benchmark GPU controller training and actual VRAM. For the deployment claim, run full compressed bank and restricted native teacher controls as in REGISTRATION.md. Use EvalMath preflight on independent dev roots before freezing confirmation. Do not treat this gym's observational proxy as a causal task result.
