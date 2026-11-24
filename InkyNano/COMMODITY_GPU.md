# Commodity GPU handoff: InkyNano

Use ordinary SSH GPU rentals for deterministic teacher capture, weight extraction and compression pilots. Preserve Silico for later post-training, including InkyNano and DataGoblin. No cloud account credentials, paid job or instance is created by this PR.

## Measured pilot and gates

Silico's native teacher peak allocation was ~54.7 GB decimal (51 GiB). Use one A100 80 GB (PCIe or SXM), H100 80 GB or equivalent for the existing full dequantized loader; do not choose a 48 GB card for this path. Choose >=64 GB host RAM (96 GB preferred), >=80 GB free persistent disk, CUDA-compatible PyTorch image and good download throughput. A 24 GB 3090 is appropriate for streamed single-expert factor experiments after an independent loader is verified; a 48 GB card is an intermediate target for offload. No assumption of full-model fit.

The pilot found ~0.888 relative nonlinear L2 error at rank 256 FP16 and ~0.895 at INT4, from one independent dev activation root. Do not run the full naive SVD atlas. Use native packed weights for residency and preserve high precision routing/recurrent controls; activation-weighted refinement is a separate experiment. Generated H0 top-8/16/32 oracle mass was 0.451/0.641/0.838, so prefer k=16–32 and quantify transfers. Neither trace mass nor reconstruction error establishes task quality.

## Step 1: rental

Use the provider console to rent one A100 80 GB, with SSH and persistent scratch. Indicative rate ~$0.80–1.19/h; set initial spend cap $5 (including storage and transfer if possible), account-level expiry/termination if available, and choose the shortest practical test. A process timeout does not stop provider billing. Stop/delete the instance explicitly after checking output transfers. Use public/synthetic rosters only on untrusted marketplace hosts. Keep private HF write credentials on the trusted workstation; pull artifacts via SSH/SCP and publish locally. Do not expose credentials to rental shell history.

## Step 2: prepare exact runtime

```bash
git clone --filter=blob:none --single-branch -b research/inky-nano-prep https://github.com/MoralityLabAI/LoopedTransformers.git
cd LoopedTransformers
bash InkyNano/commodity_setup.sh /workspace/inky
```

Requires a working compatible CUDA torch install, Python, git and pip. Uses a system-site-packages venv to preserve the provider's CUDA torch. Records the exact upstream Transformers commit and source snapshot. The setup script downloads the main shard once; source download/installation may dominate elapsed time. No access token is necessary for the public source checkpoint. Validate source inventory before launching.

## Step 3: run sharded native routing

Prepare a frozen public/synthetic JSONL roster, 128 train and 64 dev independent roots across source families for the eventual development study. Start with one shard of 16 roots; include at least one dev root per shard for the native prefix check. Family/root IDs must be globally unique across shards and disjoint across train/dev. Include root, family, split, source_kind, source, category, prompt, continuation. Use actual multi-step tasks and longer generated continuations. Cap at 512 generated tokens initially; record EOS vs token cap, actual token counts and per-root completion status. A capped continuation is not a complete turn. No sealed test data.

```bash
bash InkyNano/commodity_capture.sh /workspace/inky /workspace/rosters/shard00.jsonl 16 512 1800 shard00
```

Run the next shard only after verifying the first one's elapsed seconds, native prefix parity, all 48 visits/token, hashes and actual cost. Budget by decoded tokens/s plus teacher load and transfer; the previous H100's ~347 s/16-root 160-token-cap run does not determine A100 cost for a 512-token cap. Sharding repeats teacher load but bounds RAM and preserves completed runs. CPU analysis may require substantial RAM; start with 16 roots max and check RSS. Do not combine partially completed arms.

## Step 4: local handoff

Copy /workspace/inky/results/shard00/analysis and its manifest via SSH to a trusted machine, verify with gym_manifest.py, then terminate the rental. Use merge_demand.py to consolidate verified shards for a specific arm/horizon without losing root boundaries. Train on the 3050 or CPU.

```bash
python InkyNano/merge_demand.py --arm generated --horizon 64 --out demand_H64.npz shard00/analysis shard01/analysis
python InkyNano/gym_train.py --data demand_H64.npz --out local/mesh64 --device cuda --resident 16 --max-swaps 4
```

This trains only the small residency controller; native restricted inference and activation-aware compression still need their own tests. Prompt/full-teacher history inputs remain an offline proxy. Run the native packed/offload and restricted-router controls before declaring a 3050 fit. Preserve all data receipts and the private pilot atlas separately.

## Provenance

Cached online capture and analysis adapted from the companion InkyNano pilot code (revision e43c01156e82585402f24183d30516c346754444). Source model remains Apache-2.0. This rental wrapper and sharded workflow are untested on a rental until the first run. No full-bank SVD expansion is authorized.
