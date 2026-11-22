# INK-ROUTE-01 coordinate-2027 routing pilot

The pilot targets the fixed three-pass Thinking Machines checkpoint `thinkingmachines/Inkling-2-Mini` at `b85c3cfa615fc413cc9af3379f491b963f3b586a`. All eight prompts are public development prompts. Goodfire measured its writer on 13,233 FineWeb tokens from 400 documents and reported an average RMS-normalized coordinate of +32.4, about 46% of squared norm. This pilot uses the last token of each registered completion prompt. [Goodfire analysis](https://www.goodfire.com/research/interpreting-inkling-2-mini/) · [Pinned checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a)

## Native hooks

The writer is the output of Hugging Face layer 17 on each of its three visits. The shared `message_input` module runs 24 times. Its layer-12 calls are zero-based visits 6 and 18. In Transformers 5.24.0 the reader passes one completed block-2 state in `block_states[:, :, 0]` and block 3 as the separate `partial_block_states` argument. Its output has Q/K/V/MoE order. The probe verifies the clean writer matches the following pass's actual block-3 input. It recovers the two-source coefficient from the native output and RMS-normalized inputs, requires relative reconstruction error below `1e-3`, and checks it against the softmax of the captured query dot products. A coordinate's score contribution is `query[2027] * RMSNorm(bank)[2027] / sqrt(2304)`.

One prefill batch per root pairs all intervention lanes with one clean lane. Each edit touches only the target lane's final-token block-3 vector for one reader invocation. A private copy supplies the edited argument, and the following call checks identity and exact full-bank equality on the original argument. The five edits are zero, deterministic equal-L2 randomized-coordinate sham, norm-preserving zero, norm-preserving coordinate swap, and zero followed by replacement with the clean reader output. Receipts capture realized BF16 doses, pre-rescue routing, delivered messages, next-token log probabilities and top-token flips. The rescue requires the delivered message to equal the clean message exactly and the reference-token log probability to agree within `1e-4`. An ablation below that log-probability resolution is labeled as a null for that readout.

## Reproduce locally

Run from the repository root with Python 3.12. Install PyTorch for the target device, then install the runner dependencies. The optional Transformers smoke test uses the installed reader with synthetic weights and inputs.

```text
python -m pip install transformers==5.24.0 'accelerate>=1.10,<2' 'numpy<3' psutil
python -m unittest discover -s research_gym/inkling_routing_01 -p 'test_*.py' -v
python -m research_gym.inkling_routing_01.local --limit 2
hf download thinkingmachines/Inkling-2-Mini config.json tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json model.safetensors.index.json model-00001-of-00001.safetensors --revision b85c3cfa615fc413cc9af3379f491b963f3b586a
```

Pass the printed local snapshot path after `--snapshot`. Use a fresh `--out` for each run. Start with two roots and check all six arms, both passes, reconstruction, restoration, rescue, and resource receipts before running eight.

```text
python -m research_gym.inkling_routing_01.local --run --snapshot PATH_TO_SNAPSHOT --limit 2 --device cpu --threads 2 --out runs/ink_route_01_dev2
python -m research_gym.inkling_routing_01.local --run --snapshot PATH_TO_SNAPSHOT --limit 8 --device cpu --threads 2 --out runs/ink_route_01_dev8
```

`--device cuda:0 --cuda-budget-gib 5` selects the RTX path when it has sufficient free VRAM and compute capacity. CPU is the default for a shared GPU host. The loader checks the checkpoint configuration and tokenizer before assigning weights, verifies the full 19.5 GB weight shard by SHA-256, and uses the Transformers text decoder, attention and logits code through `Inkling2LoopForCausalLM`. Transformers 5.24.0 otherwise expands the checkpoint's mixed NVFP4/MXFP8 experts at load time. The packed backend keeps the original expert tensors memory-mapped on disk and invokes Transformers' Inkling-specific weight converter on selected experts in bounded chunks. Expert computation uses BF16 weights and FP32 accumulation, as in the eager Transformers implementation. The original pretrained checkpoint provides all text weights. This path uses software dequantization and skips the unused vision tower.

The output retains the plan hash, checkpoint and software identities, paired JSONL, per-root activation archives, resource and timing samples, failure JSONL and artifact hashes. Keep each attempt in its own directory. Empty `failures.jsonl` means the runner recorded no failures in that attempt.

## FSI-01 evidence gate

The source prompt is the experimental unit. Arms, channels and passes are correlated observations within that root. The primary contrast is the zero-versus-equal-L2-sham change in block-3 routing, with a direction toward lower routing after target removal. The dose is the realized final-token displacement at the previous-pass block-3 bank. Reference-token log probability and output rescue are secondary diagnostics. The exploratory 0.10 beta change has no calibrated matched-sham floor or inferential interpretation. Report all controls and nulls. The eight prompts have already been exposed during development and have no sealed readout. The pilot does not qualify for confirmation under FSI-01. Confirmation would require development-only sham calibration, a frozen minimally meaningful effect and comparison family, adequate independent roots, and a fresh sealed readout. [EvalMath mechanistic-resolution preflight](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) · [FSI-01](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md)
