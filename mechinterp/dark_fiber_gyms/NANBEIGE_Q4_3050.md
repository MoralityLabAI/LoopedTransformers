# Nanbeige 4-bit on RTX 3050 (4 GiB): ARG-03 experimental backend

**Status:** A two-case local NF4 pilot and a fresh matched BF16 development reference have completed and been graded. See the [2026-11-17 pilot report](../../papers/shared_weights_dark_fibers/pilot_nf4_bf16_20261117/README.md) for accuracy, recurrence-response comparisons, memory measurements, and provenance. NF4 peak CUDA allocation was 3.84 GiB, reserved memory was 4.38 GiB, and Windows reported shared GPU allocations; a strictly 4 GiB device-only fit was not established. Both backends had zero current-readout preservation error but differed in program rankings and response trajectories.

This remains a separate **quantized development-only** experiment and does not replace the registered BF16 ARG-03 evidence. The two-case accuracy floor does not establish equivalence. The 168-case held-out test remains sealed.

The official Nanbeige model card documents Q4_K_M in its llama.cpp fork. GGUF Q4_K_M is about 2.5-2.6 GB on disk, which is a useful alternative for ordinary language-model inference. The Dark Fibers gym needs access to *individual recurrent hidden tensors* to perform causal interventions; a normal llama.cpp completion endpoint does not expose the exact ARG-03 state API. The new adapter uses CUDA bitsandbytes NF4 with quantized decoder linear layers, BF16 normalization/head, and CPU token embeddings.

## Dependencies

Use the pinned original ARG-02 Nanbeige snapshot with configuration, tokenizer, original safetensors and the 4.46.3 remote source code. The two development-only split paths come from the earlier pre-unseal generator. Install bitsandbytes in the **Nanbeige 4.46.3 virtual environment**, keeping the Qwen/Transformers 5.x venv separate. The NVIDIA driver and CUDA build must support the local Windows GPU and NF4.

~~~powershell
$GPU = "$HOME/.venvs/bvg-v03/Scripts/python.exe"
& $GPU -m pip install --upgrade "bitsandbytes>=0.48,<0.51"
& $GPU -c "import torch,bitsandbytes as b; print(torch.__version__,b.__version__,torch.cuda.get_device_name(0),torch.cuda.mem_get_info())"
~~~

Free VRAM of at least 3.2 GiB is checked before loading. Other active GPU work may leave insufficient room even on the 4 GiB device. The adapter quantizes one CPU-resident original BF16 decoder layer at a time onto CUDA, leaving CPU embeddings and a BF16 GPU LM head for full-vocabulary tests. This consumes substantial **host RAM** during conversion and the initial full-vocabulary head remains sizeable. If real CUDA OOM occurs, reduce competing workloads; the fallback BF16 layer-streaming backend and GGUF inference have different memory/performance contracts.

## GPU pilot on fresh final-R development (2 cases)

~~~powershell
$C = "C:/path/to/Compactification-Theory"
$R = "runs/dark-fibers/arg03/finalr_dev_20261117"
$env:ARG02_SNAPSHOT_NANBEIGE = "C:/path/to/models--Nanbeige--Nanbeige4.2-3B/snapshots/451ed48c3273ecef7ea8faaa43c31ce529763bb1"

& $GPU -m mechinterp.dark_fiber_gyms.arg03_nanbeige `
  --compactification-root $C `
  --public-jsonl "$R/roster/development_finalR_valid_public.jsonl" `
  --out-dir "$R/q4_nf4_valid" `
  --device cuda:0 --q4-3050 --q4-microbatch 1 --limit 2
~~~

After successful checkpoint load and pilot, increase to `--limit 0` for all 48 fresh development validation cases, using a new output directory to avoid mixing the partial tape's chosen_count. Repeat for the development FIT split. Preserve source-hashed model-only tapes and grade only in a separate EvalMath process using the existing `score_final_r_future` interface.

## New per-depth time-local causal control

~~~powershell
& $GPU -m mechinterp.dark_fiber_gyms.arg03_preunseal.run_tlocal_dev `
  --compactification-root $C `
  --public-jsonl "$R/roster/development_finalR_valid_public.jsonl" `
  --out-dir "$R/q4_tlocal_valid" `
  --device cuda:0 --q4-3050 --q4-microbatch 1 --limit 2
~~~

The time-local runner verifies matched full-vocabulary readout and same-depth L2 dose at recurrent depths 2/4/8. It scores the next loop's 64-program ranking and independently grades it later with the existing `score_tlocal_future` grader. The original BF16 frozen protocol stays byte-for-byte unchanged.

## Scientific gates

1. Quantized **clean-vs-quantized patched** comparisons are the primary within-backend causal test. Start with a small validation subset shared with BF16 development receipts and report differences in native clean accuracy, 64-program rank, current readout error, and per-step perturbation responses.
2. Record the original pinned checkpoint SHA, quantizer family `bnb-nf4`, 4-bit double-quantization, microbatch size, full-vocab head dtype, GPU peak allocation and elapsed time in receipts.
3. Audit whether the CUDA 4-bit backend maintains current-logit equality under context-only interventions. A nonzero current-readout discrepancy above 1e-6 is a failed gate.
4. Label future@12 outcomes as **quantized development-only**. Any confirmation using the 168-case BF16 preregistered test requires a new, separately authorized quantization-specific protocol.

For ordinary GGUF inference, the original model describes `Q4_K_M` conversion and the Nanbeige llama.cpp fork: https://huggingface.co/Nanbeige/Nanbeige4.2-3B .
