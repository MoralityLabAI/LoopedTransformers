# InkyNano Full Expert GGUF Conversion (Google Colab CPU)

All **3,072 physical experts** (`24 layers x 128 experts`) of `thinkingmachines/Inkling-2-Mini` and a separately quantized **text-only backbone** have been converted from native **NVFP4** (layers 0–19) and **MXFP8** (layers 20–23) into individually addressable **GGML-quantized GGUF v3 tensor bundles** on a standard 2-vCPU Google Colab CPU runtime (`0` GPU hours, `0` Silico credits, `0` local RTX 5080 compute), and uploaded incrementally to private Hugging Face repository `AlephFunk/InkyNano-GGUF`.

> **Important distinction — Standalone Tensor Bundles vs Complete `llama.cpp` Model:**
> Every `.gguf` file sets `general.type = "tensors"` and `inkynano.llama_cpp_dropin_compatible = false`.
> These artifacts use **actual upstream GGML C quantization routines** (`ggml_quantize_chunk` and `ggml_get_type_traits->to_float` from `ggml-org/llama.cpp@ae6a7ad2c5754dd8db93d04c84fda70435864f6e`) inside standard GGUF v3 containers verified with `llama-gguf`, **not** a complete drop-in `llama.cpp` (`llama-cli`) model. Upstream `llama.cpp` does not implement Inkling-2-Loop's 3-pass recurrent core (`[6, 18)`), block attention residuals, recurrent message controllers, short convolutions, or soft-capped SwiGLU (`gate_cap=4.0`, `up_cap=25.0`, sigmoid on raw gate, FP32 accumulation). Use [gguf_loader.py](gguf_loader.py) for selective per-expert byte-offset loading and LDT-controlled residency on RTX 3050.

## Canonical Artifacts & Copyable URIs

- **Private Hugging Face repo:** `https://huggingface.co/AlephFunk/InkyNano-GGUF` (revision `23106ca8de1982b3de48efc9a333b52938b205e4`)
- **Master manifest (all 3,072 experts + 646 backbone tensors, 5,428,656 B, SHA-256 `0601e90943abc2320e9ea8dc9ac205fc62fe4b60df26dbe66a8708472433004e`):**
  `https://huggingface.co/AlephFunk/InkyNano-GGUF/blob/23106ca8de1982b3de48efc9a333b52938b205e4/manifest.json`
- **Model card:**
  `https://huggingface.co/AlephFunk/InkyNano-GGUF/blob/23106ca8de1982b3de48efc9a333b52938b205e4/README.md`
- **Text-only backbone (`1.080 GiB`, SHA-256 `09a22f8d3caa8489612b27348d9800ed7f333cf980b81c40cb1668d988900f58`):**
  `https://huggingface.co/AlephFunk/InkyNano-GGUF/blob/23106ca8de1982b3de48efc9a333b52938b205e4/backbone/inkynano_backbone_text.gguf`
- **24 independently downloadable layer shards (`14.892 GiB` total):**
  `https://huggingface.co/AlephFunk/InkyNano-GGUF/blob/23106ca8de1982b3de48efc9a333b52938b205e4/shards/inkynano_experts_layer_{00..23}.gguf`
- **Resumable Colab notebook:** [inkynano_colab_gguf.ipynb](inkynano_colab_gguf.ipynb) (`https://colab.research.google.com/github/MoralityLabAI/LoopedTransformers/blob/research/inky-nano-prep/InkyNano/inkynano_colab_gguf.ipynb`)
- **Committed numerical receipts:**
  - [benchmark_16_experts.json](gguf_receipts/benchmark_16_experts.json)
  - [conversion_summary.json](gguf_receipts/conversion_summary.json)
  - [backbone_receipt.json](gguf_receipts/backbone_receipt.json)
  - [manifest.json.gz](gguf_receipts/manifest.json.gz) (complete 3,072-expert manifest archive, 605 KB gzipped)

## Provenance & Native Dequantization Parity

- Source checkpoint: `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7` (Apache-2.0, `19,526,019,168` B).
- Native dequantization: reuses the verified `SilicoExperiments` PR #3 (`e43c01156e82585402f24183d30516c346754444`) path calling `transformers@d8f10a82213a3a9db48385aba3a4b57ab535de50` `Inkling2LoopDequantizeExperts` + `Interleave(dim=1)` with `loader_cast=True`.
- Cross-checked `transformers` vs standalone mirror on all 4 stratified layers (`L00/E111`, `L06/E066`, `L17/E024` NVFP4; `L23/E089` MXFP8): **`w13 max_abs = 0.0`, `w2 max_abs = 0.0`**.
- GGML C backend: `ggml-org/llama.cpp@ae6a7ad2c5754dd8db93d04c84fda70435864f6e`, `libggml-base.so` SHA-256 `37315ebe78d5ffe172fbe0eed8366bf0a84a530fc136088d9482f45d06e942dc`.

## 16-Expert Pilot Benchmark (Pure `Q4_K` vs Mixed `Q4_K`/`Q6_K`)

Measured on Google Colab CPU (`2 vCPUs`, `12.67 GiB RAM`, `CUDA=False`) over the exact 16-expert stratified roster from `SilicoExperiments` PR #3 (`layers 0, 6, 17, 23 x 4`, `seed=17`). Note that `down_proj` has `ne[0] = 1152 = 36 * 32` (`1152 % 256 = 128 != 0`), so canonical `llama.cpp` `tensor_type_fallback` (`src/llama-quant.cpp`) maps `Q4_K -> Q5_0` and `Q6_K -> Q8_0` without zero-padding; both unpadded fallbacks and zero-padded `1152 -> 1280` K-quants were benchmarked:

| Profile | NVFP4 (L0–19) | MXFP8 (L20–23) | Full Bank (GiB) | SwiGLU Unit-RMS L2 | SwiGLU Heavy-Tail L2 | 2-Expert Mix L2 | Projected 3,072 CPU min |
|---|---|---|---:|---:|---:|---:|---:|
| `pure_q4_k` | `Q4_K/Q4_K/Q5_0` | `Q4_K/Q4_K/Q5_0` | 13.764 | 0.10583 | 0.10610 | 0.10514 | 29.56 |
| `pure_q4_k_q4_0` | `Q4_K/Q4_K/Q4_0` | `Q4_K/Q4_K/Q4_0` | 12.814 | 0.13069 | 0.13073 | 0.12997 | 29.72 |
| `pure_q4_k_padded` | `Q4_K/Q4_K/Q4_K` | `Q4_K/Q4_K/Q4_K` | 13.289 | 0.12025 | 0.12042 | 0.11953 | 40.97 |
| **`mixed_q4_k_q6_k`** | **`Q4_K/Q4_K/Q5_0`** | **`Q6_K/Q6_K/Q8_0`** | **14.891** | **0.08517** | **0.08524** | **0.08431** | **27.90** |
| `mixed_q4_k_q6_k_down_q8` | `Q4_K/Q4_K/Q8_0` | `Q6_K/Q6_K/Q8_0` | 17.264 | 0.07883 | 0.07885 | 0.07830 | 28.02 |
| `mixed_q4_k_q6_k_padded` | `Q4_K/Q4_K/Q6_K` | `Q6_K/Q6_K/Q6_K` | 16.117 | 0.08121 | 0.08119 | 0.08071 | 32.89 |

- **Profile selection:** `mixed_q4_k_q6_k` cuts nonlinear soft-capped SwiGLU output rel-L2 by **19.5%** vs `pure_q4_k` (`0.08517` vs `0.10583`) for only **+8.2%** bank size (`14.891` vs `13.764 GiB`), preserves exact unpadded `[2304, 1152]` shapes, and beats the `SilicoExperiments` PR #3 rank-256 SVD pilot (`0.8876` FP16 / `0.8946` INT4) by **>10×**.
- **CPU vs GPU profiling gate:** On local Colab SSD, `dequant_time_share = 27.3%` and `projected_3072_cpu_minutes = 27.90 min` -> **`gpu_recommended = false`** (GGML K-quantization executes natively on CPU SIMD threads; zero GPU required).

## Full Conversion, Memory Bound & Backbone Precision Receipts

- **Full expert bank (24 shards, 3,072 experts, 9,216 tensors):** `15,989,637,120` B (`14.892 GiB`), `1,898.71 s` (`31.65 min`) compute time across all 24 layers:
  - NVFP4 layers 0–19 (`Q4_K/Q4_K/Q5_0`, `4,810,752` B/expert, `587.3 MiB`/shard): mean soft-capped SwiGLU rel-L2 **`0.10411`** (range `0.09935`–`0.10757`).
  - MXFP8 layers 20–23 (`Q6_K/Q6_K/Q8_0`, `7,174,656` B/expert, `875.8 MiB`/shard): mean soft-capped SwiGLU rel-L2 **`0.02548`** (range `0.02513`–`0.02561`).
- **Text-only backbone (`backbone_mixed_q4_k_q6_k`, 646 tensors, vision/MTP/experts excluded):** `1,159,361,184` B (`1.080 GiB`), `149.69 s` compute time:
  - **500 sensitive tensors preserved verbatim at native BF16/F32 (`78.08 MB`, exact `rel_l2 = 0.000000`):** 48 router tensors (`mlp.gate.*`, `14.17 MB`), 146 recurrent loop controllers & cross-pass queries (`controllers.*`, `middle_cycle_cross_pass*`, `57.07 MB`), 48 block attention residuals (`attn_res.*`, `0.52 MB`), 104 short convolutions (`*sconv*`, `2.46 MB`), and 154 norms/L2-clip/sink/relative-logit tensors (`3.87 MB`).
  - **146 quantized large matrices (`1,081.23 MB`):** `unembed` (`Q6_K`, rel-L2 `0.01784`), `attn.wo_ud` (`Q6_K`, mean `0.01830`), `attn.{wk,wv,wr}` (`Q6_K`, mean `0.01920`), `embed` (`Q4_K`, `0.07172`), `attn.{wq,gate_attn}` (`Q4_K`, mean `0.07286`).
- **Bounded memory:** Peak process RSS stayed at **`1.791 GiB`** (`1,923,190,784` B) across all 24 expert shards and the striped `[201024, 2304]` embedding/unembed matrices without ever materializing the ~49 GB dequantized model.
- **Resume & selective loader verification (Cell 8):**
  - Re-running `run_all_resumable` skipped **24/24 shards** and **1/1 backbone** via remote HF SHA-256 verification (`0` re-converted).
  - `SelectiveExpertStore` + `LDTResidencyManager` spot-checked on downloaded `layer_00.gguf`: admitted 8 experts (`38,486,016` B) via single contiguous offset reads with SHA-256 verification and finite soft-capped SwiGLU mixture output (`norm = 41.7669`).

## RTX 3050 (4 GiB) LDT Residency Fit & Local Usage

| Resident slots / layer (`k`) | Resident experts | Expert payload | Backbone + experts | Headroom before KV/activations/CUDA |
|---:|---:|---:|---:|---:|
| 8 | 192 | `0.931 GiB` | `2.011 GiB` | **`1.989 GiB`** |
| 16 | 384 | `1.861 GiB` | `2.941 GiB` | **`1.059 GiB`** |
| 32 | 768 | `3.722 GiB` | `4.802 GiB` | `-0.802 GiB` (exceeds 4 GiB) |

Every expert's 3 tensors (`ffn_gate`, `ffn_up`, `ffn_down`) are stored contiguously with zero inter-tensor padding (`uniform_expert_span_bytes == uniform_expert_payload_bytes`: `4,810,752` B for L0–19, `7,174,656` B for L20–23), enabling single-`pread` selective admission driven by `InkyNano/expert_lattice.py` and `InkyNano/stage1.py --expert-bytes`.
