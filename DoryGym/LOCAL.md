# Dory 3050 local runtime

## Source and memory

Use NVIDIA's official [GGUF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF/blob/main/README.md). Official artifact: dory-q5_k_m.gguf, 3,673,288,864 bytes, SHA256 96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0. Pin this hash; if the published file changes stop and investigate.

With 6 GB available host RAM, avoid multiple downloads/builds/runtimes simultaneously. Keep >=1.5 GB host headroom and >=512 MiB free GPU memory (measure, including desktop). Stop if OS paging pressure or CUDA OOM occurs. Build using 2 parallel jobs at most initially. Full offload recommended in official examples may not fit 4 GiB. Do not assume partial offload works until checked with this exact backend.

## Dory-enabled llama.cpp

Model card pins brianchmielNV/llama.cpp revision 98a74c97ea09ac72f6ceda1559808fc0a2f447ce. Stock binary may lack Dory support. Use a native Windows build with Visual Studio C++ tools, CMake, NVIDIA CUDA toolkit and Git, or WSL Linux CUDA with compatible driver. Follow the official pinned build instructions; Windows build is unverified by this gym.

```sh
git init llama.cpp-dory
cd llama.cpp-dory
git remote add origin https://github.com/brianchmielNV/llama.cpp.git
git fetch --depth 1 origin 98a74c97ea09ac72f6ceda1559808fc0a2f447ce
git checkout --detach FETCH_HEAD
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 2
```

Download the file once with the HF CLI or browser; keep it outside Git. Windows PowerShell: Get-FileHash -Algorithm SHA256 path/to/dory-q5_k_m.gguf. Linux: sha256sum dory-q5_k_m.gguf. Record file bytes, hash, build commit, compiler/CUDA versions, startup logs and nvidia-smi.

Launch with one loop and short context, e.g. PowerShell single line (adjust executable location):

```powershell
.\build\bin\Release\llama-server.exe -m .\dory-q5_k_m.gguf --alias dory --host 127.0.0.1 --port 8080 -ngl 16 --ctx-size 512 --parallel 1 --batch-size 64 --ubatch-size 32 --cache-type-k f16 --cache-type-v f16 --flash-attn off --no-context-shift --jinja --reasoning on --override-kv "dory.num_loops=int:1,dory.recurrent_kv_cache_mode=str:per_loop" 2>&1 | Tee-Object server1.log
```

The command is a candidate configuration, not a proven fit. Check logs for actual offload, expanded loop graph, context allocation and cache mode; monitor physical VRAM, host commit and working set. If the backend fails on partial offload, try CPU-only with short context or stop with the exact blocker. Tune -ngl conservatively; keep the same validated setting across compared depths where feasible. Changing offload is a separate factor. Server is localhost only; no public port. For a separate run, terminate the first process, change loop flag to 2 or 4 and use a fresh server log. For the cache study change only per_loop to last_loop.

From the LoopedTransformers directory:
```sh
python DoryGym/make_roster.py --out roster.jsonl
python DoryGym/bench.py --roster roster.jsonl --loops 1 --cache per_loop --server-log /path/to/server1.log --out results/1-per_loop.jsonl --max-tokens 128
```

Use the same max token cap across depths. A reasoning model can exhaust this small cap; log truncations separately and, if feasible, register a larger-context/longer-cap experiment. The script requires a unique output file, localhost URL and startup log hash. It cannot itself prove server settings or monitor system-wide resources; attach actual startup evidence and sampled resource measurements to every run. Use a fresh server after each depth/cache change. No remote API calls. No full model training at this stage.
