# Prompt-fixed expert residency

The matrix compares Goose's unrestricted Inky Q4 results with global, LDT and TRM prompt-fixed residency over the **already isolated experts**. Each restricted run selects one set per physical layer from the available prompt, assembles a compact Q4 GGUF from the shared backbone and the selected packed rows, and eagerly loads it into RAM. Loop passes share physical weights. Native top-8 routing still weights experts inside the resident set. With eight slots it can change mixture weights only. The runtime records selection, packing, loading, prefill, decode and total elapsed time.

The LDT uses per-layer lexical evidence and pooled messages across six-layer blocks. The TRM repeatedly updates high and low states with a shared update block. The two models have identical parameter counts, inputs and readouts. They predict prefill and decode demand separately, with equal weight at selection time. Only train-root traces enter fitting. Retained teacher routing mass is an offline diagnostic. Paired exact grid scores and storyworld verifier results measure the effect of residency on outputs.

`prompt_matrix.py --global-prior` starts both controllers from train-root global frequency and learns a bounded prompt-dependent correction to the prior logits. Save these fits in a separate directory to distinguish them from the unconditioned initialization experiment. Replaying their plans automatically loads the associated prior from each checkpoint.

For a fit on the RTX 3050, run `prompt_matrix.py --device cuda` in an existing CUDA-enabled PyTorch environment. The training pool stays in host memory and the fitter sends one minibatch at a time to the GPU. The run records the training device and peak CUDA allocation, then scores and saves the controllers on CPU so the native benchmark replays the same expert selections.

## Data and comparison

- `matrix_cases.py` generates 9×9 Sudoku roots with unique solutions and samples the official ARC-AGI-2 **public training** split. Its development partition uses disjoint roots from that public training split. `--compact-grids` represents each row with its original single-digit cells and no separators. This can keep complete cases within the context budget. Keep the public evaluation set unopened until the experiment is frozen. Score exact Sudoku grids and ARC pass@2 for each test grid, with success requiring all of a task's test grids. The separate `exact_grid_match` score also accepts a complete compact grid in Sudoku and single-test ARC cases. [Official ARC-AGI-2 dataset](https://github.com/arcprize/ARC-AGI-2/tree/f3283f727488ad98fe575ea6a5ac981e4a188e49)
- Outfill continues the provided prefix. Infill supplies a prefix and a suffix and asks for only the missing text. The current runner uses text-based infilling. Dependent modes share one root.
- Scoring accepts the plain answer or its native `<|content_text|>` frame. Finished native thinking messages are excluded from scoring. The receipt retains the raw completion.
- Import Goose storyworld cases as JSONL with `id`, `root`, `task: "storyworld"`, `split`, `mode`, `prefix`, `suffix`, `max_tokens` and `label`. Use `rendered_prompt` when the exact formatted Goose request is available, and enable `--parse-special` if it uses tokenizer markers. Storyworld labels may supply `required_keys` for schema checks. Use the same original benchmark verifier on native outputs when its inputs and scoring artifacts are available.
- `matrix_cases.py --hilbert-storyworld` builds fresh single-encounter authoring cases from the published [verified world](https://github.com/MoralityLabAI/HybridTRMLDT/blob/1307e7f9a38eb1afd63fb585b0cc80e297ddf1d9/MCP/results/mcp05/inky-hilbert-16k/storyworld.hilbert.json). These cases check scene identity, prose presence, choice count, and valid operators, clues and advance rules. They score the requested prose length separately. The published Goose run records multi-stage authoring and browser verification separately.
- Preserve model digest, formatting, sampler and token limits for replay. The historical Goose adapter accepts per-request rows with case/root/task/mode, prompt and model SHA-256, token counts, and available TPS and score fields. Matched comparisons require identical prompt and model digests. Exact-prompt rows with different model digests appear in `same_prompt_goose` as cross-system comparisons. Device, runtime, cache state and sampling can also differ.
- Pass `matrix_compare.py --goose-receipt` with the published [MCP-05 Storyworld receipt](https://github.com/MoralityLabAI/HybridTRMLDT/blob/1307e7f9a38eb1afd63fb585b0cc80e297ddf1d9/MCP/results/mcp05/inky-hilbert-storyworld-canvas-receipt.json) to report its aggregate throughput, cache share, long-context planning outcomes, and verified browser replay alongside the local matrix. Per-request matching uses separate prompt-level receipts.
- Compare LDT and TRM with global selection using the same prompt, bank, runtime and resident capacity. The paired report includes per-request deltas and equal-root means, with the number of roots contributing to each metric. Goose remains a separate historical comparison where a matching run is available.

## Native runtime

The runner is built against [llama.cpp a6ac41c](https://github.com/ggml-org/llama.cpp/tree/a6ac41c757c066941766164439f6876f2deaab04), which has native `inkling2_loop` support. Apply [the demand-paging patch](matrix_runtime/demand-paging.patch) to the isolated source checkout with `git apply --unidiff-zero` so the full-bank baseline does not prefault the whole GGUF. The runtime uses greedy decoding and records peak process RSS and accepted token counts. Trace capture adds instrumentation overhead, so throughput comparison uses separate uninstrumented runs.

```sh
cmake -S InkyNano/matrix_runtime -B build/matrix -DLLAMA_SOURCE=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build/matrix --target inky-matrix --parallel 2
```

For an RTX 3050 build, pass `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real` at configure time. Run the matrix with `--gpu-layers` and the available VRAM limit. llama.cpp counts logical loop slots, several of which share the same physical expert weights. Verify physical tensor placements in the native log. Before loading weights the runner checks the proposed model, context and compute buffers and leaves 256 MiB inside the cap for CUDA driver and library overhead. The per-request receipt records the planned buffers and the sampled change in free memory for CUDA0. Other applications can affect the latter because it is a device-wide measurement. When sharing WSL with another native benchmark on the same host, use the same Linux `--wsl-run-lock` path for both.

Keep linked WSL CUDA libraries in persistent Linux storage across invocations. Supply a `sha256sum` file with `--runtime-library-manifest` to check the native executable and shared libraries before replay and bind the file’s digest to the benchmark specification.

To pair a CUDA benchmark with its CPU replay, run `matrix_compare.py` with the CUDA metrics as `--matrix` and the CPU metrics as `--cpu-baseline`. Keep both `spec.json` files beside their metrics. The comparison checks the model, workload, controller checkpoints, resident plans and sampling settings. It reports missing pairs, token counts, completion equality, quality scores and per-request timing and memory.

The local bank index `inky-isolated-gguf-v1` identifies an existing export in its artifact directory. It binds `source_model_sha256` to the original Q4 checkpoint digest recorded by the exporter. The `backbone` entry contains `path`, `bytes` and `sha256` for a GGUF with architecture metadata, tokenizer, non-expert tensors and original router rows. Each `shards` entry adds `layer` and ordered `expert_ids` to those file fields. The fused `ffn_gate_up_exps` and `ffn_down_exps` names identify packed projections. An isolated shard can have one or many experts for its physical layer. A file with multiple layers can appear in the index once per layer.

The runner hashes the backbone and each accessed shard, then copies selected packed expert rows, router rows, biases and matching scale rows byte for byte. The index constrains both learned controllers and their controls to real resident IDs. The original model file is unnecessary for this path. The packer checks the compact-file budget before writing model data. Each case retains pack/load latency alongside TPS. The runtime creates a compact scratch file per case and removes it after native inference exits. Charge that staging time in end-to-end comparisons.

### Published InkyNano GGUF bank

`compose_published_bank.py` reads the [published bank](https://huggingface.co/AlephFunk/InkyNano-GGUF) directly. This export uses its own manifest and standalone gate, up, and down tensors. Supply that manifest, the matching backbone, and either local layer shards or the first eight verified contiguous spans per layer. Use the pinned source revision's configuration and tokenizer with llama.cpp's Inkling2 Loop converter.

```sh
python InkyNano/compose_published_bank.py --bank export --source-metadata source-json --llama-source /path/to/llama.cpp --out matrix/inkynano-fixed-eight.gguf
```

The adapter preserves packed expert and backbone matrix bytes and reuses the upstream converter for recurrent queries, convolution layout and model metadata. BF16 controls widen exactly to F32. A supplied `--selection` uses the same `resident_by_layer` structure as the prompt controller. The default first-eight IDs provide a composition smoke. `--unrestricted` requires all local layer shards and enough disk budget for the composed model. The native executable's `inspect` mode validates the model tensor layouts and tokenizer without allocating weight data.

The published-bank smoke composed the text backbone and 192 experts into a 2,221,144,160-byte GGUF with 704 tensors. Native CPU inspection accepted the tensor layout and tokenized an eight-token prompt at 109,576,192 bytes of peak process RSS. A separate packed GGML CPU test used synthetic activations for one early-layer Q4_K-profile expert and one late-layer Q6_K-profile expert. Their outputs differed by 1.03% and 0.96% relative L2 from float evaluation of the same dequantized weights. Generation quality, CUDA memory use, and learned-controller performance require separate runs.

The `inky-packed-expert` target accepts `cpu|cuda q4|q6 EXPERT_SPAN` for a small diagnostic independent of the full model. It checks native GGML packed execution against float matvec using rowwise dequantization of the exact same packed weights. CUDA requires a CUDA-enabled build of GGML.

A scoped teacher can include more than eight original experts per layer. Its composition receipt records both the model and bank hashes and the ordered original expert IDs. The runner expands native slot IDs into the 128-ID coordinate space before training and uses the same mapping when packing the student's selected rows. Supply the same receipt to all three commands. Report the resulting teacher control as `teacher_subset` and interpret retained mass within its available expert support.

```sh
python InkyNano/matrix_run.py capture --cases matrix/cases.jsonl --model "$TEACHER_GGUF" --published-composition "$COMPOSITION" --teacher-in-ram --native "$NATIVE" --runtime-revision "$RUNTIME_REV" --context 1024 --batch-tokens 32 --rss-limit-bytes 3221225472 --out matrix/capture
python InkyNano/prompt_matrix.py --cases matrix/cases.jsonl --routing matrix/capture/routing.jsonl --published-composition "$COMPOSITION" --out matrix/controllers
python InkyNano/matrix_run.py benchmark --cases matrix/cases.jsonl --model "$TEACHER_GGUF" --published-composition "$COMPOSITION" --teacher-in-ram --native "$NATIVE" --runtime-revision "$RUNTIME_REV" --context 1024 --batch-tokens 32 --rss-limit-bytes 3221225472 --gguf-python /path/to/llama.cpp/gguf-py --plans matrix/controllers/plans.json --capacity 8 --out matrix/bench-k8
```

Use `--teacher-in-ram` only for a bounded teacher that fits the process budget. The eager path can avoid simultaneous RSS for mapped source pages and the CPU's repacked copies. The unrestricted teacher defaults to demand paging. Use `--batch-tokens` to cap one native prefill batch when memory is tight and hold it fixed across timing comparisons. For a timing pilot, `--max-new` caps sampled output and the spec records the cap. Use the full case budget for scoring. A benchmark request can run for up to 1,800 seconds by default, with `--case-timeout` available for longer cases. Infill and outfill share a root and remain in the same split. If the teacher uses a subset, transfer up to the full bank requires a new capture and fit.

### Replay the archived RTX 3050 full-answer matrix

Run the following from the repository root in the environment that can launch the CUDA runner. Set `TEACHER_GGUF` to the scoped 12-expert GGUF identified by the archived composition receipt, `NATIVE` to the compiled runner, `RUNTIME_LIBRARIES` to its `sha256sum` manifest and `GGUF_PYTHON` to the pinned llama.cpp `gguf-py` directory. The runner checks the model and runtime library digests. The grid replay uses the archived prior-based fit. The Storyworld replay uses the mixed-task fit and the first two development requests, which are the outfill and infill of the same scene. Omit `--max-new` to use each request's full scoring budget.

```sh
archive=InkyNano/prompt_matrix_receipts
pilot="$archive/pilot"
story="$archive/mixed-storyworld"
replay_3050() {
    python InkyNano/matrix_run.py benchmark \
        --model "$TEACHER_GGUF" --published-composition "$pilot/composition.json" \
        --teacher-in-ram --native "$NATIVE" --gguf-python "$GGUF_PYTHON" \
        --runtime-revision a6ac41c757c066941766164439f6876f2deaab04+demand-paging \
        --runtime-library-manifest "$RUNTIME_LIBRARIES" \
        --context 1024 --batch-tokens 32 --threads 4 --capacity 8 --gpu-layers 28 \
        --rss-limit-bytes 3221225472 --vram-limit-bytes 2147483648 "$@"
}
replay_3050 --cases "$pilot/cases.jsonl" --plans "$pilot/controllers-global-prior/plans.json" \
    --out matrix/grid-quality-3050
replay_3050 --cases "$story/cases.jsonl" --plans "$story/controllers-cuda-global-prior/plans.json" \
    --limit 2 --out matrix/storyworld-quality-3050
```

When launching the Linux runner from Windows, add `--wsl Ubuntu` and use a shared Linux `--wsl-run-lock` if another native job uses the same workstation. Pass the published MCP-05 receipt to `matrix_compare.py --goose-receipt` alongside the Storyworld metrics to include the historical comparison. [Completed pilot report](prompt_matrix_receipts/pilot/PILOT_REPORT.md)

## Train and replay

Use the Silico export's manifest to index the already-isolated weights and its matching backbone. Supply case-aligned, per-phase routing receipts with the same original Q4 checkpoint digest. Each receipt records the exact prompt digest, pinned teacher runtime revision, normalized routing mass and visit counts per physical layer. The training loader enforces the 48-visit loop layout for every token. Set `NATIVE` to the built runner and `RUNTIME_REV` to its pinned revision. Add `--wsl Ubuntu` when Python runs on Windows and the native binary was built in Ubuntu.

```sh
python InkyNano/matrix_cases.py --arc-public-training ARC-AGI-2/data/training --out matrix/cases.jsonl
python InkyNano/prompt_matrix.py --cases matrix/cases.jsonl --routing "$ROUTING" --expert-bundle "$BANK" --out matrix/controllers
python InkyNano/matrix_run.py benchmark --cases matrix/cases.jsonl --expert-bundle "$BANK" --native "$NATIVE" --runtime-revision "$RUNTIME_REV" --context 4096 --gguf-python /path/to/llama.cpp/gguf-py --plans matrix/controllers/plans.json --capacity 8 --out matrix/bench-k8
python InkyNano/matrix_compare.py --matrix matrix/bench-k8/metrics.jsonl --goose-baseline "$GOOSE_RECEIPTS" --out matrix/summary-k8.json
```

Start a separate output directory for each capacity and memory limit. Results with 16 or 32 slots require enough indexed experts and room within the process budget. Replay commits each successful request independently and resumes with identical inputs. If the teacher runtime is already available on the source host, `matrix_run.py capture` can collect missing labels there.
