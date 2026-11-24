# InkyNano prompt-fixed routing pilot

The local pilot trained matched 20,544-parameter LDT and TRM controllers on native routing traces from a 12-expert-per-layer teacher composed from published InkyNano GGUF weights. Each request used one immutable eight-expert set per physical layer for prefill and decode. The pilot used one training root and one development root per task, each represented in infill and outfill. These results are development diagnostics.

## CPU timing

All 16 Sudoku and ARC-AGI-2 requests completed with an output cap of 24 tokens. Each policy has four requests, covering one root per task in both modes. End-to-end time includes selecting the resident set, packing its weights, model load, prefill and decode as applicable.

| Policy | Resident experts per layer | Median prefill tokens/s | Median decode tokens/s | Median end-to-end seconds |
|---|---:|---:|---:|---:|
| Scoped teacher | 12 | 3.46 | 1.59 | 98.5 |
| Global frequency | 8 | 3.85 | 1.47 | 134.4 |
| LDT | 8 | 4.22 | 1.65 | 130.6 |
| TRM | 8 | 3.71 | 1.82 | 135.0 |

The initial TRM produced the highest median decode rate. Packing and loading offset that gain in total request time. The timing cap excludes these outputs from full-answer correctness scoring. [Raw timing receipts](bench-k8/metrics.jsonl) · [CPU summary](bench-k8/cpu-throughput-overview.json)

## Full-budget quality preview

The prior-based TRM completed the Sudoku outfill request on CPU with 256 output tokens at 1.25 decode tokens/s. It hit the output limit without a parseable JSON grid or an exact compact-grid match. This is one held-out request with the eight-expert policy. [Sudoku quality receipt](quality-trm-cpu-preview/receipt.json)

The same TRM completed the held-out ARC-AGI-2 outfill request with 384 output tokens at 1.17 decode tokens/s. It hit the output limit and scored zero on the test grid under both JSON pass@2 and exact compact-grid scoring. [ARC quality receipt](quality-trm-cpu-preview-arc-agi-2-outfill/receipt.json)

The 12-expert teacher also reached 256 tokens on the same Sudoku outfill prompt and scored zero on both grid checks. [Matched teacher receipt](quality-teacher_subset-cpu-preview/receipt.json)

The eight-expert LDT with the train-root prior reached 256 tokens at 1.88 decode tokens/s and peaked at 2.36 GiB process RSS. It produced neither a parseable JSON grid nor an exact compact-grid match. All three policies tested on this held-out prompt failed both grid checks. [LDT quality receipt](quality-ldt-cpu-preview/receipt.json)

The same teacher also scored zero on the Sudoku outfill with the pinned Inkling chat template and reasoning effort set to none. It reached the 256-token cap in both prompt conditions. [Templated teacher receipt](quality-teacher_subset-cpu-preview-chat/receipt.json)

### Full-answer RTX 3050 grid replay

The full-answer matrix completed eight ARC-AGI-2 requests on one held-out root from the public training set. All reached 384 tokens and failed JSON parsing, test-grid pass@2 and exact compact-grid scoring. The four outfill completions differed. TRM repeated a digit row and LDT repeated example headings. Their outfill layouts retained 93.28% and 92.98% of teacher decode routing mass, respectively. [Complete full-budget ARC matrix](arc-full-answer-matrix.json)

| ARC outfill policy | Experts per layer | Prefill tokens/s | Decode tokens/s | Cold request seconds |
| --- | ---: | ---: | ---: | ---: |
| Scoped teacher | 12 | 14.71 | 0.97 | 506.0 |
| Global frequency | 8 | 16.39 | 1.02 | 521.6 |
| LDT | 8 | 27.26 | 0.98 | 479.9 |
| TRM | 8 | 16.94 | 1.00 | 526.1 |

The native loader allocated 1,525.64 MiB of model buffers in CUDA-pinned host RAM and 581.20 MiB on the GPU in each eight-expert replay. It reserved a 781.20 MiB CUDA compute buffer. The scoped teacher used 1,920.78 MiB of host model buffers, 663.41 MiB of GPU model buffers and a 1,002.12 MiB CUDA compute buffer. [Native buffer accounting](arc-outfill-full-answer-matrix.json)

All four ARC infill completions matched exactly, with 77 occurrences of `</after>`. The 24-token CUDA timing output was an exact prefix of the full completion for both controls in both ARC modes. LDT changed the resident set in three physical layers relative to global frequency. Its retained teacher decode mass went from 92.79% to 92.92%, and its retained prefill mass went from 90.05% to 89.91%. [Timing prefix checks](arc-global-full-answer-both-modes.json) · [Layer-level residency comparison](arc-ldt-full-answer-both-modes.json)

| ARC infill policy | Experts per layer | Prefill tokens/s | Decode tokens/s | Cold request seconds |
| --- | ---: | ---: | ---: | ---: |
| Scoped teacher | 12 | 29.78 | 0.96 | 450.2 |
| Global frequency | 8 | 25.29 | 0.86 | 542.2 |
| LDT | 8 | 19.88 | 0.81 | 574.4 |
| TRM | 8 | 19.82 | 0.87 | 562.0 |

All eight requests loaded in RAM mode. Live selection took 5–112 ms across the four learned requests. Packing and model loading took 70.4–77.6 seconds across the six eight-expert requests. Full-answer Sudoku is in progress.

Serialized reference answers passed the score checks for both modes. The ARC answer used 153 of its 384 output tokens, and the Sudoku answer used 187 of its 256. The pinned tokenizer reproduced the native prompt token counts for all four grid requests. [Reference answer budget check](answer-budget-check.json)

Each prior-based learned policy selected four distinct layouts across the four grid requests. Global frequency used one. With a reusable cache, the ARC outfill-to-infill transition would admit 4 LDT slots or 5 TRM slots. The Sudoku transition would admit 3 or 8, respectively. The corresponding packed projection estimates are 18.35 MiB and 25.19 MiB for ARC, and 16.02 MiB and 41.21 MiB for Sudoku. [Grid residency and admission estimates](controllers-global-prior/admission-estimates.json)

### Full-budget Sudoku outfill

All four CUDA policies used the same 95-token Sudoku outfill prompt, reached 256 output tokens and failed exact-grid scoring. The 45 givens in the rendered prompts yielded one solution matching the scoring reference in both modes. For both CUDA controls, the 24-token timing output was an exact prefix of the full completion. [Full-budget Sudoku outfill matrix](sudoku-outfill-full-answer-matrix.json) · [Reference uniqueness and control prefixes](sudoku-outfill-full-answer-controls.json)

| Sudoku outfill policy | Experts per layer | Prefill tokens/s | Decode tokens/s | Cold request seconds |
| --- | ---: | ---: | ---: | ---: |
| Scoped teacher | 12 | 10.02 | 0.86 | 358.2 |
| Global frequency | 8 | 6.51 | 0.79 | 466.2 |
| LDT | 8 | 10.06 | 0.80 | 406.1 |
| TRM | 8 | 10.45 | 0.86 | 384.3 |

LDT and TRM each reproduced their own full 256-token CPU completion with the same trained checkpoint. LDT decoded at 1.88 tokens/s on CPU and 0.80 on CUDA. TRM decoded at 1.25 tokens/s on CPU and 0.86 on CUDA. The 12-expert teacher completions differed across backends. All three CPU previews also failed exact-grid scoring. Full-budget Sudoku infill is in progress. [LDT CPU and CUDA pair](sudoku-outfill-full-answer-ldt-pair.json) · [TRM CPU comparison](sudoku-outfill-full-answer-matrix.json) · [Scoped teacher CPU and CUDA pair](sudoku-outfill-full-answer-teacher-pair.json)

### Full-budget Sudoku infill

The 12-expert control processed the 134-token infill prompt at 18.26 prefill tokens/s. It reached 256 output tokens at 0.82 decode tokens/s, repeated `</after>` 51 times and failed exact-grid scoring. Cold request time was 367.3 seconds in infill and 358.2 seconds in outfill. The three eight-expert infill runs remain pending. [Full-budget Sudoku control in both modes](sudoku-teacher-full-answer-both-modes.json)

## RTX 3050 CUDA smoke

The standard CUDA build generated eight tokens on the RTX 3050 Laptop GPU using the fixed eight-expert model, 32 llama.cpp offload slots, a 512-token context and a 32-token batch. It reached 3.37 prefill tokens/s and 1.23 decode tokens/s. The largest sampled device-wide free-memory drop was 1.64 GiB. The native process peaked at 1.90 GiB RSS. The 12-expert teacher exceeded the configured budget at 32 offload slots once runtime overhead was included. At 28 slots its allocation preflight passed for the 512-token timing and 1024-token quality contexts. [CUDA smoke receipt](cuda-smoke/native-cuda-smoke.json) · [Teacher preflight](cuda-smoke/native-cuda-teacher-inspect-28-1024.json)

The first matched timing pair used the 12-expert teacher for ARC outfill with 199 input tokens and the same 24-token completion on both backends. Prefill was 2.65 tokens/s on CPU and 27.38 on CUDA. Decode was 1.00 tokens/s on CPU and 0.88 on CUDA. End-to-end time was 136.2 seconds on CPU and 77.4 on CUDA, which sampled a 1.93 GiB device-wide free-memory drop and used 2.87 GiB peak process RSS. This pair measures one request. [Matched ARC receipt](cuda-smoke/first-matched-arc-timing.json) · [Matched comparison and remaining pairs](bench-k8-cuda28-initial/summary-with-cpu.json)

WSL restarted before the next CUDA request and cleared the build’s linked libraries under its temporary directory. The original timing receipt and executable checksum were preserved. A new run from the persistent build has a separate runtime identity and receipt directory.

The persistent build reproduced packed Q4 and Q6 expert execution on the 3050. One expert per profile differed from rowwise float dequantization of the same weights by 0.99% and 0.90% relative L2 on synthetic input. The fixed-eight end-to-end smoke generated eight tokens with a 1.64 GiB device-wide free-memory drop and 1.90 GiB peak RSS. The matrix uses 28 offload slots and checks the new binary and linked-library hashes before replay. [Persistent smoke receipts](cuda-smoke/persistent/runtime-identity.json)

The first request under the persistent matrix build reproduced the CPU's 24-token ARC outfill completion. CUDA reached 32.18 prefill tokens/s and 1.18 decode tokens/s, with 66.9 seconds of end-to-end latency. That request sampled a 1.93 GiB device-wide free-memory drop and reached 2.93 GiB process RSS. [Matched persistent-build replay](cuda-smoke/persistent/first-matched-grid-request.json)

All four policies subsequently completed that same ARC outfill timing request. Each GPU completion matched its own CPU completion. At eight experts per layer, packing and model load together took 69–83 seconds per CUDA replay. The scoped teacher took 66.9 seconds end to end. Cold LDT replay took 110.7 seconds, TRM 140.9 and global frequency 150.5. These are four paired policies on one request at the 24-token timing cap. [Matched ARC policy and stage timings](cuda-smoke/persistent/first-complete-case.json)

The LDT ARC infill replay diverged after 20 shared characters despite a matching prompt, checkpoint plan and token counts. Both completions contained prompt formatting tags and reached the 24-token timing cap. CUDA sped up prefill from 3.52 to 20.06 tokens/s, while end-to-end time increased from 157.8 to 171.6 seconds. This pair shows that greedy replay can produce different completions across the two backends. [ARC infill completion comparison](cuda-smoke/persistent/arc-infill-completion-divergence.json)

Across both ARC modes, seven of eight completions matched exactly. The GPU improved prefill in all eight pairs and shortened six requests end to end. CPU decode was faster in seven pairs. Infill end-to-end time increased for both learned policies, from 157.8 to 171.6 seconds for LDT and from 159.5 to 184.2 seconds for TRM. The eight CUDA requests peaked at 2.93 GiB process RSS and sampled up to a 1.93 GiB drop in device-wide free memory. [ARC infill and outfill timing pairs](cuda-smoke/persistent/arc-both-modes.json)

The four Sudoku outfill timing pairs all had faster prefill and slower decode on CUDA. LDT elapsed time fell from 120.9 to 116.5 seconds, and TRM from 112.9 to 106.0 seconds. The LDT and global completions matched exactly. The TRM and scoped teacher completions differed across backends. [Sudoku outfill timing pairs](cuda-smoke/persistent/sudoku-outfill-timing.json)

The full timing matrix completed all 16 CPU/CUDA pairs, with one development root per task, both modes and four policies. Median prefill rose from 3.71 to 19.84 tokens/s, median decode fell from 1.68 to 0.88 tokens/s, and median cold request time fell from 126.3 to 118.1 seconds. CUDA improved prefill in all 16 pairs and shortened 10 requests end to end. Thirteen completions matched exactly. GPU replays stayed within the 3 GiB RSS and 2 GiB sampled device-memory limits. The 24-token cap applies to the entire timing matrix. [Verified overview](bench-k8-cuda28-persistent/verified-overview.json) · [Paired metrics](bench-k8-cuda28-persistent/summary.json) · [Raw CUDA receipts](bench-k8-cuda28-persistent/metrics.jsonl)

## Held-out routing mass

| Policy | Initial prefill | Initial decode | With train-root prior, prefill | With train-root prior, decode |
|---|---:|---:|---:|---:|
| Global frequency | 89.60% | 91.99% | 89.60% | 91.99% |
| LDT | 69.70% | 68.87% | 89.43% | 92.07% |
| TRM | 74.22% | 74.95% | 89.67% | 92.08% |
| Hindsight | 90.06% | 94.10% | 90.06% | 94.10% |

The prior substantially improved the learned policies on this development split. Their retained mass was close to global frequency. The 24-token CPU and CUDA timing matrix used the initial checkpoints at a 512-token context. Full-answer scoring uses the train-root-prior checkpoints at a 1,024-token context with the same prompts. On the four held-out requests, the prior changed 55–56 of 192 LDT slots and 49–56 TRM slots. The global-frequency and hindsight layouts matched across fits. [Initial routing summary](offline-summary.json) · [Prior-based routing summary](controllers-global-prior/offline-summary.json)

The RTX 3050 repeated the prior-based fit on the same native Sudoku and ARC training roots with PyTorch 2.5.1+cu121. Both 20,544-parameter controllers took a combined 18.47 seconds of training. PyTorch peaked at 18.38 MiB allocated and 24 MiB reserved. Incremental process RSS peaked at 465 MiB. The CPU runner reproduced all eight learned residency plans from the GPU-trained checkpoints. All eight selected the same experts as the CPU-trained fit on these four development requests. [Training receipt](controllers-cuda-global-prior/training.json) · [Checkpoint replay and held-out mass](controllers-cuda-global-prior/offline-summary.json)

## Historical Goose reference

MCP-05 used Inky Mini IQ4_NL with two loops and an RTX 5080 Laptop GPU. Its Storyworld conveyor recorded 89 timed provider turns, with aggregate prefill of 36.01 tokens/s and decode of 9.48 tokens/s. The receipt attributed 47.27% of prompt tokens to cache hits. Its browser verifier passed the authored world and both secret endings. The three long-context planning runs reached their output cap without an accepted plan. [Published MCP-05 receipt](https://github.com/MoralityLabAI/HybridTRMLDT/blob/1307e7f9a38eb1afd63fb585b0cc80e297ddf1d9/MCP/results/mcp05/inky-hilbert-storyworld-canvas-receipt.json)

The local and historical throughputs describe different workloads, models, devices and cache conditions. The published receipt supports a historical comparison. A same-request replay needs the original Goose request payloads. The expanded local roster includes 20 new infill and outfill authoring cases derived from the verified world's encounter constraints. [Local and historical comparison](bench-k8/mcp05-comparison.json) · [Storyworld roster](../published-hilbert-storyworld-roster.json)

Goose suppressed the `<|content_thinking|>` token (ID 200008), configured a 34,816-token context and recorded cached prompt tokens. The local timing harness uses greedy sampling without logit bias and a 512-token context, with text instructions for infill. The local full-answer replay uses a 1,024-token context. [Goose runtime settings](https://github.com/MoralityLabAI/HybridTRMLDT/blob/1307e7f9a38eb1afd63fb585b0cc80e297ddf1d9/MCP/results/mcp05/inky-hilbert-storyworld-canvas-receipt.json)

## Mixed-task Storyworld fit

Four new native Storyworld traces joined the eight existing Sudoku and ARC traces. The resulting pilot has one training root and one development root for each task, each in both modes. The RTX 3050 fit both controllers in 6.23 seconds with 18.34 MiB peak allocated CUDA memory. The CPU replay reproduced all 12 learned plans from the GPU checkpoints. [Mixed-task cases](../mixed-storyworld/cases.jsonl) · [Mixed-task training receipt](../mixed-storyworld/controllers-cuda-global-prior/training.json)

On the single Storyworld development root, averaged across both modes, global selection retained 85.76% of teacher prefill routing mass and 88.50% of decode mass. The LDT retained 85.69% and 88.84%, the TRM 85.77% and 88.40%, and hindsight 86.74% and 91.83%. These results measure the 12-expert teacher's routing mass, with one resident set of eight per layer for each request. [Per-task routing results](../mixed-storyworld/controllers-cuda-global-prior/offline-summary.json)

With prefill and decode weighted equally, the learned policies stayed within 0.34 percentage points of global frequency on each of the three development roots. Their remaining eight-expert hindsight headroom ranged from 1.51 to 2.59 points. LDT gained 0.14 points on Storyworld. TRM gained 0.12 points on ARC and 0.08 points on Sudoku. [Root-level routing mass and oracle headroom](../mixed-storyworld/controllers-cuda-global-prior/offline-summary.json)

Across six held-out requests, LDT chose five distinct layouts and TRM chose six. Global frequency used one. Of the 192 resident slots per request, LDT changed 3–10 and TRM changed 4–10 relative to global frequency. For the paired Storyworld prompts, LDT kept the same layout and TRM changed one slot between outfill and infill. [Resident expert selection audit](../mixed-storyworld/controllers-cuda-global-prior/selection-audit.json)

The published bank's packed projection sizes give a cold expert payload of 999,309,312 bytes per eight-expert request, which matches the runtime receipt. With a reusable resident cache, switching between the two Storyworld modes would admit zero expert bytes for LDT and 4,810,752 bytes for TRM. These estimates count packed expert projections. The measured cold path repacks the model for every request. [Expert admission estimates](../mixed-storyworld/controllers-cuda-global-prior/admission-estimates.json)

### Full-budget Storyworld outfill

All four policies processed the same 445-token prompt on the RTX 3050. Each reached the 512-token limit and failed JSON parsing and the scene structure checks. The longest verbatim span shared with the prompt ranged from 225 to 384 characters across the four completions. The published reference scene passes the structure checks in both modes and misses the separate two-to-four-paragraph target. [Raw outfill metrics and historical Goose comparison](../mixed-storyworld/storyworld-outfill-matrix.json) · [Reference scene check](../mixed-storyworld/reference-scene-score.json)

| Policy | Experts per layer | Prefill tokens/s | Decode tokens/s | Cold request seconds |
| --- | ---: | ---: | ---: | ---: |
| Scoped teacher | 12 | 23.30 | 0.95 | 604.2 |
| Global frequency | 8 | 31.70 | 1.15 | 540.4 |
| LDT | 8 | 21.66 | 1.09 | 577.1 |
| TRM | 8 | 27.50 | 0.76 | 795.4 |

LDT exchanged 10 of 192 resident slots relative to global frequency on this prompt. It retained 85.91% of teacher decode routing mass versus 85.58% for global. [LDT outfill comparison](../mixed-storyworld/storyworld-outfill-ldt.json)

The 12-expert control peaked at 2.91 GiB process RSS. Its native sampler measured a 2,036 MiB drop in device free VRAM after CUDA initialization. A Windows sample showed 2,114 MiB device-wide in use, and the driver withheld per-process usage. The eight-expert runs stayed near 2.53 GiB RSS and a 1.69 GiB native free-VRAM drop. [Scoped resource receipt](../mixed-storyworld/first-storyworld-outfill.json)

### Full-budget Storyworld infill

Each policy processed the same 484-token infill prompt, reached 512 output tokens and failed the scene checks. The 12-expert control and global frequency returned identical text with 102 occurrences of `</replacement>`. LDT and TRM returned identical text with 102 occurrences of `</after>`. The four runs used the same 1,024-token context and CUDA placement settings. [Raw Storyworld receipts](../mixed-storyworld/quality-k8-cuda28-persistent/metrics.jsonl)

| Policy | Experts per layer | Prefill tokens/s | Decode tokens/s | Cold request seconds |
| --- | ---: | ---: | ---: | ---: |
| Scoped teacher | 12 | 41.24 | 0.94 | 607.3 |
| Global frequency | 8 | 27.53 | 0.95 | 703.2 |
| LDT | 8 | 41.66 | 0.86 | 702.5 |
| TRM | 8 | 29.08 | 0.98 | 648.1 |

The global-frequency and scoped controls spent 538.3 and 544.5 seconds decoding. Global frequency also spent 45.3 seconds packing and 59.9 seconds outside the recorded stages. The scoped run spent 1.7 seconds outside the recorded stages. [Per-stage receipts](../mixed-storyworld/storyworld-global-both-modes.json)

Across the eight local Storyworld runs, median prefill was 28.31 tokens/s and median decode was 0.95 tokens/s. All eight failed scene validation on the single held-out root. The historical Goose rates summarize its multi-turn workflow on a different model and device, with prompt caching. [Complete Storyworld overview and historical comparison](../mixed-storyworld/quality-k8-cuda28-persistent/verified-overview.json)
