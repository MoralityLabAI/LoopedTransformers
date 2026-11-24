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

## Held-out routing mass

| Policy | Initial prefill | Initial decode | With train-root prior, prefill | With train-root prior, decode |
|---|---:|---:|---:|---:|
| Global frequency | 89.60% | 91.99% | 89.60% | 91.99% |
| LDT | 69.70% | 68.87% | 89.43% | 92.07% |
| TRM | 74.22% | 74.95% | 89.67% | 92.08% |
| Hindsight | 90.06% | 94.10% | 90.06% | 94.10% |

The prior substantially improved the learned policies on this development split. Their retained mass was close to global frequency. The 16 CPU timing requests used the initial checkpoints. [Initial routing summary](offline-summary.json) · [Prior-based routing summary](controllers-global-prior/offline-summary.json)

## Historical Goose reference

MCP-05 used Inky Mini IQ4_NL with two loops and an RTX 5080 Laptop GPU. Its Storyworld conveyor recorded 89 timed provider turns, with aggregate prefill of 36.01 tokens/s and decode of 9.48 tokens/s. The receipt attributed 47.27% of prompt tokens to cache hits. Its browser verifier passed the authored world and both secret endings. The three long-context planning runs reached their output cap without an accepted plan. [Published MCP-05 receipt](https://github.com/MoralityLabAI/HybridTRMLDT/blob/1307e7f9a38eb1afd63fb585b0cc80e297ddf1d9/MCP/results/mcp05/inky-hilbert-storyworld-canvas-receipt.json)

The local and historical throughputs describe different workloads, models, devices and cache conditions. The published receipt supports a historical comparison. A same-request replay needs the original Goose request payloads. The expanded local roster includes 20 new infill and outfill authoring cases derived from the verified world's encounter constraints. [Local and historical comparison](bench-k8/mcp05-comparison.json) · [Storyworld roster](../published-hilbert-storyworld-roster.json)
