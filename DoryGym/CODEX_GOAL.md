# Goal: DoryGym Stage 1 on RTX 3050

Use LoopedTransformers branch research/dory-gym-prep. Read DoryGym/SPEC.md and LOCAL.md.

Hardware: RTX 3050 4 GiB, 6 GB currently free RAM. No paid cloud. Check RAM/VRAM/disk and active processes. Do not download Inkling or load both models. Check whether a compatible Dory-enabled llama.cpp already exists; otherwise build pinned official revision with at most 2 parallel jobs. Verify official GGUF hash. Begin with context 512, small batches, partial offload. Leave at least 1.5 GB available system RAM and 512 MiB VRAM headroom; stop on paging or OOM. Record actual resource peaks. If download/build cannot fit, finish CPU tests and report blocker.

Run the synthetic smoke roster at fixed loops 1,2,4 under identical settings, one server at a time. Save logs, raw exact-answer outputs, latency/token counts, truncations, runtime errors and paired results. If feasible, repeat with last_loop KV cache at the same context. Match offload and sampling settings. Do not call a timed-out case incorrect without separately recording failure. Run policy.py after complete matched runs; compare fixed depth and pre-request train-only allocation, with hindsight as diagnostic. Do not claim generalization from synthetic families.

Prepare a future native hook experiment for normalized recurrent state displacement, BPD and adaptive loop control. Record exact implementation, cache semantics and matched sham controls before using hooks. Do not train or alter Dory weights in Stage 1.

Commit code, tests, small receipts and report to the DoryGym PR. Keep GGUF, build binaries and large traces outside Git. Return precise local reproduction commands, actual VRAM/RAM and tokens/s, and failure blockers. Preserve partial runs separately. 
