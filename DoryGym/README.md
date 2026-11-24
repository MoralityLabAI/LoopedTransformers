# DoryGym

Dense recurrent control for InkyNano and DataGoblin. First target: RTX 3050 4 GiB VRAM, 6 GB currently free host RAM, Windows/WSL, batch 1. No paid GPU needed. Status: scaffold; no Dory inference results yet.

- [SPEC.md](SPEC.md): loop depth, cache, compute and LDT control.
- [LOCAL.md](LOCAL.md): official GGUF runtime setup and safety limits.
- [CODEX_GOAL.md](CODEX_GOAL.md): local execution handoff.
- make_roster.py: deterministic exact-answer smoke tasks (family-disjoint train/dev).
- bench.py: local OpenAI-compatible chat benchmark with response receipts.
- policy.py: root-paired depth allocation and budget analysis.
- test_gym.py: CPU-only unit tests.

```bash
python -m unittest discover -s DoryGym -p 'test_*.py' -v
python DoryGym/make_roster.py --out roster.jsonl
# Run one Dory server for each (loop, cache) configuration; see LOCAL.md.
python DoryGym/bench.py --roster roster.jsonl --loops 1 --cache per_loop --out results/1-per_loop.jsonl --server-log server1.log
# After complete matched results at depths 1,2,4:
python DoryGym/policy.py --runs results/1-per_loop.jsonl results/2-per_loop.jsonl results/4-per_loop.jsonl --out policy.json
```

Source: [Dory1.0-4B: A Looped Normalized-GPT](https://arxiv.org/html/2611.03998v1), [official GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF). Model OpenMDW-1.1; review license and attribution before redistribution. Repo contains code only. 
