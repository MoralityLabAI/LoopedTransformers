# INK-ROUTE-02 coordinate-2027 routing

This follow-up starts at [INK-ROUTE-01 merge f511aff](https://github.com/MoralityLabAI/LoopedTransformers/pull/311). It calibrates one candidate causal endpoint on the native pretrained model. The [promotion protocol](PROTOCOL.md) gives the proposed estimand, meaningful threshold, power and sealed access rule.

## Historical audit

The original eight-root CUDA run had 96 paired rows, eight tensor archives, 16 restored rescues, zero top-token flips and maximum relative reconstruction error of (1.75\times10^{-7}). Recomputed mean pass-3 K zero-minus-sham β₃ was −0.098214. The receipt audit also checked both successful two-root smoke runs and the four failed CPU attempts.

Transformers 5.24.0 calls the same reader 12 times in each of passes 2 and 3. Its input combines completed block 2 and the previous traversal's final layer-17 state. Unweighted RMS normalization precedes scaled dot-product attention. The output has Q, K, V and MoE order and the decoder unbinds it in that order. Archived output-side reconstruction and query-side softmax agree. The pre-hook supplies a private block-3 copy for one visit. The rescue supplies the clean reader output in the ablated lane for the final token. Subsequent readers use the original bank. INK-ROUTE-02 additionally verifies both banks at every read across each traversal, while allowing the model to compute its next pass's state through the native forward. [Transformers source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py)

The source checkpoint is `thinkingmachines/Inkling-2-Mini@b85c3cfa615fc413cc9af3379f491b963f3b586a`. Its full 19,526,019,168-byte shard is SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. The runner checks the tokenizer, config, index and full shard before model loading. The packed-expert backend streams selected experts through the installed Transformers converter. Text execution uses native eager attention, the recurrent decoder and logits. The vision tower is unused.

## Development evidence before the broader controls

The earlier local 60-document pass-3 K calibration used a public FineWeb train roster from `CC-MAIN-2024-51`, distinct from the eight historical prompts. This round reuses that exposed roster only for development and retains its frozen coordinate draws. The 2,880 rows include 16 reference shams and four placebos per root with two replays. The mean zero-minus-mean-sham K difference was **−0.027714**, with root SD 0.052089 and 45/60 decreases. The matched-sham 95%/95% floor was **0.220428**, above the 0.10 scientific minimum. Replay discrepancies were zero at the recorded precision. Maximum realized dose disagreement was 0.376%. The pinned EvalMath strict preflight returned **REDESIGN_OR_PILOT_ONLY**. This calibration supports no positive confirmation sample size and the sealed readout remains closed.

## Native smoke

Two fresh development prompts passed on the pinned RTX backend, with 208 rows across both passes and two replays. All eight rescues reproduced the clean message exactly. Worst relative reconstruction error was 1.669e-7 and the largest realized equal-dose discrepancy was 0.220%. Maximum rescue log-probability difference was 1.788e-7 nats. The run peaked at 4.01 GiB reserved CUDA memory and 8.65 GiB sampled process RSS. Available host RAM reached a minimum of 4.77 GiB. Full checkpoint verification and model load took 37.91 s and 18.76 s, respectively. The eight profiled forwards totaled 332.70 s under the shared workload. [Smoke receipt summary](receipts/native-smoke/summary.json)

## Reproduction

Use Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 5.24.0, Accelerate 1.16.0, NumPy 2.5.5, psutil 7.2.2, safetensors 0.8.0 and tokenizers 0.23.3. The CUDA runner checks these package pins. Before allocation the runner checks host memory, CUDA memory and physical NVIDIA headroom. The per-process allocator is capped at 5 GiB by default. Existing workloads continue running. Use a fresh output directory for every attempt.

```text
python -m unittest discover -s research_gym/inkling_routing_01 -p "test_*.py" -v
python -m unittest discover -s research_gym/inkling_routing_02 -p "test_*.py" -v
python -m research_gym.inkling_routing_02.receipts audit --history PREVIOUS_OUTPUTS --out historical-audit.json
python -m research_gym.inkling_routing_02.native --snapshot PINNED_SNAPSHOT --out smoke
python -m research_gym.inkling_routing_02.receipts analyze --receipts smoke --out smoke-analysis
python -m research_gym.inkling_routing_02.native --snapshot PINNED_SNAPSHOT --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --out development
python -m research_gym.inkling_routing_02.receipts analyze --receipts development --public-roster research_gym/inkling_routing_02/receipts/fineweb-development-roster.json --preflight EVALMATH/skills/evalmath-mechinterp-power/scripts/preflight.py --out development-analysis
python -m research_gym.inkling_routing_02.receipts bundle --receipts development --out development-bundle
```

The first native command runs two fresh smoke prompts. Estimate no scientific effect from those prompts. Validate both roots before running any public roster shard, and exclude the smoke prompts from all development floor and power estimates. For recovery use `--start OFFSET --count COUNT` with the original roster and identical software and placement. Pass nonoverlapping successful receipt directories to `analyze`. The raw paired rows contain per-forward timing, realized doses, final-token outcomes and tensor summaries. Local NPZ archives preserve the bank, query, reader output, delivered message and writer vectors. Each run emits resource snapshots, checkpoint metadata, stage failures and SHA-256 hashes. The offline CI jobs exercise synthetic contracts. They do not run the checkpoint.

The GitHub bundles contain gzip-compressed raw JSONL, the original run's SHA-256 manifest, and the metadata receipts. Verify decompressed JSONL against the original manifest. Tensor archives are retained locally and listed by hash in the run manifest.
