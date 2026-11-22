# IPB-01: Inkling 2 Mini token-conditioned position-bias probe

**Status (2026-11-22): instrumentation implemented and tested with synthetic tensors. Real checkpoint curves have not been measured.** This module is distinct from the trained-gate synthetic positive control in `research_gym/inkling_positive_control`. It does not revise any sealed LoopedTransformers, BPD, EvalMath, or RMP result.

## Question and mechanism

Which backward distances does the released Inkling 2 Mini *positional term* boost when the query is `the`, `of`, `in`, `former`, or `latter`? Its attention score includes a query-conditioned positional term:

```
feature[q, head, :] = reshape(W_r @ x_q)[head, :] * learned_scale[head, :]
feature = feature * min(1, rel_l2_clip / (L2(feature) + 1e-6))
if full_attention:
    feature *= 1 + 0.1 * log(max((position + 1) / 8192, 1))
bias[q, head, backward_distance] = feature @ learned_basis[:, backward_distance]
```

The relative-position basis has 16 channels and 1,024 backward distances, with 24 query heads. At negative distances and distances >= 1,024, the positional contribution is **zero**; the causal mask and sliding window have their own rules. This is a property of the learned relative-position term, not a claim about the complete attention distribution. In particular, it is query-conditioned: `former` and `latter` may induce different distance profiles, but the positional term alone cannot identify *which* previous token contains an antecedent. The QK term and any sink probability still matter.

For layer 0, `x_q` is reconstructible from *just* the query token embedding: `RMSNorm_attn(RMSNorm_embed(embed[token_id]))`. At layers 1–23, block-attention residuals, previous layers, and core cross-pass messages alter `x_q`. Later-layer scans therefore require **captured, post-attention-norm input activations**; they cannot be labelled as pure token-specific effects.

### Native checkpoint tensor paths

Checked against the released model's `model.safetensors.index.json`:

```
model.llm.embed.weight
model.llm.embed_norm.weight
model.llm.layers.0.attn_norm.weight
model.llm.layers.0.attn.wr_du.weight
model.llm.layers.0.attn.rel_l2_clip.scale
model.llm.layers.0.attn.rel_logits_proj.proj
model.llm.layers.0.attn.k_sconv.weight
model.llm.layers.0.attn.v_sconv.weight
```

The reader uses safetensors' random-access slices to pull only requested embedding rows plus the relative-bias parameters. A local copy of the released ~19.5 GB safetensors file is still needed. No untrusted pickle is loaded. Tensor hashes cover **the extracted arrays**, with full-checkpoint provenance supplied as a pinned model revision; the script deliberately avoids claiming to have hashed the full file.

## Run on an already downloaded checkpoint

```bash
python -m pip install numpy safetensors torch tokenizers
# Optional visualization: python -m pip install matplotlib
hf download thinkingmachines/Inkling-2-Mini \
  model-00001-of-00001.safetensors tokenizer.json \
  --local-dir /path/to/Inkling-2-Mini
python -m research_gym.inkling_position_bias.probe \
  --checkpoint /path/to/Inkling-2-Mini/model-00001-of-00001.safetensors \
  --tokenizer /path/to/Inkling-2-Mini/tokenizer.json \
  --layer 0 --form both --position 4096 \
  --model-revision PINNED_HF_COMMIT_SHA \
  --out runs/ipb01-first-layer --plot
```

The program writes `relative_curves.npz` (`curves` shape `[query_forms, 24, 1024]`, strings, token IDs, and absolute positions), `summary.json` (per-head peaks/valleys, individual tensor fingerprints, four-tap K/V magnitude summaries, limitations), and optionally a per-token heatmap PNG. It explicitly skips text forms that the **actual model tokenizer** splits into more than one ID. Both bare and leading-space token forms are measured, since those may be distinct vocabulary items.

Run the small CPU tests independently:

```bash
python -m unittest discover -s research_gym/inkling_position_bias -p 'test_*.py' -v
```

For a later layer or a particular loop pass, provide a `.npz` containing `states [N,2304]` at the **input of that layer's `relative_position_bias` module**, `labels [N]`, and optional `position_ids [N]`, then run with `--layer N --hidden-npz traced_states.npz`. `--layer > 0` rejects token-embedding reconstruction. Pin the exact pass, tokens, model revision, source prompts, and masking configuration when producing those states. This package contains no checkpoint traces.

## Causal and content-aware follow-up

The optional `--content-npz path.npz` accepts the same queries in the same order, with:

- `content_logits`: QK scores after query/key normalization and K convolution, including native QK scaling, shape `[N,24,K]`.
- `distances`: integer `query_position - key_position` shape `[N,K]`.
- `sink_logits`: optional learned sink logits, shape `[N,24]`.

The program computes the *frozen-content counterfactual* holding QK content and sink logits fixed while setting the positional term to zero. It records the mean L1 change of key+sink attention probabilities and top-destination changes, respecting causal and local masks. A **true causal intervention on model outputs** needs a second complete forward pass after perturbation, because the attention output and later hidden states can change. Call that separate experiment IPB-02, with clean accuracy gates, matched masks, layer/head-targeted and dose controls. The four-tap K/V plots are descriptive absolute weight magnitudes, not an ablation.

Prespecified test texts should include subject–verb agreement across increasing distances, prepositions, determiner/noun pairings, and `former`/`latter` with two controlled antecedents. Read positional terms and full QK scores separately. Use fixed content and swapped antecedents to distinguish distance preference from semantic disambiguation. Group examples by *source sentence template/root* when estimating uncertainty; token positions and heads are repeated measures.

## Cross-loop KV compression, separate experimental arm

**IPB-KV-00 registration; no Inkling compression benchmark has run.** LoopVQ (Cheong and Yoo, November 2026) jointly codes channel slices across **four** recurrence steps, with codebook fine-tuning and a mixed BF16 sink/recent-token path. Its Ouro/Huginn results do not transfer automatically to Inkling. Inkling 2 Mini has three passes through 12 tied core layers and 48 distinct KV cache sites across its 48 layer visits. An Inkling adaptation must compare a 3-pass shared codebook with independent-per-pass quantization; do not silently pad a four-pass LoopVQ code and count the padding as real compression. Preserve the prelude/coda caches separately. LoopKV (Zhou et al., November 2026) independently suggests per-(pass,layer) sensitivity profiles and selective precision.

Register comparison arms after a verified BF16 baseline: BF16; 4-bit independent; 2-bit independent; 3-pass grouped VQ with and without a loop-aware output-matching stage; and a sensitivity-allocated mixed-bit control. Measure KV bytes/token, wall-clock decode throughput, peak VRAM, accuracy on held-out long-context retrieval and verified reasoning, and per-pass quantization damage. Calibrate and evaluate on disjoint roots. The cross-pass **message tensors** are additional candidates for compression, but a transient message is not identical to a persistent KV cache; measure its actual residence time and bandwidth before claiming savings.

## Sources

- [Thinking Machines, Inkling 2 Mini announcement](https://thinkingmachines.ai/news/inkling-2-mini/), 2026-11-20.
- [Released checkpoint config](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/config.json) and [tensor index](https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/model.safetensors.index.json).
- [Hugging Face reference `modeling_inkling2_loop.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py), particularly `Inkling2LoopRelativePositionBias` and `Inkling2LoopSlidingDecoderLayer`.
- [vLLM, Inkling 2 Mini architecture](https://vllm.ai/blog/2026-11-20-inkling2-mini), 2026-11-20.
- [Cheong and Yoo, LoopVQ](https://arxiv.org/abs/2611.05304) and [implementation](https://github.com/mscheong01/LoopVQ), 2026-11-03.
- [Zhou et al., LoopKV](https://arxiv.org/abs/2611.04137), 2026-11-03.
