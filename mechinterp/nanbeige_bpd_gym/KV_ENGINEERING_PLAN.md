# Native two-loop cache engineering, before pretrained use

This new module is imported by no frozen worker. Its purpose is to reduce
repeated full-prefix computation for future generative wider-suite/math tasks,
while exposing both recurrent boundaries and all physical observation sites.
It is not permission to replace historical full-prefix scoring or enable donor
transfer. Changes to past hidden states require invalidating dependent caches;
the prototype has no intervention or cache-invalidation contract yet.

Use separate DynamicCache instances for the two native passes, logical layer
indices (including the BF16 reusable staging layer), correct absolute positions,
the model's own causal mask and existing final-loop normalization. Keep loop_idx=0
inside each native full-stack pass, matching the existing research backend.
Never combine K/V from different recurrent passes. A new source resets both
caches. Preserve B1, frozen weights and all 40 physical last-token observations.

Before loading pretrained tensors in any new condition, test random 22-layer,
width-64 models using the pinned native source. Engineering seeds and inputs
are fixed before the first run. FP32 uses vocabulary 4096 as a mathematical
reference; BF16/NF4 use the full 166144-token head with existing native-dtype
chunks. Prefix length 19; eight predetermined appended tokens, evaluated under
the exact same forced history for full-prefix and cached paths. No outcome or
argmax-based input selection. Test both loops and every site at layers 5/11/17/21.

Required checks: independent per-pass caches; correct length at every layer;
target parameter hashes unchanged; bit-exact initial prefill boundaries,
physical tapes and full-vocabulary logits. Incremental FP32 boundary relative
L2 and max absolute logit error must be <=1e-5. For BF16/NF4, boundary relative
L2 <=.02 and max absolute logit error <=.1 are engineering bounds only.
Preserve all per-step/loop errors and free-argmax agreements, including failures.
No bound or argmax rule may be tuned after this random fixture is observed.

Incremental single-token GEMM/GEMV/SDPA can select different kernels from
full-prefix inference, especially NF4. Incremental argmax disagreements are
reported, not hidden by changing forced inputs or claiming exactness. Passing
these checks establishes small-model execution evidence only. Pretrained use
still needs a separately frozen execution condition, same-source parity and
competence measurements, memory receipts and explicit numerical limitations.

Resource policy: no pretrained weights loaded, CUDA allocator fraction .05,
one numerical CPU thread, Below Normal priority. The large registered Prime
worker continues unchanged. Original 168-case test remains sealed. No model
competence, trained BPD quality, causal effect or equivalence claim follows.

## V2 kernel-coverage amendment, before its fixture

V1 passed and is preserved at commit b3f1386 with its exact receipt. Its MLP
width 96 triggered bitsandbytes' unaligned inner-dimension fallback for down_proj;
the real width 10752 is divisible by 64. V2 changes only the random fixture's
MLP width to 128 so every NF4 linear can exercise the aligned single-token
kernel. Keep all seeds, inputs, step counts, precision conditions, error bounds,
argmax diagnostics and failure retention unchanged. Do not retune thresholds.
