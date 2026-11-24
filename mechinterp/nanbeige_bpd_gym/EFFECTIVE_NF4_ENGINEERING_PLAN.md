# Explicit BF16 effective-weight cache engineering

This new random-model diagnostic is declared before execution. No pretrained
weights, task corpora, original sealed test or live worker files are read.
Existing KV11, cache fixtures V1/V2 and all frozen dependencies stay unchanged.

The installed bitsandbytes 0.50.2 uses shape-dependent fused NF4 kernels even
for multiple rows. Duplicating a single row does not guarantee the same
arithmetic as a long prefix. The new condition instead explicitly dequantizes
each packed NF4 matrix to BF16 and calls PyTorch F.linear for every row count.
It retains the exact packed parameters and double-quantization state. This
changes full-prefix arithmetic too; it is not asserted equivalent to the
historical/default NF4 condition. No persistent dequantized decoder is stored.

Use the V2 random fixture: seed 2026112011, 22 layers, hidden width 64, MLP 128,
4 query heads, 2 KV heads, head dimension 16, full vocabulary 166144, two native
passes, prefix 19 and eight fixed random tokens 1..99 from the original seed.
Evaluate BF16, default NF4 and explicit-effective NF4. Build the two NF4 models
with the same seed and assert their packed parameter digest is identical.
All full-prefix and cache paths receive the fixed common history; no argmax
selection, refitting, threshold tuning or dropped arms. Preserve full step/loop
metrics and any failures. Compare default versus explicit full-prefix boundary
and vocabulary errors and argmaxes separately, with no cross-policy bound.

Within each policy, prefill boundaries, all 40 physical site vectors and full
vocabulary logits must be bit exact. Incremental boundary relative L2 <= .02
and max absolute logit error <= .1 retain the V2 engineering bounds. Count and
keep every free-argmax disagreement. Run all steps even after an incremental
bound is exceeded, then fail the condition; do not truncate inconvenient rows.
Target parameter digests must stay unchanged. Report synchronized path timing,
peak CUDA allocated/reserved bytes and all differences, without attributing
cross-condition changes to caching or treating timings as isolated throughput.

Run with CUDA allocator fraction .05, one numerical CPU thread and Below Normal
priority; only after confirming >=512 MiB device headroom. The actual KV11
large-model queue continues unchanged. These are random engineering results,
zero formal inferential comparisons, no mechanistic dose/sham claim and no
donor transfer. Passing does not authorize pretrained use. A new pretrained
condition requires its own frozen sources, source cohort, accuracy endpoints,
controls, memory budget and published registration before inference.
