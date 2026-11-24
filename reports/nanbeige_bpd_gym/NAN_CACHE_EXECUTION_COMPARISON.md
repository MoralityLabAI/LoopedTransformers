# Completed BF16 and NF4 cache execution diagnostics

All three registered conditions completed 130 public development contexts from
41 source families, with independent full-vocabulary and physical-state audits.
Explicit-effective NF4 recorded zero loop-2 token-choice disagreements on its
reference histories, while measurable incremental state differences remained.
The existing full-prefix conditions remain unchanged.
Every error and disagreement in the first table compares cache versus
full-prefix execution within that condition.

| Condition | Exact registered prefills | Reference positions | Loop-2 token-choice disagreements | Maximum vocabulary logit difference | Maximum normalized-boundary relative L2 |
|---|---:|---:|---:|---:|---:|
| KV11 BF16 | 130/130 | 821 | 12 | 2.1875 | 0.035932 |
| KV11 default NF4 | 130/130 | 671 | 3 | 3.5 | 0.076402 |
| KV12 explicit-effective NF4 | 130/130 | 671 | 0 | 1.75 | 0.074452 |

Each condition supplies its own full-prefix greedy history, capped at eight
tokens/EOS, and gives that identical history to its cached path. Later histories
are not assumed matched across conditions, including the two NF4 policies.
Disagreement counts therefore retain their own denominators; they are not
paired correctness comparisons or complete-task accuracy. Zero observed token
disagreements does not establish numerical equivalence. Prefill exactness covers
the registered last-token vocabulary, normalized boundaries and all 40 physical
sites; it does not certify every intermediate token or every cache byte.

KV11 BF16 checked 1642 paired vocabulary readouts and 65680 physical vectors.
Each NF4 condition checked 1342 readouts and 53680 physical vectors. The physical
observations span layers 5/11/17/21, both native loops, residual_after_layer and
the input/output coordinates of down_proj and o_proj.

Explicit-effective NF4 dequantizes the same packed decoder weights to BF16
before every F.linear, so it changes full-prefix arithmetic as well as cached
execution. Its historical comparisons use only the common initial full-prefix
history, with all 130 contexts present in each historical comparison at
completion. No later generated trajectory is paired by its position alone.

| Initial-prefix contrast: explicit NF4 versus historical reference | Normalized-boundary relative L2 | Two-loop normalized-boundary displacement relative L2 | Layer-11 residual cosine |
|---|---:|---:|---:|
| Default NF4 | 0–0.013317 | 0–0.008567 | 0.999894–1.0 |
| BF16 | 0.077256–0.221974 | 0.079186–0.198988 | 0.925515–0.983809 |

These are ranges across the published suite/copy/loop strata of equal-root
means, not per-record extrema, confidence intervals or an overall pooled effect.
Relative distances divide by the historical reference norm. The displacement
is the difference between the two normalized native-loop boundaries; its two
loop-labelled rows repeat that same contrast. Every stratum remains in the
aggregate source. Explicit/default NF4 initial-prefix free argmaxes agree in all
reported groups at both loops. Cross-backend agreement varies by group; no new
task grade is assigned to an initial token.

| Condition | CUDA peak allocated GiB | CUDA peak reserved GiB |
|---|---:|---:|
| KV11 BF16 | 2.423 | 2.537 |
| KV11 default NF4 | 2.650 | 2.816 |
| KV12 explicit-effective NF4 | 2.646 | 2.816 |

CUDA allocator peaks are distinct from WDDM physical usage. KV12's terminal
device sample was 2975 MiB whole-device usage, with 3119206400 dedicated bytes
and 81788928 shared bytes in its worker counters; that sample is not a lifetime
physical peak or a per-tensor residency proof. NF4 retains all 22 decoder layers
and the unchanged BF16 head on CUDA, with CPU embedding lookup. BF16 streams
decoder layers. One numerical CPU thread, Below Normal priority, one large
model and the .85 CUDA cap were retained; the separate math process was untouched.
Fixed reference-then-cache timing includes observations and full-vocabulary
readout on the shared machine, so no isolated throughput advantage is claimed.

This is exposed engineering/development evidence on one checkpoint. Physical
dose is zero, causal effect/sham floors and causal power are uncalibrated, and
planned/actual formal comparisons are zero. Source families are the grouping
units; contexts, tokens, loops and features are repeated measures. This work
neither repairs the failed algebraic competence gate nor establishes mediation,
donor transfer or equivalence. The original 168-case test remains sealed.
Complete-task behavior is reported separately in the
[benchmark index](BENCHMARK_INDEX.md).

Provenance:

- [Final KV11 aggregate at 3cf9a1e](https://github.com/MoralityLabAI/LoopedTransformers/blob/3cf9a1eedc6e6230c286fabef755a8998d4c8cbe/reports/nanbeige_bpd_gym/nan_kv11_progress.json), SHA-256 `9dc90d9cf3bbe5d35adde1335c87807f057efaf097a179bb1453aa9b5d7f5cc7`; source seal `bf9cb5d06ffa94d57de11f54dc4d4d4b2ab473c4ccf9f22715fb8d6d7960de67`.
- [Final KV12 aggregate at 673ebcb](https://github.com/MoralityLabAI/LoopedTransformers/blob/673ebcb39eda450ee912b8aafbe6495316e48fb4/reports/nanbeige_bpd_gym/nan_kv12_progress.json), SHA-256 `c0aee3215bb4fa5a7e6a759607308298124157ded1965995fdefa8ea8fe7591a`; source seal `5f7d135a48aafc4e1f2afd0c2e3fbe34998e722d05d584b42ee11743b7475315`.
- Both completed aggregate reports bind the independent worker receipts. The reporter verifies closed-artifact hashes and unchanged target-parameter digests. All three completed/exact-prefill counts and the comparison ranges above were checked against those immutable committed JSON files before publication. Raw prompts, replies, token IDs, arrays and per-record derivatives remain local.

The earlier [partial-publication denominator audit](NAN_KV12_STREAMING_SCOPE.md)
remains preserved. All three terminal initial-prefix rosters now have matching
130-context coverage.
