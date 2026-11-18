# LS-SAE13: Linux/Windows matched-edit CPU replication

**Engineering PASS in both environments. Exploratory numerical-stability
analysis only.** Four deliberately selected, already-studied LS-SAE12
puzzles; no new source cohort and no confirmatory efficacy gate.

The protocol was registered at commit \`e68b2302\`, ahead of code or
execution. The Linux and Windows probes both use the same locked
float32 512-coordinate Rademacher directions at the same median-blank
site. Every edit's 2048-byte little-endian representation, SHA256,
the pretrained checkpoint SHA256, the source receipt SHA256, the four
board positions and the four batch sizes agree across both runs.

## Provenance

| | Linux CPU | Windows CPU |
|---|---|---|
| Actions | [44289989807](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44289989807) | [44291343245](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44291343245) |
| Outcome | Engineering PASS | Engineering PASS |
| CPU | Intel Xeon Platinum 8573C | GenuineIntel Family 6 Model 106 Stepping 6 |
| OS | Linux 7.0.0-1019-azure | Windows 10.0.26100 |
| Python | 3.11.18 | 3.11.9 |
| PyTorch | 2.7.1+cpu | 2.7.1+cpu |
| NumPy | 2.1.3 | 2.1.3 |
| Threads | 4 | 4 |
| Receipt SHA256 | \`ecf81f1f4b1bf0ee994914000b35c5ef9ced7e006c9b1de0795bc85fc34d1644\` | \`23c5bb1f833614e7edb3da91ac72bdd699ea8ed6b1e3ee71090076f314935e72\` |

Both runs passed six unit tests and edit-hash/receipt-shape audits. They
used the same source-only LS-SAE12 pool receipt and the original 7M
pretrained TRM checkpoint. The full raw receipts are available as
Actions artifacts \`ls-sae13-four-puzzle-cpu-pilot\` and
\`ls-sae13-four-puzzle-windows-cpu-pilot\`.

## Batch and task outcomes

Within **each** operating environment, B2, B21 and B61 produce
bit-identical float32 clean and plus/minus logit tensors at every
ACT5–ACT16 step, on all four positions. The same-forward clean
duplicates and B21/B61 filler rows also match. Each has a distinct
B1 numerical path; first clean digit changes occur as follows.

| Position | Linux first B1-vs-B>=2 clean argmax difference | Windows first B1-vs-B>=2 difference | Linux B1 / B61 ACT16 blank correctness | Windows B1 / B61 ACT16 blank correctness |
|---|---|---|---|---|
| 23 | ACT9 | ACT11 | 28/59 ; 59/59 | 59/59 ; 59/59 |
| 24 | none | none | 59/59 ; 59/59 | 59/59 ; 59/59 |
| 28 | none | none | 57/57 ; 57/57 | 57/57 ; 57/57 |
| 40 | ACT11 | ACT13 | 58/58 ; 32/58 | 33/58 ; 30/58 |

The per-sign VDD changes are scored against the same-forward clean
reference for that exact batch, so numbers across batches have
different baselines.

| Position | Linux B1 -> B61 VDD (+/-) | Windows B1 -> B61 VDD (+/-) |
|---|---|---|
| 23 | 96/246 -> 145/295 | 4/6 -> 4/20 |
| 24 | 0/0 -> 0/0 | 8/4 -> 8/4 |
| 28 | 1/0 -> 1/0 | 1/0 -> 1/0 |
| 40 | 235/223 -> 72/64 | 84/76 -> 88/215 |

Exact matching of source, edit and checkpoint controls for identity
of the intervention. The operating environment still influences
recurrent trajectory and VDD.

## Cross-platform local-response comparison

At ACT5, the **centered 81x9 logit change induced by a matched edit**
is strikingly similar between Linux and Windows. For each platform,
take edited minus the same-forward clean float32 logits, subtract its
per-cell mean over the nine digit channels, restrict to blank cells,
and compute the Pearson correlation across those cells/channels.

Correlations for **batch 61** (positive / negative signed edit):

| Position | ACT5 r (+ / -) | ACT8 r (+ / -) | ACT16 r (+ / -) |
|---|---|---|---|
| 23 | +0.99967 / +0.99950 | -0.21444 / +0.26359 | +0.08150 / -0.13694 |
| 24 | +0.99857 / +0.99651 | -0.02697 / -0.02878 | +0.49653 / -0.40217 |
| 28 | +1.00000 / +1.00000 | +0.99996 / +0.99997 | +0.99927 / +0.99547 |
| 40 | +0.99986 / +0.99978 | +0.16417 / -0.05741 | +0.51337 / +0.03944 |

The same initial near-perfect agreement holds for batch 1:
ACT5 correlations across all four puzzles and both signs range from
approximately 0.9963 to 1.0000. This is an effect comparison relative
to each platform's own clean trajectory, since the two platforms'
pre-ACT5 clean hidden states are not byte-identical.

By ACT8, the edited effect on 23, 24 and 40 has substantially
decorrelated, despite its early agreement. Position 28 retains
approximately 0.995–1.000 agreement through ACT16, consistent with
its unusually stable clean trajectory and final correctness.
Position 24 also eventually resolves the puzzle on both platforms
despite substantial transient logits disagreement.

For context, the *clean* B61 Linux/Windows max absolute digit-logit
difference at ACT5 is 0.0393 on position 23, 0.651 on 24, 0.0145
on 28, and 0.0393 on 40. Its blank-cell argmax decisions are identical
at that step. At ACT8, the corresponding maximum differences reach
7.12, 78.12, 0.047, and 16.29, respectively. The model can diverge
internally and subsequently reconverge on the same exact Sudoku
solution.

The immediate edit-response agreement is evidence for meaningful
shared short-horizon sensitivity, while final verified-deduction
displacement is sensitive to numerical path selection. These are
distinct objects; neither proves whether the structural/logit edits
are more causally important than random edits.

## Connection to ACT5 operator microtrace

A separate microtrace on the same source and original checkpoint
captured 168 paired linear projection calls during ACT5. On an
AMD EPYC 9V45 hosted runner, every B1/B2 input/output projection
hash matched. On an Intel Xeon 8573C hosted runner, the **first
difference** was the output of the first \`layers.0.down_proj\`
applied to **bit-identical input**. The maximum output difference
was 4.76837158203125e-7 on positions 23 and 40. See
[LS-SAE13 ACT5 microtrace](LS_SAE13_ACT5_MICROTRACE.md).

These runs support a batch-shape-sensitive CPU linear projection
followed by recurrent amplification. A controlled isolated
matrix multiplication test can identify what backend/reduction
scheme is responsible. Hardware, OS, Python patch version, and
CPU backend build vary across the two full pilot environments,
so the cross-platform comparison does not isolate a single cause.

## Follow-ups

1. Replay the isolated \`down_proj\` with captured identical input
   and weight bytes at B1 and B2, comparing float32 with float64
   accumulation and toggling thread count and available BLAS backends.
2. Benchmark short-horizon centered-logit agreement and margin
   certification independently of long-horizon VDD. Add paired
   negative controls for potential direction-specific signal.
3. Register a fresh unseen puzzle cohort before making new claims
   about direction-specific causal efficacy. Keep PR #164's original
   FAIL and evidence receipts fixed.
