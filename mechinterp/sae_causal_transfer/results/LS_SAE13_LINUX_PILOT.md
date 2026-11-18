# LS-SAE13: four-puzzle Linux CPU engineering pilot

**Engineering PASS; scientific inference: exploratory diagnostic.** The
four-board pilot, its six unit tests and its independent receipt/hash audit
passed. No LS-SAE12 gate was rerun, changed or rescued. The known-puzzle
sample and all findings are descriptive, with no new claim of directional
effectiveness. The protocol was committed at \`e68b2302\` before this runner
or experiment was implemented.

## Provenance

- [Actions run 44289989807](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44289989807), workflow source commit \`e9589e48e67e46e5b58f8716a0809bd80fd4ca42\` (the first version with the Linux workflow), completed successfully on November 18, 2026.
- Artifact name: \`ls-sae13-four-puzzle-cpu-pilot\`; a 4,944,378-byte uncompressed JSON receipt.
- **Receipt SHA256:** \`ecf81f1f4b1bf0ee994914000b35c5ef9ced7e006c9b1de0795bc85fc34d1644\`.
- CPU: Intel Xeon Platinum 8573C; Linux \`7.0.0-1019-azure\`; Python 3.11.18; PyTorch 2.7.1+cpu; NumPy 2.1.3; 4 threads.
- Checkpoint SHA256: \`932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3\`.
- Four source-admitted LS-SAE12 positions 23, 24, 28, 40, with model-blind median blank sites 40, 40, 41, 43. SHA256-locked float32 Rademacher edits have L2 norm approximately 0.02 and identical bytes at every batch size.
- The batch-1 ACT4 clean anchor was reused for all ACT5–16 continuations, avoiding clean-screening or site-selection drift.

## Batch-shape result

Every unedited and edited continuation at **B2, B21 and B61 is bit-identical
at every one of ACT5–ACT16**, for all four boards and both edit signs. The
two separately produced B2 clean rows also agree bit-for-bit. All clean filler
rows at B21/B61 match their batch's clean row. The batch effect is therefore
a step between B1 and B2 on this CPU/backend combination, with no additional
drift when increasing from 2 to 21 or 61.

| Position | B>=2 first clean argmax difference from B1 | Clean ACT16 correct B1 | Clean ACT16 correct B>=2 | VDD(+) B1 -> B>=2 | VDD(-) B1 -> B>=2 |
|---|---|---|---|---|---|
| 23 | ACT9 | 28/59 | 59/59 | 96 -> 145 | 246 -> 295 |
| 24 | None | 59/59 | 59/59 | 0 -> 0 | 0 -> 0 |
| 28 | None | 57/57 | 57/57 | 1 -> 1 | 0 -> 0 |
| 40 | ACT11 | 58/58 | 32/58 | 235 -> 72 | 223 -> 64 |

VDD is measured **against the same-forward clean baseline** for each edit;
thus VDD magnitudes across batch sizes are not effects against a common
fixed trajectory. Their changes also reflect clean-baseline movement.

Every board has B1-vs-B>=2 *float32 logit differences starting at ACT5*.
The initial maximum absolute ACT5 differences across 81x9 logits are:

| Position | Max absolute ACT5 clean logit difference | Clean minimum blank-cell winner margin at ACT5 |
|---|---:|---:|
| 23 | 0.0000935 | 0.503 |
| 24 | 0.000153 | 1.07 |
| 28 | 0.000233 | 0.0607 |
| 40 | 0.000120 | 0.750 |

The ACT5 differences remain far below the initial winner margins. The
margin/epsilon certificates show all blank-cell argmaxes invariant on
position 23 through ACT8 and on position 40 through ACT7; position 40
has no observed flips until ACT11.

For position 23, B1-vs-B>=2 maximum logit difference increases from roughly
9e-5 at ACT5 to 0.129 at ACT8, 2.97 at ACT9, and 81.9 at ACT11.
Blank-cell flips first occur at ACT9 (4 cells), rising to 30 by ACT11
and 31 at ACT16. B>=2 obtains the exact puzzle solution while B1
remains at 28/59 correct at ACT16.

For position 40, the maximum clean logit difference rises from 1.2e-4
at ACT5 to 0.100 at ACT8, 0.655 at ACT10, 7.20 at ACT11, and 84.7
at ACT12. Blank-cell flips first occur at ACT11 (4 cells) and reach
35 at ACT12. Here B1 is exact at ACT16; B>=2 finishes with 32/58
blank cells correct.

## Interpretation and limits

This is within-one-platform evidence of a batch-shape-sensitive forward
operator. The B1-to-B2 change creates small ACT5 numerical differences
that grow under recurrence until output decision boundaries are crossed.
The controls show that adding extra batch rows above B2 does not add any
new float32 differences in this environment. It does not identify the
first internal operation introducing the difference, or establish a
Lyapunov exponent.

The earlier registered LS-SAE12 Actions runner used an **AMD EPYC 9V45** and
reported 48/48 clean batch-61 predictions matching batch-1 screening;
this LS-SAE13 pilot ran on an **Intel Xeon Platinum 8573C**, even though
both used PyTorch 2.7.1+cpu. The processor/backend difference is an
important candidate explanation for the disparate batch-invariance
observations. More targeted traces are needed before attributing causality
to CPU instructions, MKL/oneDNN or a particular matrix multiplication.

A separately triggered Windows CPU workflow provides another platform
receipt for the exact same four locked edits. Its outcome is not included
in this Linux-only report.

## Next diagnostic

Instrument the first ACT5 H/L microcycle with a B1-versus-B2 same-anchor
comparison, including the outputs of normalization, attention and
feedforward operations, before testing deterministic CPU backend switches
and one-versus-four threads. Compare Windows-vs-Linux ACT5 centered
logit effects and winner-margin shifts on matched blank cells. The
confirmatory LS-SAE13 successor should register a fresh, previously
unused puzzle cohort only after these engineering choices are fixed.
