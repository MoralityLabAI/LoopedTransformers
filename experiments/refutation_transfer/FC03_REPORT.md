# FC-03 first preregistered LR-01 matched-control run

Completed 2026-11-13. [First CI run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43352937705)
finished successfully, in two dependent GitHub Actions jobs.

The source preregistration was committed **before** the FC-03 generator and
model-runner code at `650bd2cc5f0fd628784f672248c3a3e362e9dcdf`.
[Full public results](https://github.com/MoralityLabAI/Compactification-Theory/blob/feat/refutationbench-fc03-preregistered/research/refutation_bench/results/fc03_preregistered_paired_20261113.md)
preserve all arm and control accuracies, paired McNemar discordance counts,
false-refutation counts, exact certificate recall and inference timing.

## Main finding

Every registered four-pass arm transferred to unseen graph topologies
at the original training depth (about 98% accuracy). At greater depth
all four-pass arms collapsed to about 56% accuracy. Tied and compute-
matched untied models differed by at most one outcome among each set
of 192 paired examples. The preregistered criterion of consistent
tied-cell superiority **failed**.

| Arm | Seed | Same-depth test (out of 192) | Deep test (out of 192) |
| --- | --- | ---: | ---: |
| Tied width24, 11,521 params | 17 | 189 | 108 |
| Untied width24, 45,073 params | 17 | 188 | 109 |
| Untied width12, 11,593 params | 17 | 188 | 108 |
| Tied width24, 11,521 params | 29 | 188 | 108 |
| Untied width24, 45,073 params | 29 | 189 | 108 |
| Untied width12, 11,593 params | 29 | 189 | 108 |

The primary matched-compute comparison uses the same width and four
LR-01 cell invocations. The untied control has more parameters.
The additional width-12 untied control approximately matches parameters,
at lower arithmetic cost.

## Fixed-weight recurrence extension

Both tied models trained at 4 recurrence passes. With no weight updates,
8 inference passes scored 163/192 deep for seed 17, and 149/192 for
seed 29. At 12 passes the corresponding deep scores were 102/192 and
175/192. The pass-count response is highly nonmonotone and
seed-sensitive.

All 384 held-out tasks have independent exact propositional verdicts.
The 192 positive Lean theorems compiled in [source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43352394607).
A neural negative verdict must carry a complete, independently
verified Boolean countermodel before acceptance. The exact checker
rejects false claims, and a selective policy can abstain when verification
fails. It provides no mathematical semantic guarantee for a manuscript
unless the manuscript-to-formal mapping is itself verified.

## Immutable inputs

- FC-02 training generator commit:
  `7aca2bc3fc5586c5b7997fba868bd179db761121`
- FC-03 test generator commit:
  `c859f04cda1d142d4eb560148414b82d53f80705`
- FC-03 test JSONL SHA-256:
  `ddc04ec8553a24269b16b53e27d28b521bc2f928978bfd5ee8a9e5fc36463fdb`
- CI artifact `fc03-frozen-weights`: all six model checkpoints plus
  training SHA256 receipts.
- CI artifact `fc03-paired-43352937705`: full test score JSON and 20
  per-case prediction JSONL records (30-day retention).

Training did not access the new test generator. The test job read
and verified immutable checkpoint hashes before scoring any labels.
The experiment is public synthetic logic and its outcome is not
a result about human mathematical proofs. Future claims need a new
independent held-out test and a preregistered method for halting.
