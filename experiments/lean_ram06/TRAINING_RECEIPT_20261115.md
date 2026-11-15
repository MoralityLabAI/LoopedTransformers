# RAM-06 training-only CPU receipt | November 15, 2026

[Preregistration](PREREGISTRATION.md) commit
`f4f26ed48ed675cc7b80c73f151ba3927b46bafa` was frozen
before original train-source candidate collection or the new heldout
theorem-family evaluation.

[Completed source-only train CI #43680136177](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43680136177)
independently replayed every new Lean action effect from the original
27 MathCan source roots and each separately authored synthetic
`ExactCurriculum.lean` problem. It then trained all nine CPU models.
The separate prospective job starts only after the original
byte-verified checkpoint artifact has been published.

## Audited training corpus and provenance

| Source provenance / split | Original source roots | Genuine goal states | Candidate effects | Positive new `exact_typed` |
| --- | ---: | ---: | ---: | ---: |
| Original MathCan train-only | 22 | 171 | 4,740 | **0** |
| Generated exact auxiliary TRAIN | 12 | 12 | 249 | **12** |
| Original MathCan development | 5 | 42 | 1,176 | **0** |
| Generated exact auxiliary DEVELOPMENT | 6 | 6 | 128 | **6** |
| Combined verified corpus | 45 | 231 | **6,293** | **18** |

The 18 generated auxiliary declarations are intentionally modest
synthetic Lean tautologies with local hypothesis application, equality
symmetry, conjunction projection and implication/iff elimination.
They were all verified with the actual Lean4.35 tactic interpreter
in a separate fresh compiler session. The independent synthetic
audit checked **377 effects, 12 distinct positive typed-exact
TRAIN theorems and six distinct positive DEVELOPMENT theorems,
with zero disagreements**. Train had 19 and development
11 observed typed-exact negative candidate scripts.

Natural MathCan training and development collectively produced
**793 new typed-exact candidate checks with zero progressing
outcomes**. The generated dataset therefore establishes the
positive action mechanism and training loop, while original
source-domain transfer remains unproven before the new held-out
evaluation. This domain distinction is retained in all published
metrics. Synthetic examples count as **zero** original MathCan
certified theorems.

The original 27-theorem source collector individually verified
**5,916 true MathCan candidate outcomes**. Original FC-12
legacy 961 effects and original theorem/source hashes remain
unchanged and independently available. The fresh RAM-06
combined train corpus has SHA256:
`b807f8ba2c019a1f7544a744c8e6010ef8f138860d21bfbf170c8c8c9d7ad888`.

Auxiliary fresh independent source observation SHA256:
`320b9af8a074165b1db515acba2c9ac8d18818bb408ec74f1d8336683287b5f2`.
Independent synthetic Lean fixture SHA256:
`bd283525152ea73211d819133761fe0d3dae4962c861e64259cc0fa16d1ad6a0`.

The merged corpus recorded 183 train states and 48 development
states, with **49** natural multigoal train states and **19**
natural multigoal development states. No exact visible-state
hash overlapped between the registered optimization and
development splits.

## All nine CPU model fits

Models use the fixed 18-family grammar and the same width/loop
architecture as RAM-05, with the extra family embedding.

| Architecture | Parameters | Seed | Development top-1 usable /48 |
| --- | ---: | ---: | ---: |
| Looped32 | 85,093 | 17 | 23 |
| Looped32 | 85,093 | 29 | 22 |
| Looped32 | 85,093 | 43 | **24** |
| Looped64 | 207,045 | 17 | 19 |
| Looped64 | 207,045 | 29 | **25** |
| Looped64 | 207,045 | 43 | 20 |
| Flat64 | 182,149 | 17 | 21 |
| Flat64 | 182,149 | 29 | 25 |
| Flat64 | 182,149 | 43 | 15 |

Checkpoint selection uses only the balanced original-plus-auxiliary
development loss, **not** maximizing the top-1 counts in this
table. The selected checkpoints are:

| Model | Selected seed | Balanced development loss | Exact checkpoint SHA256 |
| --- | ---: | ---: | --- |
| Looped32 | 43 | 0.810591 | `70d1e481d0effe56bfc06c323e102b6ba305f01f716f3b5891c63c5758384258` |
| Looped64 | 29 | 0.867178 | `ac7d2fed61053cc1be9a293b5d057bb0c5d61a570889bc6fda1c0756f9865d2d` |
| Flat64 | 43 | 0.831433 | `748e18e6276af3f8a732d8cb640c310ee59593f371154f4ce21be4fcb3284ec0` |

Each model trained with the preregistered AdamW, effective batch64,
20% exact-positive independent Lean curriculum examples and
50% multigoal-input examples, with other data filling the
remaining batch. The full original optimizer/RNG states,
best/last checkpoints, loss histories, and all nine seed
receipts are retained in GitHub artifact
`ram06-frozen-audited-train-43680136177`,
artifact ID `14148087652`, 19,098,631 ZIP bytes,
SHA256 of ZIP `17c21aa9faf600841d03209f61161afa355b6cade8f9510fbe62223dbc213265`.

## CPU memory and compute

On a GitHub-hosted CPU-only Linux runner with Python3.11,
PyTorch2.6.0 CPU, Lean4.35 and two optimizer threads:

- Identical warmup/30-update Looped64 CPU calibration:
  **51.4709 optimizer updates per second**.
- Peak sampled total process-tree resident memory:
  **432,898,048 bytes (412.84 MiB)**.
- Minimum sampled available system RAM:
  **7,013,085,184 bytes**.
- All nine complete model checkpoints passed SHA256 checks,
  with no training memory-abort report.

This verifies CPU low-memory training. It does not establish
GPU throughput on a local RTX3050. The new original
MathCan family proof experiment is a separate result.
