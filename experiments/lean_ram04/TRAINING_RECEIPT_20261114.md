# RAM-04 frozen train-only Lean gym and CPU checkpoints | 2026-11-14

[Preregistration commit](PREREGISTRATION.md):
`ae7be99b6367434f0dd724ae232cfa218ed99342`.

[Original CPU training CI 43593270279](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43593270279)
completed the separate train-only job successfully before the held-out
evaluation job began. The first running workflow used original code commit
`f77b1fcc76146aead2ffa81f5193cb46c073498b`.
Its original frozen train-only archive is
`ram04-frozen-audited-train-43593270279`, 18,929,334 compressed bytes.
The archived input is derived from FC-12 training run `43547762837`,
original MathCan source `c84e2025334e9a967c1297425a490f98d5b3d189`,
and Lean4.35.0.

## Independent real Lean supervision

The RAM-03 unmodified train-only collector was run against 27 original
MathCan TRAIN theorem roots. The resulting 4,973 compiler-probed concrete
tactic effects were all independently replayed in 27 fresh Lean sessions.
The first independent audit reports `status=passed` and zero mismatches.
The 961 legacy FC-12 observations were preserved unchanged as source
inputs, without relying on original teacher proof commands.

| Split | Original theorem roots | Genuine states | States with multiple goals | Compiler-observed tactic outcomes |
| --- | ---: | ---: | ---: | ---: |
| Train | 22 | 167 | 52 | 3,939 |
| Development | 5 | 42 | 15 | 1,034 |

The four labels correspond to failed/unchanged, reduced goal count, other
real progress (including branches), and scratch closure.

Frozen RAM-04 expanded dataset SHA256:
`a84b1264e2eee76195041efc81fbcd7ec60e48c22068b7eba3c45ce85838f0b3`.
Original companion state list SHA256:
`5bee219b67a0c4fc494b69c8e05e10b5dcc389463a50a9f811f58e67315c308a`.
Frozen RAM-04 config SHA256:
`843d2973a71afbe867be88b8444c9623dad685d9050a3f890a0c2fe48eb9d96b`.

The resulting dataset reproduces the original RAM-03 per-class/source split
counts. Serialized bytes differ between the independent cloud regeneration
and the previous Windows campaign, so the receipt pins the new hash
rather than claiming a byte-identical reproduction.

## Selected training checkpoints

The nine models are fully fitted; the original full optimizer/RNG
checkpoint files were saved to the CI artifact before evaluation.

| Policy | Parameters | Selected seed | Best optimizer update | SHA256 of exact checkpoint |
| --- | ---: | ---: | ---: | --- |
| loop32 | 85,061 | 17 | 100 | `5544926c38e8b2e70faa4185eaeb51172ce7338001bb9e0742a7605131da9f2c` |
| loop64 | 206,981 | 43 | 100 | `416ea4aec151920a2e8a4ad162644b78f3f828daf3fa10e8c8b2858ff6bc00ef` |
| flat64 | 182,085 | 43 | 100 | `7a88c9a363f2e5dbc5e20a94fd459358fa183b657701baf8e2899cd636bd8de2` |

Original train-only balanced development loss for selected checkpoints:
loop32 0.85832, loop64 0.88683, flat64 0.89539.
Each selected fit ran 500 optimizer updates before early stopping,
with best development loss at update100. The selection rule remained
unchanged from RAM-03.

## CPU memory audit

The cloud CI runner was CPU-only with PyTorch 2.6.0 and two configured
training threads. Its identical 30-update pilot measured **56.93
updates/second** for loop64 in FP32. Torch reported no CUDA device.

- Maximum sampled whole-process tree RSS: **416,886,784 bytes
  (397.57 MiB)** during training.
- Minimum sampled free system RAM: **7,100,014,592 bytes**.
- No resource limit abort.
- All nine full model checkpoints passed exact saved-byte SHA256 checks.

These measurements describe the CI CPU runner, not a fresh benchmark
of the user's local RTX3050 hardware.

## Scoring isolation

The new original MathCan source families were reserved for the
separate dependent evaluation job. That job first checks all nine
checkpoint hashes and the audited train source. Only afterward can
it compile the original held-out theorems. The source-family selector
and both search budgets were preregistered before any train or score.
