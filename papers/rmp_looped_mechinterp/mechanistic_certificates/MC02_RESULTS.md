# MC-02 verified execution record

2026-11-11. Code commit: `2084913704b4c5884408cfd01dbd6ab7cc095dd0`.
PR #7 is stacked on MC-01 PR #5. No existing model or trainer was changed.

## Complete-checkout validation

GitHub Actions run [43015177173](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43015177173),
job `130504333636`, passed using the complete repository modules. The log records
68 tests passing in normal Python and 68 with `python -O`. MC-02 adds 29 tests;
the other 39 come from MC-01. Python was 3.12.16 and PyTorch 2.10.0+cpu.

| Exact control | Result |
|---|---:|
| Discriminating three-node graph/cursor challenges | 144 |
| Correct four-step counterfactual traces accepted | 144 |
| Donor-computation copies rejected | 144 |
| Ignored-intervention continuations rejected | 144 |
| Rejected donor copies that still have the correct final answer | 60 |

These controls are constructed to separate the three declared continuations at
the first future step. Their outcomes validate the relation checker and selection
contract; they are not measurements of learned reasoning.

## Native untrained integration smoke

The original LoopedChase/SEM-ECHO modules ran 16 generated items through seven
loops. Native cursor/head parity passed on all 112 visit/item comparisons with
maximum absolute error 0.0 and zero label mismatches.

The selection panel used visits 1,2,4, horizon 3, seed 940211 and a four-pair cap.
Of 48 requested receiver slots, four had correct current endpoint predictions.
Four slots were eligible, with seven possible ordered donor pairs. Four pairs
were selected before interventions; no fallback was used.

Each pair ran six arms: cursor transfer, sham, full donor with its own embedding,
constant cursor shift, and two norm-matched random cursor edits. All eight sham/
donor full-state replay checks passed, with the corresponding head-logit checks.
The archives retain patched states, resumed states, input embeddings and logits.
The full protocol defaults to eight random controls; this bounded smoke used two.

The receipt is `UNTRAINED_SMOKE`, with `scientific_summary: null` and zero trained
checkpoints evaluated. These observations establish that the original model,
intervention adapter and exact checker execute together.

## Inspected artifact identities

Artifact `13851779070`, `rmp-mc02-evidence`, contains controls.json and the smoke's
selection.json, clean_trace.npz, replays.npz and metrics.json. Its archive SHA-256
was checked after downloading:

`87b40ded38bfaf540584385e5c64726062e405cc59b51a469864df903bc04c24`.

The inspection also verified the selection/NPZ hashes against metrics.json,
matched the new source-file identities against the tested files, and independently
recomputed every reported current/future label from the saved logits.

| File | SHA-256 |
|---|---|
| selection.json | 42aa735938366561e8e3ebee183156fc40061fa4aca10d8794973e73b96a9116 |
| clean_trace.npz | b9a3ba0e5e39aa5afba577cfea096f4050cfdab724424b61623094c7eee14732 |
| replays.npz | d6fef4bed1f7735596d6436d37ce13c82caa3e3c5eed7d2f464d021f8efb72d6 |

Source hashes in the receipt include the full upstream LoopedChase and SEM-ECHO
modules. Local pre-CI tests had used source-preserving excerpts; those excerpts
were never committed. The complete-checkout CI result supplies the native
integration evidence.

## Scope and next empirical dependency

The theory/protocol is [MC02_INTERVENTIONS.md](MC02_INTERVENTIONS.md). It derives
exact counterfactual continuation under a faithful lift, gives the approximate
error recurrence, and states copy-separation and context-blind prediction bounds.
The proofs are handwritten; Lean checking has not been performed.

The external frozen Q4/Q5 checkpoint files were unavailable and were not evaluated.
The runner accepts their existing format and requires an independently supplied
checkpoint SHA-256. A trained run will measure the declared cursor lift's finite
counterfactual trace fidelity, with eligibility coverage and matched controls.
A failing cursor-only lift can coexist with a distributed internal algorithm.

No new Circom compilation, STWO proof or garbled evaluation was run in MC-02.
The code reuses MC-01's exact relation checker. Honest neural execution remains
trusted; the hashes identify evidence and do not cryptographically prove inference.
