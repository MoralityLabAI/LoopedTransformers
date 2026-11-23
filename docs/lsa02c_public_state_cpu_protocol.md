# LSA-02C: Typed public-state encoding versus raw JSON on fresh Lean roots

**Registration and implementation:** November 23, 2026. This is a CPU-only development experiment, following the [LSA-02B native teacher](lsa02b_cpu_training_pilot.md) and [root-weighted fork audit](../reports/lsa02b_cpu_fork_diagnosis_20261123.md). Locked `ordinary_test` and `ood_family` data remain unopened. No architectural superiority or mathematical theorem is presumed.

## Motivation

On 24 older training lineages, 73 of 124 decision frames had singleton public tactic menus. On 51 discriminating forks, the best *in-sample* trained checkpoint chose a shortest continuation only 51.4% of the time against a 39.4% root-weighted uniform-menu baseline. The trained models completed only 1/8 or 2/8 of the eight exposed native calibration roots versus 6/8 attainable. That result justifies testing whether the input representation leaves the key premises inaccessible.

## Matched comparison

Both conditions use the *same* `PlasticProofTransformer` Plan B, **AB block schedule**, tied parameter objects, byte encoder, eight role slots, width 32, four heads, two recurrent passes, AdamW and source task. A single explicit `input_view` checkpoint field controls the public goal text seen by the model.

- **Raw:** historical `state_input` canonical JSON, with the existing `ByteEncoder` tail crop at 384 UTF-8 bytes.
- **Typed:** deterministic, label-blind public type serialization. It preserves the current goal, pending open goals, remaining tactic checks and failed choices, then writes each named candidate's public hypothesis type and additional context hypotheses in graph-distance priority. The typed packet fits within the **same 384-byte encoder ceiling**. This is a bounded, potentially lossy abstraction, with every omitted declaration determined entirely from public types and the byte cap. It uses no shortest-proof lengths, private labels, outcomes, `thm.gen` or evaluation-root evidence.

The typed string is grounded in the same Lean `state.goals[].target` and `hypotheses[].{name,type}` values available to raw JSON. Each candidate action string is also model-visible in both arms. Unsupported private fields or type grammars abort the encoder. Format version is `lsa02c.typed_public_state.v1`, persisted with the checkpoint and checked before native inference.

## Fresh root provenance

Freeze a **new G03 suite**, with seed `20261124` and 64 roots per split, before collecting any outcomes. From its `train` shard select the first **48 independent lineages** by generator order. `lcgrt.extract.explore` builds Lean-reported shortest proofs and candidate transitions. Each reference script must pass native Lean 4.19 and an independent standalone axiom audit. The typed encoder never consumes these verified proof labels; labels enter only the common training loss, after the data corpus is frozen.

The **first 32 calibration lineages** belong to a disjoint split and are reserved for comparison. All models evaluate on exactly those roots. The unused rest of the training shard and the full `ordinary_test` / `ood_family` shards are hash-checked for integrity by the shared loader but never scored.

## Registered CPU matrix

| Axis | Contract |
| --- | --- |
| Independent model seeds | 107 and 109 |
| Views | raw, typed |
| Training order | AB for every cell |
| Initialization and source draws | identical within each seed pair; hash-verified |
| Model scale | Plan B width 32, 57,475 trainable parameters, 4 heads, 384-byte input |
| Exposure | 720 optimizer steps per cell, AdamW 0.001, gradient clip 1.0 |
| Depth | 2 tied block passes, same full-pass source supervision |
| Source | 48 independently Lean-certified training roots |
| Native development evaluation | 32 independent fresh calibration theorem roots per cell, 8 tactic checks |
| Controls | frozen AB/BA within checkpoint, public-order `auto8`, independent Lean shortest-script attainability |
| Compute | GitHub Actions CPU-only, at most 2 concurrent fitting/probe cells, 2 CPU threads on fit, one for probing |
| Outcome boundary | valid null result is accepted; no automatic promotion to larger model or gym |

The four cells are paired by **theorem root**, with repeated seed observations treated as dependent within a root. Primary endpoint: native kernel-and-standalone-certified full-proof completion under the *trained AB* schedule, typed minus raw for the same seed and same root. Secondary: strictly discriminating **training**-fork accuracy (root weighted, with uniform-menu baseline); typed byte efficiency, source collisions, task/byte truncation rates, first-action disagreement and reference-reachable fraction. A root-bootstrap on mean-seed paired effects is descriptive.

**Predeclared development threshold:** at least +0.05 absolute full-proof closure improvement over raw, with nonnegative paired effects at each seed. This is a screen for further study, not a confirmatory significance claim. The prior formal architecture competence gate remains separate: at least 32 eligible independent roots, trained-order close rate ≥0.40, and ≥0.10 over a **registered uniform-menu** benchmark. `auto8` is a public-order baseline, **not** that uniform-menu benchmark. Thus this run cannot by itself satisfy the full competence gate.

## Source custody and failure rules

Every checkpoint includes the source manifest SHA-256, training seed, input view, representation-schema version, dispatch order, initial-weight fingerprint, optimizer draws, step count and loss record. A representation that drops the current goal or exceeds the 384-byte contract aborts. Measure input collisions with conflicting labels on the source, preserving counts in the fit receipt. Retain all new calibration roots regardless of outcome.

The native runner authenticates the training census and disjoint theorem ancestry before scoring. Independently compile every successful model proof, separately compile the model-free reference scripts, and replay each action log without the neural model. If any paired root, proof certificate or artifact hash fails, the workflow fails; report as construction/verification failure instead of altering the frozen cohort.

## Execution

The [CPU workflow](../.github/workflows/lsa02c-typed-state-cpu.yml) collects the teacher, trains both views at the two pinned seeds, evaluates four checkpoints on the same fresh native roots and produces a final SHA-bound paired report. It consumes GitHub-hosted CPU/RAM, plus Lean 4.19 and CPU-only PyTorch 2.5.1. No GPU, private local machine or paid LLM calls are required. Intermediate artifacts expire according to the configured 14-day GitHub Actions retention; the report and per-cell hashes should be committed after the run.

A valid result may find either view equivalent, or that compact typing improves only in-sample fork prediction while failing to improve full proofs. In both cases the study precedes any new gym or architectural scaling.
