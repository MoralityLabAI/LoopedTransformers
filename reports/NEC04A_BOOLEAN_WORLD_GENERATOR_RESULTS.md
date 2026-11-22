# NEC-04A: precommitted Boolean-world registry and audited calibration source corpus

**Completed 22 November 2026. Evidence type: deterministic symbolic source-corpus construction and independent oracle audit.** The scientific objective is to support a later experiment on model families with measurable task headroom and physically matched yet functionally calibrated sham interventions. This deliverable builds the source grammar and corpus. No neural model or SAE was trained, and no model baseline accuracy, sham functional inertness or population causal necessity was measured.

## Exact precommit and experimental boundary

The complete generator, new world-root split, public SHA256 seed/layout commitments, independent scalar oracle, test suite, native training plan and hosted workflow were registered **before any calibration source corpus was generated**, at Git commit [71fc43a](https://github.com/MoralityLabAI/LoopedTransformers/commit/71fc43a42615f847cb7430fdae2e215902fca793).

**64 total new NEC04A world roots** are assigned to 24 calibration, 24 developmental and 16 confirmation roots, all distinct from historical NEC03L/P and original four-input confirmation roots. Seed commitments are deterministic and hash-pinned; developmental and confirmation model programs are **not instantiated** by this workflow. Because the registry uses a published deterministic salt, this is a **procedural/authoritative-code embargo, not cryptographic confidentiality**. Host-level CI refuses all development/confirmation materialization.

Original registry SHA256: **88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e**.

SHA256 of the 64 ordered seed commitments: **53f201c94b20699e81f9175fe95d86567dfd1925e2357ae519170c70207f152e**. SHA256 of ordered focal-layout commitments: **e1c69cd56acd5fb8cc62ae688e927743a533b4ef218eec1fdf76664d789b6e83**.

## Frozen grammar and complete 12-bit truth tables

Each original task consumes **five focal input bits** at a unique five-of-twelve arrangement and seven context bits, and computes an executable Boolean expression using VAR, XOR, AND, NOT and conditional MUX. Each expression has a root-specific structural seed committed before generation. The fixed five-focal parity anchor and context-gated alternative expression yield three syntactic tiers:

| Original tier | Calibration world roots | Boolean gate nodes | Program AST depth | Observed ANF degree range |
|---|---:|---:|---:|---|
| Easy | 8 | 11 | 7 | **3–4** |
| Medium | 8 | 15 | 9 | **4–6** |
| Hard | 8 | 19 | 13 | **5–8** |

Every root program passed frozen label-balance **[30%,70%]**, all-five focal-global-input influence, at least three of seven context-essentiality and at least 64 context-group balanced-focal-label requirements. The observed whole-4096-input truth-table positive frequencies were:

- **Easy:** 37.5%–62.5%.
- **Medium:** 37.5%–50%.
- **Hard:** 31.25%–50%.

All **24/24 calibration roots accepted deterministic candidate attempt index zero**, under the registered first-valid-within-64-attempt policy. The generator selected **24 different focal layouts, 24 different program hashes and 24 different complete 4096-example truth-table hashes**. This establishes symbolic variety and held split ownership. Structural gate/ANF metrics are **complexity diagnostics** and do not imply monotonic neural training difficulty.

Each root has **128 seven-bit context blocks**, partitioned deterministically into **80 source-training, 24 developmental-context and 24 reserved-context groups**. An entire **32-focal-mask counterfactual block** belongs to exactly one context split. The generator produced source labels only from the 80 groups, all original source histories unique and disjoint from later source-context splits.

## Hosted immutable source and independent oracle audit

The full [**GitHub Actions run 44983509977**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44983509977) succeeded:

1. **Precommit method and root roster:** the prior Git blob SHA1 of the generator, independent oracle, tests, protocol, draft native training plan and workflow were checked; original seed and focal-layout root commitments were computed to their locked digests; strict only-calibration materialization and complete Boolean syntax tests passed. The first job uploaded GitHub Actions artifact **14739111362** under the name **nec04a-before-materialization-64-root-seed-commitments**, preceding the second job.
2. **Calibration source generation and independent audit:** 24 calibration root programs were generated only after the preflight passed. A **bit-parallel, exact 4096-bit Boolean teacher** computed their full truth tables. The separate independent **scalar AST interpreter** recomputed **all 4096 Boolean inputs per calibration world**, checked complete truth-table hashes, verified all **61,440 source-training labels** and their original root/context/focal-mask identities, and confirmed that the 24 development and 16 confirmation roots remained seed commitments only. Hosted source artifact **14738712603**, named **nec04a-calibration-only-24-worlds-61440-examples-and-independent-oracle**, contains the source dataset, the 24 world programs, frozen 64-root commitments, source manifest and independent audit.
   
Original source dataset SHA256: **82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0**.

The original source corpus contains **24×80×32 = 61,440 canonical JSONL examples**, with exactly **2,560 source cases per calibration root**, preserving original 12-bit histories, source context groups, complete 32-mask counterfactuals, world identity and ground truth. A separate 24-world JSONL catalog carries program AST, SHA256, AST-depth and gate-count, truth table hash, full truth metrics, algebraic-normal-form degree and candidate-attempt position.

**Both hosted CI jobs passed.** No dependencies on PyTorch, native/SAE checkpoint bytes or historical NEC03P world data were required.

## Follow-up for empirical model headroom and negative controls

The [unexecuted native/SAE plan](../mechinterp/neuron_communities/NEC04A_NATIVE_TRAINING_PLAN.json) proposes calibration-root training budgets **120,240,480,900 native model steps** with source-specific 128-feature TopK16 SAE fit budgets **160,320,480**. All 24 calibration roots should be used as fixed independent model-world statistical units, with two native teacher families and frozen source-fit/rank splits within each world. The desired per-teacher, independently measured clean/full-affine accuracy bands are **60–75%, 75–90% and 90–98%**, each represented by multiple independent world roots. Neither the current corpus nor its syntactic tier label establishes those model headroom conditions.

Before any developmental model-root training/evaluation, freeze the selected native training steps/optimizer, checkpoint/SAE hash, task oracle, source geometry/rank, physical sham quality, relative dose variation, and **functional negative-control damage**. Proposed engineering gates are per-root matched-control availability ≥205/256, within-source-root median relative physical perturbation ≥0.05, physical direct 32D output-L2 matching within 1%, no amplification, and sham correct-to-wrong damage ≤2% among full-affine-correct and originally physically matched inputs. Read root-level uncertainty from independent world roots rather than repeated input masks, and preserve zero-effect unsupported cases.

This phase recorded **zero native optimizer steps, zero SAE training steps, zero neural forward passes, zero real-model baseline accuracies, zero causal sham outcome roots, zero developmental/confirmation program materializations**. The proposed model headroom is a hypothesis for a later, separately SHA-locked training study.

Original NEC03Z source groups **56–63** remain sealed, along with old NEC03 root development-evaluation/reserved contexts, all **88** independently recruited NEC03P world-root non-source data, the prior **32 five-input embargoed** worlds and **64 old four-input confirmation** worlds. The NEC03P recruited cohort still has **69/88 paired-source-eligible roots with zero independently measured causal sham outcomes**. Any future nonparametric 95/95 tolerance bound still requires at least **59 independently eligible, causally measured sham root outcomes**, after a source-specific, physically meaningful and functionally calibrated causal intervention passes its developmental gates.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
