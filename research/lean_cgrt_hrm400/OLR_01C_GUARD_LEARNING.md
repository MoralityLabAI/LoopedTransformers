# OLR-01C: native arithmetic-guard learning pilot

**Date:** 2026-11-23  
**Status:** Protocol and executable development pilot. Model outcome must be read from a completed native Lean CI run before any success claim.  
**Parent:** [OLR proposal](OLR_01_OBLIGATION_LEDGER_RECURRENCE.md), [OLR-01B structural result](OLR_01B_NATIVE_PREFLIGHT.md).

## Research question

Can a compact, weight-tied two-obligation recurrent model learn to choose between native Lean tactics whose propositional shape is identical while their computational guard differs? The ordinary Horn graph has two equal-arity rules and no way to decide which numerical equality Lean will close. A model must extract information from the public guard expression, then spend its two-check allowance on the correct rule and `rfl`.

The first pilot uses a closed arithmetic grammar: a candidate `rN : (a + b = c) → P`, where the two choices share `a,b` and one has `c=a+b`. Both worlds in a pair exchange the guard assignments between r0 and r1. The exact arithmetic oracle is a strong reference; a neural win against a fixed lexical tie is weak evidence unless it also beats direct or isolated readers. This grammar can teach the network to approximate a hand-coded arithmetic test and makes no general Lean claim.

## Source custody and native execution

[ppt_gym/olr_guards.py](ppt_gym/olr_guards.py) generates deterministic paired roots. Each of the two root tactics is physically tested through the pinned Lean 4.19 REPL. A successful root first advances to an arithmetic subgoal; Lean's `rfl` must close it and `kernel` must accept the full theorem. An unsuccessful route earns a *two-tactic menu failure* label; the `rfl` failure is not interpreted as a proof that the guard proposition is false.

A separate whitelist-only compiler accepts only a generated closed theorem statement and the pair `apply r0|r1; rfl`. It reports independent, axiom-free compilations. Candidate labels are defined by native outcomes and the original reference winner is used only for auditing. The ranker receives public normalized numerals `a/40, b/40, c/80` parsed from Lean's `raw` guard JSON, with no computed sum, equality bit, source verdict, or proof script.

The source bank uses 32 distinct source pairs and a new 12-pair development cohort, with another eight pairs from an operand range unseen in training. Both worlds of each pair remain together in their split. The generator excludes duplicate arithmetic lineages across train and dev. The main 20261124 seed, splits, and model seeds were set in CI before model scoring.

## Trainable architecture and controls

[ppt_gym/olr_guard_model.py](ppt_gym/olr_guard_model.py) embeds each candidate's three public operands, initializes two obligation slots, and updates them through a **shared/tied recurrent cell**. Each cell sees immutable encoded context, its previous slot, a message from the other candidate, and an unknown-evidence embedding. It fits a gate and produces one tactic score per slot. Repeated recurrence is the algorithm clock; native feedback only arrives after an actual action.

Four independently trained arms share source data, seed, minibatch indices, loss, optimizer and number of updates:

| Arm | Architectural change |
| --- | --- |
| cross | Tied recurrent cell exchanges cross-candidate messages |
| isolated | Same trainable parameters and compute path, cross message set to zero |
| direct | Same source encoder and output head, bypasses recurrence; lower-compute control |
| frozen_cross | Tied recurrence initialized identically but frozen, source embed and readout trained |

The direct and frozen-body controls have different *effective trainable* parameter counts. Retain exact counts from the source-fit receipt and compare per-decision actual compute in a follow-up. This pilot does not establish a strict FLOP-matched architecture winner by itself.

Loss is paired 2-class cross entropy over the unique native-verified completion. Learning reads **train source only**; dev and OOD labels are used only for consistency checks *after* each policy decides. Three model seeds (17,29,43), 80 updates, width 32 and recurrent depth 3 form the initial smoke. All model outcomes on query roots are executed again by Lean. Identical selections can share a native replay within a root while all model decisions are separately logged.

An independent numeric oracle using exact host addition and an r0-fixed guess make the ceiling and lexical tie explicit. All arms receive the same public numerals; model feature engineering itself is a simplifying computational prior.

## Primary measurements

Report individual kernel proof completions out of 24 development worlds and 16 out-of-range worlds, plus both-world pair completions, native tactic calls, source accuracy and parameter count. Both proof worlds in a pair count as **one independent lineage** in uncertainty analysis. The exact oracle should close every world with the fixed two-call budget. A useful learned recurrence signal requires complete-pair gains over isolated/direct/frozen controls across seeds, with fixed query samples, and more than an exact-information or runtime advantage. No sealed cohort has yet been opened for scientific confirmation.

A mechanism experiment must further patch the cross-candidate message direction while matching its logit-margin perturbation to a sham. The structural hypergraph control from OLR-01B remains an upper bound for fully known Horn dependencies; OLR-01C has unknown computational guards and tests only their *initial action selection*.

## Execution

From `research/lean_cgrt_hrm400`, with the native Lean 4.19 REPL built and CPU PyTorch installed:

~~~sh
python -m unittest discover -s ppt_gym/tests -v
python -m ppt_gym.olr_guards --seed 20261124 --train 32 --dev 12 --ood 8 --out runs/olr-01c-bank
python -m ppt_gym.olr_guard_fit --bank runs/olr-01c-bank --out runs/olr-01c-fit --steps 80 --depth 3 --width 32 --seeds 17 29 43 --max-seconds 600
~~~

The runner writes source and query native JSONL, immutable hashes, source-fitted model states, paired policy decisions, standalone compiled source, and kernel receipts. GitHub Actions uploads the complete bank and result directory under **olr-01c-native-guard-pilot** for audit. The CI step asserts source/eval custody, a full exact-oracle ceiling, the expected sample/seed counts, independent query certificates and that the result is *not architecture-promotion eligible*.

## Exit gates

The next phase should train on ground numerical/computational judgments with more than two candidate moves, multi-goal pending proofs, resource-constrained rollback and public raw/unknown guard statuses. It must compare with a direct supervised guard reader and exact host fold under equal data and compute, then register a fresh confirmation cohort following the EvalMath FSI-01 mechanism/sham audit. A reproducible neural gain beyond an equivalent direct reader would justify replacing the tiny independent guard scorer with a full graph-aware obligation ledger.
