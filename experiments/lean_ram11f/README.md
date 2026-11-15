# RAM-11F | Native Lean certificates for historical tactic-fibre splices

**Completed November 15, 2026.** This document is a retrospective supplement to RAM-11's source-only worklist, committed after native verification. It preserves the earlier negative/censored RAM-09/10/11 outcomes and makes no fresh-heldout theorem claim.

## Immutable provenance

- MathCan original **TRAIN** source commit: `c84e2025334e9a967c1297425a490f98d5b3d189` (Lean `v4.35.0`).
- Original RAM-08 independently replayed 20,930 compiler observations: `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`.
- Original RAM-11F 76-script plan SHA256: `f5876539862a0f4f913dc3af620be0d038555efcdd3611754e002dc353c3b2aa`.
- Public MathCan native [successful run #43712039711](https://github.com/patrickdugan/MathCan/actions/runs/43712039711), artifact `ram11f-native-old-mathcan-proofs-43712039711`, ZIP SHA256 `536ebffc38c399b730cb3fe14ebdeb3a6a72590518885a9e52e8cd26f20f9b06`.
- GitHub [MathCan replay PR #76](https://github.com/patrickdugan/MathCan/pull/76) contains a standalone replay implementation, immutable worklist and native Lean GitHub Actions workflow.

The first native job [#43711565504](https://github.com/patrickdugan/MathCan/actions/runs/43711565504) compiled and verified the first 60 candidates, then aborted on original term-mode theorem syntax. The corrected runner added `:= term` proof-body replacement and passed ten local tests. The complete next run successfully evaluated the **original, unchanged 76 candidate scripts**. Both runs are disclosed; the correction was to the source renderer and not to the candidate selection.

## Kernel-checked results

| Measure | Verified |
|---|---:|
| Original MathCan TRAIN theorem-root cases | 22 |
| Proposed alternative proof scripts | 76 |
| Double native Lean4.35 complete-source proof compiles | 152 |
| Rejected proof scripts | 0 |
| Compiler timeouts/unknowns in completed run | 0 |
| Disallowed transitive axioms among accepted proofs | 0 |
| Independent artifact audit of every original script and compilation | 76/76 |

The runner *replaces the original target theorem's proof body* with the candidate script, preserves its source statement and dependencies, compiles the whole original module twice via fresh Lean invocations, verifies byte-identical target sources and the transitive axiom allowlist (`propext`, `Classical.choice`, `Quot.sound`). The second, offline audit independently checked 76/76 generated proof bodies against their proposed tactic lists, two source SHA256s, compiler exit codes, two logs and the printed axiom inventories. Source-target proof copies, synthetic assumptions, `sorry`, `admit`, and all prohibited axioms are excluded from certified cases.

This verifies **76 distinct alternative scripts** for **22 existing theorems**. It establishes neither 76 newly discovered mathematical theorems nor capability transfer to the untouched prospective tests.

## Exact-prefix completion lemma

Let a particular Lean source environment `S` elaborate the original theorem statement `P`. Suppose a specific script `A = [a₁,...,aₖ]` replaces the original proof body and its resulting theorem term is accepted by the Lean kernel. For each `i < k`, the exact script prefix `[a₁,...,aᵢ]` is **extendable** under the same source environment by suffix `[aᵢ₊₁,...,aₖ]`. This follows because concatenating those literal lists reconstructs the source script whose complete elaboration was checked. The result is an existential completion witness for that exact *script prefix*. It does **not** assert transfer of the suffix to another prefix with the same printed goals or provide a new snapshot of each intermediate Lean state.

**Corollary on the certified cohort:** the 76 accepted scripts contain 282 tactic occurrences. Deduplicating exact `(original root, script-prefix, next action)` pairs produces **230 positive sequence labels across 176 distinct script-prefix contexts**. A comparison to the independent RAM-08 TRAIN compiler corpus finds 138 already observed positive actions (44 label-1 and 94 label-2) and **92 additional script-prefix/action pairs** absent from that original collector. All 92 have a valid complete-script completion witness. Their intermediate Lean goal states have not been newly extracted. None is a fresh negative action label.

The verified curriculum is stored in `curriculum/` with SHA-pinned JSONL files and exact proof IDs linking each record to the original output archive. Each row is labeled as *source-verified existential completion*, without suggesting a per-step compiler observation or solved new heldout theorem.

## Scope and next experiment

This is source-only historical TRAIN data: it has **zero overlap with the original RAM-08 DEV theorem roots**. The RAM-08 PEF 120-case holdout and the RAM-10 untouched source-family heldout both retain **zero scores from this audit**. The old graph's goal-quotient path-splice warning remains correct in general: the successful 76 examples were selected for testing and verify specific concatenations, not a blanket quotient-state transport theorem.

A future compact reasoner can use the 230 positive sequence labels to train prefix-conditioned tactic selection, with a separate preregistered source-family holdout, an independently verified compiler-state gym, and complete kernel-checked proof certification. Preserve source-hash and theorem-root boundaries before any new training fit or hyperparameter selection.

## Reproduce the independent audits

Use the public native Actions artifact, original RAM-08 frozen archive and original public MathCan locked worklist:

```bash
python experiments/lean_ram11f/independent_artifact_audit.py <ram11f_native.zip> <locked_worklist.jsonl> <independent_audit.json>
python experiments/lean_ram11f/build_curriculum.py <ram11f_native.zip> <locked_worklist.jsonl> <independent_audit.json> <new_empty_corpus_dir>
python experiments/lean_ram11f/overlap_audit.py <ram08_frozen_train.zip> <certified_positive_prefix_action_labels.jsonl> <overlap_result.json>
```

The checked-in receipts preserve those exact already-run results. The raw native archive remains available through MathCan Actions, including all 152 distinct compiler logs and generated source texts.
