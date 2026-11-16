# RAM-14: source-visible tactic generation resolves historical GramianTransfer failures

**Date:** 16 November 2026  
**Status:** Independently triple-certified **posthoc diagnostic**; original RAM-13 prospective benchmark unchanged.

## Finding

[RAM-13](../lean_ram13/RESULTS_REPORT_20261115.md) completed 156 Lean4.35 policy/theorem searches against a frozen twelve-theorem MathCan source-family cohort. Every original policy missed both `GramianTransfer` roots, while native, global frequency, Looped64 and Flat64 each solved ten of the other ten. The last two failures therefore offer a discriminating *action-space* test. Their gold proof source and failure outcome were inspected before the RAM-14 design.

RAM-14 reuses exactly the pinned original public source `MathCan@f427e0ef4a61093304d12ca2c0dc943d8d2429e5`, Git blob `f07e99a3f27acba22c2c0c6b3f675a68ab774ab2`. It generates no reference proof commands and inserts no extra axioms. The proposed tactics are derived from (a) typed target statement binders, (b) previously declared definitions/lemma statements, (c) inverse-law structure field names.

Three new structural action families were tested:

| Structural generator | Candidates | Independently triple-compiled full proofs |
|---|---:|---:|
| `nat_induction` | 18 | **2** |
| `simp_prior` | 12 | 0 |
| `rewrite_prior` | 5 | **2** |
| **Total** | **35** | **4** |

**Two distinct original MathCan theorems** were proved. `transport_iterate` has 1 new proof, and `observable_transport` has 3. The full original-source certificates each passed twice in [first MathCan CI 43791363290](https://github.com/patrickdugan/MathCan/actions/runs/43791363290) and a **third** time in [independent CI 43791713791](https://github.com/patrickdugan/MathCan/actions/runs/43791713791), totaling 12 successful complete-source compilations. All allowed axiom inventories contained only `propext`; 31 candidates rejected, zero unknown, zero false certificates, eight standalone source-leakage tests passed.

The compact, type-derived **induction + normalization** certificate:

```lean
induction n generalizing x <;> simp_all [iterateState, transport, g.left_inv]
```

was accepted on both targets. The previous-lemma tactic

```lean
simp [transport_iterate, g.left_inv]
```

proved `observable_transport`, with another accepted `simpa` variant. The corresponding gauge-conjugacy identities are mathematically standard; RAM-14's result concerns an *algorithm for proposing kernel-verifiable proof scripts*.

## What failed in the original model experiment

The 20-class candidate grammar in [RAM-13 `grammar.py`](../lean_ram13/grammar.py) offered `induction_list` only when a compiler-printed local context contained a List, and `rw_local` only when it contained an applicable named local hypothesis. Its original templates could not directly select induction over a visible `n : Nat` or cite the already-proven `transport_iterate` lemma at a subsequent theorem root. Increasing a search budget over the **same candidate grammar** did not solve these targets.

This deficiency is logically prior to model ranking: a policy only reorders the candidate set it receives. For a registered state `s` and grammar `G`, let `C_G(s)` be its concrete generated actions. If a proof requires an action sequence outside the closure of `C_G`, improving the score function over `C_G` alone cannot manufacture the missing actions.

The new one-shot structural proof macros are **not** comparable to the original 20s/32-evaluation RAM-13 searches as a matched causal test. The RAM-14 candidates were designed after the original test and reference source became visible. The reported 2/2 theorem coverage is an audited retrospective proof-generation result, with **zero** credit in RAM-13's original registered performance.

## Next: RAM-15 prospective shared-grammar arena

Freeze a fresh, entirely different MathCan module cohort **using paths and source blobs before reading goal states**, excluding every RAM-08, RAM-10, RAM-13 and RAM-14 original TEST root. Use a *shared concrete grammar* across original native/global-frequency/shape-frequency controls and recurrent/flat policies. The new grammar may add `Nat` induction, source-header prior-lemma simplification and the generic one-shot structural tactic macros; rankers may only reorder concrete proposals.

### Explicit model-compatibility decision

The original RAM-08 recurrent/flat policies use a **20-row learned family embedding**. The three new RAM-14 family names require a new embedding or a transparent, frozen semantic proxy:

| Extended candidate | Original embedding proxy |
|---|---|
| `nat_induction` | `induction_list` |
| `rewrite_prior` with `rw [...]` | `rw_local` |
| `rewrite_prior` with `simp [...]` / `simpa [...]` | `simp_all_only_visible` / `simpa_local` |
| `simp_prior` | `simp_all_only_visible` |

The proxy preserves all original parameter bytes and exposes the script tokens to the original action tokenizer. It is a *frozen inference adapter*, not evidence of having trained for Nat induction. The alternative is new training with a wider 23-family embedding, requiring an independent train/DEV split and new source-only labels. Preregister this choice before opening fresh TEST goal states, and disclose the added candidate count and consumption of each theorem's real Lean budget.

**Safety/integrity:** no private original checkpoint or compiler TRAIN corpus goes into public MathCan. The original RAM-08 PEF and RAM-10 first registered assessments remain separate and unscored.

## Sources

- [RAM-14 MathCan generator, protocol and receipt](https://github.com/patrickdugan/MathCan/tree/research/ram14-structural-lean-grammar-20261116/research/ram14)
- [First native proof run 43791363290](https://github.com/patrickdugan/MathCan/actions/runs/43791363290)
- [Independent third compilation 43791713791](https://github.com/patrickdugan/MathCan/actions/runs/43791713791)
- [Original RAM-13 negative result](../lean_ram13/RAM13_RESULT_20261115.json)
