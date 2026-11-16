# RAM-17 final source-only report: independently certified but diverse induction gate failed

**16 November 2026. Frozen RAM-17 registration, operational Amendment A, full native TRAIN/DEV evaluation completed.**

The source-collection and triple replay ran under Lean 4.35 on [GitHub Actions #43816331953](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43816331953). **The registered admission gate failed.** The independent third compiler passed all **20 of 20** twice-certified positive original-source scripts. The TRAIN subset comprises **18 positives from 10 theorem roots across five source modules**, including 15 prior-definition simplifications and three prior-lemma rewrites. Two further simplification positives belong to DEV.

| Observed evidence | Count |
|---|---:|
| Source-selected original TRAIN/DEV roots | 63 (51 TRAIN, 12 DEV) |
| Original source action attempts | 326 |
| Fully twice-compiled macro positives | 20 |
| Triple-compiled positive macros | 20/20 |
| Triple-compiled TRAIN / DEV positives | 18 / 2 |
| Nat-induction candidates tested | 69 |
| Theorem roots yielding any Nat-induction candidate | 5 |
| Nat-induction complete source proofs | **0** |
| TRAIN rewrite positives / source modules | 3 / 1 |
| TRAIN simplification positives | 15 |
| RAM-17 model fits and TEST scores | **0 / 0** |

The positive TRAIN modules are `DeferredObservability`, `FundedHooks`, `HookMultiTick`, `NOR09Asynchronous`, and `TacticFibers`. The three certified prior-lemma rewrites occur in `TacticFibers` on two theorem roots. The two DEV examples are from `ObserverPermutationNull`. The known exposed Gramian source contributes zero observations.

The preregistration required at least two induction-bearing TRAIN modules with independently certified source closures, rewrites across two modules, and all three macro families. The observed deficits were two induction modules, two induction roots, one rewrite module, and the absent induction family. The remaining positive-count and positive-module thresholds passed. The failure is a **structural supervision coverage** result, rather than a new proof-search result or evidence for recurrence.

### Diagnostic scope

The registered grammar found candidate induction scripts on **only five of 63** original theorem roots. It recognized a narrow set of explicitly parenthesized Nat binders, and the source-order selection provided little induction breadth. This is an exploratory diagnosis using the unchanged first-run candidate/action receipts, not a causal demonstration that broader induction syntax would succeed. Every one of the 69 generated Nat-induction scripts failed to close a complete original theorem in the first source run. No censored compiler attempt was converted into a negative training label.

The positive results are exact source-level compiler facts. The transcript and byte hashes of both original successful compilations, the independent third source, the native printed initial goals, and transitive permitted-axiom inventories are retained in artifact `ram17-source-first-43816331953` (ID `14203020987`) and `ram17-independent-third-gate-43816331953` (ID `14203055828`). Machine-readable hashes and the source outcome are in [RAM17_RESULT_20261116.json](RAM17_RESULT_20261116.json).

### Stopping rule and next iteration

The source compiler passed its operational run after [Amendment A](AMENDMENT_A.md) repaired intentional goal-probe shutdown handling. The independent native gate was a registered **negative**, so the dependent twelve-fit CPU training workflow stayed sealed. No RAM-17 prospective TEST theorem source was opened. The published RAM-16, RAM-10 and RAM-08 results remain untouched.

A **new** source-locked induction curriculum can extend the binder reader to support leading `∀ n : Nat` clauses and source-visible induction variants, then select induction-relevant original theorem roots from independent source modules. Such an update belongs to RAM-18 with a new registration, fresh provenance, and separately identified prospective tests. Reusing this failed gate as if it were an admitted training corpus would change the original criterion after the observations.
