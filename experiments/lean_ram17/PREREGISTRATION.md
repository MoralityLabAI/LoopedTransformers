# RAM-17 | Diverse source-root structural curriculum and paired Lean proof search

**Registered on 2026-11-16. Phase: predata design.** Based on RAM-16 merged commit `840f8e1688d0577e3ea64f9c4642d9d327dce87a`. RAM-16's first prospective theorem search failed its recurrence-specific objective (Looped64 task-only 6/10, structural 5/10, native 6/10). RAM-17 makes that failure its control, while changing *where* independently compiler-certified structural positives arise.

## Frozen source separation

All source hashes and module assignments are in [config.json](config.json); the source tree and blob SHA-1 values were selected using Git metadata. TRAIN/DEV source is `patrickdugan/MathCan@f427e0ef4a61093304d12ca2c0dc943d8d2429e5` (Lean `leanprover/lean4:v4.35.0`). TEST source is `patrickdugan/MathCan@3749c84cc6d47297faaf94404af6d1fd36916812`. The original source selectors take the first four TRAIN and first three DEV top-level theorem/lemma declarations per file, ignoring nested comments and strings. Ten TEST candidates are the first two declarations in each of the five registered TEST modules, selected **after checkpoint freeze**, including two chronologically new NOR modules.

TRAIN modules (14): BasinCodes, BlockRemainder, ChromaticCutCodes, DeferredObservability, DeletionMemory, FiniteHorizonGap, FundedHooks, HookMultiTick, NOR07Volterra, NOR09Asynchronous, PhaseGaugeCapacity, PrecisionEntropyRobust, SpectralAliasing, TacticFibers.

DEV modules (4): InvariantBall, OpcodeRefinement, TOEYangMills, ObserverPermutationNull.

Sealed TEST modules (5): OrderedTangent, SpectralStability, PhaseBlindObservers, NOR19StreamedVerifier, NOR20BigintWidthBounds. `NOR19StreamedVerifier` and `NOR20BigintWidthBounds` are absent from the pinned old TRAIN tree. The three earlier TEST modules must be physically removed from the TRAIN checkout alongside RAM-16's four older protected TEST modules. Nothing under the prospective TEST source may be read, selected by proof content, or imported before every paired checkpoint is frozen. During training, purge original RAM-08 transition rows from all five RAM-17 TEST modules, even if they appear in the historical archive. Stop if any TEST-root statement hash or source file reaches TRAIN/DEV.

Each of the 14 TRAIN and four DEV modules is distinct from the modules used in RAM-16's macro TRAIN/DEV split. `GramianTransfer` is absent from this new curriculum and cannot satisfy its admission gate. This is a whole-module disjoint transfer protocol; chronological novelty applies to the two NOR modules only. Root selection is mechanical; no difficulty filtering by scored source proof is permitted.

## Original compiler labels and admission gate

Reuse the identical RAM-16 / RAM-14 source-header-only candidate grammar: at most 48 scripts per theorem root and at most 16 per each of `nat_induction`, `simp_prior`, and `rewrite_prior`. All source scripts must be generated from visible target statements and preceding declarations only. Replace the reference proof, then compile the full MathCan source twice in separate Lean 4.35 processes. Reconstruct the original source, native printed goal, logs, source SHA, and allowed transitive axioms in a separate third compiler audit. Only a genuine three-compile closure is a positive `label=3`; failure/timeout is censored. Preserve every attempted source, outcome, and process log, including zero-result families.

Before any training or TEST opening, require eight triple-certified TRAIN positive actions spanning at least four different source roots and four TRAIN modules; `nat_induction` must close in at least two modules and two roots, `rewrite_prior` in at least two modules and two roots, with at least one `simp_prior` closure. If any gate fails, publish the negative TRAIN-only certificate and stop. Keep the original source and toolchain pinned and enforce the 90-minute collection wall cap, 15-second per compile cap, and 6 GiB resource guard. Pre-register any operational repair as a dated, separately labeled amendment.

## Paired CPU learning

Start with the same original audited RAM-08 task effects, 20 old family IDs and 23-row RAM-16 embedding initialization. Two architectures: tied four-pass width-64 Looped64 and matched flat width-71, roughly 207k and 208k parameters. Each architecture crosses the task-only objective and ten-percent verified structural curriculum for **all three paired seeds 17, 29, 43**. Run 400 AdamW steps per fit (batch64, 2 CPU threads, LR 0.002, weight decay 0.01, clip 1); 12 total fits. Check byte-identical initial state tensors for paired objective arms and archive both best and last checkpoint hashes. DEV checkpoint selection uses only the pinned task DEV loss, evaluated every 100 steps, with earlier update as tie breaker.

For the 2,560 structural-positive slots, sample round-robin over present macro families, then independently cycle source modules and theorem roots within each family, then select a certified script within the root via seeded shuffle. Record observed sampling counts by family, module, and root; enforce a maximum one-slot family imbalance. Remaining batch slots sample the original audited task effects. The task-only arm uses its own original task sampling and the same training budget. The curriculum serves only positive action supervision, without invented negative labels.

## Original TEST outcome and paired criterion

After a publicly committed source-only freeze records the twelve checkpoint byte SHA256 values, paired DEV histories and exact TEST declaration names, evaluate the **same 16 controllers** on each original TEST theorem: native source ordering, TRAIN global frequency, TRAIN shape frequency, the historical frozen recurrent model, and all 12 architecture/objective/seed combinations. Every policy gets the same original 23-family concrete tactic pool, 30 s / 48 requested Lean actions / 10 accepted transitions / eight pending goals / 3 s per request / two restarts, with two full-source compiles inside each policy wall deadline. A separate independent compiler reruns every accepted proof for the third compilation and checks the transitive axiom list. Save failures and source bytes as well as successes.

**Registered recurrent-specific criterion:** at least one original TEST theorem is certified by structural Looped64 for at least two of the three paired seeds, while native, global-frequency and shape-frequency controls all miss it and task-only Looped64 and structural Flat71 miss it at those same seeds. No false full-source certificates. Report an architecture/objective-specific outcome for every seed and theorem, alongside the per-seed paired change in theorem coverage, total successful action requests, elapsed wall time, and the across-seed mean paired change. A new success cannot come from a test theorem scored in RAM-16. Report a negative outcome verbatim if the criterion fails.

Secondary 60 s / 96 requests and old-20-family grammar ablations are descriptive; their repairs remain distinct from the original primary. Preserve score provenance from first prospective attempts, exact request budgets, executor errors, and source replay hashes. No pretrained checkpoint or TEST proof oracle may contribute candidate ranking features. Preserve the original RAM-16, RAM-10, and RAM-08 results without modification.

## Current status

This registration freezes sources, sampling, action limits, and success criteria. It is **not** a native Lean proof result, source-gate admission, trained checkpoint, or prospective evaluation. The source-only collector, third audit, training and evaluator must produce explicit receipts before any empirical claims are made.
