# RAM-16 | Original Lean structural-macro supervision and fresh theorem transfer

**Preregistered 2026-11-16 before reading TEST theorem statements or goal states, and before collecting new structural training outcomes.** Follow-on to RAM-15's negative recurrent-only result. The RAM-14 Gramian outcome is previously exposed and belongs exclusively to TRAIN.

## Immutable sources and split

TRAIN/DEV source: `patrickdugan/MathCan@f427e0ef4a61093304d12ca2c0dc943d8d2429e5`, Lean `leanprover/lean4:v4.35.0`. Select first **six top-level theorem/lemma declarations** per module, in original source order; fewer if unavailable.

| TRAIN module | Git blob SHA-1 |
|---|---|
| Intervention | 7d522bca485cc5a877a65897ce65246ba5b0d52b |
| Hodge | 1207f7604ab9e04188188764b80d655866a5fa71 |
| HookState | bd147aab11aa0c39f74ffc34199500cc89e4cdf9 |
| TOEGraph | c3aee2943e9f53739de097756bc4608e339eb182 |
| GramianTransfer | f07e99a3f27acba22c2c0c6b3f675a68ab774ab2 |
| FilteredOrbit | 050583a1c578cf208a57e3d3e88955784aac33b3 |
| OrbitStitching | a7d2c88e2e4b7f5757df8adf1303e0a1c40028bf |
| GraphAuthorization | 502ada45d84639a926267d950288806e6b604082 |
| MultiBasin | 20b1109dd0c98220e774c677e895ae1edf41f5d8 |
| PhaseMemory | 615f20a32b0e190408ebee0b3b2a4f9bb695ee02 |

DEV, whole-root disjoint: BasinCapture `60182b4e1f8e92bbec5ba7af17b3c4540ca70202`; TreeReconstruction `1e493152fcba3c5702d61efdfd3625734fe024a6`; MemoryProjection `fa1a8d2522452e7dd366fac8a4f62c43ac91c27e`. Each uses the same first-six selector, restricted to DEV selection.

TEST source: `patrickdugan/MathCan@fdbe1581e95427016699169fd19c9e915a40bcf9`. Select first **two top-level theorem/lemma declarations per TEST module** after stripping nested Lean comments and strings, for up to 10 original TEST cases:

| TEST module | Git blob SHA-1 |
|---|---|
| NOR18RReciprocalReuse | 6be0c1cdd1ceb1068399888cc55729be4412c293 |
| TemporalCutCapacity | 620d901005ed025368ebe0a2a12ecfc833ebe05c |
| PhaseLatch | c4589a577dd4a50b35ea80f0c0350e258bd27d5f |
| ObserverLabelAlignment | 0e357909b9a012eb27a905fea2258337f7210ca8 |
| ContractionAliasing | e9eb7911bec05ee05c83d3dcc970cffc99bab3d1 |

Exactly **one** TEST module, NOR18RReciprocalReuse, is new since the TRAIN source; the others are source-file-disjoint and excluded from RAM-16 TRAIN/DEV despite existing earlier. This is a **module-disjoint prospective test with one chronological new module**, not five time-new source families. These choices were made from Git tree paths and blobs alone. Before TRAIN collection, remove the four older protected TEST module files from the TRAIN checkout and build only whitelisted source modules/dependencies, aborting on any protected import. Do not access any TEST goal state/proof body until all training gates and checkpoint freezes pass. Source oracles must never supply proof commands to student candidate generation.

## Independently compiler-certified structural labels

Use unchanged RAM-14 source-header-only Nat induction, prior-lemma rewriting and prior-definition simplification grammar, pinned to MathCan RAM-14 merge `0e899d07d077146973c20ba7777a2d7d4e9b8abd`. For each source-selected TRAIN and DEV root, generate at most 48 candidate macro scripts, at most 16 per structural family. For each candidate reconstruct the native printed Lean goal list and visible definitions before the action, replace only the target proof, and compile the entire original MathCan source under Lean4.35. **Positive label 3** requires an independently successful second complete-source compilation, identical bytes and permissible transitive axioms (`propext`, `Classical.choice`, `Quot.sound` or none), followed by a **third post-collection compiler audit** of all positives. Compilation failure or timeout is *censored nonclosure*, not a verified rejected single-step action. Never invent negative state-action labels.

Registered TRAIN admission gate (before any model fitting or TEST checkout): at least **two distinct TRAIN roots** with triple-compiled `nat_induction` macros, at least **one TRAIN root** with triple-compiled `rewrite_prior`, at least **six unique** new-family positive actions across **three TRAIN roots**. Report all family counts, including zeros; report the share from the known GramianTransfer roots. If the gate fails, stop with a named negative TRAIN-only receipt and **0 new TEST cases**. A source or compiler timeout retains its unknown status. Collect up to 35 minutes on GitHub-hosted 2-CPU-thread/6GiB RAM, 15s maximum native compiler per candidate.

## Matched learning experiment (conditional on gate)

Original RAM08 independently replayed training ZIP SHA-256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`, TRAIN compiler effects SHA-256 `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`. Retain old 20-family row order and labels; expand the model action-family embedding to **23** families, preserving the first 20 IDs. The three new rows initialize as the corresponding original-family semantic proxy rows plus deterministic zero-mean noise (standard deviation 0.01). Compare a tied four-pass recurrent width64 model and a **parameter-matched flat width71** model under two objectives each: task-only original RAM08 supervised labels, and identical task plus **10% separately certified source-macro positives**. No negative from unknown actions.

Use **three seeds 17/29/43 per model/objective**, **twelve total training fits**, each 400 CPU AdamW updates with batch64, LR0.002, weight decay0.01, clip1, deterministic 2 CPU threads. Within each architecture and seed, both objectives start from byte-identical initialized weights; explicitly SHA-check. Fit data never include TEST sources. Evaluate DEV every100 updates, and choose the minimum fixed weighted 4-class CE + 0.2 signed-goal-delta SmoothL1 loss; ties by smaller seed then earlier step. Freeze all best/last checkpoints, parameter totals, per-seed DEV metrics and their hashes before TEST access. Distinguish absolute task-only model performance from paired structural-curriculum gains.

## First TEST proof-search protocol

Eight primary policies on every original selected TEST theorem: original native source-order control, original TRAIN-only global-frequency, original TRAIN-only shape-frequency prior8, recurrent task-only, recurrent+macro supervision, parameter-matched flat task-only, flat+macro supervision, and historic unchanged RAM08 loop64 with frozen family proxy. All receive **the same source-visible 23-family concrete tactic pool**, at most 32 additional macro candidates per state. Fixed 30s/48 genuine Lean action requests/10 accepted transitions/8 pending goals/3s request/2 restarts per policy/theorem. Complete original theorem requires **two** byte-identical full-source compiles inside each individual wall deadline, plus a separate independent **third** Lean4.35 source/axiom replay after scores freeze.

Secondary: exact same TEST cases with native, shape-frequency, recurrent+macro and flat+macro at 60s/96 requests/16 transitions. Original-20-family ablation: native and recurrent+macro at 30s/48 requests. Save complete all-failed cases and CPU, call and process accounting. Report matched model sizes and per-theorem paired effects.

**Primary success criterion:** a theorem certified by recurrent+macro and missed by all three strong nonneural primary controls **and** by recurrent task-only, with zero false certificates. Recurrence-specific success further requires flat+macro to miss the same original theorem. No TEST-driven threshold, seed or grammar changes. Prior RAM13/15 target roots are excluded from prospective credit. The original RAM08 PEF and RAM10 first registered tests remain unscored by this experiment.

Theorem source and private model data stay in private LoopedTransformers Actions. MathCan public CI supplies compiler/source only; no private checkpoints or historical compiler corpus may be transferred to MathCan. Archive all native logs, label and checkpoint hashes, original full proofs and independent compiler audit. This protocol does not authorize merge or paid compute.
