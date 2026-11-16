# RAM-15: prospective expanded structural grammar on new MathCan NOR theorem families

**Preregistered 16 November 2026 before inspecting this cohort's theorem statements, original proofs, compiler states, or candidate outcomes.** The protocol and exact module Git blob identifiers below were selected using a Git tree-path-and-blob inventory comparison alone. The older study [RAM-14](../lean_ram14/GRAMIAN_DIAGNOSIS_20261116.md) examined GramianTransfer and is a posthoc diagnosis. RAM-15's original benchmark must not include those two exposed theorems.

## Frozen new-source cohort

Original public MathCan source commit: `0e899d07d077146973c20ba7777a2d7d4e9b8abd`; Lean `leanprover/lean4:v4.35.0`. The seven listed `MathCan/*.lean` modules did **not exist** in the previous RAM-13 source commit `f427e0ef4a61093304d12ca2c0dc943d8d2429e5`, as confirmed by comparing Git tree paths only.

| Registered source module | Git blob SHA-1 |
|---|---|
| `MathCan/NOR15MaskCapacity.lean` | `7283488dc452135e02cfae126ceae2873896a562` |
| `MathCan/NOR15TrajectoryObservability.lean` | `45c8e4eb2a325b85d063144041b3e4ab913a3de8` |
| `MathCan/NOR15VerifierAlias.lean` | `b6240c8f8bb0d2f5e4552657fdeee43ef13b2b4a` |
| `MathCan/NOR16PublicOutputCertificate.lean` | `dafaaf3f89b21e1f21cabbcdc45a1fb3bb1b5f54` |
| `MathCan/NOR16PublicQueryConsensus.lean` | `8f8d8e8733f301e6a1898f8d32a512a4ff48266a` |
| `MathCan/NOR17BudgetedPublicCertificate.lean` | `df4542309bd6794627bc7f7f05f11b5c67e2308a` |
| `MathCan/NOR18CachedVerifierWork.lean` | `908873c5ab43263400e788ae37833353f52c4dbc` |

Frozen target selector: **first two top-level `theorem` or `lemma` declarations in source order from each module**, ignoring Lean nested comments and quoted strings; at most 14 cases. A module with fewer than two eligible declarations contributes whatever it contains; never replace missing cases or select a later theorem posthoc. Record the exact names and SHA-256 in a locked pre-scoring manifest immediately after reading the source and before extracting any goal states or evaluating candidates. These source modules are public, and this holdout is procedural and time-ordered, not cryptographic secrecy. Intra-source imports may share vocabulary, so evaluation units are theorem roots nested in seven source families.

## Frozen six-policy comparison and ablation

**No new neural fitting or checkpoint selection.** Freeze the three original RAM-08 best policies:
- `loop32` width32/four tied passes, seed29; SHA-256 `4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f`;
- `loop64` width64/four tied passes, seed43; `5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac`;
- `flat64` width64/no recurrent passes, seed29; `2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8`.

Original train-only compiler effects and frequency controls: RAM-08 artifact `ram08-frozen-audited-train-43689874169`, source Actions run `43689874169`, original ZIP SHA-256 `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc` and data SHA-256 `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`. Exclude original RAM-08 DEV from training-only frequency statistics. The original RAM-12 residual-head weights are not used for RAM-15, thereby fixing six policies and keeping the model comparison balanced.

**Expanded shared grammar:** original 20 RAM-08/13 source-visible tactic families plus the three RAM-14 type-directed families `nat_induction`, `simp_prior`, and `rewrite_prior`. All six primary controllers receive the **identical concrete actions**, provenance, and same expanded candidate count at every Lean state. The six controllers are native deterministic order, original TRAIN-only global-frequency ranker, original TRAIN-only shape-frequency (prior8), frozen original `loop32`, `loop64`, and `flat64`. At most 32 source-derived extra candidates per state; deduplicate by concrete script spelling. For the **native** ordering, interleave one structurally generated tactic after every four original old-family proposals (without changing relative order inside each set). Global and shape frequency controls rank new candidates using their declared 20-family semantic proxy; models use the same proxy indices for their learned 20-family embeddings. Preserve token spellings and old model tensors unchanged.

Structural candidate provenance is restricted to (a) compiler/target-statement-visible typed binders such as `n : Nat`, (b) source definitions and theorem **headers** declared strictly before the target, and (c) explicitly declared fields on an earlier structure. It must never read a reference proof body for a candidate, introduce a lemma, add a hypothesis, use a future declaration, or use the target theorem name/filename as a ranking feature. All generated tactics must pass the RAM-14 source-only whitelist, and the entire candidate set is logged. A post-freeze ablation compares the same six policies against the original 20-family grammar for the *same* theorem set, separately labeled as a grammar ablation, not counted toward the main six-policy score.

## Matched Lean search and certification budgets

Primary: every selected theorem × all **six expanded-grammar policies**, max **25 seconds total wall per theorem/policy**, max **48 real Lean tactic probes**, max **ten successful tactic transitions** per path, max **eight pending goals**, max 3 seconds per request, at most two process restarts. Policy order rotates deterministically across case index. Secondary: same theorem set × (native, global-frequency, shape-frequency, `loop64`) under max **50 seconds / 96 actual tactic evaluations / 16 transitions**. Include counts of source-external macro candidates, process restarts, failures, timeouts and CPU time; earlier/later probes must not cross the individual deadline. Original 20-family ablation: same theorem set × (native, `loop64`) at original 25s/48/10 budget, with separately identifiable scores.

A proof receives credit only after completing the **entire original theorem**, independent full-source Lean4.35 compilation twice **inside** the same registered per-case wall budget, byte-identical proof source and permitted `#print axioms` inventory (`propext`, `Classical.choice`, `Quot.sound` or none). A third separate post-score full-source Lean compiler audit checks every accepted source SHA and exact executable tactic script and leaves scores fixed. Reference original proofs are used solely as source-module oracles outside policy search and never as student input.

**Registered primary criterion:** at least one complete original fresh theorem certified by `loop32` or `loop64` and missed by all three matched expanded-grammar non-neural controls, with zero false source certificates. Report recurrence-specific cases missed by `flat64`. Separate structural-grammar gain: more complete theorem certificates from the expanded grammar than from the original grammar for the matched policy, with results and time/call costs reported per theorem. A native ordering benefit from a new action is useful grammar coverage evidence, not a recurrent-only discovery. Negative results are permanent.

## Execution and trust boundary

Use GitHub-hosted CPU on **private LoopedTransformers Actions** (the RAM-14 private adapter tests successfully ran). Checkout pinned public MathCan source and compile with Lean 4.35. Restore private RAM-08 archive from the original GitHub Actions run into the private workflow job; verify exact bytes and each checkpoint SHA before opening TEST goal states. Never copy original private model weights or RAM-08 training examples into the public MathCan repository or an untrusted public artifact. Use separate private-only model/evidence artifacts for native full-search results. Preserve data and weights checksum receipts, full scored case matrix, original failed scripts and independent Lean recompile logs. Automatic merging is disabled; requests to integrate prior research do not authorize merging an unverified future scientific result.

If hosted CI runner cannot access the original private artifact, report an infrastructure blocker and preserve the source-only scorer and fresh cohort. Do not substitute a public reference proof or label heldout cases solved before an executed private model search.

The original RAM-08 PEF and RAM-10 first registered evaluations remain unchanged and independently unscored by RAM-15.
