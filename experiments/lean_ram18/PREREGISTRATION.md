# RAM-18 source-only preregistration: quantified Nat induction and diversified verified macros

**16 November 2026. Predata. Separate experiment following the registered negative RAM-17 source gate.**

RAM-17 independently triple-compiled 20 source proofs, including 18 TRAIN positives, while its induction family produced zero closures from 69 attempts covering only five of 63 source roots. All three family types and prior-rewrite module diversity were required, so the registered gate failed. This is a new study with source-only structural grammar changes and a newly locked TRAIN/DEV collector; RAM-17's earlier receipts and failure remain immutable.

## Immutable source roles and original proof isolation

TRAIN/DEV: `patrickdugan/MathCan@f427e0ef4a61093304d12ca2c0dc943d8d2429e5`, Lean `leanprover/lean4:v4.35.0`. The 22 TRAIN modules and four DEV modules are listed with their exact Git blob SHA-1 values in [config.json](config.json); every original source is verified byte-for-byte before tactics. TRAIN takes the first eight top-level `theorem` or `lemma` declarations per module in source order, DEV the first six, without difficulty filtering or postscore root replacement. Modules with fewer declarations contribute their complete original set.

The five protected TEST module Git blobs and TEST source revision remain the earlier unopened RAM-17 heldout: `3749c84cc6d47297faaf94404af6d1fd36916812`. `NOR19StreamedVerifier` and `NOR20BigintWidthBounds` are chronologically new relative to the old TRAIN checkout. The three existing older protected TEST modules, together with the four pre-existing RAM-16 protected TEST modules, are physically removed before any compiler probe. Prospective TEST goals, reference tactics and theorem statements are not accessed during this source collection. The two chronologically new source modules must be absent from TRAIN. Historical RAM08 task rows from every RAM18 TEST module must be purged before any model fit or controller-prior construction.

This is a **post-RAM17 redesign of TRAIN**, not a fresh prospective training-source discovery. The TEST cohort is still sealed and unscored.

## Source-derived tactic generation and full native certificate

Preserve RAM14's pinned original action prefix and RAM17's general header-only `simp_prior` and `rewrite_prior` generator. Extend only the Nat induction family by identifying source-visible leading universal quantifiers of type Nat (`∀ n : Nat, ...`, `∀ (n : Nat), ...`), alongside the existing typed parenthesized binders. A quantified `n` permits deterministic atomic composite scripts `intro n; induction n` with optional `<;> simp`, `<;> simp_all`, `<;> omega`, or `<;> simp_all <;> omega`. A previously introduced explicit Nat binder permits `induction n` variants. Every such action is a **single tested complete-source script**, never a mined reference proof. Script grammar has an explicit allowed whitelist, exactly 48 candidates/root and at most 16/family. Preserve source-only witnesses, script order and any zero generated candidate counts.

For each original root, observe a real Lean-printed goal before action, replace only the original target proof, and compile the entire unmodified remainder of the MathCan file twice in independent Lean 4.35 processes. Third verifier reconstructs original source bytes, checks target theorem and action hashes, observes a fresh native original goal, triple-compiles every twice-accepted source, and checks permitted transitive axioms (`propext`, `Classical.choice`, `Quot.sound`). Full-source closure is the only positive label; timeout or compiler rejection is censored as nonclosure, with every source and log preserved.

## Gate, fixed budgets and stopping rule

Before any model fitting or TEST access, require the same structural diversity criterion used by RAM-17: at least two distinct TRAIN Nat-induction roots in two modules; at least two prior-lemma rewrite roots in two modules; all three macro families nonempty; at least eight independently triple-compiled TRAIN positive scripts across at least four TRAIN theorem roots and four TRAIN modules. The source collector uses a 90-minute GitHub 2-thread CPU cap, at most 15 seconds per individual native Lean compiler process, and the existing 6 GiB process memory guard.

**If the gate fails, archive a named negative receipt, mark TEST unopened, and stop at zero model fits.** Thresholds, source modules and grammar are frozen before this new compiler trial.

## Conditional paired training and scoring

A gate-admitted follow-up uses the same 23 family embedding IDs, Looped64 and parameter-matched Flat71 models, task-only and 10% verified-positive objectives, seeds 17/29/43, identical initialization within architecture/seed, 400 CPU AdamW updates per arm, and family/module/root-balanced positive sampling. Run all twelve best checkpoints through a new source-only pretest lock, then commit their SHA256 digests and deterministic TEST declaration names before opening any TEST goal. The original sixteen policies share the same 48 real-Lean-request, 30-second/ten-transition primary budget. The recurrent-specific criterion remains the RAM-17 two-paired-seed theorem condition against native, global-frequency, shape-frequency, task-only Looped64 and structural Flat71, with third independent source replays.

The present implementation stage is a **new TRAIN-only source-gate experiment**. It claims neither a source gate pass nor any trained proof-search improvement.
