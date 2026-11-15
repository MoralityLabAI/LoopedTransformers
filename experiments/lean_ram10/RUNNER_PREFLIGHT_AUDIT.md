# RAM-10 runner and artifact integrity audit, November 15, 2026

**Scope:** Post-registration engineering repairs to the existing RAM-10 frozen
train/test execution. The scientific preregistration, original MathCan source
commit and TRAIN/DEV/TEST module split, 20-family grammar, all model selection
rules, runtime budgets and theorem criteria are unchanged. This audit reads no
held-out theorem statements or proof bodies and makes no Lean proof claim.

## Runner status

The original Actions run `43702011161` failed in
`original-train-only-compiler-mine-and-freeze` with `runner_id=0` and no
executed steps. The dependent theorem TEST job was skipped. Other RAM-10
attempts in the same interval show the same immediate hosted-runner
assignment failure. The underlying GitHub platform reason is unknown. The
current ChatGPT local sandbox has no Lean 4.35 executable and its shell does
not resolve github.com for the official release download. The first genuine
RAM-10 compiler collection, all nine real-model fits and every original
TEST theorem score remain pending, in separate registered jobs.

## Two pretest gaps closed

1. **Imported dependency mutation.** Previously `verify_checkout` checked
   the pinned Git `HEAD`, `lean-toolchain` and existence of the four TEST files
   in the TRAIN checkout. It also checked each selected target module's raw
   source Git blob. A dirty tracked import dependency or build configuration
   could still affect genuine compiler outcomes. The corrected check rejects
   any tracked worktree change except the four *exact registered deletions*
   for masked TRAIN. The pristine TEST checkout admits zero tracked changes.
   Added untracked `MathCan/*.lean` or Lean toolchain/build-input source files
   abort. Build products in `.lake/` are permitted.

2. **Unselected checkpoint mutation.** Previously `verify_frozen` checked
   the training `complete` receipt and three selected `best.pt` file hashes.
   Now it requires the exact nine original architecture/seed combinations,
   recomputes **all eighteen** `best.pt`/`last.pt` hashes, loads their saved
   provenance/optimizer/RNG metadata, compares recorded DEV history to the
   `best_score` and `best_step`, then reconstructs the DEV-only winning seed.
   A mutation to one **unselected** checkpoint also blocks the entire TEST
   job before the first original theorem source is checked out.

Neither new check changes the original admission gate or proof search. These
are audit gates on the exact original training artifact, not a new chance to
select weights or inspect/test against PEF source.

## Verification

The original nine RAM-10 synthetic/contract tests plus three new tests pass
locally using the preserved source-compatible software-only RAM-08 model
harness; the new tests simulate a nine-checkpoint artifact and dirty MathCan
imports. A separate Lean4.35 compiler replay remains required. Synthetic
fixtures are not added to original TRAIN and never count toward registered
`exact`, `rw` or `rcases`/`induction` minima.

The WSL two-stage runner and CI both perform this pretest audit before access
to prospective TEST goals. The first registered held-out run must still score
all selected original theorems under the frozen primary and secondary
budgets and a third original Lean source/axiom audit.
