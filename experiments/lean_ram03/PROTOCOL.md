# RAM-03 protocol

Freeze before new corpus collection, fitting or proof evaluation. Preserve
RAM-02 a3e7be130334ecf86b8f0befc912f4864418d385 and its artifacts. Use a separate
branch and D:\Research Engine\RAM-03. Reuse existing Python/PyTorch2.6 and
Lean4.35 from RAM-02. No upgrades, torch.compile or GPU compilation. Same caps:
6 GiB total process-tree RSS,2.5 GiB CUDA allocation,2 GiB available RAM/commit,
512 MiB dedicated GPU headroom. Sample resources continuously.

## Supervision

Use only the27 theorem roots selected in the old FC-12 TRAIN corpus, across
its seven original modules. Original target reference proof commands and
teacher-prefix actions are not used. Each root and every descendant state
is observed by Lean. Explore at most10 visible states per theorem, depth4,
at most8 simultaneous goals. Probe the registered grammar at each state.
Prefer retaining both branching and nonbranching descendants. Preserve every
positive/negative attempt, complete ordered before/after goals, visible
definition names, source/script hashes and exact action prefixes.

Five complete theorem groups listed in config.json are development-only.
All their descendants remain held out from optimization. Require positive
multi-goal input states in train and development before fitting. Remove exact
visible-state hash overlap across splits from training, preserving an exclusion
receipt. Never overwrite or relabel the original961 compiler observations.

A second, fresh Lean session independently replays EVERY new observation.
Require agreement on acceptance and canonical complete goal-list signatures.
No training until the audit passes. Scratch observations and server verdicts
are untrusted; they are not full theorem certificates.

Labels distinguish failed/unchanged, reduced remaining-goal count, other real
state progress (including branch creation), and all-goal scratch closure.
An auxiliary head predicts clipped net goal-count decrease. Inputs contain
only the active context/goal, pending-goal conclusions, candidate script,
family, and observed goal count. No theorem ID, file, original command, action
prefix, future state or target outcome is a feature.

Fit width32/64 recurrent models with four shared passes, and flat64 control,
seeds17/29/43. Effective batch64 from16x4; half the samples drawn from observed
multi-goal input states. AdamW .002/.01, gradient cap1.0, at most1,600 updates,
validate every100, patience4. Optimize class-weighted CE plus0.2 goal-delta
SmoothL1. Select checkpoints using balanced single/multi-goal development CE
plus the registered auxiliary loss, with training-only class weights. Select
one seed per architecture by that development score; report all seed fits.
Calibrate CPU/CUDA on identical updates first; require1.15x CUDA speedup.

## Grammar and search

Retain FC-12 tactics. Add registered simplification/unfolding actions using
only definition constants actually occurring in Lean goal/hypothesis types,
not arbitrary theorems or reference proofs. Add a fixed core Int multiplication
simplification script, all-goals closing variants, case splitting on visible
And/Or/Exists/Bool or original enum parameters, and list induction generalizing
visible scalar/state variables. Every policy gets the identical candidate pool.
Cap generated arguments per family and record the exact ordered pool.

Persistent Lean probing restores a saved root before evaluating each entire
candidate prefix. Reset heartbeat accounting per request; reject logged errors.
Count logical tactic evaluations separately from actual processes. Each policy
starts a fresh server, with startup charged to its theorem budget; no cross-policy
verdict cache. Prefixes and all sibling goals are preserved; canonical visible
state signatures deduplicate states. Explore bounded depth-first alternatives,
with an8-goal cap and backtracking. All goals must close.

Primary budget:30 seconds,48 logical evaluations,8 successful transitions per
path. Secondary:60 seconds,128 evaluations,16 transitions. Both are registered
now, not enlarged after scores. Per-request timeout3 seconds. Reserve0.5 seconds
for shutdown. Final compile and independent replay each require their own actual
Lean process and axiom audit. Primary compares native fixed ordering, train-only
family frequency, and the development-selected loop32/loop64/flat64 checkpoints.
Secondary compares native and development-selected loop64 only, on the same cases.
Allow at most two server restarts after request timeouts, counted and charged.

Policy utility:3*P(all-closed)+1.5*P(reduced-goal-count)+0.3*P(other-progress)
-0.6*P(failed)+0.4*clip(predicted goal-count decrease,-3,3). Fixed native ordering
tries arithmetic/reflexive/simplification/visible unfolding before branching.
Frequency uses only audited training labels. No policy gets reference actions.

## Proof evidence and evaluation

MathCan stays pinned to c84e2025334e9a967c1297425a490f98d5b3d189. FC-13's16
published cases are post-hoc regression. The prospective set is the first three
source-order theorem declarations in each of FlashAccounting, V4Transient,
TOEFormSectors and TOEMultipletKinetics, frozen by name in config.json. Only
declaration names, not target proof bodies or outcomes, were inspected before
this selection. These modules are disjoint from FC-11/12/13 selected families.
Do not inspect prospective goals or compile selected reference proofs until
training, configuration and search implementation are frozen.

For every accepted proof, separately compile the complete original theorem in
its original preceding context, with only selected tactics. Reject sorry/sorryAx
and dependencies outside propext,Classical.choice,Quot.sound. Independently
recompile the exact final source bytes. Full certificates and replay determine
coverage; cached predictions never do. Report prospective and post-hoc results
separately, including native-only/model-only solved sets, calls, evaluations,
search/verification cost, branch failures, all seed metrics and hardware peaks.
An improvement over RAM-02 native6/16 is not a learned-policy gain unless it
also exceeds the contemporaneous RAM-03 native comparator at identical budgets.
Preserve negative results. Publish a separate feature PR; never merge.
