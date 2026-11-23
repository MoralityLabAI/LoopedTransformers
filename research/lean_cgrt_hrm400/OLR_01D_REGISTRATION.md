# OLR-01D registration: delayed guard feedback with pending Lean obligations

**Date:** 2026-11-23  
**Status:** registered before the new native scorer or its outcomes are evaluated  
**Lineage:** [OLR-01C](OLR_01C_COMPUTE_ABLATION.md) used source seed 20261124 and model seeds 17/29/43.

## Claim under test

Can the OLR experiment stack use a *public Lean goal state* to maintain two simultaneous pending proof obligations, make a reversible branch choice, and spend a diagnostic check early enough to recover a certified proof within a tight budget? The exact arithmetic guard remains visible, so a host arithmetic solver can settle every root without a probe. This gate establishes real native control semantics and cost accounting. The frozen OLR-01C learner is an exploratory ranking control, with **no online learned feedback update** in this gate.

## Frozen design

- **New cohort:** 8 root pairs, both worlds, generated with `olr-guard-v1` seed **20261127** and operand range **9..16**. Exclude every (a,b,delta) tuple from the 20261124 OLR-01C source, dev, and OOD bank. This source freeze is performed before scoring any OLR-01D model.
- **New theorem:** `∀ (P Q R : Prop) (h0 : P) (h1 : Q) (r0 : (a+b=c0) → P → Q → R) (r1 : (a+b=c1) → P → Q → R), R`. Across pair worlds, c0 and c1 swap. Lean must expose *three active goals* after the first `apply rN`: a computational equality followed by P and Q.
- **Budget:** compare 4 and 6 **tactic calls**, with one allowed host rollback; count host `focus` separately. Each candidate path needs exactly four tactics to close.
- **Schedule:** `probe_early` checks the arithmetic guard with native `rfl` before solving P and Q; `probe_late` solves both open propositional obligations first (with `focus`) and tests the guard last. The early schedule can recover from a wrong initial branch within six calls, while the late schedule has already spent four calls before discovering the mistake.
- **Root policies:** frozen OLR-01C `cross`, `cross_onepass`, `direct`, `isolated`, `frozen_cross` at each of seeds **17, 29, 43**; host exact-addition oracle; deterministic `r0` tie. Keep weights frozen and report every seed. All policies consume only the raw Lean predecision guard expressions.
- **Learning evidence:** the previous OLR-01C source bank and model archive stay fixed. No OLR-01D proof, verdict or test-set data may be used for training or model selection.
- **Controls:** rerun both schedules under both budgets; native verifier feedback alone determines rollback, without a hindsight proof label in the action selection. The immediate oracle and fixed-tie reference bound success and repair cost. The OLR-01C matched-margin sham is retained as an earlier caveat, not claimed anew on this distribution.

## Required verifications before reporting

1. Register and hash all world statements, pair membership, toolchain hashes and source-bank checksum before model scoring. Verify menus exactly `apply r0`, `apply r1`.
2. Force each root branch and validate in fresh native Lean that it creates three simultaneous goals, one `rfl` pass and one rejection per pair world. A rejected `rfl` is a failed tactic attempt, not proof that the equality is mathematically false.
3. In successful trajectories call Lean's `kernel`, require an empty axiom list, replay the full tactic/focus/rollback ledger in a fresh server state, and compile every distinct successful proof independently against its **full original theorem**.
4. Score only fully paired roots; report native closures, mean calls, host focus and rollback counts, cost utilities separately, and exact reachable coverage. Group roots by pair and retain every model seed.
5. Keep the registry and live native results in a downloadable CI artifact with exact SHA-256 content hashes. Classify any result as **engineering / exploratory** under FSI-01 until an independent mechanism-sham and power design exists.

## What counts as progress

A verified early-probe rollback must recover at least one wrong-branch proof within six native checks while the matched late probe cannot. The exact arithmetic oracle should close both worlds under four checks and is a higher-value baseline than a neural guard classifier on this grammar. An OLR-specific benefit requires future fresh pair families in which native feedback changes the right action under partially observable semantics, a learned adaptive budget controller outperforms exact and direct-reader baselines at equal inference cost, and message removal has a direction-specific effect beyond the matched-sham floor.

**This protocol registers the live search/custody gate. It does not assert that the current recurrent classifier possesses verified obligation-tracking ability.**
