# PPT-G03: live Lean proof-search and replay-gated training data

**Status:** executable research implementation, 2026-11-22.
**Branch:** research/ppt-g03-live-lean-20261122.
**Companion:** [PPT Plan A/B gyms](PPT_GYM_RUNBOOK.md).

## Scope

PPT-G03 advances the compact recurrent specialists from archived tactic-menu scoring to **live execution against Lean 4.19.0**. It reuses the existing Lean-CGRT JSON-lines proof-state server, the Lean kernel, and the independent standalone certifier in lcgrt.

The initial cohort is freshly generated **propositional Horn theorems**. It is not fresh Mathlib source-position evaluation. The generated families also occur in older Lean-CGRT research, and overlap with earlier datasets requires a separate ancestry audit.

The model sees the active goal, local hypotheses, candidate tactics, prior failed choices, and remaining check budget. It never sees the generator reference proof, certificate, or outcomes of tactics it has not tried. Candidate retrieval uses Lean-reported hypothesis types to match conclusions to the current goal. Allowed actions are restricted to apply hN, apply rN, and constructor.

Each attempted tactic uses one budgeted Lean run call. A rejection leaves the goal unchanged. An accepted tactic can create subgoals. A completed goal triggers a separate kernel verification, followed by **standalone compilation** of the complete script with an axiom audit. Full proof credit requires both checks.

## Matched arms

| Arm | Decision rule |
| --- | --- |
| auto8 | Deterministic typed hypothesis order with eight tactic attempts |
| auto4_then_ppt | Four automation checks followed by recurrent tactic ranking |
| frozen | PPT alone with fixed parameters |
| dram | PPT plus CPU fast weights updated from training support |
| retrieval | PPT plus retrieved examples from the same support evidence |
| sham | PPT plus online weights trained on deliberately inverted support labels |

All arms start from fresh independent states for the same theorem and have the same limit on Lean tactic calls. They may spend different inference compute. Plan A and Plan B share this interface, including message/depth experiments conducted separately.

The fast learning head and retrieval store receive only **train-support** feedback. Accepted nonterminal tactics are typed transition examples; the binary full-closure head updates on rejected choices and certified terminal closures only. Query feedback never changes fast weights. For an unseen OOD family, its fast head begins empty.

## Freeze and execute

From the repository root:

~~~sh
cd research/lean_cgrt_hrm400
python -m ppt_gym.live_cli freeze --seed 20261122 --per-split 24 \
  --out runs/ppt-g03-suite

cd lean && lake build && cd ..

python -m ppt_gym.live_cli run \
  --suite runs/ppt-g03-suite \
  --checkpoint runs/ppt-a-checked/checkpoint.pt \
  --support-roots 12 --query-roots 12 --query-split calibration \
  --checks 8 --depth 3 --max-seconds 360 \
  --out runs/ppt-g03-calibration
~~~

The suite has train, calibration, ordinary_test, and ood_family partitions, with source hashes, theorem-lineage deduplication and pinned generator revisions. Keep historical c4 Mathlib evaluation cases out of all training lanes.

The runner requires the Lean 4.19.0 executable and lgym binary supplied through lcgrt.config. Set LCGRT_LEAN_HOME for an alternate toolchain installation. Each experiment retains a checkpoint hash, server binary hash, Lean version, raw tactic receipts, verified proof scripts, and a per-policy scorecard. The default wall and request timeouts bound execution, with a process-tree memory cap left to the external operating-system job wrapper.

## Independently replay and export

~~~sh
python -m ppt_gym.live_cli replay \
  --suite runs/ppt-g03-suite --session runs/ppt-g03-calibration \
  --out runs/ppt-g03-replay

python -m ppt_gym.live_cli export \
  --suite runs/ppt-g03-suite --session runs/ppt-g03-calibration \
  --replay-receipt runs/ppt-g03-replay/replay.json \
  --out runs/ppt-g03-training-export
~~~

Replay reloads the same frozen theorems, checks binary hashes, re-executes every recorded tactic, checks its before/after goal states, and reruns the terminal kernel verification.

The exporter requires a passing independent native replay plus the original compiled proof certificates. It reads only train-support episodes and emits SFT, outcome, transition, repair, and identical-state preference streams. Entries bind to source root, predecision packet, exact tactic and verifier receipt. A four-presentation limit applies across streams for each theorem lineage. Original raw episodes remain intact.

Training admission remains pending an external source-ancestry and rights audit. Synthetic protocol fixtures are never admitted as Lean proof evidence.

## Evaluation and known limits

The test suite includes fake-server protocol checks that carry simulated status. The hosted native workflow builds Lean 4.19.0, executes a proof, audits a standalone certificate, and runs a tiny generated suite with replay and export. Only the run results establish native execution.

The current search follows one committed proof path. A valid tactic can lead to a dead branch, so the next research step is bounded branch restoration and explicit node-expansion accounting. A second follow-up will adapt Mathlib source-position execution through the existing certified select path. The proof-model objective, online learning controller and variable-depth scheduler can then be trained on fresh, independently certified proof-search episodes.

Promotion requires a paired improvement over the strongest same-compute baseline on lineage-disjoint proof roots, with a proven certificate for every counted closure.
