# PPT-G03B: rollback-aware Lean proof-tree gym

**Status:** implemented research prototype, 2026-11-22.
**Parent:** [PPT-G03 native Lean stepper](PPT_G03_RUNBOOK.md).
**Branch:** research/ppt-g03b-backtracking-20261122.

## Motivation

The earlier type-directed tactic menu reduces invalid choices. The first live run revealed a different problem: a valid apply may open an unprovable premise. The old policy stays on that dead branch even when another candidate at the parent might prove the goal. PPT-G03B restores previously saved, immutable Lean states when it exhausts a branch.

Each tactic attempt consumes one Lean check. A rollback consumes a separate host search action, with its own cap and recorded cost. The policy still has exactly the predeclared check budget.

## Model-visible search

At each proof state, derive candidates from Lean's public hypotheses and the active target. Try a tactic under the frozen ranker or deterministic ordering. If Lean accepts a nonterminal transition, save the parent state and append the proposed tactic to the provisional script. If the child has no candidates, restore the parent state, remove the abandoned tactic from the provisional script and mark the branch as exhausted. A valid terminal result goes through the Lean kernel; successful scripts then undergo independent standalone compilation.

The ledger stores the exact public proof state, tactic menu, chosen action, before/after hashes, restored parent identity, tactic-call number, and rollback count. Replay constructs new Lean states and checks each action and rollback without depending on the original model.

Plan A and Plan B checkpoints use the same proof action interface. This study crosses two underlying policies, auto8 and frozen PPT, with rollback enabled or disabled, for **four paired arms on the same frozen roots**. The zero-rollback arms receive the identical tactic-call budget and stop when an accepted branch has no available tactics. The query cohort never updates DRAM weights.

## Run on the registered live theorem suite

From research/lean_cgrt_hrm400, after building the native lean server, and using a previously created PPT checkpoint:

~~~sh
python -m ppt_gym.search_cli run \
  --suite runs/ppt-g03-suite \
  --checkpoint runs/ppt-a-checked/checkpoint.pt \
  --split calibration --cases 12 \
  --checks 8 --rollbacks 8 --depth 2 \
  --max-seconds 360 --out runs/ppt-g03b-search

python -m ppt_gym.search_cli replay \
  --suite runs/ppt-g03-suite \
  --session runs/ppt-g03b-search \
  --out runs/ppt-g03b-replay
~~~

Outputs: episodes.jsonl, certificate.json, optional standalone.lean, and manifest.json. The report includes certificate counts, Lean tactic-call counts, host rollback counts, and model calls for each policy. Source and binary hashes are retained. The selected split must be evaluation-only, with calibration used for initial model selection and ordinary/heldout families held for confirmation.

## Controls and validation

The simulated CPU test contains a dead-end rule and a valid alternative. With backtracking enabled, the control uses three tactic attempts and one rollback, preserving only the successful script. The same deterministic policy with rollback disabled spends one tactic call on the dead branch, so its proof remains open. Tampered rollback and check counts must fail replay.

A hosted workflow builds the pinned Lean 4.19.0 proof server, creates the same problem inside native Lean, runs the rollback-aware policy, verifies kernel closure, and replays the full search. It also runs a tiny frozen-suite comparison with standalone certificate accounting and independent replay.

These measurements are a search-controller engineering baseline. The generated Horn task distribution overlaps prior Lean-CGRT synthetic studies, and model inference compute is not yet matched across arms. No architecture improvement is claimed before a paired, unsaturated, preregistered comparison.

## Next gates

The next experiment measures backtracking against the preexisting single-path policy on independent fresh roots at matched tactic-check and inference budgets. A verified proof-distance target could then train a branch-value head under fixed and shuffled controls. Mathlib source-position proving requires the separate, original-position certified select facility. Any future DRAM learning must use train-support feedback with independent replay and an external ancestry audit before admission.
