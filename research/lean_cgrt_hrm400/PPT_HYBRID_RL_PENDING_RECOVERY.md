# Transferring the recovery controller to pending Lean obligations

The branch-recovery study trained the hybrid controller to judge a speculative tactic that leaves one active goal. This study freezes those checkpoints and evaluates a different shape of proof. The speculative tactic creates two simultaneous goals. Both worlds can solve the active goal along the same proof path. Swapping facts that support the pending goals determines whether to keep the speculative branch or restore the checkpoint.

The 64 layout pairs combine chains, binary joins, and ternary joins. A productive branch costs seven checks, including the first tactic. Taking it after a rollback costs eight because the speculative tactic has already consumed one check. Spending a second check in the rejected branch makes recovery too late. Lean independently tries both actions and compiles the winning proof in each world before scoring any checkpoint.

The registry identifies the original recovery cohort and hashes all 15 frozen sessions. The scorer loads the same five policies at seeds 103, 151, and 197 with the original tactic, CPU, and arithmetic budgets. All policies receive the shared first tactic and use the same direct-fact continuation rule. The models receive the public Lean state, including pending goals, and make a single keep-or-abandon decision. This experiment performs zero training updates.

The primary outcome is independent certification of both proofs in a layout pair. Four policy comparisons evaluate hybrid RL against verified-search imitation, hybrid readout calibration, graph TRM RL, and graph TRM calibration. Three additional comparisons perturb the hybrid RL controller at its history readout. Removal zeros the history contribution. The common shift changes both scores by the same amount. The signed-margin sham perturbs the candidate difference in a frozen direction selected for each layout and seed. Both shams match the Euclidean norm of removal. The signed sham can change the selected action, which tests sensitivity to a perturbation that acts at the same decision.

This is a development transfer study. The layouts have distinct paired topologies and reuse individual subtree templates. The seven comparisons use a five-percentage-point minimum and group both worlds and all seeds by layout. Independence and paired-label exchangeability are unverified. The previous reserved layouts remain sealed.

## Reusing the native branch traces

After the forced first tactic and the recovery decision, every attempted tactic in this cohort has a singleton menu. The choice replay checks that invariant in both recorded branches of every world. It passes the recorded public observation to each frozen controller, then takes the certified outcome of the selected native trajectory. The replay also checks the arithmetic budget for the opening and recovery forwards. The native run independently executes each controller and measures its CPU time. The audit can compare recovery choice, proof script, certificate, tactic count, arithmetic estimate, and applied edit between the two runs.

The same traces also provide a structural baseline. A fixed Horn solver chooses the cheapest root proposal using the public rules and facts, then keeps or restores the branch accordingly. This measures how much of the task is solvable with a deterministic algorithm specialized to the generated fragment.

## Reproduction

```text
python -m configs.freeze_ppt_pending_recovery OLD-COHORT TRANSFER-COHORT SEED-103 SEED-151 SEED-197
python -m configs.run_ppt_pending_recovery TRANSFER-COHORT RUNS SEED-103 SEED-151 SEED-197
python -m configs.replay_ppt_pending_choices TRANSFER-COHORT RUNS/witness.json.gz REPLAY SEED-103 SEED-151 SEED-197
python -m configs.analyze_ppt_pending_recovery TRANSFER-COHORT RUNS SUMMARY EVIDENCE --choice-replay REPLAY
```

The analysis requires EvalMath's FSI-01 module on the Python path. The runner uses the original Lean verifier and CPU-only PyTorch. The campaign runs under a 6 GiB process-tree cap, with 2 GiB and two PyTorch threads per worker.
