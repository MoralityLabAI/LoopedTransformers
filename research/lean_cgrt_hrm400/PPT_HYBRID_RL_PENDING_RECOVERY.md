# Transferring the recovery controller to pending Lean obligations

The branch-recovery study trained the hybrid controller to judge a speculative tactic that leaves one active goal. This study freezes those checkpoints and evaluates a different shape of proof. The speculative tactic creates two simultaneous goals. Both worlds can solve the active goal along the same proof path. Swapping facts that support the pending goals determines whether to keep the speculative branch or restore the checkpoint.

The 64 layout pairs combine chains, binary joins, and ternary joins. A productive branch costs seven checks, including the first tactic. Taking it after a rollback costs eight because the speculative tactic has already consumed one check. Spending a second check in the rejected branch makes recovery too late. Lean independently tries both actions and compiles the winning proof in each world before scoring any checkpoint.

The registry identifies the original recovery cohort and hashes all 15 frozen sessions. The scorer loads the same five policies at seeds 103, 151, and 197 with the original tactic, CPU, and arithmetic budgets. All policies receive the shared first tactic and use the same direct-fact continuation rule. The models receive the public Lean state, including pending goals, and make a single keep-or-abandon decision. This experiment performs zero training updates.

The primary outcome is independent certification of both proofs in a layout pair. Four policy comparisons evaluate hybrid RL against verified-search imitation, hybrid readout calibration, graph TRM RL, and graph TRM calibration. Three additional comparisons perturb the hybrid RL controller at its history readout. Removal zeros the history contribution. The common shift changes both scores by the same amount. The signed-margin sham perturbs the candidate difference in a frozen direction selected for each layout and seed. Both shams match the Euclidean norm of removal. The signed sham can change the selected action, which tests sensitivity to a perturbation that acts at the same decision.

This is a development transfer study. The layouts have distinct paired topologies and reuse individual subtree templates. The seven comparisons use a five-percentage-point minimum and group both worlds and all seeds by layout. Independence and paired-label exchangeability are unverified. The previous reserved layouts remain sealed.

## Reusing the native branch traces

After the forced first tactic and the recovery decision, every attempted tactic in this cohort has a singleton menu. The choice replay checks that invariant in both recorded branches of every world. It passes the recorded public observation to each frozen controller, then takes the certified outcome of the selected native trajectory. The replay also checks the arithmetic budget for the opening and recovery forwards. The native run independently executes each controller and measures its CPU time. The audit can compare recovery choice, proof script, certificate, tactic count, arithmetic estimate, and applied edit between the two runs.

The same traces also provide a structural baseline. A fixed Horn solver chooses the cheapest root proposal using the public rules and facts, then keeps or restores the branch accordingly. This measures how much of the task is solvable with a deterministic algorithm specialized to the generated fragment.

## Native transfer results

Each cell counts layouts with certificates for both worlds out of 64. The recorded-branch replay agreed with all 3,072 live trials, including the seven comparisons and the unedited recovery policy.

| Policy | Seed 103 | Seed 151 | Seed 197 |
| --- | ---: | ---: | ---: |
| Hybrid recovery RL | 34 | 64 | 63 |
| Hybrid verified-search imitation | 61 | 64 | 63 |
| Hybrid readout calibration | 51 | 59 | 61 |
| Graph TRM recovery RL | 0 | 0 | 0 |
| Graph TRM readout calibration | 0 | 0 | 0 |

Both graph TRM policies certified 64 individual proofs out of 128 at each seed. The public Horn solver selected the certified branch in all 128 worlds. The hybrid advantage over graph TRM transferred to pending goals. The earlier RL advantage over imitation did not transfer. Imitation led by 27 pairs at seed 103 and tied at the other two seeds. Calibration also led hybrid RL on the seed-103 checkpoint and trailed it at seeds 151 and 197.

Removing the full history reduced the mean fully solved pair rate by 17.2 percentage points. The equal-norm common shift preserved all choices. The randomized margin sham reduced the mean by 76.0 points. The large response to randomized perturbation limits what the whole-history comparison can establish about the particular recurrent signal. The campaign averaged 4.9 logical cores across witness preparation and scoring and peaked at 2.37 GiB of committed process-tree memory.

The transfer runs produced 2,360 certified proof receipts with 128 distinct theorem-and-tactic sources. A separate Lean recheck matched every source to its registered theorem, then compiled all 128 distinct proofs in four isolated batches and checked their axioms. This recheck leaves the original live-search timings and certificate records intact.

## Diagnosing the recurrent history readout

The component probe compares the original single-goal development layouts with the pending-goal development layouts. At the recovery decision it separates the linear fact projection, the premise-interaction readout, and the arrival-history projection. It removes each contribution individually and compares the effect with an equal-norm randomized change to the same action margin. It checks the reconstructed margin against a physical edit on one world per seed and family and on any near-tie.

The probe scores whether both worlds select the continuation that the Lean witness showed was viable. This is a recovery-choice endpoint. Some single-goal traces still require further model-selected tactics after recovery. The probe makes no changes to the checkpoints or the reserved layouts.

On the 24 single-goal layouts, removing the premise-interaction readout reduced viable-branch pairs from 10, 14, and 13 to 3 at every seed. On the 64 pending-goal layouts, removing only the arrival projection changed complete viable-branch pairs from 34, 64, and 63 to 52, 64, and 62. The direction and size differ by seed. A common logit offset also contributes to the Euclidean norm without changing the action margin. The component analysis therefore reports a second randomized sham matched to the absolute change in the keep-or-abandon margin. For the seed-103 arrival projection, the norm-matched sham moved the margin by a median of 3.5 times the removal dose, while the margin-matched sham left 27 complete pairs.

## Steering the next training

The next variant keeps the original arrival projection for a single open goal. With pending goals, a learned gate scales that projection using the parent and current branch scores. It starts with an arrival multiplier of about 0.12. The premise-interaction path remains available. In a given branch, both worlds provide the same number of facts at each premise depth. In one world those facts support a single pending rule, and in the other they are split across competing rules. Training uses two-premise pending rules and keeps the original recovery and mixed-corpus training cases. Development includes three-premise pending rules. The same online verifier loop compares the routed architecture under RL, imitation and calibration, the unrouted hybrid under RL, and both graph TRM controls.

The new cohort and its seed registry freeze the comparison before scoring. It is a development study motivated by the observed seed sensitivity. The earlier reserved layouts remain sealed.

The static audit found one seven-check branch in each of the 128 new train and development worlds. It also found identical linear support for the two worlds' root actions through eight evidence updates. The native witness verified both recovery choices on all 32 development pairs. The first training comparison uses seed 103, which had the largest transfer loss.

The training audit separates recovery pairs with unequal verifier returns from pairs where both returns agree. A tied return contributes zero to the paired action objective. The critic and optimizer can still change parameters on that batch. The audit also counts direct action contrasts that include an uncertified world, including the subset where neither world has a certificate. The evaluation compares complete certified pairs on the same layouts for all six arms.

In the completed seed-103 routed RL training ledger, 41 of 64 recovery pairs supplied a direct action contrast involving an uncertified world. Each of those pairs also contained one certified world. The remaining 23 had tied returns and produced parameter updates. The same training ledger records 128 LoopedTransformers recovery episodes, 64 native BPD episodes, 12 native EvalMath episodes, and 40 MathCan archived replay episodes.

Two seed-103 arms hit a SQLite write lock during final scoring when a live audit held a read transaction. Their training checkpoints and committed proofs are intact. The runner can resume interrupted arms from their ledgers, retries a transient lock once, and records resource use by attempt.

### First routed training result

Both interrupted arms resumed from committed checkpoints and finished. Each row below reports the seed-103 final score on 32 newly generated development pairs.

| Policy | Complete pairs | Individual Lean proofs |
| --- | ---: | ---: |
| Routed hybrid RL | 9 | 40 |
| Unrouted hybrid RL | 7 | 37 |
| Routed hybrid readout calibration | 6 | 38 |
| Routed hybrid verified-search imitation | 10 | 42 |
| Graph TRM RL | 0 | 10 |
| Graph TRM calibration | 0 | 11 |

Eight initial continuations hit the CPU limit. Their recorded tactics were prefixes of the longer continuations, and all recorded initial recovery choices agreed within each architecture. All final trials stayed within the CPU budget. The routed hybrid improved on unrouted RL and calibration at this seed. Imitation led by one complete pair.

Routed RL selected a viable branch in 41 worlds and certified 40. Imitation selected a viable branch in 42 worlds and certified all 42. Both policies certified both worlds of every pair where they selected both viable branches. The seed-103 pair difference arose at the recovery decision.

## Reproduction

```text
python -m configs.freeze_ppt_pending_recovery OLD-COHORT TRANSFER-COHORT SEED-103 SEED-151 SEED-197
python -m configs.run_ppt_pending_recovery TRANSFER-COHORT RUNS SEED-103 SEED-151 SEED-197
python -m configs.replay_ppt_pending_choices TRANSFER-COHORT RUNS/witness.json.gz REPLAY SEED-103 SEED-151 SEED-197
python -m configs.analyze_ppt_pending_recovery TRANSFER-COHORT RUNS SUMMARY EVIDENCE --choice-replay REPLAY
python -m configs.recheck_ppt_pending_certificates TRANSFER-COHORT EVIDENCE CERTIFICATE-RECHECK
python -m configs.probe_ppt_recovery_components OLD-COHORT OLD-WITNESS TRANSFER-COHORT RUNS/witness.json.gz COMPONENTS SEED-103 SEED-151 SEED-197
python -m configs.freeze_ppt_pending_alias OLD-COHORT PENDING-ALIAS-COHORT
python -m configs.run_ppt_branch_recovery PENDING-ALIAS-COHORT MATHCAN-TRAIN.zip INITIALIZATION ROUTED-RUNS --pending-arrival-router --seed-study SEED-STUDY --seed SEED
python -m configs.analyze_ppt_pending_alias PENDING-ALIAS-COHORT ROUTED-RUNS INITIALIZATION SUMMARY EVIDENCE --seed-study SEED-STUDY --seed SEED
python -m configs.combine_ppt_pending_alias PENDING-ALIAS-COHORT SEED-STUDY COMBINED EVIDENCE-103 EVIDENCE-151 EVIDENCE-197
```

The analysis requires EvalMath's FSI-01 module on the Python path. The runner uses the original Lean verifier and CPU-only PyTorch. The campaign runs under a 6 GiB process-tree cap, with 2 GiB and two PyTorch threads per worker.
