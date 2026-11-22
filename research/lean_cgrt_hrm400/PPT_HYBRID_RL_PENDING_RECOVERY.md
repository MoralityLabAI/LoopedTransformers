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

### Routed training results

Both interrupted arms resumed from committed checkpoints and finished. Each cell reports complete pairs out of 32 and individual certificates out of 64 on the new development layouts.

| Policy | Seed 103 pairs / proofs | Seed 151 pairs / proofs | Seed 197 pairs / proofs |
| --- | ---: | ---: | ---: |
| Routed hybrid RL | 9 / 40 | 14 / 45 | 19 / 51 |
| Unrouted hybrid RL | 7 / 37 | 14 / 45 | 18 / 50 |
| Routed hybrid readout calibration | 6 / 38 | 8 / 39 | 8 / 40 |
| Routed hybrid verified-search imitation | 10 / 42 | 15 / 47 | 18 / 50 |
| Graph TRM RL | 0 / 10 | 0 / 19 | 0 / 17 |
| Graph TRM calibration | 0 / 11 | 0 / 16 | 0 / 15 |

Eight initial continuations hit the CPU limit. Their recorded tactics were prefixes of the longer continuations, and all recorded initial recovery choices agreed within each architecture. All final trials stayed within the CPU budget. The routed hybrid improved on unrouted RL and calibration at this seed. Imitation led by one complete pair.

Routed RL selected a viable branch in 41 worlds and certified 40. Imitation selected a viable branch in 42 worlds and certified all 42. Both policies certified both worlds of every pair where they selected both viable branches. The seed-103 pair difference arose at the recovery decision.

Seed 151 also placed routed RL one pair behind imitation and ahead of calibration. Routed and unrouted RL closed the same 14 pairs. The seed-151 routed checkpoint was exported with CPU weights and its search configuration. Reloading it reproduced a seven-check native Lean proof from the audit, including its recovery decision and tactic sequence.

Seed 197 placed routed RL one pair ahead of imitation and unrouted RL. Across the three seeds, routed RL completed 42 pairs, imitation 43, unrouted RL 39, and calibration 22. Both TRM arms completed zero. The 96 seed-layout observations share 32 layouts. The combined audit groups the results by layout.

### Stronger pending-rule readout control

The history-only intervention zeros the direct parent/current score. The routed model still uses those inputs in its arrival gate. On seed 103, routed RL finished 31 of 32 pairs, imitation 32, calibration 28, and unrouted RL 30. This is a frozen development intervention on the same layouts as the main comparison.

On seed 151, the history-only intervention brought routed RL, unrouted RL and calibration to 30 of 32 pairs. Imitation again certified all 32. On seed 197, all four hybrid checkpoints certified all 32 pairs under this intervention. Calibration certified 24 worlds with newly executed and replayed trajectories and reused 40 unchanged trajectories from the native audit. These controls raise the baseline for subsequent RL variants.

### Progress in failed searches

Some recovery pairs end with two uncertified proofs at the same tactic cost. Their original action returns tie. The progress audit counts discharged goals along the surviving sequence of Lean-accepted tactics and discards any discharges undone by a rollback. On the seed-103 and seed-151 routed training ledgers, progress favored the structurally viable branch in 31 of these 48 pairs. The other 17 remained tied. A shortest-public-Horn reference assigned viability for this audit and agreed with the independently certified development witness in every world. A follow-up audit on seed 197 found seven aligned pairs, two that favored the structurally nonviable branch, and nine ties. Goal discharge supplies a noisy preference signal.

The next transfer cohort keeps all 166 existing training theorems and moves development to 64 new pairs that open three goals after the shared first tactic. A wrong branch either lacks a fact for its second goal or has support split across competing rules for its third goal. Both mechanisms keep a legal continuation after the first tactic. Static proof-cost checks identified one seven-check branch in all 128 new worlds. The original reserved layouts stay sealed.

Lean verified both recovery choices on all 64 transfer pairs. The registered training comparison adds a 0.125 action preference only when both in-budget searches fail at equal tactic cost and one retains more completed goals. The RL critic keeps the original certificate-and-cost target. The progress-aware imitation control receives the same preference as an action label. The eight-arm comparison also retains both terminal-reward controls, routed readout calibration, an unrouted hybrid, and the two graph TRM controls.

The history-only diagnostic also covers all six hybrid checkpoints in the three-obligation comparison, including the terminal-reward controls. It uses the audited session and weight hashes and reports paired certificate changes on the same layouts. A changed recovery or continuation decision requires its own native Lean rollout and replay.

The arrival decomposition uses the exact eight channel coordinates of the trained arrival projection. The diagnostic reconstructs its native contribution across eight hops, then swaps the donor world's early, late, or full arrival contribution into the matched recipient. The recipient keeps its own router gain and other scores. A common logit shift of the same norm controls for the intervention magnitude. The result measures decision changes on audited development states.

On the 64 two-goal development worlds at seed 151, both early and late swaps moved the RL and imitation margins toward the donor decision in every world. The full swap changed zero decisions. Its median margin change was 0.00071 for RL, compared with a median native margin of 0.427, and 0.00019 for imitation, compared with 0.528. The frozen calibrated arrival projection supplied zero change. These results support testing the larger history and readout contributions before increasing the arrival budget.

In a native interim readout test on the three-goal cohort, the progress-trained RL checkpoint certified 62 of 64 pairs and calibration certified 60. The viable recovery choices were 64 and 61 pairs, respectively. Full proof completion includes subsequent tactic choices. The remaining history-only controls require the complete matched audit.

## Native BPD rewrite transfer

BPD-LG-02's original rewrite declarations now enter the same bounded Lean search used for Horn proofs. A rewrite action moves the left side of an equality through one local equality premise. Both recurrent policies receive the same public bidirectional graph, with evidence seeded at the right side and the current left side marked as the active target. Search, backtracking, independent proof compilation, and replay use this tactic menu.

The frozen study adds 64 original BPD rewrite training declarations to 86 LoopedTransformers, BPD exact, and EvalMath training declarations. The CPU trials also consume the audited MathCan training archive. The 16 original BPD rewrite development declarations cover four structural groups. The public shortest-path reference certified all eight valid declarations within eight checks. Every search replayed in Lean 4.19.0. The five model arms compare hybrid RL, hybrid calibration, hybrid imitation, graph TRM RL, and graph TRM calibration under the same budgets. The symbolic reference sets the ceiling for this family.

The warm hybrid checkpoint certified one of the eight valid development declarations before training. It revisited an accepted proof state on seven of those eight declarations. The completed prefix recorded four updates, including four native rewrite rollouts. The visit counts provide a diagnostic for testing loop avoidance with the same tactic budget.

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
python -m configs.probe_ppt_pending_alias_history PENDING-ALIAS-COHORT ROUTED-RUNS EVIDENCE-103 HISTORY-EVIDENCE --native
python -m configs.export_ppt_pending_alias_checkpoint PENDING-ALIAS-COHORT ROUTED-RUNS/hybrid_recovery_rl.sqlite EVIDENCE-151 CHECKPOINT
python -m configs.audit_ppt_recovery_progress PROGRESS-AUDIT --run RUNS-103/hybrid_recovery_rl.sqlite EVIDENCE-103 --run RUNS-151/hybrid_recovery_rl.sqlite EVIDENCE-151
python -m configs.freeze_ppt_three_goal_recovery PENDING-ALIAS-COHORT THREE-GOAL-COHORT
python -m configs.run_ppt_branch_recovery THREE-GOAL-COHORT MATHCAN-TRAIN.zip INITIALIZATION PROGRESS-RUNS --pending-arrival-router --progress-study PROGRESS-STUDY --workers 2
python -m configs.analyze_ppt_pending_alias THREE-GOAL-COHORT PROGRESS-RUNS INITIALIZATION SUMMARY EVIDENCE --progress-study PROGRESS-STUDY
python -m configs.probe_ppt_pending_alias_history THREE-GOAL-COHORT PROGRESS-RUNS EVIDENCE HISTORY-EVIDENCE --progress-study PROGRESS-STUDY --native
python -m configs.probe_ppt_recovery_arrival PENDING-ALIAS-COHORT ROUTED-RUNS EVIDENCE-151 ARRIVAL-EVIDENCE
python -m configs.freeze_ppt_bpd_rewrite ROOT-ADAPTATION-COHORT REWRITE-COHORT
python -m configs.check_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE
python -m configs.run_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE MATHCAN-TRAIN.zip INITIALIZATION REWRITE-RUNS
python -m configs.analyze_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE REWRITE-RUNS INITIALIZATION REWRITE-AUDIT
```

The analysis requires EvalMath's FSI-01 module on the Python path. The runner uses the original Lean verifier and CPU-only PyTorch. The campaign runs under a 6 GiB process-tree cap, with 2 GiB and two PyTorch threads per worker.
