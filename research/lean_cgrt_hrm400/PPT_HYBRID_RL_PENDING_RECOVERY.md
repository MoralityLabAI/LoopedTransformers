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

| Three-goal development arm | Certified pairs out of 64 | Certified proofs out of 128 |
|---|---:|---:|
| Routed progress RL | 38 | 101 |
| Hybrid calibration | 36 | 99 |
| Progress imitation | 42 | 106 |
| Graph TRM RL | 4 | 53 |
| Graph TRM calibration | 2 | 49 |
| Unrouted progress RL | 39 | 102 |
| Routed terminal-reward RL | 37 | 100 |
| Terminal-reward imitation | 40 | 104 |

The audited seed-151 run gave routed RL 15 direct progress preferences during training. Routed RL gained one certified pair over its terminal-reward control. Progress imitation gained two pairs over its terminal-reward control and finished four pairs ahead of routed RL. The pinned rerun reused six source-matched ledgers and retrained both terminal-reward controls from the original revision. Peak committed process-tree memory stayed below 1.3 GB.

The history-only diagnostic also covers all six hybrid checkpoints in the three-obligation comparison, including the terminal-reward controls. It uses the audited session and weight hashes and reports paired certificate changes on the same layouts. A changed recovery or continuation decision requires its own native Lean rollout and replay.

The arrival decomposition uses the exact eight channel coordinates of the trained arrival projection. The diagnostic reconstructs its native contribution across eight hops, then swaps the donor world's early, late, or full arrival contribution into the matched recipient. The recipient keeps its own router gain and other scores. A common logit shift of the same norm controls for the intervention magnitude. The result measures decision changes on audited development states.

On the 64 two-goal development worlds at seed 151, both early and late swaps moved the RL and imitation margins toward the donor decision in every world. The full swap changed zero decisions. Its median margin change was 0.00071 for RL, compared with a median native margin of 0.427, and 0.00019 for imitation, compared with 0.528. The frozen calibrated arrival projection supplied zero change. These results support testing the larger history and readout contributions before increasing the arrival budget.

With the direct readout suppressed, routed progress RL, unrouted progress RL, and terminal-reward RL each certified 62 of 64 pairs. Calibration certified 60. Progress imitation and terminal-reward imitation each certified all 64 pairs. Both routed RL checkpoints chose the viable branch in all 128 worlds. Each finished 126 proofs, so the remaining gap was in continuation after the recovery choice. The earlier RL and calibration receipts matched the audited sessions and were reused.

## Native BPD rewrite transfer

BPD-LG-02's original rewrite declarations now enter the same bounded Lean search used for Horn proofs. A rewrite action moves the left side of an equality through one local equality premise. Both recurrent policies receive the same public bidirectional graph, with evidence seeded at the right side and the current left side marked as the active target. Search, backtracking, independent proof compilation, and replay use this tactic menu.

The frozen study adds 64 original BPD rewrite training declarations to 86 LoopedTransformers, BPD exact, and EvalMath training declarations. The CPU trials also consume the audited MathCan training archive. The 16 original BPD rewrite development declarations cover four structural groups. The public shortest-path reference certified all eight valid declarations within eight checks. Every search replayed in Lean 4.19.0. The five model arms compare hybrid RL, hybrid calibration, hybrid imitation, graph TRM RL, and graph TRM calibration under the same budgets. The symbolic reference sets the ceiling for this family.

The warm hybrid checkpoint certified one of the eight valid development declarations before training. It revisited an accepted proof state on seven of those eight declarations. The completed prefix recorded four updates, including four native rewrite rollouts. The visit counts provide a diagnostic for testing loop avoidance with the same tactic budget.

The shared revisit control gives an unvisited equality target priority when one is available. It changed the warm hybrid from one to five certified proofs and the warm graph TRM from zero to three. On valid declarations, accepted state revisits fell from 46 to one for the hybrid and from 51 to 12 for TRM. The control leaves the policy scores among unvisited candidates intact. Applying it to every audited final checkpoint measures what training contributes after the same search improvement.

| BPD rewrite arm | Final certificates | Final certificates with revisit control | Certificates in 128 training rollouts |
|---|---:|---:|---:|
| Hybrid RL | 0 | 3 | 7 |
| Hybrid calibration | 1 | 5 | 10 |
| Hybrid imitation | 3 | 4 | 29 |
| Graph TRM RL | 0 | 4 | 1 |
| Graph TRM calibration | 0 | 4 | 2 |

The revisit control also changes the amount of feedback available during training. On the 64 original rewrite training declarations, paired sampling from the frozen hybrid certified two proofs with ordinary exploration and 12 with the revisit control. The registered follow-up applies the rule during rollouts in all five arms. Each updated policy uses the eligible actions from its recorded rollout to compute the training probabilities. Lean replays those choices, and final development is scored with both inference settings.

| BPD rewrite arm trained with the revisit rule | Final certificates | Final certificates with revisit control |
|---|---:|---:|
| Hybrid RL | 4 | 7 |
| Hybrid calibration | 1 | 5 |
| Hybrid imitation | 0 | 4 |
| Graph TRM RL | 0 | 4 |
| Graph TRM calibration | 0 | 4 |

Seed 151's Hybrid RL checkpoint improved from one to four certified proofs under ordinary inference. It finished two proofs ahead of calibration and three ahead of imitation and graph TRM under the shared revisit control. The two registered replications used the same original BPD developments at different model seeds.

At seed 103, ordinary inference certified three proofs for hybrid RL and zero for each control. Under the shared revisit rule, the warm hybrid and warm graph TRM each certified two. Training raised hybrid RL to five, tied with hybrid imitation. Both graph TRM arms and hybrid calibration certified three.

| Exploration training arm | Ordinary inference, seeds 103 / 151 / 197 | Shared revisit rule, seeds 103 / 151 / 197 |
|---|---:|---:|
| Hybrid RL | 3 / 4 / 1 | 5 / 7 / 2 |
| Hybrid calibration | 0 / 1 / 2 | 3 / 5 / 4 |
| Hybrid imitation | 0 / 0 / 0 | 5 / 4 / 5 |
| Graph TRM RL | 0 / 0 / 0 | 3 / 4 / 4 |
| Graph TRM calibration | 0 / 0 / 0 | 3 / 4 / 4 |

The hybrid and graph TRM had 1,663,266 and 1,663,362 parameters, respectively. Each arm consumed 192 BPD, 96 LoopedTransformers, 12 EvalMath, and 36 archived MathCan training episodes. Hybrid RL led both graph TRM arms at every seed under ordinary inference. Under the shared revisit rule it tied hybrid imitation at 14 certificates across all seeds, and calibration led hybrid RL at seed 197. These runs reuse eight valid declarations and four structural groups.

The parameter substitutions kept the same native inference budget. At seed 151, moving the learned tied cell and message bank into the calibrated checkpoint raised its count from five to seven. Moving the calibrated recurrence and bank into RL reduced its count from seven to two. The tied cell alone reproduced those two changes. Moving only the message bank into calibration left it at five, and moving the calibrated bank into RL reduced it to three. Transplanting the graph, fact-transport, or action-head parameters between RL and calibration left their respective counts at seven and five.

At seed 103, the learned tied recurrence and bank raised the calibrated checkpoint from three to five. The reciprocal substitution reduced the RL checkpoint from five to three. Together, these interventions locate the advantage over calibration in learned recurrent computation on the original BPD layouts. The comparison with imitation still depends on seed and inference setting.

The portable seed-151 RL checkpoint reloads on CPU and reproduces a six-check native Lean proof in each inference mode. The export carries the exact model weights and the audited search settings.

The registered tied-cell training comparison freezes every other model parameter in its RL and imitation arms. The head-only calibration arm starts from the same warm checkpoint. Search and reward keep the earlier exploration protocol. The audit compares the final frozen weights with their initialization in addition to verifying both inference modes.

At seed 151, cell-only RL and imitation both certified one proof with ordinary inference, matching the calibrated head. They certified four and three proofs under the shared revisit rule, respectively. The calibrated head certified five. The final weight hashes verified that both cell-only runs changed the tied cell and preserved every weight outside it. The seven-proof full-model RL result therefore motivates training the cell together with the rest of the policy.

The component intervention reconstructed the ordinary-training hybrid's eight transport channels at every hop. Using only their fact-transport scores with the revisit control certified all eight valid proofs for the ordinary-training RL, calibration, and imitation checkpoints. The score isolates a stronger control than the end-to-end calibrated policy on these declarations.

## Fresh BPD-style rewrite readout

The new readout freezes the seed-151 exploration checkpoints for all five arms. It draws 96 fresh anchored equality forests with three-to-seven rewrite spines. Each layout pairs a valid theorem with a degree-matched disconnected variant. Lean independently certified all 96 valid proofs within eight checks and replayed the native reference on both worlds.

The primary comparison scores one valid theorem per layout under the shared revisit rule. Ordinary inference is recorded as a secondary readout. EvalMath's FSI-01 scores the paired outcomes against the four preregistered controls at a five-point effect floor. The layout schema is exclusively for evaluation.

| Frozen seed-151 policy | Certificates with the shared revisit rule | Certificates with ordinary inference |
|---|---:|---:|
| Hybrid RL | 23 | 9 |
| Hybrid calibration | 20 | 3 |
| Hybrid imitation | 21 | 2 |
| Graph TRM RL | 17 | 1 |
| Graph TRM calibration | 12 | 0 |

The shared-rule gains over calibration and imitation were 3.1 and 2.1 percentage points. FSI-01 classified these two comparisons as inconclusive and the comparisons with both graph TRM arms as suggestive. The readout replayed all 960 outcomes under the registered search budgets. The stronger transfer claim remains open.

The seed replication fixes the five exploration checkpoints at seeds 103 and 197 before evaluation. It reuses the same 96 transfer layouts and Lean reference, so the comparison tests the sensitivity to model seed on a fixed layout population. The primary comparison uses the shared revisit rule. Ordinary inference remains secondary. The analysis groups both seeds by layout and keeps the original five-point minimum effect.

The seed-103 follow-up exchanges the tied proof cell between the fully trained hybrid RL checkpoint and its head-calibrated control. Each recombined policy replays the same 96 layouts with the shared revisit rule. A second control uses only the calibrated hybrid's typed transport and keeps the same tactic and CPU ceilings.

At seed 103, hybrid RL certified 43 of 96 layouts with the shared revisit rule. Calibration certified 19, imitation 13, graph TRM RL ten, and graph TRM calibration 12. Ordinary inference yielded 31, five, one, one, and one, respectively. On the shared rule, hybrid RL gained 29 individual proofs and lost five relative to hybrid calibration. The RL cell raised the calibrated policy from 19 to 36 certificates. The calibrated cell lowered RL from 43 to 29. Isolated calibrated transport certified all 96 proofs. This family distinguishes the learned complete policy from ordinary readout calibration and leaves a strong fixed inference control. The portable seed-103 CPU weights reproduced two independently certified transfer proofs, including one without revisit masking.

At seed 197, the shared-rule counts were 26 for hybrid RL, 24 for hybrid calibration, 27 for hybrid imitation, 14 for graph TRM RL, and ten for graph TRM calibration. Across both replicated seeds they were 69, 43, 40, 24, and 22. FSI-01 rated the combined comparisons suggestive and grouped the 192 seed-layout outcomes by their 96 layouts. The advantage over imitation varied by seed.

The archived BPD training comparison found 28 matched forks with distinct returns across seeds 103, 151, and 197. Twenty-six were at the opening. All of the unequal returns included at least one certified trajectory. The interior-fork collection chooses the earliest eligible continuation after an accepted prefix using the seed-103 training ledger. It reruns that prefix in Lean for every eligible action and uses the same frozen continuation policy to obtain each return.

Lean replayed 108 native branches at 51 interior forks in the original training split. Seven branches certified, and five forks supplied nine unequal return comparisons. Two other forks supplied different costs when Lean independently certified extensions of the failed eight-check prefixes. These extensions use at most four extra checks as training evidence. Held-out certification keeps the eight-check budget.

The target-contrast diagnostic selects 12 of those layouts by a fixed hash, with three each at proof depths four through seven. At each rewrite decision, it subtracts the graph score with the target evidence bit cleared from the score with that bit present. Node identities, edges, and the active left-hand symbol stay fixed. This removes the target-independent graph score from the action margin. Each policy uses the same revisit rule, and the native gym charges both graph evaluations. The diagnostic also evaluates the ordinary graph score, isolated typed transport, and target contrast with typed transport suppressed.

| Frozen policy | Ordinary graph | Target contrast | Target contrast without typed transport | Isolated typed transport |
|---|---:|---:|---:|---:|
| Hybrid RL | 0 | 12 | 3 | 12 |
| Hybrid calibration | 3 | 12 | 3 | 12 |
| Hybrid imitation | 5 | 12 | 5 | 12 |
| Graph TRM RL | 2 | 0 | | |
| Graph TRM calibration | 1 | 2 | | |

Each count is a standalone Lean certificate on the same 12 positive layouts. Every new trajectory replayed within the original CPU and tactic budgets. All three trained hybrid checkpoints reached the reference ceiling through isolated typed transport on these layouts. The learned contrast was highest for imitation.

## Bundle opening feasibility

The next proof-search task presents two local conjunctions of equality premises. Opening either conjunction leaves the target in place. One opening reveals a seven-rewrite route. Opening the other first spends the extra check needed to finish within eight. Independent Lean compilation certified the viable eight-check route in both worlds of all eight synthetic BPD-style layouts. The failed route in each world also replayed, including the immediate opening of the viable bundle after the first choice.

The gym now scores prospective equality graphs for the two public bundle openings. Each view contains the equalities Lean has exposed and the equalities declared in that candidate bundle. The same view is available to the hybrid and graph TRM. The hybrid uses its gated recurrent updates and fact transport to score the view. The arithmetic ledger charges the extra graph passes.

The frozen comparison adds 32 paired training layouts and 24 new paired development layouts to the mixed BPD rewrite training declarations. It holds the rewrite continuation fixed across the five arms so that the learned decision is the first opening. During training, each arm executes both opening proposals in both worlds and receives credit from the final native certificate. Imitation sees the matched successful opening. Calibration trains the readout with the recurrent weights frozen. Development performance requires independent Lean certificates in both worlds of a layout. The gym replayed all 96 opening proposals across 48 development worlds. In each world only the registered opening reached a certificate within eight checks.

The frozen hybrid and graph TRM each picked the viable opening in 26 of 48 worlds before bundle training. Those choices covered five complete pairs for the hybrid and three for graph TRM. The hybrid's lexical score covered two pairs, its graph preview covered four, and suppressing typed fact transport left its full score at five. The graph TRM preview alone covered eight. These counts compare opening choices with the native witness. The witness executes the subsequent proofs.

The full mixed-source run finished with 17 of 24 certified pairs for graph TRM RL, five for hybrid imitation, four for hybrid calibration, three for graph TRM calibration, and two for hybrid RL. Every arm consumed 192 BPD, 12 EvalMath, 96 LoopedTransformers Horn, 128 bundle, and 52 archived MathCan episodes. Lean independently certified 64 bundle training rollouts per arm with no CPU censoring. The campaign peaked at 1.39 GiB of committed memory. The target-membership cue in this first cohort limits the interpretation of the graph TRM gain.

In this first cohort, the viable bundle is the only bundle that mentions the equality target. The matched cohort freezes 32 new training pairs and 24 new development pairs with disjoint spur layouts. Both options now contain two chains and mention the target exactly once. Swapping the terminal edges changes whether the start is connected to the target. The number of appearances of each atom and the in-degree and out-degree profiles stay fixed. The gym checked those properties in all 48 Lean development states, replayed both choices in every state, and independently compiled 48 proofs using eight checks each.

On the matched layouts, the frozen hybrid and graph TRM each selected the viable opening in 24 of 48 worlds and in zero complete pairs. The graph preview alone reached one complete pair for each architecture. Every opening exposes exactly one initial rewrite. Subtracting the scores obtained after suppressing typed fact transport therefore isolates the transport score. That frozen score selected the independently verified opening in all 48 worlds and all 24 pairs. The cue is present in the original checkpoint and obscured by the full readout.

The shared-branch experiment collects both verified training proposals in each world once. It gives the five update rules identical native outcomes and compares their restored weights on the frozen development observations. This experiment adapts the warm checkpoints on the bundle branch traces. The mixed-corpus run continues to include the original Lean and archived MathCan training schedules. The shared-branch experiment also executes a fresh Lean rollout in both worlds of one layout per restored arm.

The shared-branch experiment used 128 Lean rollouts and 64 independently compiled training proofs. After 32 paired updates, hybrid calibration selected the verified opening in 26 of 48 development worlds and in both worlds of two pairs. Hybrid RL, hybrid imitation, graph TRM RL, and graph TRM calibration each selected the verified opening in 24 worlds and in zero complete pairs. All five checkpoints changed, reloaded, and reproduced the native outcomes in the two-world checks.

The paired policy gradient increased the hybrid's median absolute mean opening margin within a layout from 0.638 to 40.463. The median half-difference between the two worlds increased from 0.0096 to 0.0121. Isolating the trained hybrid's transport contribution still recovered all 24 pairs.

Fitting the paired action-value differences on the same 128 traces reduced the hybrid's median absolute mean margin to 0.249 and increased its half-difference to 0.0230. Hybrid RL selected the viable opening in both worlds of two pairs and in 26 of 48 worlds. Hybrid calibration reached one pair and 25 worlds. Hybrid imitation and both graph TRM controls reached zero pairs and 24 worlds. The five restored checkpoints reproduced their fresh two-world Lean checks. The hybrid advantage over the corresponding calibration arm was one pair. The isolated frozen transport control remained at 24 pairs.

The deadline cohort keeps the degree controls and makes the target reachable after either opening. The two public routes require seven and eight rewrites. Since opening the bundle costs one tactic check, only the seven-rewrite route fits the eight-check budget. The paired worlds exchange which side reaches the shorter route. Both bundles have the same degree profile, with spurs spread across one approach to the target and concentrated on the other. The gym measured both distances from Lean's public contexts, replayed both openings in all 48 worlds, and independently compiled all 48 viable proofs. The frozen hybrid and graph TRM each selected the verified route in 24 worlds and zero full pairs. The frozen hybrid's isolated transport selected the shorter, verified route in every world.

## Equality deadline transfer

Lean independently certified the viable first rewrite in both worlds of 24 new equality layouts within eight checks. The other first rewrite spent the budget without closing. Each layout keeps the same two root actions and switches the viable branch. In half the layouts, fixed diffusion prefers the branch that needs a ninth check.

The frozen exploration checkpoints chose both viable branches in zero pairs with their full scores. Each hybrid's isolated fact transport chose both viable branches in all 12 diffusion-aligned pairs and zero misleading pairs. Counting only support that arrived in seven hops recovered both viable branches in all 24 pairs for all three hybrid objectives. This diagnostic measures root decisions using the native witness's feasibility labels. Its fixed cutoff provides a control for subsequent deadline training.

## Mixed equality and proposition costs

The next fixture presents two Lean rules for the same conclusion. Each rule opens an equality goal followed by a pending proposition. The two paths require eight and nine checks, including the opening tactic. Within a pair, the equality chains stay fixed and the proposition rules exchange which opening fits within eight checks. The choice therefore depends on the total cost of both obligations.

Lean replayed both openings in all 16 worlds and independently compiled the 16 proofs that fit the eight-check budget. Choosing on equality cost alone or on proposition cost alone found eight of the 16 valid openings and zero complete pairs. These eight layouts are a development fixture for a policy that combines both obligations.

The mixed-obligation policy projects each proposed rule into its two public goal graphs and runs the same evidence recurrence on both. Its readout combines the maximum action score and active-goal value from each graph. Training used 60 traces from 15 paired layouts with an equality-cost gap of one, including 30 independent certificates. All eight development layouts have a larger gap. After one pass through the traces, hybrid RL, hybrid imitation, and hybrid calibration finished with zero complete pairs. Graph TRM RL finished with two, and graph TRM calibration with one. All five CPU checkpoints reloaded and reproduced two fresh Lean trajectories each.

The phase-cost model predicts the two tactic counts separately and selects the opening with the lower predicted total. Its update takes the equality count from Lean's first one-goal state. A certified continuation supplies an exact proposition count. An unfinished proposition supplies a lower bound of one more check than the branch already spent. The calibration control fits the same targets with the recurrent graph frozen. Imitation uses the same policy architecture and the successful opening labels.

After four passes through the same traces, the phase-cost hybrid RL, hybrid imitation, and graph TRM RL each finished with zero complete pairs. Hybrid calibration finished with one and graph TRM calibration with two. Replacing the learned estimate for either phase with its exact witness cost still gave zero complete pairs for every checkpoint. For hybrid RL, the mean absolute errors were 1.61 equality checks and 1.47 proposition checks. The next readout estimates each phase from changes in target-conditioned evidence across one to four recurrent passes. Both architectures and their frozen-encoder controls receive the same four depths.

The four-depth experiment yielded two of eight complete pairs for hybrid calibration and one for graph TRM RL. Hybrid RL, hybrid imitation, and graph TRM calibration finished with zero. All five checkpoints restored and reproduced their two native trajectories.

## Conditional rewrites with delayed goals

Both initial equality routes now have the same length. After an unconditional opening, two later rewrites each create a proposition obligation. The paired worlds keep the equality graph fixed and exchange which branch's conditions can be discharged before the eight-check limit. Lean replayed both branches in all 16 development worlds and independently compiled all 16 feasible proofs. A deterministic policy that sees only the equality graph misses one world in each pair.

The conditional policy places proposition facts and guarded equality edges in the same recurrent graph. A guarded edge requires both its equality successor and its proposition premise. Once Lean opens an obligation, the graph also marks it as pending. Both recurrence variants load their previous checkpoints without adding parameters. Before training, the conditional and ordinary scores each picked a valid branch in eight worlds and in zero complete pairs. The new curriculum provides 108 independently replayed branches from 27 structurally disjoint training pairs, including 54 standalone Lean proofs.

After four passes over those traces, hybrid RL and hybrid imitation each selected nine certified branches and finished one of eight complete pairs. Hybrid calibration and both graph TRM controls finished zero pairs. The isolated calibrated transport in the joint graph selected all 16 viable branches, as did the public conditional-cost reference. The next native check tests quantified rewrite rules whose premises become concrete after Lean matches the term. For `h0 : ∀ n : Nat, n = c → f n = g n`, rewriting `f a = g a` with `h0` leaves the obligation `a = c`. Lean compiled the two-check proof using the instantiated rule and the local equality.

## Dependent rewrite search

The two routes rewrite the same binary function application to the same target. A rule can swap the two arguments. Later guards bind to the arguments in their new positions, so the choice of earlier rewrites changes the subsequent equality obligations. The paired worlds keep the quantified rules fixed and exchange the costs of proving the guards. One branch completes in eight checks and the other requires a ninth.

The shared recurrent graph instantiates each public rule in both argument orders and connects the resulting term edge to its guard. Pending Lean equality goals mark their corresponding guard nodes. This preserves the original hybrid and graph TRM parameter counts. The warm checkpoints and the 768 public tactic observations are shared across objectives. The 24 training layouts supply both native opening outcomes in each world and 48 independently compiled proofs. Eight disjoint development layouts include longer term paths.

The primary evaluation lets the policy choose among eligible tactics in a fresh Lean session, with the same eight-check limit and the same restriction on immediately undoing the preceding rewrite. The opening-only readout also measures the first choice against the frozen branch witnesses. The fixed controls use the calibrated model's isolated transport and a public guard-cost reference. Both search and branch witnesses require native replay and standalone Lean certificates.

On the first dependent layouts, the seed-103 hybrid RL policy completed both worlds in one of eight held-out layouts. It certified seven of 16 proofs, hybrid imitation certified eight, hybrid readout calibration certified seven, and graph TRM RL certified two. Only hybrid RL completed a pair among these trained arms. The isolated transport and the public guard-cost reference each certified all 16 proofs. This fixture separated the trained arms on complete pairs and remained solvable by the frozen transport.

The dead-end equality study uses 24 new training layouts and eight new development layouts, disjoint from the first study. Each world adds a single equality connecting one argument to an otherwise unused natural-number variable. Both members of a layout receive the same extra equality, and Lean independently verified the expected winning branch in all 64 worlds. The extra edge leaves the eight-check shortest proof intact. It dilutes transport evidence by adding an incoming message from the dead end to the uniform average.

Uniform frozen transport certified eight of 16 development proofs and zero complete pairs. A fixed gate that weights messages by the mean and weakest premise evidence increased that to ten proofs and two pairs. The public guard-cost reference certified all 16 proofs and all eight pairs. The trainable gate uses the same two premise features with two learned weights. Zero initialization reproduces the uniform model's scores exactly, and the matched uniform RL arm keeps the same warm checkpoint and training trajectories. The gated replay uses the same eight-check limit, independent Lean certificates, and arithmetic ceiling.

A separate ridge calibration fit jointly updated all three linear readouts of the warm hybrid with its encoder frozen. It selected the regularization strength by leaving out one training layout at a time, including both worlds in each fold. On the eight held-out dead-end layouts it certified eight of 16 proofs and zero complete pairs, matching the frozen transport count.

The seed-103 gated hybrid RL, gated hybrid imitation, gated hybrid readout calibration, and uniform hybrid RL each certified eight of 16 proofs and zero complete pairs. Both graph TRM arms certified zero complete proofs. Every trained arm chose a certified opening in eight worlds and zero complete pairs. After one pass over the training layouts, the gate weights were about 0.0021 and 0.0019 under RL and 0.0216 and 0.0215 under imitation. For this pilot, the learned gate tied uniform transport, and fixed support weighting retained the strongest score among the transport models. All six checkpoints were reloaded on CPU and hashed, and every live tactic sequence passed replay.

### Routed transport and proof cost

On the training layouts, stronger support weighting alone chose the certified opening in both worlds on 13 of 24 pairs. Adding the unchanged seed-103 recurrent scores reduced that count to zero at the same gate values. The routed model keeps access to both scores and adjusts the recurrent workspace's contribution independently of the support gate. It inherits the same shared recurrent cell and typed evidence bank and adds one scalar to the two gate weights. The three routing parameters are the only weights changed during fitting.

Both warm hybrid checkpoints came from the mixed CPU exploration campaign. Each recorded 192 BPD, 96 LoopedTransformers, 12 EvalMath, and 36 audited MathCan training episodes. The graph TRM comparison has 1,663,362 parameters and receives the same native dependent-rewrite training layouts.

Lean certified both paths in the 48 training worlds, including the path that needs a ninth check. The first eight tactic results match the original registered traces. Behavioral imitation uses every eligible choice in both certified prefixes through check eight. The return objective assigns 0.8 to an eight-check proof and −0.2 to the branch whose proof falls outside the eight-check gym budget. Both fits select from the same 36 routes and start with the same warm CPU weights. The return fit maximizes the greedy policy's mean observed opening return, using complete paired layouts to break return ties. Imitation minimizes action cross-entropy on the certified prefixes. The fit uses the recorded suffixes to assign opening returns, and the held-out test lets the policy choose its own continuation in Lean.

The registered development cohort contains eight layouts disjoint from both earlier cohorts. The primary endpoint is a pair in which both complete proofs independently certify within eight checks. A meaningful exploratory gain requires at least two more complete pairs than each learned control and both frozen routing controls. Proof counts, opening counts, and the exact public-cost reference are reported separately. A positive pilot still calls for replication on fresh layouts and another warm seed.

| Seed-103 pilot | Complete pairs out of 8 | Certified proofs out of 16 |
| --- | ---: | ---: |
| Routed proof-return fit | 7 | 15 |
| Routed imitation | 1 | 9 |
| Frozen uniform route | 0 | 8 |
| Frozen support-weighted route | 3 | 11 |
| Ridge readout calibration | 0 | 8 |
| Graph TRM RL | 0 | 0 |
| Exact public guard cost | 8 | 16 |

The 1,663,269-parameter seed-103 pilot met the exploratory gain threshold. Every pair completed by either trained or frozen routing control was also completed by the return fit. The return objective set the mean-support weight to zero, used a weakest-support weight of four, and retained 1/64 of the recurrent workspace score. The seed-151 fit independently selected the same three values. The pilot tested 16 worlds at one seed, and the public guard-cost reference still achieved the highest score.

The component audit reused the seed-103 return checkpoint and the same held-out worlds. Uniform aggregation reduced the result from seven to zero complete pairs. Restoring the full recurrent score reduced it to one, and isolating the learned transport reduced it to five. Moving the weight of four from weakest support to mean support produced six pairs. Each intervention used fresh native searches, tactic replay, and standalone Lean certificates. The best result in this cohort retained both the learned gate and a small recurrent contribution.

The confirmation cohort holds out 12 further layouts and uses the seed-151 warm checkpoints. It checks the same eight-tactic outcome and requires at least three additional complete pairs over each learned or frozen routing control. Two controls also test the stronger transport representation directly: isolated frozen support weighting and ridge calibration with the support-weighted encoder frozen. The exact public guard cost remains the automation reference. The two routing fits for this seed were frozen before running the confirmation cohort.

| Seed-151 confirmation | Complete pairs out of 12 | Certified proofs out of 24 |
| --- | ---: | ---: |
| Routed proof-return fit | 5 | 17 |
| Routed imitation | 1 | 13 |
| Frozen uniform route | 0 | 11 |
| Frozen support-weighted route | 2 | 14 |
| Ridge readout calibration | 0 | 12 |
| Isolated frozen support weighting | 4 | 15 |
| Support-weighted ridge calibration | 4 | 12 |
| Graph TRM RL | 0 | 2 |
| Exact public guard cost | 12 | 24 |

The return fit again led imitation, graph TRM, and ordinary readout calibration. Each of the two stronger support controls completed four pairs, all included in the five completed by the return fit. The gain over those controls was one pair, below the registered three-pair threshold. The complete-search result for the return fit agreed with its opening-only result on both cohorts. The two cohorts establish gains over ordinary calibration and graph TRM for these layouts. The advantage over the stronger support controls remains inconclusive.

In a diagnostic cross-score after confirmation, both frozen return checkpoints completed seven pairs on the pilot cohort and five on the confirmation cohort. They shared six successful pilot pairs and four successful confirmation pairs. The lower count followed the cohort in this check, while changing the warm seed changed individual successes. All of these held-out searches and both router fits ran with one CPU thread per process. The evaluation runs peaked at 0.51 GiB and 0.54 GiB of committed memory for seeds 103 and 151, respectively.

### Learning serial proof work from live returns

To separate arrival depth from tactic work, the next cohort holds term path length and the longest guard proof fixed within each theorem. The losing opening needs one extra rewrite in the shorter guard. The public dependency graph gives the two alternatives the same earliest synchronous arrival in every world. Lean replayed both openings and certified the eight-check path in all 48 training and 24 held-out worlds. The shorter and longer proofs are saved independently for imitation, including the ninth tactic on the longer path.

The new return channel shares one update across eight graph hops. It interpolates between the minimum premise support, which models concurrent arrival, and the product of discounted premise supports, which counts the work across obligations. A learned power controls how alternatives aggregate. The channel adds its discounted edge score to the established workspace and fact-routing scores. Only the interpolation, selection power, and additive readout weight can change. The seed-103 routed checkpoint provides all inherited weights.

For each training layout, the return learner tries both public openings in both worlds against the current policy. The policy chooses its own continuation. Lean outcomes, tactic counts, replay, and independent certificates determine the per-opening returns. A policy-gradient update maximizes expected return after each layout. Matched imitation uses the same 24 layouts with both independently verified prefixes and the same trainable parameters. The held-out test gives every policy the original eight-check and CPU limits.

The primary endpoint remains the number of pairs with two independently certified proofs. The exploratory gain threshold on the 12 held-out layouts is three additional complete pairs over each learned or frozen routing control. The controls include the initial channel, a frozen product channel, calibration of the same product representation including its affine return readout, isolated support weighting, the previous return router, and graph TRM trained on the matched layouts. The exact public guard-cost solver is the automation reference. The fitted checkpoints are frozen before the new held-out searches.

| Tied-arrival layouts | Complete pairs out of 12 | Certified proofs out of 24 |
| --- | ---: | ---: |
| Live return learning | 12 | 24 |
| Imitation | 11 | 23 |
| Serial ridge calibration | 12 | 24 |
| Frozen product channel | 11 | 23 |
| Frozen initial channel | 9 | 21 |
| Previous return router | 7 | 19 |
| Isolated support weighting | 12 | 24 |
| Graph TRM RL | 0 | 2 |
| Exact public guard cost | 12 | 24 |

The live learner probed 96 openings and updated after each of the 24 training layouts. It moved the serial mixing fraction from 0.50 to 0.88 and the readout weight from 0.04 to 1.95. The imitation fit reached a mixing fraction of 0.56 and a readout weight of 0.04. The live return update improved over the initial channel and previous router. It tied serial ridge calibration and isolated support weighting, so the equal-arrival test fell short of the registered threshold. The training and evaluation jobs each used one logical CPU and peaked at 0.47 GiB and 0.49 GiB of committed memory.

### Reusing a rewrite before splitting a guard

The next cohort puts both equality obligations inside one focused conjunction. Each route has four term rewrites. Once split, each equality needs two more rewrites, so either route costs nine checks when the policy splits immediately. One route can rewrite the shared subject in both conjuncts with a single `rw` before `constructor`, reaching the eight-check limit. The paired world changes which route has that shared rewrite. The longer proof remains independently certifiable with a ninth check.

The guard planner enumerates finite equality and conjunction states under single-lemma rewrites, `constructor`, and `rfl`. It finds the shortest guard suffix for each registered term prefix. Lean executes the result, replays it from a fresh state, checks the proof in the kernel, and compiles it independently. Every reference action appears in the public menu reconstructed from the actual Lean state. Train and development layouts vary both argument permutations and the position of the guarded rewrite. Their roles and source roots are disjoint.

The stored witnesses cover 48 training worlds and 24 development worlds. Both branches independently compiled in all 72 worlds, for 72 eight-check proofs and 72 nine-check proofs. All 144 proofs cost nine checks under immediate splitting. The training witness includes the public observations and candidate menus along both verified paths.

The decoy cohort adds 12 further layout pairs. Both branches have three outgoing rewrites on the guard subject and a rewrite that can affect both conjuncts. The decoy route still needs an extra tactic to finish one of the two obligations. Lean replays both branches and independently verifies their eight- and nine-check proofs. The learner uses the original 24 training layouts. The learner checkpoints are frozen before either held-out learner evaluation. The controls are evaluated independently of those fits.

The model keeps one recurrent bank for individual equality obligations and another for the focused conjunction. Each transition in the joint bank applies one public substitution to the entire conjunctive target. A split commits to the two individual proof values and pays for `constructor`. The resulting joint value contributes to the term rewrite that would create that guard. The routed hybrid still scores the lexical and structural proof context, and the joint channel adds its recurrent return estimate. Its three trainable scalars adjust the value readout, the preference for splitting, and the pooling of alternative rewrites. The inherited workspace and fact transport stay frozen during this update.

The live learner tries both openings in each training world and picks its own eligible continuations. It updates after each layout using native proof returns. Matched imitation sees the independently certified prefixes of both branches. The other controls use the same test menu and budgets. They include the same joint bank with its initial parameters, ridge calibration of the initial bank and affine readouts, calibration of just the joint readout gain, a factorized bank that treats both obligations independently, and graph TRM fitted to both native opening receipts. Both calibration procedures select regularization by holding out complete training layouts. The max-product joint planner provides a structural reference. A positive exploratory result requires three additional complete pairs over every learned and frozen control on each held-out cohort. The structural planning ceiling is reported separately.

| Shared-rewrite policy | Ordinary pairs out of 12 | Decoy pairs out of 12 |
| --- | ---: | ---: |
| Live return learning | 12 | 0 |
| Imitation | 0 | 0 |
| Ridge calibration of affine readouts | 12 | 0 |
| Calibration of joint readout gain | 12 | 0 |
| Frozen shared bank | 0 | 0 |
| Factorized bank | 0 | 0 |
| Matched graph TRM RL | 0 | 0 |
| Max-product joint planner | 12 | 12 |

The live learner ran 96 native opening probes and updated after each of the 24 training layouts. It increased the pooling logit from zero to 2.04, decreased the split log weight to −2.01, and increased the joint readout weight from 0.15 to 2.21. On ordinary layouts the live learner and both calibrators certified all 24 proofs. On decoy layouts the live learner certified five individual proofs. Both calibrators and imitation certified zero. The paired result on the decoy cohort stayed at zero, below the registered threshold.

In the decoy component check, keeping the trained gain and restoring the initial recurrence reduced the correct openings from five to zero. Restoring the initial pooling logit alone also returned zero, and restoring just the split parameter retained two. An exploratory intervention increased the pooling logit to eight with the other learned parameters held fixed. It certified all 24 decoy proofs and all 12 pairs in new native searches. The fitted learner’s lower pooling strength left it short of that result. The fit and evaluation jobs used one logical CPU each, and the largest process-tree memory peak was 0.49 GiB.

The follow-up data separates 12 decoy training layouts and 12 fresh confirmation layouts from all earlier layouts and from each other. Each split contains 24 worlds and 48 independently compiled proofs, half closing in eight checks and half closing in nine. The training split includes the public observations and legal menus along both certified prefixes. This gives subsequent proof-return fitting direct exposure to decoys and leaves an independent cohort for measuring transfer.

The decoy continuation starts both learners, the frozen baseline, and the calibration controls from the previous live-trained hybrid. The return learner compares the incumbent pooling logit with two bounded perturbations on both worlds of each training layout. Radii cycle through six, three, and one and a half. Every trial chooses its entire tactic path in Lean. It retains an update only when the paired native proof return improves. Only the pooling logit can change. Imitation starts from the same checkpoint and uses the certified prefixes of both branches. Graph TRM continues from its prior matched fit and receives the same decoy training layouts.

The confirmation cohort remains sealed until the continuation checkpoints are frozen. Head-only calibration, the prior bank, the fixed sharpening intervention, and calibration on the sharpened bank are evaluated under the same eight-check budget. The fixed intervention uses the previously diagnosed pooling logit of eight. The primary score counts complete certified pairs and retains the three-pair exploratory margin over each learned and frozen control. The ordinary development layouts serve as a retention check.

| Decoy continuation | Fresh confirmation pairs out of 12 | Ordinary retention pairs out of 12 |
| --- | ---: | ---: |
| Live pooling search | 12 | 12 |
| Imitation continuation | 10 | 12 |
| Ridge calibration of the prior bank | 0 | 0 |
| Calibration of just the prior joint gain | 12 | 12 |
| Prior live checkpoint | 0 | 12 |
| Independent work | 0 | 0 |
| Continued graph TRM RL | 0 | 0 |
| Fixed sharpening | 12 | 12 |
| Ridge calibration of the sharpened bank | 12 | 12 |
| Max-product joint planner | 12 | 12 |

The live search ran 72 native proof trials. On the first decoy layout it selected 8.04 over the inherited 2.04 and retained that setting. The selected policy closed both proofs in every training layout and in every independent confirmation layout. Imitation closed 22 of 24 confirmation proofs. The prior checkpoint closed five. Ridge calibration of the prior bank found all 24 correct openings but closed zero proofs. The gain-only control selected a much larger weight, 283.01, and closed all 24. The live learner recovered the effect of the hand-set sharpening and tied the strongest controls, so this confirmation also fell below the registered margin. Each evaluation process used one logical CPU and peaked at no more than 0.52 GiB of committed memory.

### Choosing a paid normalization with local definitions

The alias cohort holds the raw goal, hypothesis types, and starting tactic menu fixed within each pair. The two worlds swap the values of two local definitions. Lean displays those values in the public printed context. The definition values determine which route can finish its conjunctive guard with a shared rewrite. Each opening menu includes either term rewrite and either single-name `dsimp only` action, so the policy can normalize first or commit to a branch first. The in-budget opening sets are disjoint across the paired worlds.

The projected guard search tracks whether the conjunction has been split and charges one check to normalize a local definition when it is used in the target. It enumerates legal single substitutions, `constructor`, and `rfl`. For each of the four initial tactics, the generator replays a shortest projected plan in Lean and independently compiles it. It checks both the live result after eight tactics and the eventual proof. The raw-only observation check includes the initial tactic menu and omits the printed definition values. This establishes a boundary for calibrating the existing raw-only policies under this grammar. The printed context is available to both an alias-aware learner and any future control that encodes it.

The stored cohorts contain 48 training and 24 development worlds with 288 independently compiled opening plans. Both existing models received the complete public packets for the opening diagnostic. The adapted hybrid and continued graph TRM each produced exactly identical opening scores in all 24 training pairs and all 12 development pairs. Each chose an in-budget opening in one world per pair. A deterministic readout recalibration on the same raw-only representation still gives the same action to both members, so its complete-pair upper bound in this protocol is zero.

### Alias-aware live return study

The seed-103 shared-rewrite hybrid and matched graph TRM continue from the frozen decoy checkpoints. Both receive the local Nat definitions printed by Lean. Their term graphs include the same directed edges for those definitions. The hybrid can propagate the cost of a single-name normalization through its coupled guard state. At term-level states it also scores the conventional rewrites available after one such normalization. The representation uses public observations at every step.

The live learner tries all four openings in each of the 48 training worlds. It updates three scalars in the recurrent return calculation using the mean-centered returns of the actual continuations. Closure receives credit after independent Lean certification, and each issued tactic costs 0.025 return. Imitation fits the same three scalars to the first eight actions of all certified training paths, including paths that close beyond the eight-check limit. The graph TRM updates its inherited weights using the certified eight-check opening labels and the continuations from in-budget paths. All fits leave the 12 development layouts out.

The controls include the alias-aware frozen hybrid, the continued alias-aware graph TRM, and a hard maximum over the same alias-aware return state. Two additional controls fit the frozen hybrid's output to the verified opening labels, once with a nonnegative return gain and once with the action heads and return readout jointly. Their penalties are chosen by holding out each complete training layout in turn. All arms receive the same tactic grammar, eight tactic checks, six process-tree CPU seconds and 2.4 billion estimated multiply–accumulates per proof. The larger compute envelope was chosen after a training-only smoke check hit the previous 1.2 billion estimate before issuing the last two tactics of a registered eight-check proof.

The primary outcome is the number of development layout pairs where both members close in budget and pass live replay plus independent certification. Record viable openings per world and per pair as diagnostics. A single-seed result is a candidate gain if the live learner completes at least three more pairs than every trained and frozen learned control. A tie with the hard reference means that exact pooling still explains the attainable success on this cohort. The raw-only opening invariance remains a representation diagnostic.

The seed-103 fit is frozen before the alias-aware development readout. The 192 live opening trials yielded 96 independently certified in-budget proofs. Imitation used 1,536 training decisions, and graph TRM used 720 supervised opening and continuation targets. The live update changed the normalization logit from 1.39 to 2.10, the preview logit from 0 to 1.54, and the pooling logit from 8.04 to 9.77. The archived trial records bind the fitted weight hashes to the train split. The hybrid warm lineage comes from the audited mixed-source run with 192 BPD episodes, 96 LoopedTransformers Horn episodes, 12 EvalMath episodes, and 36 MathCan archived replay episodes.

| Alias-aware development arm | Viable openings | Certified proofs | Complete pairs |
| --- | ---: | ---: | ---: |
| Live return learning | 24 / 24 | 24 / 24 | 12 / 12 |
| Imitation | 24 / 24 | 24 / 24 | 12 / 12 |
| Frozen alias bank | 24 / 24 | 24 / 24 | 12 / 12 |
| Gain calibration | 24 / 24 | 24 / 24 | 12 / 12 |
| Affine calibration | 24 / 24 | 24 / 24 | 12 / 12 |
| Hard maximum reference | 24 / 24 | 24 / 24 | 12 / 12 |
| Continued graph TRM with local definitions | 12 / 24 | 0 / 24 | 0 / 12 |

The six successful arms issued identical tactic scripts on all 24 worlds. The frozen alias-aware return computation is sufficient to reproduce the live learner's proofs, and the live update supplied no additional certified pair on this cohort. The independent experimental unit is the paired layout. The 12-layout, single-seed comparison is exploratory. The raw traces and standalone certificate sources are in `results/ppt_hybrid_rl/bpd-dependent-shared-alias-dev-results-103.json.gz`. The evaluation used one logical processor and peaked at about 1.19 GiB of committed memory.

### Persistent two-slot proof probe

The smallest policy uses two 32-wide tokens, two passes of one shared transformer block, and one gated carry update. Each decision encodes the focused goal, the pending goal when present, the public local hypotheses, the candidate tactic, and the remaining checks. The selected prediction is updated from Lean's actual successor. A forced tactic updates the carry too.

The held-out set contains two worlds of each structure. The in-budget fork choice switches within each pair.

| Proof structure | Forced prefix | Competing moves | In-budget / late checks |
| --- | --- | --- | --- |
| Cross-goal rewrite | Split the conjunction | Rewrite both goals with one of two equalities | 4 / 5 |
| Guarded application | Split the conjunction | Apply one of two implications that replace the active equality with different premises | 4 / 5 |
| Split timing | Rewrite the joined target | Share another rewrite before splitting, or split and rewrite each goal as needed | 5 / 6 |

A bundled rewrite spends one check. The server implements it with one focused rewrite for each goal and restores their order before the policy sees the state. The CPU limit charges the whole operation. Standalone Lean certification checks the script with its literal `all_goals rw` tactic. The suffix grammar offers a matching local `exact`, the registered bridge equality, or a required split.

Both fits use the native returns of both branches on six separate renamings and binder permutations of the cross-goal rewrite template. The held-out guarded and split-timing structures are excluded from fitting and calibration. The persistent learner backpropagates through the forced prefix and fork. Frozen calibration fits an affine actor on the same initial policy's carried features and the same returns, with regularization selected by leaving out one training pair at a time. Slot reset evaluates the persistent checkpoint with the carry discarded between every tactic. All three arms use the same tactic limit within each pair and the same compute cap. The evaluation reports individual proof counts, complete certified pairs, and each structure's pair outcome.

The seed-404 pilot used the committed structural cohort and the frozen fit with SHA-256 `5f232b6cffc3cd8712bc6e1561d1aef507e39016b1d6f426af113c04247de272`. Each arm completed three of six development proofs and zero of three complete pairs. The persistent and slot-reset policies selected identical tactics in the six worlds. Calibration chose the other guarded application and finished the opposite member of that pair. Both fitted arms also chose only six of the twelve correct training forks. The result identifies a training-fit problem before a claim about transfer or carry is warranted. The observational unit is the proof structure, giving three development units, with two prespecified contrasts against the persistent arm. This is a power-limited pilot with no matched sham population. The raw development traces and Lean certificates are stored in `results/ppt_hybrid_rl/ppt-two-slot-structural-dev-404.json.gz`. Each stage used one logical processor.

An optional TRM control fits the same archived native returns with the same 12 AdamW updates. It reads the same two goal tokens, typed public context and tactic menu, and alternates two latent updates with one answer update at each tactic. Its width of 28 keeps the parameter count within three percent of the 32-wide persistent cell. Its weights bind to the parent fit digest, and the optional evaluation scores it under the same proof caps. Any comparison on the already opened seed-404 cohort is exploratory.

The exploratory TRM matched the three original arms at three certified proofs and zero complete pairs. It had 22,709 parameters compared with 22,177 for the persistent model. The rerun exactly reproduced the earlier trajectories and certificates for the three original arms. The four-arm record is in `results/ppt_hybrid_rl/ppt-two-slot-structural-dev-with-trm-404.json.gz`.

A training-only contrast probe measured the change in the fork logit margin when the supporting local equality switched sides. In the learned model the median within-pair change was about `7.3e-8`, with a median absolute margin of `0.011`. A least-squares fit to the 12 training margins could rank all 12 correct choices, but required a head norm of about four million compared with 0.62 for the trained head. This interpolation has no held-out result. The diagnosis points to preserving which local hypotheses link a candidate tactic to each goal before recurrence and calibration fit the action scores.

The optional binding input projects the local effect of a registered tactic on each goal, then records literal equality incidences between the projected targets and the public hypotheses. It includes direct and reverse support and the presence of other outgoing edges. It performs no proof search and consumes no witness outcomes. The same projection is available to the frozen readout and matched TRM. After an accepted tactic the persistent model reconciles the selected slots with the incidences in Lean's actual successor. On the previously opened six worlds, this is a development diagnostic of a representation change. The endpoints are training fork accuracy, sensitivity to the paired binding swap, and complete certified development pairs under the existing caps.

A separate continuation cohort allocates 12 layout pairs to training and 24 to held-out development. The two worlds in a layout swap the route whose two subproofs fit within seven tactics. Both routes are provable. The slower route requires eight tactics. The layouts vary which goal is cheaper, the cost distribution of the slow route, whether goal paths share an intermediate node, and where dead-end rewrites occur. The grammar admits forward equality moves after the bundled fork and closes a matching local fact directly. A legal move can now leave the policy in a dead end after a correct opening. The native witness uses a public directed shortest-path reference solely to verify the two branches and the tactic costs. Learned policies receive the unranked menu and public Lean state.

The binding-input development rerun chose eight of 12 correct training forks. Its median paired margin change was 0.0219. Frozen calibration chose ten of 12 and reached a median change of 0.0562. On the reused three development pairs the persistent and slot-reset arms each closed one complete pair, and frozen calibration and TRM each closed two. The new input makes binding information accessible before the readout. The reused cohort supplies no gain over either fitted control.

The fresh continuation fit starts from the frozen binding-aware two-slot and matching TRM checkpoints. The live learner tries both openings per training world and samples subsequent choices at temperature 0.4. It updates after each layout with the expected paired opening return and a policy gradient on subsequent sampled tactics. Lean-certified closure and issued checks determine the terminal return. The frozen comparison calibrates an affine head on all the same visited decisions and returns, with penalties selected by leaving out one training layout at a time. The imitation arm trains on the successful scripts in those same trials. The matched TRM fits the recorded action returns. The warm two-slot policy and a slot-reset replay of the live learner complete the controls. The primary development outcome is complete certified pairs within seven checks, using the same six CPU seconds and arithmetic cap throughout. The structural reference and 24-layout cohort define this exploratory comparison before any fitted policy sees the development states.

The 24 paired layouts are the analysis units. An exploratory gain requires three additional complete pairs over each learned or frozen control. An exact shortest-path reference sets a structural ceiling. There is no matched sham population or population-level confirmation in this single-seed comparison. Lean certified all 48 short and all 48 late reference paths, with at least one continuation choice in each world. The frozen native witness is `results/ppt_hybrid_rl/ppt-two-slot-continuation-witness.json.gz`.

The seed-831 fit ran 96 live training episodes across two passes. It sampled 68 continuation decisions in the first pass and 72 in the second. Lean independently certified four and five completed proofs, respectively. The frozen checkpoint digest is `8763ef6f9e1ebf24141afd50c8c3b73e82c9039f00403872373fb182386eaf74`. The training archive `results/ppt_hybrid_rl/ppt-two-slot-continuation-fit-831.json.gz` records every sampled path, terminal return, certificate, and fitted model hash. The calibrated weights outside the actor match the warm checkpoint exactly. The fit and all controls were frozen before the new development readout.

| Continuation control | Viable openings out of 48 | Certified proofs out of 48 | Complete pairs out of 24 |
| --- | ---: | ---: | ---: |
| Live return learning | 24 | 13 | 0 |
| Frozen affine calibration | 24 | 12 | 0 |
| Verified-script imitation | 24 | 13 | 0 |
| Matched TRM | 24 | 17 | 0 |
| Warm two-slot model | 24 | 7 | 0 |
| Slot reset | 24 | 13 | 0 |
| Directed shortest-path reference | 48 | 48 | 24 |

Every learned arm selected only one viable opening in each of the 24 paired layouts. The live learner gained six certificates over the warm policy through its continuations and tied imitation and slot reset. The independently verified traces are in `results/ppt_hybrid_rl/ppt-two-slot-continuation-dev-831.json.gz`. Evaluation used one logical processor.

The one-step typed incidence vectors matched between the two actions in every world and were unchanged by swapping the winning route in all 12 training layouts and all 24 development layouts. At this fork each bundle creates two indirect equality goals and merely reaches the start of both paths. The action-relevant differences occur over multiple edges. The terminal return also favored the structurally longer route in 12 of the 39 training comparisons where both continuations failed, because that attempt exited sooner and paid fewer tactic checks. Among those failures, seeing the pending goal favored the shorter route in ten comparisons, favored the longer route in ten, and tied in 19. The findings identify multi-step evidence propagation and the cost assigned to failed searches as separate targets.

### Directed path bank and certificate-conditioned cost

A directed evidence bank starts at the target term and propagates backward along the public equality edges. Every tied update runs two graph hops and reads both the strongest reach and its latest change into each action-conditioned goal slot. An edge gate learns from whether the edge reaches the target and whether its endpoints have multiple outgoing moves. The same gate and readout are reused at each hop. The selected slot state still carries across actual Lean tactics. Graph TRM receives the same bank and number of graph hops with its latent updates. The reward keeps the tactic cost on successful proofs and assigns zero to every failure.

The joint change reuses the same binding-aware warm checkpoint, 12 training layouts, rollout count, and six arms. Frozen calibration and TRM receive the same recorded rewards and typed information. The 24 later continuation layouts are disjoint from both training and the opened development cohort. Lean verifies both paths on these layouts before any learned scoring. Complete certified pairs are the endpoint, with a three-pair exploratory gain floor relative to every learned or frozen arm. The directed shortest-path witness stays a separate structural ceiling. This joint comparison can test the intervention as a whole. It leaves separate reward and architecture effects for controlled ablations.

The later-layout witness records all 48 seven-check closures and 48 eight-check late branches, each with an independent Lean certificate. Before fitting, multiplying the two slots' reach values from a uniform 0.8 discount over four graph hops ranked all 48 viable openings correctly. This is a structural diagnostic of which information the bank exposes. The frozen native witness is `results/ppt_hybrid_rl/ppt-two-slot-continuation-confirm-witness.json.gz`.

The frozen joint fit certified 14 viable-route trials in its first pass and 23 in the second, with 24 eligible worlds per pass. The two passes sampled 83 and 98 nonforced continuation decisions. In the 11 both-failed comparisons the revised return gave the two branches identical credit. The bank weights changed during learning and the calibration arm preserved every non-actor warm weight exactly. The fit digest is `c9aabb0828672e506965a0aee80afca560906135ccb2df6e6cb58857e85a5ecb`. The archived fit is `results/ppt_hybrid_rl/ppt-two-slot-path-fit-831.json.gz`.

| Later-layout arm | Certified proofs out of 48 | Complete pairs out of 24 |
| --- | ---: | ---: |
| Live return learning | 35 | 16 |
| Frozen affine calibration | 34 | 15 |
| Verified-script imitation | 37 | 18 |
| Matched TRM | 27 | 4 |
| Warm path-bank model | 26 | 9 |
| Slot reset | 36 | 17 |
| Directed shortest-path reference | 48 | 24 |

Every viable opening led to a certificate in each learned arm. Live learning gained 12 complete pairs over TRM and one over frozen calibration. Against calibration it won three layouts and lost two. Imitation and slot reset both finished more pairs, so the result falls below the registered gain floor. The fresh traces and certificate sources are in `results/ppt_hybrid_rl/ppt-two-slot-path-confirm-831.json.gz`. Evaluation used one logical processor.

An opening-only diagnostic on the opened cohort exactly reproduced the native fork choices. Holding the recipient's lexical cues and real prefix carry fixed, swapping only the path-bank input to its paired counterpart shifted 38 of 48 margins toward the donor for live learning, calibration, and imitation. The swap reduced viable live openings from 35 to 13, compared with 24 after removing the bank at the fork. The same directional count for all three fitted arms establishes the scope of the diagnostic. It supplies no evidence of a live-learning-specific mechanism. The raw scores are in `results/ppt_hybrid_rl/ppt-two-slot-path-bank-swap-831.json.gz`.

The mixed-source warm stage uses the completed, digest-checked seed-103 training ledger. It deduplicates successful episodes by root and retains matching candidate menus, verifier outcomes, and usable observations from BPD, LoopedTransformers Horn, EvalMath, and the audited original MathCan replay. Native source episodes require their independent axiom-free certificate. MathCan entries retain their archived-closure provenance. Longer goal contexts are encoded in bounded UTF-8 chunks and pooled without discarding text. The hybrid and TRM receive the same round-robin source batches of verified tactic imitation and remaining-return targets. A later live fit can load both warm weights with `--mixed-warm`, preserving matched initial data exposure before learning from new Lean searches.

Both tiny models consumed 459 decisions from 73 distinct audited source roots. This includes 181 decisions from 32 BPD roots, 260 from 35 LoopedTransformers Horn roots, 15 from three EvalMath roots, and three from three MathCan roots. The receipt `results/ppt_hybrid_rl/ppt-two-slot-mixed-warm-831.json` records the source hashes, exact example digest, and both output weight hashes. This stage establishes matched mixed-source training exposure. It did not participate in the later-layout results above.

The original warm data offered multiple tactics at 36 BPD, 70 LoopedTransformers, and three MathCan decisions. Every consumed EvalMath state offered a single tactic. A supplemental MathCan cost pool uses the original audited training archive to find different-length proof routes from the exact same recorded prefix. It retains two-slot states with an archived completion inside eight checks and at least two distinct completion costs. The result contains 59 decision states across 20 roots and 14 modules, with 225 archived action completions. The 135 censored alternatives stay in the receipt without return labels. Counts of states and completions refer to repeated observations from the same roots.

The optional `--mathcan-cost-archive` warm start uses the same supplement for the hybrid and TRM. The loss puts equal weight on the fastest recorded successful tactics and fits the measured cost for every archived completion in the selected menu. The original four-source schedule and its example digest remain available. The supplemental data and source hash are in `results/ppt_hybrid_rl/ppt-two-slot-mathcan-cost-archive.json.gz`. The archive supplies training-only recorded continuations, with the original audit and toolchain provenance.

Native replay in Lean 4.19 independently certified two fast and slow proof routes each for the original MathCan training theorems `epsilon_sixth_equiv` and `full_horizon_recovers`. The arithmetic theorem takes one or four tactics. Allowing the standard Lean axioms admits both routes, while excluding `Classical.choice` admits only the four-tactic route. The list theorem takes one or two tactics and both proofs use `propext`. An empty axiom inventory admits none of these four routes. This supplies a source-grounded example where the proof contract changes which tactic sequence earns credit. The receipts in `results/ppt_hybrid_rl/ppt-two-slot-mathcan-native-costs.json.gz` preserve the original source hash, native states, and standalone axiom inventories.

### Matched native continuation interventions after mixed replay

The next fit starts both architectures from the audited mixed warm checkpoint. It retains the four-hop directed bank and charges tactic cost only on certified proofs. The hybrid writes the gated difference between its prediction and Lean's observed successor, because the next input already includes that observation. Slot reset zeroes that residual across tactics.

At each forced training opening, the policy samples the suffix and locates its first multi-action continuation decision. Lean replays the exact prefix for every other legal tactic at that decision. The original and alternate paths share index-aligned random draws for later choices. The learner uses the expected native return over the local choices and uses the best observed return per opening for its paired opening update. Later sampled decisions on the original path still receive terminal credit. Frozen calibration and TRM fit the same recorded returns, including the best observed opening values. Imitation sees every independently certified script in the same collection.

The counterfactual split contains another 24 paired layouts, disjoint from both opened splits and training. The models use seven tactic checks and the same CPU and arithmetic caps. The endpoint remains full certified pairs, with an exploratory three-pair floor over every fitted or frozen arm. The finite directed shortest-path reference remains a structural ceiling. Mixed replay, residual carry, and local intervention are joint changes in this comparison.

The native witness froze 48 axiom-free seven-check proofs and 48 axiom-free eight-check alternatives before learned scoring. Its archived digest is `7d5fe1a2e2961d4d9bb1277e743a945f80b21b15fe58c9c8f062c94e881790d7`.

The joint fit executed 96 original paths and 144 paired alternatives. The first pass certified 12 original paths and four alternatives. In four training worlds only an alternate continuation rescued the viable opening. The second pass certified every viable original opening, 24 in total. The paired probes produced 16 unequal-return suffix forks in the first pass and 24 in the second. Frozen calibration preserved every non-actor warm weight. The fit receipt `results/ppt_hybrid_rl/ppt-two-slot-counterfactual-fit-831.json.gz` binds the exact replayed prefixes, shared draws, and model hashes to the mixed warm checkpoint and the unopened development split. The frozen fit digest is `ae1e85b7dfa02909013afcbeba3a33b40890635985d6efd589dbadfd8500f2d4`.

| Mixed-source counterfactual arm | Viable openings out of 48 | Certified proofs out of 48 | Complete pairs out of 24 |
| --- | ---: | ---: | ---: |
| Live return learning | 40 | 40 | 17 |
| Frozen affine calibration | 40 | 40 | 17 |
| Verified-script imitation | 41 | 41 | 18 |
| Matched TRM | 33 | 32 | 8 |
| Warm path-bank model | 32 | 32 | 10 |
| Slot reset | 40 | 40 | 17 |
| Directed shortest-path reference | 48 | 48 | 24 |

Live learning and calibration each completed two pairs missed by the other. Live learning gained nine complete pairs over TRM. Resetting the residual carry preserved all 48 native tactic sequences. The extra continuation feedback improved the sampled training search and did not produce the registered held-out advantage. The exact trajectories and Lean certificates are in `results/ppt_hybrid_rl/ppt-two-slot-counterfactual-dev-831.json.gz`.

The archived trajectories show that live learning and frozen calibration used identical scripts in 40 of 48 worlds. Every difference began at the opening, where each policy won four proofs the other missed. With the path gates fixed at initialization, summing the per-slot support after four hops recovered all 48 openings and all 48 certified scripts from the stored public states. The winning cue was already available in the frozen graph representation. This retrospective audit identifies an available cue and leaves the learned policies' use of it to an intervention test. The deeper witness cohort moves the opening evidence outside that four-hop budget. The audit includes per-world choices and support values in `results/ppt_hybrid_rl/ppt-two-slot-counterfactual-path-audit-831.json.gz`.

### Shared dependent witness

The next native fixture leaves the directed equality grammar. Applying `Exists.intro` and splitting its predicate produces two proof obligations sharing one unresolved natural-number witness. Proving the first obligation with `exact h0` or `exact h1` instantiates the witness and changes the type of the pending obligation. The server now removes metavariables already assigned by that focused tactic from the list of pending goals. This lets the native proof close with the same script that standalone Lean accepts.

The two worlds exchange which witness gives a direct proof of the pending obligation. Their axiom-free scripts take four checks for the viable choice and five for the other choice. The fork's effect is a Lean unification across goals, beyond the local equality edges used by the preceding bank. The native fixture and standalone certificates are recorded in `results/ppt_hybrid_rl/ppt-shared-existential-witness-native.json.gz` with the rebuilt server and compiler hashes. All 57 tests passed with native Lean enabled. The receipt measures the verifier transition and tactic costs. Dependent-witness policy scoring remains a separate experiment.

### Shared-witness continuation comparison

The new cohort combines existential instantiation with four serial implications, a split conjunction with a subproof, or an implication followed by a split conjunction. Legal distractor implications lead to goals with no support in the local context. Each layout pairs two worlds that exchange the viable witness. A viable proof takes eight tactics. The other witness requires nine. The fit uses 12 layouts, and the development comparison uses 24 distinct layouts. Lean verified and independently certified both witness routes in all 48 development worlds. The frozen receipt is `results/ppt_hybrid_rl/ppt-shared-witness-horn-dev.json.gz`.

Both proof models use a bank that substitutes a candidate witness into the pending goal and propagates evidence over implication and conjunction nodes. A conjunction receives evidence only when both premises have support. The persistent model carries its two goal tokens from one verified tactic to the next. Native receipts preserve the deferred witness goal and both proof goals before instantiation. The model reads two goal tokens with the shared metavariable in their types.

The fit uses the same mixed-source checkpoint as the counterfactual comparison. It uses paired continuation feedback with a neutral return for failed proofs and residual carry between tactics. Controls include the initial policy, the frozen bank and backbone with a calibrated affine head, successful-script imitation, the same parameters with carry reset after each tactic, and parameter-matched TRM fits to the same action returns and to the same paired policy objective. The models receive the same public observations, tactic menu, and eight-check, six-CPU-second, 150-million-MAC caps. The endpoint is complete axiom-free proof pairs, with an exploratory floor of three additional pairs over every learned and frozen control. This is a single-seed development comparison. The reference proofs establish native transitions and cost, and no model has been scored on these cases.

The public-cost audit uses the same projected goal and Horn rules as the witness bank. It assigns one tactic to a fact or implication and one for splitting a conjunction, plus the cost of both conjuncts. Full propagation recovers the independently certified eight- and nine-tactic costs in all 48 worlds. With three synchronous rounds it identifies 33 viable openings and recovers the 33 corresponding certified scripts across 11 complete pairs. Four rounds recover every viable opening and every eight-check script across all 24 pairs. At each choice the fixed rule selects the only tactic predicted to meet the deadline, and the archived Lean proof verifies the resulting script.

The two models already budget four message hops per decision. On this cohort, that budget suffices for a fixed public-cost rule to solve the task without learned carry or reward feedback. The result is a finite structural baseline. Whether the learned policy improves over frozen calibration remains unmeasured. The per-world costs and input receipt digest are in `results/ppt_hybrid_rl/ppt-shared-witness-horn-public-cost.json.gz`.

### Deeper shared-witness continuations

The deeper cohort switches which remote endpoint supports a witness while preserving the number of rules and facts. Its 12 training and 24 development layouts are pairs of different structural templates, selected before drawing names and witness numerals. The templates vary the location of a conjunction, the order of its subgoals, and the placement of a legal dead-end tactic. Serial and conjunctive routes both consume ten checks for the faster witness and eleven for the slower one. The audit requires four rounds of public support propagation to leave both choices unresolved at the opening fork.

The `--deep-witness` comparison uses the existing two-slot learner and its frozen calibration, imitation, slot-reset, and matched TRM controls. All arms have the same ten-check, six-CPU-second, 150-million-MAC caps. The same audited mixed-source warm checkpoints supply the initial weights. The primary outcome is complete certified pairs, with the three-pair exploratory floor over every fitted and frozen model control. The exact public proof-cost reference is separate. This comparison tests whether learning improves decisions when the existing fixed message budget cannot reach the relevant support at the fork. The encoder retains all named proof hypotheses, which contain the full hypothesis graph, and the native receipts retain the predicate declarations. This reduced arithmetic for longer contexts. This is a development cohort, and proof structure alone supplies no estimate of the calibration or TRM score.

Lean certified the ten- and eleven-check scripts in all 48 development worlds. Four rounds reached neither witness at any opening. Five rounds recovered 40 on-time scripts across 16 complete pairs. Six recovered all 48 scripts across 24 pairs. The fixed max-product bank also had zero pending-goal evidence for both witness choices after its four registered message hops at each fork. On these reference paths, the largest charges under the runner's arithmetic accounting were 96.1 million MACs for the hybrid and 56.0 million for TRM. The paired native proofs and the public-cost control are in `results/ppt_hybrid_rl/ppt-deep-witness-horn-dev.json.gz` and `results/ppt_hybrid_rl/ppt-deep-witness-horn-public-cost.json.gz`.

The deep fit includes a stateless two-pass TRM that starts from the same weights and learns from the same paired returns as the ordinary TRM control. Its second pass gives it eight rule hops in one decision with the same 22,965 trainable parameters. Eight fixed hops recover the faster witness at all 48 registered openings. The two-pass TRM's largest reference-path charge is 57.0 million MACs under the shared cap. The evaluation records the actual charges, pass counts, and complete-layout wins, losses, and ties against every control. Its registered three-pair floor is an exploratory effect size.

### Carrying proof evidence across tactics

The optional `--witness-workspace` fit carries partial Horn proof evidence alongside the two recurrent goal tokens. Candidate actions and slots read projected goals from a shared graph of the current public proof hypotheses. At the next verified tactic the model reuses evidence only for propositions in the same local hypothesis context. The same trainable rule gates still run four hops per tactic, and training through a replayed trajectory can differentiate through the carried evidence. The change adds no trainable weights.

The frozen calibration and imitation arms get the same workspace as the live learner. The workspace reset keeps the learned goal-token carry, while the slot reset clears both. A TRM arm carries the same proof workspace and fits the same paired returns, using TRM's usual per-tactic answer and latent updates. The ordinary stateless TRM arms remain in the comparison.

The fixed-gate audit replayed choices on the frozen native traces. With support carried, the unique highest-product choice recovered all 48 certified scripts and all 24 pairs. Resetting support at every tactic recovered zero scripts. Both used four hops per tactic and at most 27 evidence values on these traces. The retained prefix computation suffices for this finite cohort with fixed gates, so it isolates the value of carrying the workspace. It leaves the trained policy's gain over calibration and TRM unmeasured. The raw paired choices are in `results/ppt_hybrid_rl/ppt-deep-witness-workspace-audit.json.gz`.

The BPD probe uses the matching scored checkpoint and native fork receipts. It replays the forced prefix and reproduces the scored opening before intervening. On each of the two passes, it replaces the pending goal's current and newly arrived support with the paired world's values for the same two tactics. It leaves the focused goal, original context, and tactic menu in place. A self patch must preserve scores exactly. The probe reports donor-directed decision flips in both worlds of each layout for the live learner, frozen calibration, and the workspace TRM. Eligible pairs have both viable natural openings. The probe retains logits, paired readouts, and censored pairs. The decision flip is the endpoint. Margin changes help diagnose smaller shifts, and patched proof completion would require a separate native replay. This development intervention uses a single checkpoint seed and a deterministic identity control.

The fixed-rule reference points to the opposite world's witness after the pending-support swap at all 48 native fork snapshots. Synthetic intervention tests verify that the focused slot remains identical, a self patch preserves the model's scores exactly, and the original scores return after the hook is removed.

### Credit for slower certified proofs

At a binary fork with one positive return and one zero return, maximizing expected return gives an update in the same direction as successful-action imitation, scaled by the probability of the successful action. The optional `--late-proof-credit` deep fit reveals the cost of the other valid witness. Once the ten tactic checks are spent, Lean can try a single additional tactic if the menu is a singleton. This tactic contributes to the independently certified proof and its cost, with no policy call or replay loss on that extra step. The held-out scoring budget remains ten tactics.

The live learner, frozen calibration, imitation, and the TRM controls receive the same training receipts. Imitation sees both successful paths. The additional supervised control fits observed returns through the same hybrid architecture and warm initialization as the live learner. This lets the comparison measure the effect of the return objective alongside the effect of changing the network's weights. The branch costs and resulting advantage are exploratory training conditions. The held-out endpoint remains complete certified pairs.

The paired return-regression control replays the same ordered layout batches as the live learner, starting from a copy of the same hybrid weights. It uses the same certified returns at the opening and at paired continuation decisions, with one AdamW step per layout batch. It minimizes return-prediction error where the live learner maximizes expected return. The longer per-world regression fit remains available as a second supervised control. Each model's actual AdamW update count is recorded with the checkpoint and the deep-witness score report.

Native replay verified the credit protocol in all 24 training worlds from the 12 disjoint layouts. The ten-tactic and eleven-tactic proof returns were 0.750 and 0.725. All 48 legal decoy attempts ended without an available tactic and received zero under the certificate-cost rule. A forced replay with a stand-in policy made ten policy calls with either setting. The native proofs and reward audit are in `results/ppt_hybrid_rl/ppt-deep-witness-horn-train.json.gz` and `results/ppt_hybrid_rl/ppt-deep-witness-late-credit-audit.json.gz`.

### Seed-831 deep-witness fit and BPD

The one-logical-CPU fit used the original four-source mixed warm checkpoint, paired continuation interventions, residual slot carry, the persistent witness workspace, and late-proof credit. All 24 training worlds supplied the complete 0.750 versus 0.725 opening contrast in each pass. Original sampled traces improved from 29 to 41 successful proofs out of 48 across the two passes. The live learner and paired supervised return control each took 24 AdamW updates. In order to score the development cohort, the fit was frozen at SHA256 `b725f20ea65fb1d7bd8ff92519e29c5251aa5f5c269b7ae2a7ae03ba4feb8e57`.

| Arm | Viable openings / 48 | Proofs / 48 | Complete pairs / 24 |
| --- | ---: | ---: | ---: |
| Live RL | 24 | 24 | 0 |
| Frozen calibration | 29 | 29 | 5 |
| Imitation | 37 | 37 | 13 |
| Supervised return regression | 31 | 31 | 7 |
| Matched paired return regression | 23 | 20 | 0 |
| TRM return regression | 24 | 23 | 0 |
| TRM paired returns | 24 | 24 | 0 |
| TRM with workspace | 24 | 24 | 0 |
| Two-pass stateless TRM | 24 | 10 | 0 |
| Warm hybrid | 24 | 15 | 0 |
| Slot reset | 24 | 23 | 0 |
| Workspace reset | 24 | 23 | 0 |

Live RL selected `exact h0` in all 48 worlds, and the workspace TRM selected `exact h1` in all 48. The live learner therefore missed the exploratory gain floor. The calibration, return-regression and imitation advantages reflect actual standalone Lean certificates. The held-out run stayed under 1.10 GiB peak committed and the fit under 1.06 GiB, measured by the Windows process-tree accounting.

The BPD interchange reproduced all natural openings and all identity-patch scores exactly. Swapping the pending-goal evidence toward the paired donor shifted all 48 margins in the donor direction for each arm. Mean margin shifts were +0.0673 for live RL, +0.0114 for frozen calibration and +0.0548 for workspace TRM. It flipped zero live RL or workspace TRM openings. Calibration flipped ten of its 29 viable openings, both directions in each of its five eligible pairs. The margin response establishes sensitivity to the code-level intervention. The live policy's constant choice survived the intervention. The graded training opening contrast was 0.025, while the certified-versus-dead-end suffix contrasts were 0.725 or 0.750. This scale difference is a candidate explanation for the observed opening bias and requires a separately controlled fit to test.

The raw training, native evaluation and BPD receipts are `results/ppt_hybrid_rl/ppt-deep-witness-workspace-late-{fit,dev,bpd}-831.json.gz`. This is a single-seed developmental result.

### Opening loss balance on fresh witness layouts

The next comparison fixes a 30-fold opening-loss weight before evaluating a model on the next 24 deep layouts. That multiplier puts the observed 0.025 opening contrast on the scale of the 0.750 suffix contrast. It leaves tactic costs, the 10-tactic test cap, replay observations, model architecture, and the original four-source warm weights unchanged. The witness layouts exclude the 12 training and 24 previously scored layouts. The reference proofs establish the fixed 10-check versus 11-check costs.

Live learning collects fresh paired native returns. An unweighted hybrid replay isolates loss weighting on the same traces and 24 optimizer steps. Matched TRM replay and the paired supervised return fit receive the same opening-loss weight as the candidate. Frozen calibration, imitation, the longer supervised return fit, and slot and workspace resets remain in the comparison. The exploratory endpoint is at least three additional complete pairs over each of these controls. The fixed-rule structural reference and opening-only BPD are separate diagnostics. The design remains single-seed developmental evidence.

#### Seed-831 transfer results

The weighted fit was frozen at SHA256 `d6af076f01ff1b8c07c5d8e122615344f1aeb328b5ede3e054c0ddf436202302` before native scoring. The opening returns were 0.750 and 0.725 in all 24 training worlds in each pass. The live sampler produced 29 and 41 certified proofs across 48 original trials in passes one and two. Fit and evaluation averaged 0.83 and 0.89 logical cores, respectively, on logical CPU 2. Peak process-tree committed memory was 1.15 GiB for fit and 1.10 GiB for evaluation.

| Arm | Viable openings / 48 | Proofs / 48 | Complete pairs / 24 |
| --- | ---: | ---: | ---: |
| Live RL, 30-fold opening loss | 24 | 24 | 0 |
| Unweighted RL replay | 24 | 24 | 0 |
| Frozen calibration | 28 | 28 | 4 |
| Imitation | 41 | 41 | 17 |
| Return regression | 34 | 34 | 10 |
| Paired return regression | 18 | 10 | 0 |
| TRM | 24 | 22 | 0 |
| Paired TRM | 24 | 22 | 0 |
| Workspace TRM | 24 | 24 | 0 |
| Deeper TRM | 24 | 9 | 0 |
| Slot reset | 24 | 23 | 0 |
| Workspace reset | 24 | 23 | 0 |
| Warm hybrid | 24 | 14 | 0 |

Weighted and unweighted RL chose `exact h0` in all 48 worlds. Workspace TRM chose `exact h1` in all 48. The weighted fit missed the registered gain floor. Separately, the persistent fixed-gate reference recovered every action on all 48 certified scripts, while resetting that reference at each tactic recovered no complete scripts.

| BPD arm | Mean donor margin shift | Donor-induced tactic flips / 48 | Binder-rename tactic flips / 48 |
| --- | ---: | ---: | ---: |
| Weighted RL | +0.09193 | 0 | 36 |
| Unweighted RL | +0.06714 | 0 | 36 |
| Frozen calibration | +0.01114 | 8 | 9 |
| Workspace TRM | +0.16119 | 0 | 10 |

The pending-goal interchange reproduced natural openings and exact identity-patch scores on all 48 worlds for each arm. All donor shifts were positive. Calibration flipped in both directions in each of its four eligible layouts. The post-hoc binder intervention exchanged `h0` and `h1` throughout the public opening prefix and fork, keeping their types and positions fixed. It preserved the two pending-goal readout channels exactly after aligning candidate actions by their proof types. Reversing the candidate menu alone changed any aligned score by at most `4.77e-7`. Larger margin shifts under weighting therefore coexisted with persistent sensitivity to the printed binder names. The next architecture comparison should remove those names from the learned opening score and give the same information to the frozen, supervised and TRM controls.

The training, native evaluation and opening diagnostics are in `results/ppt_hybrid_rl/ppt-deep-witness-balance-{fit,dev,bpd}-831.json.gz`. The result is seed-831 developmental evidence.

### Binder-invariant opening score

The two-slot candidate masks proof-hypothesis names in its text encoder from the first tactic onward. When each menu option instantiates a shared witness in the pending goal, it gives the two options the same lexical tactic cue. The typed proof bank still sees the original proof state and computes the distinct substitutions and pending-goal messages. Lean receives the original tactic names. This isolates scoring at the witness decision without adding parameters, changing tactic costs, or providing extra lookahead.

The seed-831 fit starts from the same four-source checkpoint with the fixed 30-fold opening weight and the original 12 training layouts. All frozen, regression, imitation, reset and TRM arms use the same anonymous observation rule. A named hybrid replays the candidate's complete on-policy receipts with the same initialization, 24 optimizer steps and opening weight. The unweighted anonymous hybrid replays those receipts at the original weight. The witness receipt freezes the next 24 distinct layouts, both witness alternatives and the ten-check evaluation budget before any model evaluation.

The primary outcome remains complete certified proofs in both worlds of a layout. A candidate gain requires at least three additional complete pairs over each learned and frozen learned control. Report the fixed-gate structural reference separately. The BPD comparison exchanges the two pending readout channels on both passes with the opposite world, records the self patch, reverses menu order, and exchanges the two witness names throughout the prefix. The rename test holds the proof types and aligned typed channels fixed. On every arm using anonymous observations the aligned opening logits must agree to numerical precision after renaming. The primary comparison covers 24 layouts with one fitted checkpoint. Use the results to guide design, with FSI-01 `REDESIGN_OR_PILOT_ONLY` for any population or mediation claim.

#### Seed-831 binder-invariant fit and BPD

The mixed warm checkpoint covered 459 decisions from BPD, EvalMath, LoopedTransformers and MathCan. The fit was frozen at SHA256 `983138dd35dd0b7e6d0e3baa8c7d2ae8a2f3672c141eacf325e1a093beb7c5a7` before model scoring. The new witness receipt has SHA256 `cf928d27622576e74cfea0ff8d8b24dedf19e79e9b1154ee3f2dc2bb78a3cf01`. The 24 training worlds again supplied both graded opening returns in each pass. The live sampler closed 29 of 48 original trials in the first pass and 41 in the second. Fit and native evaluation used logical CPU 2 and peaked at 1.10 GiB of committed process-tree memory.

| Arm | Viable openings / 48 | Proofs / 48 | Complete pairs / 24 |
| --- | ---: | ---: | ---: |
| Binder-invariant live RL | 48 | 48 | 24 |
| Unweighted binder-invariant RL | 48 | 48 | 24 |
| Named hybrid replay | 24 | 24 | 0 |
| Frozen calibration | 48 | 48 | 24 |
| Imitation | 48 | 48 | 24 |
| Return regression | 48 | 48 | 24 |
| Paired return regression | 0 | 0 | 0 |
| Standard TRM | 24 | 23 | 4 |
| Paired TRM | 20 | 19 | 5 |
| Workspace TRM | 48 | 48 | 24 |
| Deeper TRM | 48 | 31 | 10 |
| Slot reset | 25 | 24 | 4 |
| Workspace reset | 25 | 24 | 3 |
| Warm hybrid | 0 | 0 | 0 |

The invariant live learner gained 24 pairs over the named replay with the same training records and 20 over standard TRM. It tied imitation, the longer return-regression fit, frozen calibration, workspace TRM and its own unweighted replay. The fixed-gate persistent workspace recovered all 48 certified scripts, while the reset fixed-gate reference recovered zero. Masking proof-hypothesis names and sharing the lexical opening cue improved the hybrid on this layout family. The comparison misses the registered gain floor for the RL objective.

| BPD arm | Mean donor margin shift | Donor-induced tactic flips / 48 | Binder-rename tactic flips / 48 |
| --- | ---: | ---: | ---: |
| Binder-invariant live RL | +0.10206 | 48 | 0 |
| Unweighted binder-invariant RL | +0.07447 | 48 | 0 |
| Named hybrid replay | +0.09274 | 0 | 34 |
| Frozen calibration | +0.00472 | 48 | 0 |
| Workspace TRM | +0.31775 | 48 | 0 |

The self patch reproduced every score exactly. The binder swap and menu-order sham changed anonymous scores by at most `4.77e-7` after candidate alignment. On the four anonymous arms, swapping the pending evidence changed the choice in both directions of every layout. On the named replay, it moved every margin toward the donor without changing the opening. The small margin shift for frozen calibration was sufficient to flip all its choices. These are measured code-level interventions at the opening of one developmental fit. The fit, native evaluation and BPD receipts are in `results/ppt_hybrid_rl/ppt-deep-witness-invariant-{fit,dev,bpd}-831.json.gz`.

### Sharing a Lean subproof across two obligations

The reuse fixture keeps the individual proof costs identical for both witnesses. In one route, both conjuncts use the same proved fact. In the other route, the conjuncts use distinct facts with equally long proofs. The two worlds exchange which witness has the shared fact. At the opening the current proof-tree reference estimates 15 checks for either witness. With a 14-check budget, either witness can close directly in 15 checks. In each world the shared route fits by introducing one local lemma with `have`, using it for both conjuncts, and discharging its proof once.

Lean replayed and independently compiled all 12 combinations of world, witness and tactic plan. The three plans finish directly, cache the first candidate fact, or cache the second one. Each world has exactly one in-budget plan among those tried. The three frozen binder-invariant policies give the two witnesses identical pending readouts on both passes. Their opening logit differences have magnitude at most `1.20e-7`. This verifies a two-world development fixture where the current encoder loses useful information about the cost of reusing a proof. Its public declaration, literal tactics, Lean certificates and frozen scores are in `results/ppt_hybrid_rl/ppt-witness-reuse-feasibility.json.gz`.

An architecture comparison on this family needs to give the two-slot learner and controls the same public cache proposals. Vary the cost of constructing a shared lemma and the benefit of reusing it across conjuncts. Include a strong frozen readout that can see shared proof structure and a structural reference that counts the cost of materializing it. Evaluate the cache decision as well as witness selection. The two verified worlds establish feasibility for this test design.

### Learning witness and cache decisions together

The new pilot predefines 16 training and 24 evaluation layouts. Two worlds in each layout exchange the routes available to the witnesses. The routes have equal direct proof costs. The layouts balance a single sharing route against cases with two competing sharing routes. They vary fact construction and downstream proof length, the available checks, and which cache proposal names the reusable fact. The generator chooses distinct route layouts before binder and cache names are assigned. Each world exposes two witness actions and three plans after the witness, giving 480 native paths with standalone Lean certificates and one in-budget plan per world.

The two-token planner projects the pending conjunction from public witness types at the opening. Its plan features give both obligations their direct proof distance, whether each can use the cache, and the distance to the cache and its construction cost. Each option is scored by the inherited tied proof cell with a new input projection. A learned pooling step summarizes the possible plans for each witness. The selected witness's two tokens carry into the later cache decision. The registered opening projections match the public observations before the cache decision. The feature extractor uses no certificate outcomes.

The seed-831 fit inherits the trained cell and readouts from the mixed BPD, EvalMath, LoopedTransformers and MathCan checkpoint. The return learner differentiates the expected terminal success under its witness and cache probabilities on the exhaustive native training table. Imitation and supervised return regression share the initial hybrid and the same records, layout batches, and 192 optimizer steps. The matched TRM receives the same public plan features and returns. Frozen controls fit ridge readouts after the hybrid encoder, over its alternative plans, and on the untransformed public plan features. Ridge penalties are chosen by leaving out both worlds of each training layout. The slot-reset control removes the carried tokens at the cache decision. The exact feasibility reference chooses among the previously certified paths.

Evaluation runs the chosen witness and cache plan back through native Lean under the registered tactic limit. The two decisions are learned. After the cache choice the scorer follows its corresponding registered continuation. The primary outcome counts layouts where both worlds close within budget. The exploratory gain threshold is three additional pairs over each learned and frozen control. One training seed and 24 layouts can guide design under FSI-01 `REDESIGN_OR_PILOT_ONLY`.

The cache BPD fixes the naturally selected witness and its public cache menu. It transplants the corresponding witness's two carried tokens from the opposite world. It checks a self-patch, a donor with the same route structure, and a channel-rotated edit with the same norm as the opposite-route transplant. The readout records cache changes and whether the selected plan was certified within budget. The TRM arm shares the same intervention interface and scores cache plans without that carry.

#### Seed-831 cache planning results

The native plan table was frozen at SHA256 `afea330adc2b6ca7a9abb08119d76e843740716daa823b9cd98e8a299616312c`. Each trained arm took 192 optimizer steps. The return policy objective included a quarter-weight auxiliary term for expected cache success across both witnesses. The fit was frozen at SHA256 `22c79a85a2798ca6dc797f84a57606968cadf99e1f5dbc6cda3c509e3854a7c5` before evaluation. The native evaluation used logical CPU 2 and peaked at 0.93 GiB of committed memory.

| Arm | Viable witnesses / 48 | Certified proofs / 48 | Complete pairs / 24 |
| --- | ---: | ---: | ---: |
| Proof-return policy gradient | 48 | 48 | 24 |
| Imitation | 48 | 48 | 24 |
| Supervised return regression | 48 | 48 | 24 |
| TRM proof-return policy gradient | 48 | 48 | 24 |
| Slot reset | 48 | 48 | 24 |
| Ridge calibration of witness and cache heads | 40 | 40 | 20 |
| Ridge calibration with maximum option score | 44 | 44 | 22 |
| Ridge calibration on the public features | 44 | 44 | 22 |
| Warm planner | 24 | 0 | 0 |
| Fixed structural reference | 48 | 48 | 24 |

The strongest frozen calibrations lost two pairs to the fitted hybrid. Every miss by a calibrated policy arose at the witness choice. Regrouping the public distances into the cost of constructing a lemma once and then using it reproduced all 288 held-out plan costs exactly. The learned arms that trained their encoders also reached the ceiling. The registered three-pair margin over the controls was missed. This run compares policy optimization from a fully observed return table.

BPD reproduced each natural opening and cache decision. The opposite-route carry transplant moved at least one RL cache logit by 13.52 in the most affected world. The equal-norm sham moved one by 14.97. The same-route donor moved any logit by at most `1.4e-6`. None of the interventions changed the chosen cache or a certified outcome for the hybrid, frozen calibration, or TRM. Together with the slot-reset result, this shows that the carried tokens were dispensable for the learned decisions on this cohort. A future comparison needs a tactic that reveals action-relevant evidence after the witness choice so a stronger search-time memory can be tested.

The training, native evaluation and BPD receipts are in `results/ppt_hybrid_rl/ppt-reuse-plans-{fit,dev,bpd}-831.json.gz`.

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
python -m configs.probe_ppt_pending_alias_history THREE-GOAL-COHORT PROGRESS-RUNS EVIDENCE HISTORY-EVIDENCE --progress-study PROGRESS-STUDY --native --prior-native MATCHING-PRIOR-NATIVE-EVIDENCE
python -m configs.probe_ppt_recovery_arrival PENDING-ALIAS-COHORT ROUTED-RUNS EVIDENCE-151 ARRIVAL-EVIDENCE
python -m configs.freeze_ppt_bpd_rewrite ROOT-ADAPTATION-COHORT REWRITE-COHORT
python -m configs.check_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE
python -m configs.probe_ppt_bpd_rewrite_revisits REWRITE-COHORT REWRITE-REFERENCE WARM-REVISIT-EVIDENCE --initialization INITIALIZATION
python -m configs.probe_ppt_bpd_rewrite_revisits REWRITE-COHORT REWRITE-REFERENCE TRAIN-EXPLORATION-EVIDENCE --initialization INITIALIZATION --sample-training
python -m configs.run_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE MATHCAN-TRAIN.zip INITIALIZATION REWRITE-RUNS
python -m configs.analyze_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE REWRITE-RUNS INITIALIZATION REWRITE-AUDIT
python -m configs.probe_ppt_bpd_rewrite_revisits REWRITE-COHORT REWRITE-REFERENCE FINAL-REVISIT-EVIDENCE --sessions REWRITE-RUNS --audit REWRITE-AUDIT
python -m configs.run_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE MATHCAN-TRAIN.zip INITIALIZATION EXPLORATION-RUNS --exploration-study EXPLORATION-STUDY
python -m configs.analyze_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE EXPLORATION-RUNS INITIALIZATION EXPLORATION-AUDIT --exploration-study EXPLORATION-STUDY
python -m configs.run_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE MATHCAN-TRAIN.zip SEED-INITIALIZATION SEED-RUNS --exploration-study EXPLORATION-STUDY --replication-study REPLICATION-STUDY --seed SEED
python -m configs.analyze_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE SEED-RUNS SEED-INITIALIZATION SEED-AUDIT --exploration-study EXPLORATION-STUDY --replication-study REPLICATION-STUDY --seed SEED
python -m configs.probe_ppt_bpd_rewrite_components REWRITE-COHORT REWRITE-REFERENCE REWRITE-RUNS REWRITE-AUDIT FINAL-REVISIT-EVIDENCE COMPONENT-EVIDENCE
python -m configs.probe_ppt_bpd_rewrite_transplant REWRITE-COHORT REWRITE-REFERENCE EXPLORATION-RUNS EXPLORATION-AUDIT EXPLORATION-REVISITS TRANSPLANT-EVIDENCE
python -m configs.probe_ppt_bpd_rewrite_transplant REWRITE-COHORT REWRITE-REFERENCE EXPLORATION-RUNS EXPLORATION-AUDIT EXPLORATION-REVISITS LEXICAL-TRANSPLANT-EVIDENCE --lexical
python -m configs.export_ppt_bpd_rewrite_checkpoint REWRITE-COHORT REWRITE-REFERENCE EXPLORATION-RUNS EXPLORATION-AUDIT EXPLORATION-REVISITS CHECKPOINT EXPORT-RECEIPT
python -m configs.run_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE MATHCAN-TRAIN.zip INITIALIZATION CELL-RUNS --exploration-study EXPLORATION-STUDY --cell-study CELL-STUDY
python -m configs.analyze_ppt_bpd_rewrite REWRITE-COHORT REWRITE-REFERENCE CELL-RUNS INITIALIZATION CELL-AUDIT --exploration-study EXPLORATION-STUDY --cell-study CELL-STUDY
python -m configs.freeze_ppt_bpd_rewrite_transfer REWRITE-COHORT EXPLORATION-AUDIT FRESH-REWRITE-COHORT
python -m configs.check_ppt_bpd_rewrite_transfer FRESH-REWRITE-COHORT FRESH-REWRITE-WITNESS
python -m configs.evaluate_ppt_bpd_rewrite_transfer FRESH-REWRITE-COHORT FRESH-REWRITE-WITNESS EXPLORATION-AUDIT EXPLORATION-RUNS FRESH-READOUT
python -m configs.replicate_ppt_bpd_rewrite_transfer register FRESH-REWRITE-COHORT FRESH-REWRITE-WITNESS AUDIT-DIRECTORY REPLICATION-REGISTRY
python -m configs.replicate_ppt_bpd_rewrite_transfer run FRESH-REWRITE-COHORT FRESH-REWRITE-WITNESS AUDIT-DIRECTORY REPLICATION-REGISTRY RUNS-PARENT REPLICATION-RESULTS
python -m configs.probe_ppt_bpd_transfer_cell FRESH-REWRITE-COHORT FRESH-REWRITE-WITNESS AUDIT-DIRECTORY REPLICATION-REGISTRY RUNS-PARENT REPLICATION-RESULTS TRANSFER-CELL.json.gz
python -m configs.probe_ppt_bpd_rewrite_forks REWRITE-COHORT EXPLORATION-AUDIT EXPLORATION-RUNS INTERIOR-FORKS.json.gz --collect
python -m configs.probe_ppt_bpd_rewrite_suffixes REWRITE-COHORT INTERIOR-FORKS.json.gz TRAINING-SUFFIXES.json.gz
python -m configs.check_ppt_bpd_conditional_rewrites CONDITIONAL-WITNESS.json.gz
python -m configs.collect_ppt_bpd_conditional_training CONDITIONAL-WITNESS.json.gz CONDITIONAL-TRAIN.json.gz
python -m configs.probe_ppt_bpd_conditional_policy CONDITIONAL-WITNESS.json.gz CONDITIONAL-TRAIN.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS CONDITIONAL-RESULTS.json.gz CONDITIONAL-WEIGHTS
python -m configs.check_ppt_bpd_dependent_rewrites dev DEPENDENT-WITNESS.json.gz
python -m configs.check_ppt_bpd_dependent_rewrites train DEPENDENT-TRAIN.json.gz
python -m configs.probe_ppt_bpd_dependent_policy DEPENDENT-WITNESS.json.gz DEPENDENT-TRAIN.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS DEPENDENT-RESULTS.json.gz DEPENDENT-WEIGHTS
python -m configs.probe_ppt_bpd_dependent_controls DEPENDENT-WITNESS.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS DEPENDENT-CONTROLS.json.gz
python -m configs.check_ppt_bpd_dependent_rewrites dev SPUR-WITNESS.json.gz --spurs
python -m configs.check_ppt_bpd_dependent_rewrites train SPUR-TRAIN.json.gz --spurs
python -m configs.probe_ppt_bpd_dependent_policy SPUR-WITNESS.json.gz SPUR-TRAIN.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS SPUR-RESULTS.json.gz SPUR-WEIGHTS
python -m configs.probe_ppt_bpd_dependent_controls SPUR-WITNESS.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS SPUR-CONTROLS.json.gz --training SPUR-TRAIN.json.gz
python -m configs.check_ppt_bpd_dependent_returns SPUR-TRAIN.json.gz SPUR-RETURNS.json.gz
python -m configs.check_ppt_bpd_dependent_rewrites dev ROUTING-WITNESS.json.gz --routing
python -m configs.probe_ppt_bpd_dependent_routing fit SPUR-TRAIN.json.gz WARM-HYBRID.pt ROUTING-FIT.json.gz SPUR-RETURNS.json.gz
python -m configs.probe_ppt_bpd_dependent_routing evaluate SPUR-TRAIN.json.gz WARM-HYBRID.pt ROUTING-RESULTS.json.gz ROUTING-WITNESS.json.gz ROUTING-FIT.json.gz MATCHED-TRM.pt SPUR-RESULTS.json.gz ROUTING-WEIGHTS
python -m configs.probe_ppt_bpd_dependent_routing components ROUTING-WITNESS.json.gz ROUTING-RESULTS.json.gz ROUTING-WEIGHTS/routed_rl.pt ROUTING-COMPONENTS.json.gz
python -m configs.probe_ppt_bpd_dependent_routing transfer OTHER-ROUTING-WITNESS.json.gz ORIGIN-ROUTING-RESULTS.json.gz ORIGIN-ROUTING-WEIGHTS/routed_rl.pt ROUTING-TRANSFER.json.gz
python -m configs.check_ppt_bpd_dependent_rewrites dev ROUTING-CONFIRMATION-WITNESS.json.gz --routing-confirmation
python -m configs.probe_ppt_bpd_dependent_policy SPUR-WITNESS.json.gz SPUR-TRAIN.json.gz SEED-AUDIT SEED-RUNS SEED-TRM-RESULTS.json.gz SEED-TRM-WEIGHTS --arms trm_rl
python -m configs.check_ppt_bpd_dependent_rewrites train EFFORT-TRAIN.json.gz --serial-effort
python -m configs.check_ppt_bpd_dependent_rewrites dev EFFORT-WITNESS.json.gz --serial-effort
python -m configs.check_ppt_bpd_dependent_returns EFFORT-TRAIN.json.gz EFFORT-RETURNS.json.gz
python -m configs.probe_ppt_bpd_dependent_policy EFFORT-WITNESS.json.gz EFFORT-TRAIN.json.gz SEED-AUDIT SEED-RUNS EFFORT-TRM-RESULTS.json.gz EFFORT-TRM-WEIGHTS --arms trm_rl
python -m configs.run_ppt_discounted_rewrite train EFFORT-TRAIN.json.gz ROUTED-HYBRID.pt EFFORT-WEIGHTS EFFORT-FIT.json.gz EFFORT-RETURNS.json.gz
python -m configs.run_ppt_discounted_rewrite evaluate EFFORT-TRAIN.json.gz ROUTED-HYBRID.pt EFFORT-WEIGHTS EFFORT-RESULTS.json.gz EFFORT-WITNESS.json.gz EFFORT-FIT.json.gz EFFORT-TRM-WEIGHTS/trm_rl.pt EFFORT-TRM-RESULTS.json.gz
python -m configs.check_ppt_bpd_shared_conjuncts train SHARED-CONJUNCT-TRAIN.json.gz
python -m configs.check_ppt_bpd_shared_conjuncts dev SHARED-CONJUNCT-WITNESS.json.gz
python -m configs.check_ppt_bpd_shared_conjuncts dev SHARED-CONJUNCT-DECOY-WITNESS.json.gz --decoy
python -m configs.check_ppt_bpd_shared_conjuncts train SHARED-CONJUNCT-DECOY-TRAIN.json.gz --decoy
python -m configs.check_ppt_bpd_shared_conjuncts dev SHARED-CONJUNCT-DECOY-CONFIRMATION.json.gz --decoy-confirmation
python -m configs.run_ppt_shared_rewrite trm SHARED-CONJUNCT-TRAIN.json.gz SHARED-TRM-REPORT.json.gz SEED-AUDIT SEED-RUNS SHARED-TRM.pt
python -m configs.run_ppt_shared_rewrite train SHARED-CONJUNCT-TRAIN.json.gz SHARED-FIT.json.gz ROUTED-HYBRID.pt SHARED-WEIGHTS
python -m configs.run_ppt_shared_rewrite evaluate SHARED-CONJUNCT-TRAIN.json.gz SHARED-RESULTS.json.gz ROUTED-HYBRID.pt SHARED-WEIGHTS SHARED-CONJUNCT-WITNESS.json.gz SHARED-FIT.json.gz SHARED-TRM.pt SHARED-TRM-REPORT.json.gz
python -m configs.run_ppt_shared_rewrite components SHARED-CONJUNCT-TRAIN.json.gz SHARED-COMPONENTS.json.gz ROUTED-HYBRID.pt SHARED-WEIGHTS SHARED-CONJUNCT-DECOY-WITNESS.json.gz SHARED-FIT.json.gz
python -m configs.run_ppt_shared_rewrite adapt SHARED-CONJUNCT-DECOY-TRAIN.json.gz DECOY-FIT.json.gz ROUTED-HYBRID.pt DECOY-WEIGHTS --prior-fit SHARED-FIT.json.gz --prior-checkpoints SHARED-WEIGHTS
python -m configs.run_ppt_shared_rewrite trm SHARED-CONJUNCT-DECOY-TRAIN.json.gz DECOY-TRM-REPORT.json.gz SEED-AUDIT SEED-RUNS DECOY-TRM.pt --prior-trm SHARED-TRM.pt --prior-trm-audit SHARED-TRM-REPORT.json.gz
python -m configs.run_ppt_shared_rewrite evaluate SHARED-CONJUNCT-DECOY-TRAIN.json.gz DECOY-RESULTS.json.gz ROUTED-HYBRID.pt DECOY-WEIGHTS SHARED-CONJUNCT-DECOY-CONFIRMATION.json.gz DECOY-FIT.json.gz DECOY-TRM.pt DECOY-TRM-REPORT.json.gz --prior-fit SHARED-FIT.json.gz --prior-checkpoints SHARED-WEIGHTS
python -m configs.check_ppt_bpd_shared_aliases train SHARED-ALIAS-TRAIN.json.gz
python -m configs.check_ppt_bpd_shared_aliases dev SHARED-ALIAS-WITNESS.json.gz
python -m configs.probe_ppt_bpd_shared_aliases dev SHARED-ALIAS-WITNESS.json.gz ROUTED-HYBRID.pt DECOY-WEIGHTS/live_rl.pt DECOY-TRM.pt SHARED-ALIAS-INVARIANCE.json.gz
python -m configs.run_ppt_alias_policy train SHARED-ALIAS-TRAIN.json.gz ALIAS-FIT.json.gz DECOY-WEIGHTS/live_rl.pt ALIAS-WEIGHTS
python -m configs.run_ppt_alias_policy trm SHARED-ALIAS-TRAIN.json.gz ALIAS-TRM-FIT.json.gz DECOY-TRM.pt ALIAS-TRM.pt
python -m configs.run_ppt_alias_policy evaluate SHARED-ALIAS-TRAIN.json.gz ALIAS-RESULTS.json.gz DECOY-WEIGHTS/live_rl.pt ALIAS-WEIGHTS SHARED-ALIAS-WITNESS.json.gz ALIAS-FIT.json.gz ALIAS-TRM.pt ALIAS-TRM-FIT.json.gz
python -m configs.run_ppt_two_slot verify TWO-SLOT-WITNESS.json
python -m configs.run_ppt_two_slot fit TWO-SLOT-FIT.pt
python -m configs.run_ppt_two_slot evaluate TWO-SLOT-FIT.pt TWO-SLOT-RESULTS.json
python -m configs.run_ppt_two_slot fit-trm TWO-SLOT-FIT.pt TWO-SLOT-TRM.pt
python -m configs.run_ppt_two_slot evaluate TWO-SLOT-FIT.pt TWO-SLOT-TRM-RESULTS.json --trm TWO-SLOT-TRM.pt
python -m configs.run_ppt_two_slot fit BINDING-TWO-SLOT-FIT.pt --bindings
python -m configs.run_ppt_two_slot verify CONTINUATION-WITNESS.json --continuations
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt CONTINUATION-FIT.pt
python -m configs.run_ppt_slot_continuations evaluate CONTINUATION-FIT.pt CONTINUATION-RESULTS.json
python -m configs.run_ppt_two_slot verify CONFIRMATION-WITNESS.json --continuations --confirm
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt PATH-FIT.pt --paths --certificate-cost --confirm
python -m configs.run_ppt_slot_continuations evaluate PATH-FIT.pt PATH-RESULTS.json
python -m configs.pretrain_ppt_slot_mixed BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt BPD-COHORT.json SOURCE-AUDIT.json COMPLETED-HYBRID-RL.sqlite MIXED-WARM.pt
python -m configs.prepare_ppt_mathcan_costs MATHCAN-TRAIN.zip MATHCAN-COST-TRAIN.json.gz
python -m configs.check_ppt_mathcan_native_costs MATHCAN-TRAIN.zip FiniteHorizonGap.TRAIN.lean MATHCAN-NATIVE-COSTS.json.gz
python -m configs.pretrain_ppt_slot_mixed BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt BPD-COHORT.json SOURCE-AUDIT.json COMPLETED-HYBRID-RL.sqlite MIXED-COST-WARM.pt --mathcan-cost-archive MATHCAN-TRAIN.zip
python -m configs.run_ppt_two_slot verify COUNTERFACTUAL-WITNESS.json --continuations --counterfactual
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt COUNTERFACTUAL-FIT.pt --mixed-warm MIXED-WARM.pt --paths --certificate-cost --residual-carry --paired-continuations --counterfactual
python -m configs.run_ppt_slot_continuations evaluate COUNTERFACTUAL-FIT.pt COUNTERFACTUAL-RESULTS.json
python -m configs.audit_ppt_slot_counterfactual COUNTERFACTUAL-RESULTS.json COUNTERFACTUAL-WITNESS.json.gz COUNTERFACTUAL-PATH-AUDIT.json.gz
python -m configs.check_ppt_dependent_witness SHARED-WITNESS.json
python -m configs.check_ppt_dependent_witness --cohort dev SHARED-WITNESS-HORN-DEV.json
python -m configs.check_ppt_dependent_witness --audit-receipt SHARED-WITNESS-HORN-DEV.json SHARED-WITNESS-PUBLIC-COST.json
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt SHARED-WITNESS-FIT.pt --mixed-warm MIXED-WARM.pt --shared-witness --certificate-cost --residual-carry --paired-continuations
python -m configs.run_ppt_slot_continuations evaluate SHARED-WITNESS-FIT.pt SHARED-WITNESS-RESULTS.json
python -m configs.check_ppt_dependent_witness --deep-cohort dev DEEP-WITNESS-DEV.json
python -m configs.check_ppt_dependent_witness --audit-receipt DEEP-WITNESS-DEV.json DEEP-WITNESS-PUBLIC-COST.json
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt DEEP-WITNESS-FIT.pt --mixed-warm MIXED-WARM.pt --deep-witness --certificate-cost --residual-carry --paired-continuations
python -m configs.run_ppt_slot_continuations evaluate DEEP-WITNESS-FIT.pt DEEP-WITNESS-RESULTS.json
python -m configs.check_ppt_dependent_witness --audit-workspace DEEP-WITNESS-DEV.json DEEP-WITNESS-WORKSPACE-AUDIT.json
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt WITNESS-WORKSPACE-FIT.pt --mixed-warm MIXED-WARM.pt --deep-witness --certificate-cost --residual-carry --paired-continuations --witness-workspace
python -m configs.run_ppt_slot_continuations evaluate WITNESS-WORKSPACE-FIT.pt WITNESS-WORKSPACE-RESULTS.json
python -m configs.probe_ppt_witness_workspace WITNESS-WORKSPACE-FIT.pt WITNESS-WORKSPACE-RESULTS.json WITNESS-WORKSPACE-BPD.json
python -m configs.check_ppt_dependent_witness --deep-cohort train DEEP-WITNESS-TRAIN.json
python -m configs.check_ppt_dependent_witness --verify-late-credit DEEP-WITNESS-TRAIN.json DEEP-WITNESS-LATE-CREDIT.json
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt LATE-CREDIT-FIT.pt --mixed-warm MIXED-WARM.pt --deep-witness --certificate-cost --residual-carry --paired-continuations --witness-workspace --late-proof-credit
python -m configs.run_ppt_slot_continuations evaluate LATE-CREDIT-FIT.pt LATE-CREDIT-RESULTS.json
python -m configs.check_ppt_dependent_witness --deep-cohort transfer DEEP-WITNESS-TRANSFER.json
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt BALANCED-OPENING-FIT.pt --mixed-warm MIXED-WARM.pt --deep-witness-transfer --certificate-cost --residual-carry --paired-continuations --witness-workspace --late-proof-credit
python -m configs.run_ppt_slot_continuations evaluate BALANCED-OPENING-FIT.pt BALANCED-OPENING-RESULTS.json
python -m configs.probe_ppt_witness_workspace BALANCED-OPENING-FIT.pt BALANCED-OPENING-RESULTS.json BALANCED-OPENING-BPD.json --opening-diagnostics
python -m configs.check_ppt_dependent_witness --deep-cohort invariant DEEP-WITNESS-INVARIANT.json
python -m configs.run_ppt_slot_continuations fit BINDING-TWO-SLOT-FIT.pt BINDING-TWO-SLOT-TRM.pt INVARIANT-OPENING-FIT.pt --mixed-warm MIXED-WARM.pt --deep-witness-invariant --certificate-cost --residual-carry --paired-continuations --witness-workspace --late-proof-credit
python -m configs.run_ppt_slot_continuations evaluate INVARIANT-OPENING-FIT.pt INVARIANT-OPENING-RESULTS.json
python -m configs.probe_ppt_witness_workspace INVARIANT-OPENING-FIT.pt INVARIANT-OPENING-RESULTS.json INVARIANT-OPENING-BPD.json --opening-diagnostics
python -m configs.check_ppt_witness_reuse WITNESS-REUSE-FEASIBILITY.json --checkpoint INVARIANT-OPENING-FIT.pt
python -m configs.freeze_ppt_reuse_plans REUSE-PLANS.json.gz
python -m configs.run_ppt_reuse_plans fit REUSE-PLANS.json.gz INVARIANT-OPENING-FIT.pt REUSE-FIT.pt
python -m configs.run_ppt_reuse_plans evaluate REUSE-PLANS.json.gz INVARIANT-OPENING-FIT.pt REUSE-DEV.json.gz --fit REUSE-FIT.pt
python -m configs.run_ppt_reuse_plans bpd REUSE-PLANS.json.gz INVARIANT-OPENING-FIT.pt REUSE-BPD.json.gz --fit REUSE-FIT.pt --evaluation REUSE-DEV.json.gz
python -m configs.probe_ppt_bpd_target_contrast FRESH-REWRITE-COHORT FRESH-REWRITE-WITNESS EXPLORATION-AUDIT EXPLORATION-RUNS FRESH-READOUT TARGET-CONTRAST.json.gz
python -m configs.check_ppt_bpd_mixed_cost MIXED-COST-WITNESS.json.gz
python -m configs.probe_ppt_bpd_mixed_policy MIXED-COST-WITNESS.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS MIXED-RESULTS.json.gz MIXED-CHECKPOINTS
python -m configs.probe_ppt_bpd_mixed_policy MIXED-COST-WITNESS.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS PHASE-COST-RESULTS.json.gz PHASE-COST-CHECKPOINTS --training-evidence MIXED-RESULTS.json.gz --phase-credit --epochs 4
python -m configs.probe_ppt_bpd_mixed_policy MIXED-COST-WITNESS.json.gz EXPLORATION-AUDIT EXPLORATION-RUNS ARRIVAL-RESULTS.json.gz ARRIVAL-CHECKPOINTS --training-evidence MIXED-RESULTS.json.gz --phase-credit --arrival-history --epochs 4
python -m configs.check_ppt_bpd_bundled_opening BUNDLED-OPENING-WITNESS
python -m configs.freeze_ppt_bpd_bundled REWRITE-COHORT BUNDLED-COHORT
python -m configs.check_ppt_bpd_bundled_gym BUNDLED-COHORT BUNDLED-REFERENCE
python -m configs.freeze_ppt_bpd_matched_bundles BUNDLED-COHORT MATCHED-BUNDLE-COHORT
python -m configs.check_ppt_bpd_bundled_gym MATCHED-BUNDLE-COHORT MATCHED-BUNDLE-REFERENCE
python -m configs.probe_ppt_bpd_bundle_components BUNDLED-COHORT BUNDLED-REFERENCE WARM-COMPONENTS --initialization INITIALIZATION
python -m configs.probe_ppt_bpd_bundle_components MATCHED-BUNDLE-COHORT MATCHED-BUNDLE-REFERENCE MATCHED-WARM-COMPONENTS --initialization INITIALIZATION
python -m configs.probe_ppt_bpd_bundle_components MATCHED-BUNDLE-COHORT MATCHED-BUNDLE-REFERENCE TRANSPORT-CONTROL --transport-report MATCHED-WARM-COMPONENTS
python -m configs.probe_ppt_bpd_bundle_credit MATCHED-BUNDLE-COHORT MATCHED-BUNDLE-REFERENCE TRAINING-TRACES.json.gz
python -m configs.probe_ppt_bpd_bundle_credit MATCHED-BUNDLE-COHORT MATCHED-BUNDLE-REFERENCE SHARED-BRANCH-RESULTS.json.gz --evidence TRAINING-TRACES.json.gz --initialization INITIALIZATION --checkpoints CHECKPOINTS
python -m configs.probe_ppt_bpd_bundle_credit MATCHED-BUNDLE-COHORT MATCHED-BUNDLE-REFERENCE ACTION-VALUE-RESULTS.json.gz --evidence TRAINING-TRACES.json.gz --initialization INITIALIZATION --checkpoints ACTION-VALUE-CHECKPOINTS --action-value
python -m configs.freeze_ppt_bpd_deadline_bundles MATCHED-BUNDLE-COHORT BUNDLE-DEADLINE-COHORT
python -m configs.check_ppt_bpd_bundled_gym BUNDLE-DEADLINE-COHORT BUNDLE-DEADLINE-REFERENCE
python -m configs.run_ppt_bpd_rewrite BUNDLED-COHORT BUNDLED-REFERENCE MATHCAN-TRAIN.zip INITIALIZATION BUNDLED-RUNS
python -m configs.analyze_ppt_bpd_rewrite BUNDLED-COHORT BUNDLED-REFERENCE BUNDLED-RUNS INITIALIZATION BUNDLED-AUDIT
python -m configs.freeze_ppt_rewrite_deadline REWRITE-COHORT DEADLINE-COHORT
python -m configs.check_ppt_rewrite_deadline DEADLINE-COHORT DEADLINE-WITNESS
python -m configs.probe_ppt_rewrite_deadline_choices DEADLINE-COHORT DEADLINE-WITNESS REWRITE-COHORT REWRITE-REFERENCE EXPLORATION-RUNS EXPLORATION-AUDIT DEADLINE-CHOICES
```

The analysis requires EvalMath's FSI-01 module on the Python path. The runner uses the original Lean verifier and CPU-only PyTorch. The campaign runs under a 6 GiB process-tree cap, with 2 GiB and two PyTorch threads per worker.
