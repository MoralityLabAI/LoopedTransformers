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

The confirmation cohort holds out 12 further layouts and uses the seed-151 warm checkpoints. It checks the same eight-tactic outcome and requires at least three additional complete pairs over each learned or frozen routing control. Two controls also test the stronger transport representation directly: isolated frozen support weighting and ridge calibration with the support-weighted encoder frozen. The exact public guard cost remains the automation reference. The two routing fits for this seed were frozen before running the confirmation cohort.

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
python -m configs.check_ppt_bpd_dependent_rewrites dev ROUTING-CONFIRMATION-WITNESS.json.gz --routing-confirmation
python -m configs.probe_ppt_bpd_dependent_policy SPUR-WITNESS.json.gz SPUR-TRAIN.json.gz SEED-AUDIT SEED-RUNS SEED-TRM-RESULTS.json.gz SEED-TRM-WEIGHTS --arms trm_rl
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
