# Hybrid Lean RL development results

The 1,647,874-parameter hybrid and the 1,646,850-parameter TRM control used the same proof-state and tactic encoders, action head, critic, candidate generator, six CPU-second deadline, 1.2 billion estimated-MAC allowance and eight-check limit. The hybrid used four tied-cell passes and TRM one answer recursion with two latent updates. Each proof counted here has live Lean 4.19 kernel closure, independent axiom-free compilation and a separate native transcript replay.

## Native-only preflight

| Development population | Roots | Hybrid before RL | Hybrid after RL | TRM before RL | TRM after RL |
|---|---:|---:|---:|---:|---:|
| LoopedTransformers alt-cycle | 6 | 2 | 1 | 1 | 0 |
| LoopedTransformers tree-deep | 6 | 2 | 3 | 2 | 3 |
| EvalMath / BPD-LG-02 exact | 4 | 4 | 4 | 4 | 4 |
| All | 16 | 8 | 8 | 7 | 7 |

After 16 actor-critic updates, the hybrid had two wins, one loss and 13 ties versus TRM. The one-proof aggregate gap was already present before reinforcement learning in this seed. Both models certified all four BPD exact development roots. Removing the pass-two recalled bank changed zero selected proof paths. The original slot-permutation check was permutation-invariant under attention. The subsequent mixed-source pilot corrected the sham to permute channels and checked that the intervention actually changes the next pass while preserving the bank norm.

Native reward training used only executed tactic trajectories, independent proof closures and tactic cost. The original local EvalMath experiment generator at `78fe9cc9d11e98c459a4ba367f2e8fb821a835b0` matched the LoopedTransformers BPD-LG-02 generator after newline normalization. The later independently named EvalMath fixture was left outside this cohort.

## Mixed-source development run

| Development population | Roots | Hybrid actor-critic | TRM actor-critic | Frozen TRM backbone | Hybrid imitation |
|---|---:|---:|---:|---:|---:|
| LoopedTransformers alt-cycle | 6 | 3 | 2 | 2 | 3 |
| LoopedTransformers tree-deep | 6 | 3 | 1 | 3 | 1 |
| EvalMath / BPD-LG-02 exact | 4 | 4 | 4 | 4 | 4 |
| All | 16 | 10 | 7 | 9 | 8 |

On the same frozen roots the hybrid gained two completions relative to its own untrained checkpoint. TRM full training matched its initial count. Frozen-backbone TRM gained two completions by fitting its policy and critic heads, so this pilot has not established calibration resistance. The hybrid's final paired comparisons were three wins and zero losses against full TRM and three wins and two losses against head-only TRM. The imitation comparison is exploratory because that arm consumed native Lean episodes only. The other three arms had the same archived MathCan replay schedule.

Each mixed arm sampled eight exact-prefix replay episodes from the original SHA-pinned MathCan TRAIN archive. The hybrid observed one recorded closure and one censored continuation. Targeted verification separately replaced the original proof body of `MathCan.FiniteHorizonGap.epsilon_sixth_equiv` with the four archived steps and compiled the entire module twice in Lean 4.35. Both returned the registered axioms `propext` and `Quot.sound`. This verifies a training-source witness. The scored development completions are the new Lean 4.19 trajectories.

Neither pass-two bank removal nor the channel-permutation sham changed a greedy proof path on these 16 roots, even though the channel permutation changed the next pass state in the model test. This readout provides no causal case for steering training towards that message. The improvement warrants independent replication before an efficacy or mechanism claim.

## Evidence limits

The study is a development pilot with one paired model seed and reused evaluation roots. One more alt-cycle proof than the calibrated TRM cannot establish a gain on the family. A confirmed gain requires independent paired seeds, a fresh family-disjoint evaluation, a calibrated-recurrence control and a pre-registered evidence threshold. Behavioral parameter decomposition still requires a causal contrast whose effect clears the matched control on new development roots.

The [replication cohort](results/ppt_hybrid_rl/replication-cohort.json) freezes 24 fresh alt-cycle and 24 fresh tree-deep roots for seeds 29, 43 and 71. Its lineages exclude the original hybrid study and all five prior G04 cohorts. It was frozen and checked for leakage before any policy was scored on it.

## Three-seed replication

All nine registered runs completed with matched per-seed training-root schedules, certificates, and final saved weight hashes. Each root is one statistical unit with three paired model-seed observations.

| Task | Root-seed observations | Hybrid RL | Full TRM RL | Frozen TRM backbone |
|---|---:|---:|---:|---:|
| Alt-cycle, primary | 72 | 32 | 22 | 29 |
| Tree-deep, secondary | 72 | 28 | 40 | 35 |

Alt-cycle favored the hybrid against full TRM in each seed. The hybrid gained seven completions over its own initial policy. Full TRM lost seven and the frozen-backbone control finished with the same aggregate count it started with. The paired hybrid result was 19 wins and 9 losses against full TRM, and 8 wins and 5 losses against the frozen-backbone control. EvalMath FSI-01 classifies the primary comparison with full TRM as suggestive. The gain over the frozen-backbone control is 4.17 percentage points, below the registered five-point floor, and the result is inconclusive. The replication is still development evidence. Aggregate peak resident memory for the complete Windows process tree was 2.16 GiB, inside the enforced 6 GiB commit cap.

## Mechanistic diagnostics and the next candidate

On the earlier 16 development roots, removing attention value content at all bank reads changed three greedy proof paths, while the equal-norm channel-subtraction control changed one. Both arms independently certified nine proofs, down from ten unedited. A pass-two-only edit changed no paths in either arm. The controls preserve the keys and match the perturbation norm at each realized bank state. These results establish sensitivity of some paths to the all-read intervention on this checkpoint. They do not support a message-specific proof-completion improvement or a training-steering claim.

A separate finite Lean witness identifies an input bottleneck. Two public premise graphs expose the same target, immediate candidate rules and lexical inputs to both original policies. Their distant source facts differ. For each world, the two first actions were forced separately. Exactly one direction closed with a replayed, independently compiled proof on the eighth check, and it was the opposite direction in the other world. A policy or calibration that sees only the original encoded input has a 50 percent ceiling on this equally weighted pair. That ceiling is scoped to the recorded input and eight-check limit.

The `--graph-context` candidate supplies the full visible Horn facts and implications to both the hybrid and the TRM comparator. The hybrid uses two gated premise-to-conclusion message updates per tied pass, while TRM attends over tokens for the same atoms and implications. The frozen-backbone control trains the lexical and graph readout heads. Outside parsable Horn goals the policy falls back to the lexical route, including the archived MathCan observations. A separate [paired development cohort](results/ppt_hybrid_rl/graph-alias-pilot-cohort.json) varies source-fact placement, chain length and nuisance graph shape, and holds out longer chains. Each paired context is one unit. Training mixes the shorter pairs with BPD-LG-02 and audited MathCan replay. The graph change provides the missing observation, and its comparative proof performance requires a matched run.
