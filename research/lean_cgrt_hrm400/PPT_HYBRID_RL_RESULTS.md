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
