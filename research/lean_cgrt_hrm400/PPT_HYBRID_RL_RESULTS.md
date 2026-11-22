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

## Evidence limits

The study is a development pilot with one paired model seed and reused evaluation roots. A positive result on one of six alt-cycle roots cannot establish a gain on the family. A confirmed gain requires independent paired seeds, a fresh family-disjoint evaluation, a calibrated-recurrence control and a pre-registered evidence threshold. Behavioral parameter decomposition still requires a causal contrast whose effect clears the matched control on new development roots.
