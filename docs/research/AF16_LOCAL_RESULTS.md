# AF-16: novel Horn-composition rule-priority transfer, final local result

**Registered scientific verdict: FAIL.** **Local independent engineering audit: PASS.** This is the completed frozen 4-seed × 3-arm matrix under Python 3.13.5 / Torch 2.10 CPU. The separately registered Python 3.12 / Torch 2.5.1 / Lean 4.19.0 hosted workflow [#43705227868](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43705227868) failed before any preflight steps ran, so that environment and formal Lean checks remain **unverified**.

- **Frozen scientific source**: [`bd43ad9765d9a78c32b5ee670968ac9762ede7ca`](https://github.com/MoralityLabAI/LoopedTransformers/commit/bd43ad9765d9a78c32b5ee670968ac9762ede7ca).
- **Preregistered protocol**: [PROTOCOL.json](../../experiments/activation_funnel_af16/PROTOCOL.json), SHA-256 `c5a72cdab59dc4434d25f0b36b31d3a14821eb8b72aa3cd12ec992226424ff8a`.
- **Frozen cases**: [fresh_manifest.json](../../experiments/activation_funnel_af16/fresh_manifest.json), SHA-256 `a4e701e1132d2cc19b56c79939ec24587dd3adef352c6ff43f7ed1453a8d59db`.
- **Parent**: AF-15 [draft PR #77](https://github.com/MoralityLabAI/LoopedTransformers/pull/77), frozen initial scientific source `eedb938`. This new experiment is independently stacked on it.
- **AF-16 draft**: [PR #82](https://github.com/MoralityLabAI/LoopedTransformers/pull/82).

## Design and independent reference

AF-16 reuses AF-15's exact 5,234-parameter original TypedLoop rule ranker. A separate, host-owned Horn transition checks whether all premises for the selected physical rule are already proved, then adds only its declared head. This ensures every accepted inference remains within the least Horn fixed point. The neural model can choose to stop early, and success means deriving the goal within the exact minimum number of rule firings plus one.

Training combines AF-15's original 672 training cases with 256 new `swap_support` and `choice_detour` graphs. Development contains 352 cases. New held-out structures include `cross_requisite` (Q0 is a premise for Q1), `cross_join` (both branch relays are premises for Q0), and `shortcut_fork` (an alternate direct route to the goal). There are 64 new-layout diagnostics and **192 cases at each of held-out depths 4 and 5**. Public permutations, fact inputs, rule connections, roles and enabled masks remain model inputs. The exact proof path, shortest distance, goal answer and source closure labels remain host-only.

Four seeds (17, 29, 43, 61) each train `imitation` and `frontier` for 240 fixed AdamW updates with identical model initialization and the same sample stream. `frontier` adds a softplus rank-margin loss over nonoptimal **legal** actions. The third arm, `imitation_extra`, receives **320 updates**, including the same first 240 draws, as a compute-advantaged reference. The two 240-update arms have matched native forward-call counts; the margin arm consumes extra backward arithmetic. This does not establish exact FLOP matching.

The scoring audit independently reconstructs graph edges from the observed physical incidence matrices and computes shortest legal proof distance by breadth-first search, separate from the memoized shortest-path DP used for training. It checks all accepted proof edges, the raw model logits, chosen actions, budgets, original targets, saved scores, and the exact checkpoint SHA-256 plus Torch/NumPy equality. Each audit independently reproduces sampled model outputs from the frozen weights.

## Development eligibility

The preregistered eligibility gate requires both the imitation and frontier arms to reach **75% original training** and **65% development** proof success for each seed. The full matrix is retained:

| Seed | Imitation train/dev | Frontier train/dev | 320-update imitation train/dev | Eligible? |
|---:|---:|---:|---:|---|
| 17 | 48.78% / 48.00% | 48.78% / 48.00% | 84.63% / 84.67% | No |
| 29 | 48.78% / 48.00% | 48.78% / 48.00% | 84.63% / 84.67% | No |
| 43 | 48.78% / 48.00% | 55.61% / 55.33% | 84.63% / 84.67% | No |
| 61 | **84.63% / 84.67%** | **84.63% / 84.67%** | 84.63% / 84.67% | **Yes** |

All models were frozen after their registered final step. Training uses only the predetermined training cases; development determines eligibility. The new heldout families were evaluated after checkpoint publication.

## Held-out proof-search performance

Accuracy is solved / 52 provable goals in each depth family. The 192 physical cases per family include two layouts for each logical case, so the 52 positive cases correspond to 26 unique logical input patterns represented twice.

| Seed | Depth | Imitation 240 | Frontier 240 | Imitation 320 | Fixed physical order | Shortest oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 4 | 0.0000 | 0.0000 | 0.5962 | 0.8654 | 1.0000 |
| 17 | 5 | 0.0000 | 0.0000 | 0.5962 | 0.7885 | 1.0000 |
| 29 | 4 | 0.0000 | 0.0000 | 0.5962 | 0.8654 | 1.0000 |
| 29 | 5 | 0.0000 | 0.0000 | 0.5962 | 0.7885 | 1.0000 |
| 43 | 4 | 0.2885 | 0.2885 | 0.3077 | 0.8654 | 1.0000 |
| 43 | 5 | 0.2885 | 0.2885 | 0.3077 | 0.7885 | 1.0000 |
| **61** | **4** | **0.5385** | **0.5962** | **0.5962** | **0.8654** | **1.0000** |
| **61** | **5** | **0.5385** | **0.5962** | **0.5962** | **0.7885** | **1.0000** |

The primary scientific gate requires each eligible seed to gain **at least +0.05 versus imitation and +0.02 versus the extra-update imitation** on *both* novel depths, reach ≥0.70 absolute proof success, remain within 0.03 of fixed rule order and never accept a false proof. Seed 61 gains **+0.057692** on each depth against 240-update imitation, but **0.000000** versus the 320-update control. Its frontier absolute success is **0.596154**, below 0.70 and substantially below fixed order. This eligible failure makes the aggregate **FAIL**. Seeds 17, 29 and 43 are excluded from primary positive/negative claims by the frozen eligibility gate.

## Mechanism inspection, exploratory

At both heldout depths, **17 and 29** choose STOP immediately on all 52 provable examples with their 240-step frontier models. Their 320-step imitation controls solve 31/52. This is a training-budget sensitivity result with exactly the same original initializations and first 240 minibatches; the longer training runs receive more updates and their different optimizer paths are part of that comparison.

Seed 61's frontier improvement is concentrated entirely in the new **`shortcut_fork`** composition. It solves 15/24 provable shortcut examples at each depth, versus 12/24 under imitation. The other composition groups remain unchanged: 6/12 for `cross_requisite` and 10/16 for `cross_join`. There are only **3 extra successful physical-layout examples** per depth, and the 26 underlying logical input patterns are shared across the two depth-family layouts. These outcomes support a narrow learned priority effect, not statistically established general-purpose proof-search transfer.

The symbolic fixed-rule-order control is stronger than every neural model on these heldout compositions. Structural host soundness prevents any false accepted proof in all arms; early STOP and inefficient ordering still cost proof success. Improvements in rule ranking can therefore be studied separately from proof validity.

## Evidence and engineering boundary

The independent local audit accepted **10,752 original-host episode rows** across 4 seeds × 6 policy/baseline arms × 448 held-out cases and replayed **147 model calls** directly from frozen weights. All 12 Torch checkpoints and NumPy named-array exports match their retained SHA-256 identities. **Zero false accepted Horn proofs** were recorded. The source hashes of the original AF-15 ranker and gym, AF-05 reference grammar, AF-02 native adapter, and Lean reasoner match the GitHub branch's Git blob identities. Nine Python tests passed in both standard and optimized mode, including tamper rejection after modifying and rehashing a saved action record.

The source carries a Lean 4.19.0 Boolean-host-soundness file, but GitHub Actions [run #43705227868](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43705227868) failed before preflight runner execution, leaving the pinned Torch and Lean checks outstanding. This local result is a completed *local* scientific experiment, not a claim of hosted reproduction or Lean certification of neural outputs.

The registered negative result is retained. A future AF-17 could train a state-dependent Bellman distance-to-go with an explicit STOP opportunity-cost target; the associated exact ranking-gap lemma is in [AF16_BELLMAN_RANKING.md](AF16_BELLMAN_RANKING.md). This would be a separately registered objective, not a modification to AF-16.
