# LSA-02J: Branch viability and recovery-cost learning against public Horn DFS

**Date:** November 24, 2026. **Status:** native Lean data, matched CPU training, standalone proofs and replay **PASS**. Preregistered learned-versus-sham development screen **FAIL**. Strong-control learned-competence gate **FAIL**. [PR #331](https://github.com/MoralityLabAI/LoopedTransformers/pull/331) merged as fbf546f3d55d0bd9c5de140220ec2b6c49d7f0d5. [Complete GitHub Actions run 45269861563](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45269861563) passed, along with the existing PPT Gym CPU, native Lean G03 and LSA-02I backtracking regressions.

The [registered protocol](../docs/lsa02j_branch_utility_cpu_protocol.md) follows [LSA-02I](lsa02i_native_bounded_backtrack_cpu_20261124.md), in which a model-free uniform DFS policy already achieved 65% native whole-proof completion under sixteen charged tactic/backtrack work units. LSA-02J tests whether learned viability and recovery-cost supervision provides an advantage over the same frozen model, a label-shuffled auxiliary head and a stronger deterministic symbolic control.

## 1. Native Lean 4.19 source and targets

A new Lean-CGRT Horn suite was frozen at generator seed **20261203**, 64 roots per train/calibration/ordinary-test/OOD split. Real Lean 4.19 supplied all tactic transition verdicts and minimum successor proof lengths on **64 independent training roots**. Every training reference script independently compiled, and **55/64** required at most eight tactics.

| Training-only evidence | Count |
| --- | ---: |
| Independent theorem roots | 64 |
| Public shortest-reference training frames | 347 |
| Public candidate actions | 588 |
| Finite viable native actions | 407 |
| Rejected or unprovable actions | 181 |
| Mixed viable/dead action-choice states | **125** |
| Training roots with mixed viable/dead choices | **61** |

Each action's training label is finite viability plus minimum whole-state recovery cost, defined as one tactic plus the native shortest cost of all successor and still-pending goals. A null cost marks a rejected or unprovable action. This is a native tactic-count lower bound: it does not model the additional cost of wrong choices or backtracks. The source label generator checked shortest-continuation agreement against the original Lean teacher.

**Source hashes:** native teacher manifest SHA-256 9cbb141cbc397dd998e69e798044c5cc6378c3e9097a1994ba87ab255ee83d24; branch-label manifest SHA-256 e20ece23266fdf9794097d8fbe2616fcdd063158bd3e503d712a20268c514c8c; frozen suite manifest SHA-256 8d1b6da6bfe1114dc69862aee5789fc694dcb39e5fa06bb134f8c863029d8f89.

The actual source and its Lean certificates remain archived in [native training artifact 14864217602](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45269861563/artifacts/14864217602).

## 2. Matched CPU models and learned readout

Both frozen original Plan B backbones used **57,475 parameters**, width 32, four heads, two tied AB recurrence passes, and the 384-byte pp96 typed/public-pretty state. Registered seeds were **183 and 191**. Each backbone was trained for 720 FP32 AdamW updates **twice on one CPU worker**, passing exact tensor, checkpoint-byte and optimizer-trace checks.

Each backbone then remained frozen while two small auxiliary viability/recovery-cost heads were trained for **480 steps** apiece. The head receives 32 frozen Plan B candidate features plus eight normalized public premise/count statistics. It predicts finite-viability logits and normalized finite recovery cost, trained by binary cross-entropy plus 0.25 times cost regression. The live policy adds a fixed 5-times viability-minus-half-cost adjustment to the unmodified frozen Plan B tactic logit.

The **shuffled-sham** head has the same architecture, initial tensors, optimizer, source frames and root/frame draw schedule. Native viability/cost label pairs are deterministically permuted among the *same frame's candidate tactics*. Forced singleton menus remain unchanged. Every learned and sham head fit was independently repeated, and all four duplicate pairs passed exact checkpoint/optimizer-trace verification. Training used one Torch thread, deterministic algorithms, disabled MKLDNN, pinned CPU PyTorch 2.5.1 and **zero calibration-root optimizer updates**.

[Same-worker training/checkpoint artifact 14865120681](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45269861563/artifacts/14865120681).

## 3. Native whole-proof completion under charged search

All policies used the identical **32 fresh calibration theorem roots**, actual Lean 4.19 tactic adjudication, 16 effort units per episode with one unit for every attempted tactic or explicit RECONSIDER, complete search-journal replay and independent standalone axiom-free compiler checks for terminal scripts. The five-stream model-free uniform DFS control ran 160 episodes on the same 32 roots and closed **101/160 = 63.125%**, with 504 charged backtracks.

The separate **public_horn_dfs** policy reads only publicly declared Lean Horn hypothesis types, derives minimum proof costs through a deterministic forward-chaining fixed point and selects the lowest-cost finite tactic. It does not use a neural network, training labels, the native hidden transition table, or the generator's proof.

| Search policy | Seed 183 | Seed 191 | Matched mean |
| --- | ---: | ---: | ---: |
| Frozen Plan B DFS | 22/32 | 18/32 | **62.500%** |
| Learned viability/cost DFS | **24/32** | **26/32** | **78.125%** |
| Shuffled-label sham DFS | 24/32 | 25/32 | **76.5625%** |
| Model-free public Horn DFS | **32/32** | **32/32** | **100.000%** |
| Model-free five-stream uniform DFS | 101/160 across five random streams | | **63.125%** |

The structural solver took only **201 work units across 32 theorem roots per evaluation**, with zero backtracks or model calls. It achieved full proof closure with average **6.28** work units per theorem, exposing how efficiently a classical algorithm can exploit the explicitly visible Horn dependency graph. The original reference oracle independently found all **32/32** calibration theorems attainable within the 16-unit budget.

The learned head's advantage over frozen DFS is **+15.625 percentage points**, with descriptive theorem-root-bootstrap 95% interval **[0, +29.6875] points**. Its advantage over the shuffled sham is only **+1.5625 points**, interval **[0, +4.6875] points**. Against model-free public Horn DFS, the learned head is **−21.875 points**, interval **[−37.5, −9.375] points**.

Repeated model seeds are paired observations on the same 32 theorem lineages, and random streams are repeated stochastic evaluations on those same roots. Neither 64 model-seed measurements nor 160 uniform episodes are independent theorem samples.

## 4. Post-scoring native viability and cost calibration

**Only after every native proof-policy run completed**, a separate job used real Lean to re-explore the 32 calibration theorems and independently compile all 32 reference proofs. It yielded **201 reference-path public states and 341 candidate actions**, with **79 mixed viable/dead choice states across 31 independent roots**. The frozen learned and sham heads were scored on exactly those same states with zero gradient/optimizer updates.

| Root-weighted calibration metric | Seed 183 utility | Seed 183 sham | Seed 191 utility | Seed 191 sham |
| --- | ---: | ---: | ---: | ---: |
| Finite-viability classification accuracy | 65.62% | 76.43% | 73.27% | 76.43% |
| Viable-choice accuracy on mixed action forks | 44.84% | 44.84% | 40.27% | 40.27% |
| Normalized finite recovery-cost MAE | 0.1185 | 0.1148 | 0.1012 | 0.1023 |

The learned branch head shows **no held-out development choice advantage over its shuffled-label sham** on the actual mixed-choice states. The ordinary per-action accuracy measure is lower for the learned head than for the sham, and the cost errors are comparable. None of these metrics establishes specific recovery-cost or viability competence.

The public Horn fixed-point solver's *exact predicted minimum whole-state tactic costs matched the native Lean graph on **every one of the 201 calibration reference-path states***. The fixed-point algorithm's matching costs and 32/32 full-proof completion show that the problem family exposes enough public structure for a complete nonlearned solution.

[Post-scoring Lean calibration and paired-root evidence artifact 14863933226](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45269861563/artifacts/14863933226). The native closed-proof receipts for the two model seeds are [183 artifact 14865725398](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45269861563/artifacts/14865725398) and [191 artifact 14863774671](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45269861563/artifacts/14863774671).

## 5. Gate decision

**Registered development screen: FAIL.** The learned head exceeds unchanged frozen DFS by +15.625 points, surpassing the required +5, but exceeds the label-shuffled sham by only **+1.5625 points**, below the second required +5 threshold. Native completion remains nonnegative versus frozen at both matched seeds.

The older native model preflight of ≥40% standalone closure and ≥10 points over independently sampled random DFS is met by the learned probe (78.125% versus 63.125%). However, the **LSA-02J registered learned-competence preflight** requires at least +10 over *both* random DFS and the model-free public Horn solver. It **fails by a wide margin** against the structural 100% baseline. This prevents claiming learned search superiority or promoting the architecture/gym.

All measurements are development-only synthetic Horn theorem results. Source ancestry outside the generated corpus, Mathlib transfer, independent sealed confirmation, FSI-01 and cross-worker bitwise numerical invariance remain unverified.

## 6. Research consequence

The current small Horn theorem gym supports a **complete exact classical algorithm from the public local hypotheses**. A learned probe trained on finite-viability/cost labels can help the frozen DFS search procedure, but matched random-label supervision produces nearly the same gain, while the classical public solver reaches 100% with fewer work units. Enlarging recurrent depth, heads, training hours or model size on this benchmark would not provide convincing evidence of improved learned reasoning.

The useful next research step is to register a harder native Lean proof-task family whose full solvability cannot be obtained by one forward-chaining pass over publicly enumerated Horn premises. Examples include generated tasks with dependent rewrites, equality reasoning and tactic choices that require validating an intermediate transformation, or a properly sourced Mathlib benchmark with certified train/evaluation separation. Retain an explicit public-only deterministic symbolic competitor in every new experiment. Track learned benefit over that competitor, not just a random tactic menu.

The experimental modules, native-source compiler receipts and parent workflow are indexed from [the LSA-02J protocol](../docs/lsa02j_branch_utility_cpu_protocol.md). GitHub Actions artifacts expire under the configured 14-day retention policy. This Markdown report preserves the methodological and scientific outcome.
