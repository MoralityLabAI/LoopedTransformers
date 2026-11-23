# LSA-02I: Native Lean bounded backtracking with explicit reconsideration cost

**Date:** November 24, 2026 (Manila). **Scientific status:** development screen **PASS** for charged backtracking over frozen greedy; formal learned-policy competence **BLOCKED** by the independently sampled random-DFS margin. **Implementation:** [PR #330](https://github.com/MoralityLabAI/LoopedTransformers/pull/330), squash merge `4ce9ea4eab434d64ebe58c5407188d03ac7edc49`. **Evidence:** [successful complete CI run 45265151419](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419). This is a CPU-only study using generated Lean Horn theorems; ordinary and OOD sealed test proofs remain unopened.

## Research question and registered intervention

[LSA-02H](lsa02h_recoverable_offpath_cpu_20261123.md) found 93 new Lean-certified recovery training states, all of them forced single-choice states. This exposed the need for an explicit way to **undo an accepted tactic that leads to an unprovable or cyclic continuation**.

LSA-02I compares one frozen Plan B ranker under **greedy tactic selection** against the **same ranker with depth-first proof-state reconsideration**. The live Lean 4.19 `lgym` interface supplies public goal states, type-directed candidate actions, tactic verdicts and terminal proof kernel checks. The policy receives neither generator labels nor the Lean reference solver's reachable/dead branch annotations during inference.

The runner retains the actual native Lean state identifier, public goal fingerprint, attempted tactic set and accepted root-to-state proof prefix at each stack frame. After the current public menu is exhausted, or the public state repeats an ancestor, a registered **RECONSIDER** event pops that frame and restores its predecessor. The abandoned action stays marked tried at the predecessor. Any successful final script discards all abandoned tactics, and this final script is separately compiled against the original full theorem statement.

**Cost contract:** one work unit per attempted tactic, regardless of acceptance; one additional work unit per RECONSIDER. Every arm has the same fixed **16-work-unit allowance**. The model's remaining-budget token saturates at eight, exactly as required by the existing public pp96 input contract. Each episode records attempts, reconsiderations, work units, model calls and terminal state. A rollback is a charged search operation; it never counts as a Lean theorem tactic in the final script.

## Source and matched CPU checkpoint training

A fresh G03 suite was frozen at source seed `20261202`, with **64 independently compiled training theorem roots** and **32 disjoint calibration roots**. Native Lean 4.19 extracted **338 shortest-continuation source frames**, independently compiled all 64 reference proofs with zero extra axioms, and found **60/64** training reference scripts fit an eight-tactic budget.

The frozen theorem suite manifest SHA-256 is `58f7feccfaf4b04d6891eb4e5478bdcaec9ffe1c97a2a3df23aea6e2f0dd31ef`; the Lean-certified teacher manifest SHA-256 is `a16c10a49eaa601b84b0e512c1afe8934924447421a35a3c87e022ab3bfdc9fb`. The Lean 4.19 executable fingerprint is `92c3d35b5bfaa5e0fea413a775d504cf46cd95e1345df61c2274f76779e7e023`, and the `lgym` server binary fingerprint is `2f62adfeb75301449cf7f332691eb294af5dc0446801ac52e9bd58cfc15a5bc4`.

Model seeds **173** and **179** each trained a **57,475-parameter Plan B** on the same frozen training corpus, width 32, four heads, two AB recurrent passes, pp96 public typed/prettified state, 384-byte input budget and 720 FP32 AdamW optimizer updates. Each fit was repeated on **one physical GitHub-hosted CPU worker**. Both same-worker duplicate pairs passed exact saved tensor, checkpoint-file and optimizer-step trace equality; frozen evaluation did not update any model parameters.

The two trained checkpoint SHA-256 values are `c34e9d36e72cb32499a98803d6e146fd4957f066e637f025328966344316f6e5` and `581eba595856ac77f3d34c9fe2298d447c54a15dda34cbd51e11c0d1e9f80989`.

## Whole-proof native Lean results

All policies saw the **same 32 calibration theorem lineages** under the **same 16-work-unit budget**. Native Lean independently certified every successful whole-proof terminal state; every winning script also passed standalone axiom-free Lean compilation and model-free replay. A shortest-reference oracle, computed only after live policies finished scoring, found **32/32** theorems attainable within 16 tactic actions. The prior eight-tactic test results have a different resource budget and are not directly comparable.

| Policy | Seed 173 | Seed 179 | Mean certified closure |
| --- | ---: | ---: | ---: |
| Frozen Plan B, greedy, 16-unit cap | 6/32 | 8/32 | **14/64 = 21.875%** |
| **Same frozen Plan B with charged DFS rollback** | **19/32** | **23/32** | **42/64 = 65.625%** |
| Deterministic public-order DFS | 19/32 | 19/32 | **38/64 = 59.375%** |
| Native uniform-menu DFS, 5 seeded streams | See below | | **104/160 = 65.000%** |
| Independently verified reachable reference | 32/32 | 32/32 | **100% ceiling** |

The five **model-free** uniform DFS streams were fixed in advance at `17,31,47,61,83`. Their certified root counts were **22, 18, 19, 23, 22** of 32 respectively, totaling **104/160** native proofs. The random control used no model calls and paid exactly the same tactic and backtracking costs as the learned DFS. These are 32 independent theorem roots with five repeated stochastic outcomes, not 160 independent theorems.

### Resource accounting

| Policy | Completed episodes | Charged tactic attempts | Charged reconsiderations | Mean effort spent per episode | Neural forward calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen greedy | 14/64 | 247 | 0 | **3.86 / 16** | 247 |
| Frozen DFS | 42/64 | 548 | 208 | **11.81 / 16** | 548 |
| Public-order DFS | 38/64 | 554 | 230 | **12.25 / 16** | 0 |
| Uniform DFS | 104/160 | 1,437 | 572 | **12.56 / 16** | 0 |

The larger DFS success rate comes with additional evaluated tactics and charged rollbacks. The greedy policy often terminates early in a dead branch with unspent budget; it cannot recover. The experiment identifies a **search-procedure benefit** under equal maximum allowance and explicitly reports the different effort actually consumed.

## Primary development screen and uncertainty

The preregistered outcome was native standalone-certified full-proof success for frozen DFS minus the **same checkpoint** under greedy search. Scores were paired by theorem within seed, averaged over seeds within each of **32 unique theorem lineages**, then evaluated with a descriptive 2,000-draw theorem-root bootstrap.

| Root-paired contrast | Difference | Descriptive 95% root-bootstrap interval |
| --- | ---: | ---: |
| **Frozen DFS minus frozen greedy** | **+43.750 percentage points** | **+31.250 to +57.8125 points** |
| Frozen DFS minus deterministic public-order DFS | +6.250 points | −9.375 to +23.4375 points |
| Frozen DFS minus native uniform DFS | **+0.625 points** | Approximately **−9.688 to +10.625 points** |

The **+5-point** preregistered development screen passed by a wide margin, with positive DFS-minus-greedy improvement at **both model seeds**. The second and third comparisons do not establish a meaningful advantage over model-free search. Their intervals include zero.

## Native post-scoring dead-branch audit

Only **after** all model and random proof evaluations did a separate Lean graph auditor inspect the accepted tactic transitions and determine which led to unprovable successor subgoals. The online policy used only public state, the candidate menu, earlier Lean verdicts and its own path history.

| Post-scoring event count | Frozen greedy, 64 episodes | Frozen DFS, 64 | Auto DFS, 64 | Uniform DFS, 160 |
| --- | ---: | ---: | ---: | ---: |
| Accepted tactic transitions with unreachable successor | 73 | **142** | 170 | 407 |
| Reconsiderations that discarded a dead-branch script prefix | 0 | **166** | 204 | 465 |
| Reconsiderations triggered by a public-state cycle | 0 | **17** | 10 | 45 |

These are **event counts on correlated trajectories**, not counts of unique theorem roots or one-to-one causal rescues. A branch may have multiple nested reconsiderations. The overall outcome nevertheless directly verifies that the runner can unwind previously accepted, unproductive Lean states and produce standalone-verified proofs.

The full native journals are protected by hashes; independent replay verifies the public tactic menu, state fingerprints, history stack, each RECONSIDER cause, every one-unit cost, surviving proof prefix and terminal native kernel outcome.

## Scientific gate and next falsifiable experiment

The model's **65.625% whole-proof closure passes the 40% absolute completion floor** on 32 independent calibration roots. The separate competence gate also requires at least **+10 points above independently sampled native uniform DFS**. Its measured margin was only **+0.625 points**, so the formal result remains `BLOCKED_INCOMPETENT_CHECKPOINT`, specifically for failure to establish **learned action-selection superiority** over random-search DFS.

The new search mechanism is usable, and its native verifier/cost/replay infrastructure is complete. The result does **not** show that the Plan B network itself reasons better than random branch selection. The theorem generator, public candidate grammar and finite search depth remain limited to synthetic Lean Horn proofs, with external dataset overlap and Mathlib transfer unverified.

A prospective **CPU-only successor** should train a native-Learned *branch-viability / recovery-cost* ranking head on **train-only Lean graph evidence**. Its learning target would identify a public tactic whose accepted Lean successor is provable within the remaining budget, while maintaining explicit `unknown` or unreachable labels for actions requiring backtracking. Compare it against an equal-weight shuffled-label sham, the existing shortest-path learner, and **the now strong 16-work-unit uniform DFS control**, with fresh independent theorem roots and explicit verifier work costs. Keep native standalone compilation, source ancestry checks and the FSI-01/sealed confirmation gate.

## Reproduction and artifact custody

- [Merged implementation PR #330](https://github.com/MoralityLabAI/LoopedTransformers/pull/330)
- [Preregistered resource and competence protocol](../docs/lsa02i_native_backtrack_cpu_protocol.md)
- [Bounded native proof search and journal replay](../research/lean_cgrt_hrm400/ppt_gym/backtrack_search.py)
- [Proof compilation and model-free native replay campaign](../research/lean_cgrt_hrm400/ppt_gym/backtrack_campaign.py)
- [Matched frozen CPU training](../research/lean_cgrt_hrm400/ppt_gym/backtrack_train.py)
- [Post-scoring native branch and root-cluster audit](../research/lean_cgrt_hrm400/ppt_gym/backtrack_compare.py)
- [Complete green CI run 45265151419](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419)
- [Native teacher and frozen suite artifact 14861533754](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419/artifacts/14861533754)
- [Same-worker CPU checkpoint artifact 14862538291](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419/artifacts/14862538291)
- [Seed 173 native model proof artifact 14863036082](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419/artifacts/14863036082)
- [Seed 179 native model proof artifact 14863435723](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419/artifacts/14863435723)
- [Five-stream uniform DFS proof artifact 14860434616](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419/artifacts/14860434616)
- [Root-paired proof effects and dead-branch oracle audit artifact 14863750271](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45265151419/artifacts/14863750271)

All artifacts are registered with **14-day retention**. Source hashes, per-root outcomes, backtrack cost ledgers, accepted scripts, native proof compiler receipts and model-free independent replay remain available in their manifests. This report preserves the benchmark statistics and scientific decision after the artifacts expire.
