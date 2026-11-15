# AF-12 local evaluation | 2026-11-15

**Local scientific gate: FAIL.** The sole eligible seed (17) fails the preregistered primary depth-4 and depth-5 closure-transport improvement requirements. Seed 29 is ineligible because its `relay_final` comparator missed the train/dev goal-accuracy threshold. The local implementation and its independent replay **passed**.

**Official registered-runtime verdict: NOT RUN.** GitHub Actions could not assign a runner. Original [AF-12 run 43692333571](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43692333571) failed twice before executing any step, each with runner ID 0 and an empty step list. A workflow-only alternate-runner trial on Ubuntu 22.04, [run 43692840481](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43692840481), failed the same way. There are no hosted Python/Torch 2.5.1 or Lean 4.19.0 results for AF-12. The numerical outcomes below use **Python 3.13.5 / Torch 2.10.0+cpu**, with the same frozen algorithm, seeds, objectives, weights and train/dev/heldout rosters. Do not substitute this local evaluation for a successful pinned hosted study.

- **Frozen scientific source:** [`6743d9d3731a24222590eaacdca74996ddabbfee`](https://github.com/MoralityLabAI/LoopedTransformers/commit/6743d9d3731a24222590eaacdca74996ddabbfee).
- **Workflow-only runner repair:** [`12d96e6fa300838b726e5c7ddf9353fae9f399b0`](https://github.com/MoralityLabAI/LoopedTransformers/commit/12d96e6fa300838b726e5c7ddf9353fae9f399b0). Scientific source and preregistration unchanged.
- **Preregistered protocol SHA-256:** `1c1f7719662f55947b4e55efeba3cea58cf25c0c7450f8891548470b2faaace7`.
- **Prospective feature-manifest SHA-256:** `8212687f37f0bd8cc158efc9ab037ecb61b541eee89607b48538cde651e22254`.
- **Original AF-06 qualified parents:** seed17 artifact `14116102941`, SHA-256 `c8cf6c6c8264cf313e755c4d3008c26ae72cdd6ef20583fadc112bf02169ffdd`; seed29 artifact `14115887129`, SHA-256 `660b71acd5dc95bbfd200e55f206f609dcf6cc141ccdbb131d74bb7e5f0d4ebf`.
- **Draft PR:** [AF-12 #63](https://github.com/MoralityLabAI/LoopedTransformers/pull/63), stacked on AF-11 #62.

## The tested mathematical distinction

AF-11 demonstrated that supervising downstream Boolean influence can leave the *upstream* Q0/Q1 representation incorrect on long Horn chains. AF-12 treats the upstream computation as repeated applications of a monotone closure map

`Φ_R(S) = S ∪ {head(r) : premises(r) ⊆ S}`.

All arms reuse the same source TypedLoop, the same exact AF-06 qualified warm-up weights and the AF-10 structural Q-code cut. Each trains for **160 AdamW updates**. `terminal` gets goal/Q/counterfactual supervision. `relay_final` adds final-prefix atom truth; `closure_rollout` supervises every intermediate atom after each recurrence step using the independent synchronous Horn oracle and adds a probability-monotonicity loss.

Training uses the unchanged AF-05 depth-2/3 cases. Prospective inputs use physical layouts 14–19, disjoint by frozen feature hashes from all AF-05/AF-11 inputs. The same old logical AND/OR templates are reused, so this is a depth/layout generalization test within a synthetic Horn DAG family. It does not establish broader Lean proof discovery.

## All local development qualifications

| Seed | Arm | Development original goal | Development joint Q |
|---:|---|---:|---:|
| 17 | terminal | 1.0000 | 1.0000 |
| 17 | relay_final | 1.0000 | 1.0000 |
| 17 | closure_rollout | 1.0000 | 1.0000 |
| 29 | terminal | 0.8348 | 0.8973 |
| 29 | relay_final | 0.8259 | 1.0000 |
| 29 | closure_rollout | 0.9732 | 1.0000 |

Seed 17 meets both required ≥0.90 train/dev goal and joint-Q thresholds for the two main comparator arms. Seed 29's comparator is ineligible, leaving its held-out improvements as exploratory.

## New held-out results

The primary target is **upstream exact joint-Q accuracy** at the cut boundary. The second metric is final-prefix all-atom truth accuracy (15 physical atoms per case). Each deep family has 256 canonical input cases, one example per unique feature-hash within the family.

| Seed | New depth | Final-relay joint Q | Stepwise joint Q | Final-relay atom truth | Stepwise atom truth |
|---:|---:|---:|---:|---:|---:|
| 17 | 4 | **1.0000** | 0.5625 | **1.0000** | 0.5000 |
| 17 | 5 | **1.0000** | 0.2500 | **0.9000** | 0.6333 |
| 29 | 4 | 0.0625 | 0.2500 | 0.0333 | 0.2333 |
| 29 | 5 | 0.0625 | **0.5625** | 0.3833 | **0.6333** |

For the eligible seed 17, stepwise closure supervision **loses 43.75 percentage points** at depth4 and **75 percentage points** at depth5 against final-relay supervision. The final-relay arm reaches perfect held-out joint-Q inference at both depths, even though ordinary terminal supervision scores only 39.06% (depth4) and 25.00% (depth5). The process arm's depth5 informative counterfactual goal accuracy is 0.50 versus 0.75 for final-relay, and its irrelevant-patch goal retention also falls to 0.50 from 0.75.

Seed 29 moves in the other direction: stepwise supervision improves joint-Q from 0.0625 to 0.2500 (depth4) and to 0.5625 (depth5), while both main arms have only 0.50 clean goal accuracy at each depth. Its registered comparison remains ineligible.

All six models retain the original AF-10 exactly latched Q codes. All scored cases in the new depth3 layout achieve joint-Q accuracy 1.0 for seed 17. Across the new corpus, exact entire-trajectory all-atom correctness is **zero** for every arm, including the process-supervised arm. This suggests the native recurrent update and the imposed one-wave-per-call Boolean timetable are poorly aligned, or that the shared atom readout lacks the required per-stage semantics; those are hypotheses requiring a separate experiment.

## Independent local audit

Two complete training runs and two complete held-out evaluations passed. The independent scorer reconstructed source Horn truth, fresh input hashes, every synchronous closure stage, the goal and Q labels, both matched donor identities per case, the source-bound saved logits and the AF-10 protected-code identities. Its numerical replay checked 72 sampled actual Torch forward records per seed.

- **Six completed trained model weights**, each saved as a Torch checkpoint and a matching named NumPy export, with full per-file SHA-256.
- **3,936 clean cases** containing actual step-by-step atom logits, plus **7,872 donor-counterfactual rows**.
- **144 independently replayed forward records** from both seeds.
- **10 unit tests** passed in regular and optimized Python modes, including a rehashed-mutated-prediction rejection.
- **Deterministic local evidence ZIP:** 2,038,315 bytes, SHA-256 `96fc585f7b03aa8da1b91ef80c79bf8c55aa7e6079d4693ec0e1da0346a184a7`. Its internal manifest has a SHA-256 digest for every member and distinguishes local runtime from the unexecuted hosted study.

The local registered-rule reducer returns FAIL for the eligible seed 17, with seed 29 INCONCLUSIVE. This verdict is explicitly scoped to the observed local runtime.

## Mechanism implication

Final-relay supervision is stronger than stepwise process supervision for the eligible seed 17, even though both have identical development accuracy. It appears possible to teach the compact recurrent cell to encode the **final** Horn subgoal state across longer dependency chains without reproducing the exact Boolean wave sequence after each individual neural call.

A subsequent distinct preregistration should test *time-reparameterized closure consistency*: allow a monotone neural time index to align internal updates with logical closure waves, or supervise first-reach times with a monotone partial order rather than forcing step `t` to encode precisely `Φ^t(S)`. This proposal is descriptive and is not part of AF-12's frozen objective.

AF-01 through AF-11 and their scientific outcome histories remain unchanged. Leave AF-12 #63 draft and unmerged.
