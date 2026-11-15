# AF-07 final scored research results | 2026-11-15

**Registered width-2 causal-locality advantage: INCONCLUSIVE.** All three width-2 shared warm-ups missed the preregistered exact-joint-Q training/development gate. Every seed, failed comparison and post-hoc diagnostic is retained. The strongest apparent depth-locality improvement belongs to an ineligible seed.

**Scientific source:** [`6c85f02c`](https://github.com/MoralityLabAI/LoopedTransformers/commit/6c85f02c289201c9666fd99056515d68e62fb545). **Protocol SHA-256:** `ed3bb99a2de0915f7ff89b8c94af11eacc9162f7b6f6cf2315e496ccbf901da7`. **Parent:** AF-06 [PR #46](https://github.com/MoralityLabAI/LoopedTransformers/pull/46). **AF-07:** [draft PR #48](https://github.com/MoralityLabAI/LoopedTransformers/pull/48).

Original [AF-07 CI #43675750007](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43675750007) passed preflight, 6 independent training jobs, and 6 independently replayed evaluation jobs. The original final aggregate failed **before scoring** because its runner lacked `torch`; the separately pinned [aggregation repair #43676432842](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43676432842) passed on those exact original six evaluation artifacts. The repaired `results.json` SHA-256 is `c195aefa130f194eda1bc46afbbc59844628f992ea3eb62c6962b8d59b066f9b`. Independent local reconciliation produced the identical JSON contents.

## Experiment

AF-07 uses AF-05's unchanged Horn DAG and the original source-hash-checked native TypedLoop. It trains with 480 shared status-warm-up updates per seed/width, followed by equal 160-step `status`, `ramp`, and `locality` continuations. All three continuation arms share weights at the split, exact input/donor RNG streams, and recurrent-call counts. The `locality` arm alone adds recurrence-resolved KL and compressed-code MSE penalties for the semantically unaffected subgoal. This adds extra loss/backpropagation FLOPs, so the arms are **forward-call matched**, not full-compute matched.

All three arms supervise the derived Q truth at each downstream recurrence step. The additional penalty attempts to preserve the unaffected Q atom's actual carried code as the other Q is transplanted. Evaluation retains three-call and five-call stage traces at depths 4 and 5. The five-call extension is a tensor-only diagnostic, and Lean's 4,176 generated closed Boolean theorems certify the Horn oracle rather than student proof terms.

## Frozen complete seed matrix

Each development cell is exact original-goal / joint-Q accuracy. The main intervention columns report **untouched Q accuracy** after three downstream calls, ramp → locality.

| Seed | Width per token | Warm-up dev G/Q | Status dev G/Q | Ramp dev G/Q | Locality dev G/Q | Depth4 ramp → locality | Depth5 ramp → locality | Registered gate |
|---:|---:|---|---|---|---|---|---|---|
| 17 | 1 | .491/.277 | .509/.277 | .705/.491 | .795/.455 | .750 → .688 | .750 → .688 | Inconclusive |
| 29 | 1 | **1.000/1.000** | .786/.397 | **1.000/1.000** | **1.000/1.000** | .875 → .875 | .875 → .875 | Inconclusive |
| 43 | 1 | .491/.277 | .509/.205 | .795/.277 | .795/.205 | .500 → .500 | .500 → .500 | Inconclusive |
| 17 | 2 | .491/.277 | .509/.277 | .491/.277 | .491/.277 | .500 → .500 | .500 → .500 | Inconclusive |
| 29 | 2 | .848/.647 | **1.000/1.000** | **1.000/1.000** | .897/.781 | .750 → .563 | .750 → .625 | Inconclusive |
| 43 | 2 | .830/.679 | .951/.897 | .951/.897 | .951/.897 | .625 → **1.000** | .625 → **.875** | Inconclusive |

Eligibility required warm-up, status, ramp and locality models to all achieve at least 0.80 G and exact joint Q accuracy separately on both train and dev. **No width-2 seed qualifies.** Seed 29 width 1 also fails because its status continuation lost joint-Q accuracy. The preregistered scientific aggregate therefore stays **inconclusive**.

## Mechanistic observations

**Seed 43, width 2** has the strongest exploratory improvement: +.375 untouched-Q accuracy on depth4 and +.250 on depth5, along with informative-goal-flip accuracies improving from .7578 to 1.0000 on depth4 and from .7344 to .9375 on depth5. Protected-Q correctness remains **1.000** through all five downstream calls on depth4 and **.875** through all five on depth5. Its irrelevant-patch goal retention is .9375 on depth4 and **.875** on depth5, which also misses the preregistered .90 threshold in the latter family. Its warm-up exact joint-Q score (.679) prevents a primary positive claim.

**Seed 29, width 2** shows worse locality under the new objective: untouched-Q accuracy falls .750→.5625 on depth4 and .750→.625 on depth5. Informative-goal-flip accuracy also decreases .500→.4609 and .6875→.4063, respectively. This is a material negative control on any claim that adding locality penalties reliably helps.

**Seed 17, width 2** exhibits a particularly informative collapse: protected-code displacement after three downstream steps is below `1.5e-8` in the locality model on depth4, while untouched-Q accuracy remains **.500**. Near-zero code drift is therefore **insufficient evidence of correct semantic transport**. Constant or degenerate encodings can satisfy a code-consistency penalty. An explicit noncollapse condition and semantic competence gate would be required to separate a stable proof-state variable from a stable but uninformative code.

## Evidence and verification

- **6 shared warm-up checkpoints** and **18 continuation models** at one/two carried coordinates for each of 29 tokens.
- **8,928 canonical held-out clean predictions**, **17,856 original three-condition donor records**, and **27,072 recurrence-resolved traces**, preserving exact raw logits, corresponding source identities, Q code displacements and three-/five-step histories.
- All six independent evaluators recomputed Horn labels, guarded case rosters, reproduced source-code patches and replayed saved Torch states, including 108 sampled recurrence traces each. Each evaluation job passed.
- The repaired aggregate returned the full seed matrix with exact independent source hashes and saved-row reconciliation. All negative, ineligible and failed cases remain included.
- The original aggregation failure was strictly an environment dependency omission, repaired with the pinned source and original artifacts. Its failure stays documented.

## Interpretation and next mathematics

AF-07 establishes a runnable, differentiable recurrence-locality regularizer and an independently replayable method for distinguishing semantic subgoal accuracy from mere numerical code stability. It **does not establish a reliable depth-generalizing improvement**. The most useful next test would retain a **qualified pretraining checkpoint before adding stepwise-locality losses**, then explicitly prevent collapse through a discriminative Q-status and counterfactual-contrastive condition. This needs a new frozen registration, with the unchanged AF-06/07 negative results retained.

See the [mathematical formulation](AF07_MATHEMATICS.md) for the commuting transport conditions and the code-vs-semantic error separation.
