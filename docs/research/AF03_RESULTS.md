# AF-03 final scored results | 2026-11-14

**Scientific verdict:** registered ten-coordinate causal-training advantage **FAILED**.  
**Engineering verdict:** full six-job training/evaluation matrix, five tests in both Python modes, and cross-seed evidence reconciliation **PASSED**.  
**Frozen experimental source:** [`459552ae`](https://github.com/MoralityLabAI/LoopedTransformers/commit/459552ae942edf42ecffb599373cd1402881e5d8).  
**AF-03 preregistration:** [PROTOCOL.json](../../experiments/activation_funnel_af03/PROTOCOL.json), SHA-256 `cb6bf085509befa550c0dc49d788b6829a493be9f4c5b962eccc76865975d746`.  
**Original AF-02 model and data:** verified 2026-11-14 CI archive, source pin `94bb2be`, [AF-02 evidence release](https://github.com/MoralityLabAI/LoopedTransformers/releases/tag/af02-ci-evidence-94bb2be-20261114).  
**Primary run:** [AF-03 GitHub Actions #43574697388](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43574697388).

## Study

All runs fine-tune the same 5,590-parameter native `TypedLoop` cell from each corresponding frozen AF-02 teacher seed. New codebooks have one or two float32 coordinates per atom/rule token (five or ten total), while all decoded working states have 80 coordinates. The task, status, and causal arms start from identical model weights and train-only PCA codec initialization, using identical canonical input minibatches and donor RNG draws for 240 steps.

The causal arm backpropagates through addressed atom/rule code patching at the first preview boundary. A composed-history oracle supplies labels, while an other-branch patch serves as a negative control. The experiment exposes original-host task and branch-audit accuracy separately from counterfactual patch fidelity.

The causal arm uses additional actual forwards and gradients per optimization step. Equal training steps provide a paired objective comparison, not a compute-matched claim.

## All seed-width outcomes

| Seed | Carried coordinates | Task baseline dev audit-exact | Status baseline dev audit-exact | Causal dev audit-exact | Status addressed-patch first accuracy | Causal addressed-patch first accuracy | Causal held-out depth task/audit | Causal 22-visit synthetic task/audit | Primary registered seed gate |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 17 | 5 | .444 | .778 | .556 | .648 | .667 | .722 / .556 | .722 / .556 | Inconclusive |
| 29 | 5 | .222 | .444 | .444 | .667 | .667 | 1.000 / .444 | 1.000 / .444 | Inconclusive |
| 43 | 5 | .444 | .444 | **1.000** | .667 | **1.000** | **1.000 / 1.000** | **1.000 / 1.000** | Inconclusive |
| 17 | 10 | .778 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000 / 1.000** | **1.000 / 1.000** | **Fail** |
| 29 | 10 | .444 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000 / 1.000** | **1.000 / 1.000** | **Fail** |
| 43 | 10 | .556 | .222 | .444 | **1.000** | **1.000** | **1.000 / 1.000** | **1.000 / 1.000** | Inconclusive |

*The registered gate uses exact per-branch, per-order train/development eligibility, and compares the status and causal arms. The five-coordinate results were a registered stress test. Every scored row and every ineligible seed is retained.*

### Why the registered advantage failed

At ten coordinates, seeds 17 and 29 qualify in both comparison arms, and the status-supervised baseline already achieves perfect addressed-patch correctness on depth_context and delay4. The causal objective also achieves perfect patch fidelity and clean status accuracy, with 1.0 selectivity and no extra tainted attempts. Its improvement over status supervision is **0.00**, below the preregistered +0.05 threshold in each required family. This gives two unambiguous eligible failures, making the cross-seed primary aggregate **fail**.

Seed 43 at ten coordinates has perfect held-out status and patch scores in both the status and causal arms, but the dev-order qualifications fail. It is **inconclusive** under the registered comparison, not evidence for an aggregate positive.

### The five-coordinate observation

Native seed 43 learns perfect original task/branch-status readouts and addressed patch outcomes through 22 recurrent visits under causal supervision, starting from the same AF-02 source state as the controls. The status-only baseline has .667 first-branch patch accuracy, while the causal arm attains 1.000, a **+0.333** absolute gain on the changed-prefix pairs. The causal model's untouched-branch retention and addressed-versus-other selectivity are also 1.000.

The status-only baseline failed the registered dev eligibility test; consequently, this **single-seed intervention result remains exploratory**. In seed 17, the five-coordinate causal objective fails to maintain clean accuracy and the patch gain is small; seed 29 also remains poor on audit states. This variability is an important target for the next experiment.

## Mechanics and evidence integrity

- **18 trained models**: 3 seeds x 2 code sizes x 3 objectives. The causal first-step gradient reached the original native cell in every seed-width combination, measured positive from 0.962 to 61.532 in L2 norm.
- **4,536 clean scored rows** across five evaluation families and all arms. Each trained model has 108 depth-context inputs plus 36 inputs for each of two registered delay families and two tensor-only extensions.
- **9,072 changed-status counterfactual rows** comprising 27,216 addressed/other/whole patch-condition predictions, including actual carried codes and logits. Each family reuses the independent AF-02 compositional oracle.
- **Whole-code patch/direct-hybrid equality** passed with maximum logit error zero in the observed matrix. All six Torch checkpoints and named NumPy exports, source manifests, and scored row files passed independent SHA-256 checks after extraction from GitHub artifacts.
- **CI** passed preflight, all six seed-width training jobs, and the final receipt aggregator. The aggregation code independently reconciles source identity, frozen train receipts and complete model/output hash rosters.

The extra delay5 and delay8 tensors carry 16 and 22 native recurrent visits, respectively. These variants extend AF-02's delay4 histories by legitimate W vectors while retaining the original branch targets; **they are tensor-only extrapolation diagnostics and receive no new original Lean host certificate**.

## Interpretation

AF-03 provides a functioning **mechanistically interpretable training signal** for the original native TypedLoop: a differentiable intervention objective grounded in exact donor/recipient composed histories, with controls for address specificity and untouched branch retention. It shows a potentially useful gain at the severe five-coordinate bottleneck in one qualified causal model. The primary causal-vs-status advantage **was not established** at ten coordinates, where the status baseline already saturated in two eligible seeds.

This is a public-addressed receipt-memory gym with host-generated proof evidence. The results concern recurrent causal memory and are not demonstrations that a small model has independently discovered latent Lean proof search.

### Next registered question

To assess whether causal interchange adds generalizable reasoning ability, train on *derived* intermediate Horn/Lean proof-state variables rather than publicly supplied branch status. Include held-out dependency-graph compositions, a compute-matched status baseline, and harder settings where exact status supervision itself has unsaturated causal-patch accuracy. Preserve AF-03 unchanged and register this separately before training.

## Artifact access

The exact primary CI ZIPs are named `af03-s{seed}-k{width}-43574697388`, alongside `af03-aggregate-43574697388`; they remain downloadable through the [primary CI run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43574697388) until scheduled expiry on **2026-12-14**. See the AF-03 pinned release link in the PR once archival publication completes.
