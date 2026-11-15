# AF-11: Boolean-influence training through causal-cut recurrence

**Registered scientific verdict: FAIL.** **Engineering verdict: PASS.** Six of six GitHub Actions jobs passed, including the Lean 4.19.0 Boolean-influence and AF-10 causal-cut proofs, two source-bound training jobs, both full held-out audits, and independent cross-seed aggregation. Every scored outcome and ineligible source is preserved.

- **Frozen scientific source:** [`dab61799`](https://github.com/MoralityLabAI/LoopedTransformers/commit/dab61799dadc2e820dc424c714b70b1e7e05e771).
- **Protocol:** [PROTOCOL.json](../../experiments/activation_funnel_af11/PROTOCOL.json), SHA-256 `12174aac2d9ff47fdfba93042868678f5054e8c8b9b1f260e1cf875b171aea20`.
- **Fresh input-feature receipt:** [fresh_manifest.json](../../experiments/activation_funnel_af11/fresh_manifest.json), SHA-256 `07a4561dc64e528057a6966c2da7acd0979c8547e361e2639bcd728837e5302b`.
- **Primary CI:** [run 43690038920](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43690038920).
- **Independent aggregate:** `results.json` SHA-256 `285b579f01cb712129067b3e72c2f1b6e5f5c00d00cf43416ec09800ddb30abc`.
- **Draft PR:** [AF-11 #62](https://github.com/MoralityLabAI/LoopedTransformers/pull/62), stacked on [AF-10 #59](https://github.com/MoralityLabAI/LoopedTransformers/pull/59).

## The prospective intervention experiment

AF-10's structural cut restores both actual carried Q codes after every downstream recurrent update. AF-11 trains the downstream **goal's response** to a change in one selected derived subgoal. For original Boolean Q values and a donor with a changed selected subgoal, the exact signed finite difference is

\[
G(q_j\leftarrow q'_j)-G(q)=(q'_j-q_j) I_j(G),\quad
I_j=\begin{cases}q_{1-j} & \mathrm{AND},\\
1-q_{1-j} & \mathrm{OR}.\end{cases}
\]

Lean proves the four finite Boolean derivative identities, while the separate AF-10 theorem proves the structural code-preservation condition. The neural model receives initial facts and public Horn graph incidence. Training targets and audits are recomputed from the same original finite Horn least-fixed-point oracle.

Both seeds 17 and 29 start from the exact SHA-256-verified and previously qualified **AF-06 shared 480-update, 58-coordinate** checkpoint. Each receives a shared 160-update cut adaptation, followed by four 160-update continuation models: `status_cut`, `counterfactual_cut`, `influence_cut` and `influence_ordinary`. Within a seed, all continuation arms start from identical adapted weights and receive identical cases and donors under the same number of native cell calls. The influence objective adds arithmetic and backward FLOPs. The strong registered comparator is `counterfactual_cut`, which already receives the same exact addressed counterfactual goal and Q supervision.

The original AF-05 train/dev set has 672/224 cases at depths 2/3. AF-11's 656 **new physical graph-layout inputs** were fixed before any full optimization: 128 at depth3 layout12, 16 OR/OR→AND cases at layout13, and 256 each at depth4 (layouts8/9) and depth5 (layouts10/11). Logical AND/OR rule templates remain familiar. The claim concerns new layouts and extrapolated chain depth, rather than entirely unseen inference rules.

## Official seed-level eligibility

| Seed | Original AF-06 warmup | Shared cut-adaptation dev goal / exact Q | Each continuation dev goal / exact Q | Registered comparison |
|---:|---|---|---|---|
| 17 | 1.000 / 1.000 | **1.000 / 1.000** | 1.000 / 1.000 | **Eligible** |
| 29 | 1.000 / 1.000 | **0.785714 / 1.000** | 1.000 / 1.000 | **Ineligible** |

The frozen registration requires the shared adapted checkpoint and both compared continuations to reach at least 0.90 on both original goal and exact joint-Q accuracy separately on train and dev. Seed 29's shared adapted checkpoint misses the **goal** threshold, despite all continuations later qualifying. It remains *inconclusive* for scientific advantage. Differences in an exploratory local Torch 2.10 run cannot replace the registered Torch 2.5.1 CPU CI result.

## Held-out goal-flip comparison

All four arms are independently scored on two exact changed-status donor interventions per case. The primary accuracy is the fraction of **informative** selected-Q changes for which the final goal prediction matches the independently recomputed counterfactual Horn truth.

| Seed | New held-out depth | Counterfactual-cut baseline | Influence-cut model | Difference | Primary seed verdict |
|---:|---:|---:|---:|---:|---|
| 17 | 4 | 0.5859375 | 0.5781250 | −0.0078125 | Fail |
| 17 | 5 | 0.2656250 | 0.3593750 | +0.0937500 | Fail |
| 29 | 4 | 0.5000000 | 0.5000000 | 0 | Inconclusive |
| 29 | 5 | 0.5000000 | 0.5000000 | 0 | Inconclusive |

Seed 17 fails the required >=0.05 influence gain on **both** depths and the >=0.80 absolute informative goal-flip requirement. At depth4 its influence model scores 0.828125 on noninformative-patch goal retention (required >=0.90), clean goal accuracy decreases from 0.8515625 to 0.734375 (maximum allowed decline 0.03), and false-positive goal attempts rise from 12 to 64. At depth5 the influence model attains 0.578125 noninformative retention and only 0.546875 clean goal accuracy, with 128 false-positive patched goals. Seed 29's descriptive heldout performance also fails absolute thresholds.

Both seeds' principal cut-trained arms achieve **1.000** informative-flip accuracy on the new depth3 layout, showing the heldout failure appears with longer dependencies rather than merely changed physical layout.

## Where the failure occurs

The independent readout tracks exact **joint upstream Q correctness** before the downstream cut, alongside counterfactual goal correctness and conditional performance.

| Seed | Depth4 upstream exact Q | Depth5 upstream exact Q |
|---:|---:|---:|
| 17, influence cut | 0.765625 | **0.140625** |
| 29, influence cut | 0.062500 | 0.250000 |

Thus a structural proof-state transport guarantee can faithfully carry an incorrect upstream inference. The cut protects the actual codes, while the learner must still infer the right Horn closure for unseen longer chains. Seed17's good training/development accuracy and weak depth5 upstream Q jointly locate the main extrapolation problem. The goal head can sometimes predict the correct goal even with incorrect Q readouts, so these are empirical decompositions rather than a strict logical upper bound on goal accuracy.

This result weakens the hypothesis that adding an explicit Boolean-influence loss to an already counterfactually supervised causal-cut model solves deeper Horn proof generalization. The failed eligible seed determines the aggregate **FAIL** under the frozen gate.

## Independent engineering and formal evidence

The six successful CI artifacts contain **eight trained continuation models plus two shared cut-adaptation checkpoints**, complete named NumPy and Torch parameter exports, train/development receipts, and **10,496 prospective heldout counterfactual rows**. Each of two independent evaluation jobs verifies 5,248 saved donor cases and **64 sampled original model forwards** against the stored raw logits and the source-bound weights. Every row is checked against both the original finite Horn closure and separately reconstructed Boolean AND/OR truth and donor influence. Training/evaluation source and corpus hashes, actual carried-code cut invariance, and model file digests are guarded. All checks passed.

Lean 4.19.0 validated AF-11's four Boolean derivative theorems and AF-10's source-cut theorem. Those proofs apply to the finite algebra and structural operator. They are **not proofs that the trained model performs independent Lean theorem discovery**.

A pre-registered and independent next question is whether a deeper upstream recurrence curriculum, with an explicit proof-state convergence/saturation check, can maintain the Q abstraction before downstream influence training. AF-01 through AF-10 remain unchanged. AF-11 stays draft and unmerged.
