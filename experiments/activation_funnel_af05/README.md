# AF-05: depth-stable causal interchange for inferred proof state

This is a **new, separately preregistered** study continuing AF-04's negative depth-3 result. Its parents and all earlier evidence remain immutable. Read [PROTOCOL.json](PROTOCOL.json) before interpreting scores. AF-04 is [PR #43](https://github.com/MoralityLabAI/LoopedTransformers/pull/43) with its frozen [negative result](../../docs/research/AF04_RESULTS.md).

## Question

A small native `TypedLoop` receives a public Horn DAG and initial-fact bits. It must derive the truth of hidden Q0/Q1 intermediate proof nodes, preserve those values through a recurrence boundary, and predict the truth of a downstream Boolean-Horn goal. A matched donor changes one branch of the initial facts while leaving the other branch unchanged. The target for replacing Qj's **actual carried code** is recomputed independently using `do(Qj := donor Qj)` and the recipient's untouched Q node, not inferred from the student.

AF-05 trains depth-2 and depth-3 Horn chains in balanced batches and alternates **4/6 upstream recurrent visits** at every optimization update. The status arm receives supervised intermediate Q labels and a negative-control patch; the causal arm additionally trains on addressed Q interchange, including the downstream goal. Both share a pre/post-state commuting loss and the same number of native recurrent calls. At evaluation, unseen dependency depths **4 and 5** require **8/10 upstream visits**, respectively. Held-out composition and unseen graph layouts are additional diagnostics.

This tests whether learned proof state supports **depth-stable causal abstraction**. The source `TypedLoop` has 5,024 native-cell parameters and remains trainable in every arm. With 15 atom slots, 14 rule slots, 16 working floats per token and 1/2 stored floats per token, there are 464 decoded working floats and 29/58 carried floats.

## Inputs and frozen splits

| Family | Depth | Permutation | Unique inputs | Donor pairs |
|---|---|---|---:|---:|
| train | 2,3 | 0,1,2 | 672 | 1,344 |
| dev | 2,3 | 3 | 224 | 448 |
| held_composition | 3 | 4 | 16 | 32 |
| held_layout | 2,3 | 5 | 224 | 448 |
| held_depth4 | 4 | 6 | 128 | 256 |
| held_depth5 | 5 | 7 | 128 | 256 |

All four Boolean leaf-fact bits are enumerated. The model observes only original facts, node-role bits, rule arities, stage markers and physical incidence; hidden derived Q and G labels are excluded. Canonical input hashes and model/target receipts are frozen in [corpus_manifest.json](corpus_manifest.json).

## Model and objective

For upstream call count \(k\), the original TypedLoop computes recurrent state \(H_k\) and code \(Z_k = E(H_k)\), then carries \(D(Z_k)\). The downstream goal is read after three further calls with only downstream incidence active.

The causal objective patches the code at the selected Q token with the code from a matching donor. It minimizes the original goal loss, Q-state supervision, addressed-patch goal and Q cross entropy from the independently computed Horn counterfactual, and a common commuting loss from stage1 Q readout to stage2 Q readout. The status comparator performs the same amount of recurrent work with an irrelevant-branch patch and original targets. The task-only arm uses a self patch. All arms use 240 AdamW steps, batch32, three seeds 17/29/43 and widths1/2; the primary comparison is width2.

All optimized code paths and scientific cutoffs are fixed before scored held-out training. We retain every seed/arm, train-dev record, task/Q logits, counterfactual patch logits, actual transported Q code, model weights, independent row-level replay and Lean-generated Boolean-Horn certificates. A green integrity workflow has no implication for the scientific verdict.

## Reproduce

From repository root with Python 3.12, CPU Torch2.5.1, numpy and Lean4.19.0:

```bash
python -m unittest discover -s experiments/activation_funnel_af05/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af05/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af05.experiment manifest --out /tmp/af05-manifest.json
cmp /tmp/af05-manifest.json experiments/activation_funnel_af05/corpus_manifest.json
python -m experiments.activation_funnel_af05.experiment lean --split held_depth5 --out /tmp/af05-held-depth5.lean
lean /tmp/af05-held-depth5.lean
python -m experiments.activation_funnel_af05.experiment train --seed 17 --width 2 --out /tmp/af05-train
python -m experiments.activation_funnel_af05.experiment evaluate --seed 17 --width 2 --training /tmp/af05-train --out /tmp/af05-eval
python -m experiments.activation_funnel_af05.verify --seed 17 --width 2 --train /tmp/af05-train --evaluation /tmp/af05-eval --out /tmp/af05-eval/independent_audit.json
```

The full matrix and strict preregistered aggregate execute on GitHub Actions. Graph certificates establish the correctness of the synthetic Boolean oracle, not the ability of a model to generate a novel Lean proof.
