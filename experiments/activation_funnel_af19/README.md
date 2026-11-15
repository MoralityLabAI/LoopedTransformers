# AF-19: graph-ancestry STOP readout

AF-19 compares two policies with **6,036 identical-shaped trainable parameters** and the same inherited TypedLoop(16) graph-attention cell. Both compute the entire backward goal-ancestry mask from public Horn rule incidence; one attends only rules immediately concluding the goal, and one attends every backward-ancestor rule. The same attention heads aggregate rule tokens into the STOP/reachability/distance state. Every Horn action is host-checked, with illegal rules masked before selection.

The prospective training/dev split is unchanged from AF-17 (928 / 352 cases). Fresh AF-19 heldout layouts 62–66 contain 416 cases with depths 3, 4, and 5, and are checked for disjoint public feature hashes from AF-16/17/18. The rule-language templates remain unchanged.

The primary scientific claim requires >=0.05 host-proof success gain for **both** ancestor policies versus same-loop local controls on **both** new depths, plus >=0.70 absolute success, no added false proofs, no substantial increase in premature STOP and an informative mechanistic contrast. All four seeds must qualify for a positive aggregate. Scientific failure never causes engineering CI to fail.

Mathematical boundary: the original goal-only readout has a causal radius of 2L incidence edges. A rule-state ancestry pool provides direct access to more distant rule representations at the end of L TypedLoop calls. The new readout can be sensitive even when the original goal atom remains insensitive. This proves an **expressivity/access difference**, with training required to produce useful proof policies. The exact irrelevant perturbations in the registered test touch **padded rule slots or disconnected atoms**. Connected but logically irrelevant subgraphs may still have neural feedback pathways; we make no exact invariance assertion for them.

## Reproduce

Run from the repository root. PyTorch 2.10 CPU on local Python 3.13 was used for engineering tests; CI requests Python 3.12, Torch 2.5.1 CPU and checksum-pinned Lean 4.19.0.

```bash
python -m unittest discover -s experiments/activation_funnel_af19/tests -p 'test_*.py' -v
python -O -m unittest discover -s experiments/activation_funnel_af19/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af19.experiment manifest --out /tmp/af19_manifest.json
python -m experiments.activation_funnel_af19.experiment train --seed 17 --out artifacts/af19/s17/training
python -m experiments.activation_funnel_af19.experiment evaluate --seed 17 --training artifacts/af19/s17/training --out artifacts/af19/s17/evaluation
python -m experiments.activation_funnel_af19.verify --seed 17 --training artifacts/af19/s17/training --evaluation artifacts/af19/s17/evaluation --out artifacts/af19/s17/independent_audit.json
python -m experiments.activation_funnel_af19.aggregate --root artifacts/af19 --out artifacts/af19/results.json
```

Repeat for seeds 29, 43 and 61 before aggregation. Each nonempty output directory must be fresh; manifests and checkpoints are SHA-256 bound. Host proof success and causal readout probes are scored in independent ledgers.

## Study lineage

AF-18 [PR #93](https://github.com/MoralityLabAI/LoopedTransformers/pull/93) reports a failed recurrent-depth advantage and limited goal-only sensitivity despite causal cone access. AF-19 tests an attention readout across the backward dependency graph using matched controls, preserving all prior frozen protocols and outcomes.
