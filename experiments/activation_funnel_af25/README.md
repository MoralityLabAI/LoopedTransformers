# AF-25: four-initial-fact transfer against prospectively hard graph controls

AF-25 follows [AF-24B](../../docs/research/AF24B_RESULTS.md). The previous frozen-model test failed, but the stronger symbolic ancestor and BFS16 policies were perfect, saturating the comparative gate. AF-25 isolates a different experimental question using four known facts on **every** source test case, matching the original AF-23 positive training roots, while holding all original AF-23 TypedLoop weights frozen.

## Protocol and independent structural gate

[PROTOCOL.json](PROTOCOL.json) was committed first at `adc50ba`, with pre-generator amendments `6d6c628` and `7f7ddc6` before any AF-25 data generator or evaluation script. Both amendments explain distinct root branching and the preregistered bad-goal physical tie-break. No neural model results were used when designing the new cases.

Source-only [CI #43828215369](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43828215369) passed; its full [structural manifest artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43828215369) was generated before the original AF-23 checkpoints were accessed. The frozen manifest SHA-256 is `636373237bb44371281e82c827b57f7703b1808e3d8836358dd821ed3b783507`.

Each depth has **192 cases: 80 provable and 112 unprovable**, with exact shortest proof depth 5 or 6. All cases begin with **exactly four known atoms**, including unprovable cases. The negative cases replace a critical root fact with one decoy rather than reducing the root's input cardinality. Public physical hashes are distinct across splits, from AF-23's train/dev/test, and from AF-24B's test.

| Original symbolic policy | Depth 5 solved /80 | Depth 6 solved /80 |
|---|---:|---:|
| Goal-ancestor greedy | 32 | 32 |
| Additive closure | 32 | 52 |
| BFS, 4 expansions per decision | 32 | 32 |
| BFS, 16 expansions per decision | 32 | 32 |
| **BFS, 64 expansions per decision (higher-cost diagnostic)** | **80** | **80** |
| First-legal fixed | 3 | 4 |
| Shortest-proof oracle | 80 | 80 |

The hard family contributes 48 positives at each depth, and the balanced goal-shortcut family contributes 32. All four preregistered **inexpensive** controllers fail at least 20/80 provable cases and pass at least 20/80. BFS64 solves the original 80-case roster and is kept as a **higher-expansion diagnostic** that limits any broad neural-search advantage interpretation.

The two goal alternatives are short shared-prefix productive conjunctive proof and longer decoy conjunctive proof. Goal-ancestor reverse-hop distances favor the decoys even though their joint prerequisites are more expensive. Additive closure may double-count the shared prefix. The hard goal's physical tie-break was set prospectively.

## Neural and symbolic arms

AF-25 loads only the eight original **6,036-parameter** AF-23 Track B final checkpoints from [CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916). ZIP artifact SHA-256, source/training receipts, Torch/NumPy weight parity and independent forward replay are required. Every policy is evaluated on the same held-out physical incidence and shortest+1 action budget.

The scored neural methods are original Direct logits, VI-1 learned successor values and VI-2 two-step min-plus Bellman backup. The symbolic modes are first-legal, ancestor greedy, additive closure, BFS4, BFS16, BFS64 and exact shortest-proof upper control. The four low-budget methods are the only ones in the preregistered comparative gate; BFS64 is always shown for context.

A **single seed** passes only if VI-2 solves at least **eight more than the best of the four cheap controls** at both depths, scores at least 40/80 positive completions at each, selects correct initial STOP on at least 101/112 unprovable goals per depth, and produces no false accepted Horn proof. At least **six of eight** original fixed seeds are needed for a registered neural-control advantage. These thresholds were frozen before scoring.

Source-only structural selection targets the listed controls. Therefore any subsequent result is conditional on this deliberately challenging synthetic family. Success cannot imply general-purpose proof search, and BFS64's perfect completion is important evidence for that limitation. Actual neural forward evaluations and symbolic rule examinations/state expansions are recorded separately.

## Reproduce

For the complete source-only structural certificate:

```bash
python -m unittest discover -s experiments/activation_funnel_af25/tests -v
python -m experiments.activation_funnel_af25.gym > af25_structural_manifest.json
```

With pinned Python 3.12, PyTorch 2.5.1 CPU and original AF-23 immutable GitHub seed ZIPs (e.g. `original/af23-branch-seed-17.zip`):

```bash
python -m experiments.activation_funnel_af25.evaluate --seed 17 \
  --original-zip-dir original --out /tmp/af25-seed-17
python -m experiments.activation_funnel_af25.audit --seed 17 \
  --original-zip-dir original --output-dir /tmp/af25-seed-17
```

Source-locked CI runs smoke, every seed's fresh proof episodes, independent physical-incidence/Bellman validation and aggregate. **Do not merge or reinterpret AF-23/AF-24/AF-24B scientific verdicts** on the basis of an AF-25 result.
