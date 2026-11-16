# AF-24: public-graph symbolic challenge to AF-23

AF-23 Track B passed its **registered 6/8 seed** comparison against first-legal and blind randomized Horn actions in [original CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916). That result stands. AF-24 tests a **different question**: whether a simple deterministic nonlearned solver can do the same work directly on the fully public Horn incidence.

## Registered source

[PROTOCOL.json](PROTOCOL.json) was committed before AF-24 data scoring. All eight original AF-23 checkpoints stay untouched; this experiment needs **no training** and imports their archived model results by exact aggregate checksum.

Controls: a reverse goal-ancestor greedy heuristic, five independently randomized ancestor policies, bounded online BFS limited to **4, 16, or 64 host-state expansions per action**, a public forward-closure derivation, original first-legal, and an evaluation-only exact shortest-proof oracle. The 64-cap BFS and exact oracle are higher-cost diagnostics; low-budget controls are reported separately.

Every controller sees only the public physical Horn-rule graph, current proved atoms, and the marked goal. STOP requires either a goal, no enabled rules, or a completed public graph exploration establishing impossibility. Exact truth labels and shortest-proof oracle outputs are withheld from controller inference. The scorer separately uses independently verified physical BFS costs to set the same shortest+1 proof budget as AF-23.

### Registration gate

For one **identical named symbolic arm** to match the neural source at both depths in a seed, it must solve at least **76/80 depth-4 provable goals**, matching original Direct (80/80) within five percentage points, and at least **AF-23 seed-specific depth-5 VI-2 completions minus 4 goals**. The registered outcome is `SYMBOLIC_CONTROL_MATCH` when a single primary symbolic arm clears that gate in at least **six of eight** seeds. Such an outcome prevents a claim of neural improvement over these public-graph controls. The alternative `NO_SYMBOLIC_CONTROL_MATCH` only excludes the tested controls and their budgets.

The experiment reports full per-case proofs, correct STOP on 112 unprovable cases/depth, node expansions, rule examinations, and same-runtime symbolic latency. Historical neural inference was run on a different job and measured using different units, so wall-time equality or matched FLOPs cannot be asserted.

### Source-only preflight

```bash
python -m unittest discover -s experiments/activation_funnel_af24/tests -p 'test_*.py' -v
python -m compileall -q experiments/activation_funnel_af24
```

### Source-locked prospective CI

The [AF-24 symbolic workflow](../../.github/workflows/activation-funnel-af24.yml) has an explicit source-lock followed by eight seed jobs and an independent aggregate. Original AF-23 results are downloaded by fixed artifact ID and verified by ZIP and member SHA-256 before scoring the comparison.

The symbolic policy and parser are source-frozen at the request commit preceding the run marker; changing source requires an amended registration and new results. AF-23 and AF-22 scientific verdicts remain unchanged.

### Follow-up grammar generalization

A separate AF-24b proposal can expose frozen AF-23 models to **genuinely new** Horn rule families where goal-ancestor relevance alone fails, including misleading ancestor paths, shortcut swaps, and jointly needed facts. That data must have its own source-registered structural manifest before scoring. Even a symbolic match on the original fixed grammar only establishes that AF-23's current public-graph task admits a cheap constructive solver.
