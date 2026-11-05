# Lean data pipeline (lcgrt-v1)

Lean 4.19.0 (commit 6caaee842e94) is the authority on every training fact. Python generates theorem *statements*, drives the exploration, and solves the AND-OR graph that Lean reports. It never decides whether a tactic applies or what goals a tactic leaves.

```
fragment.generate ──statement──▶ lgym (Lean) ──goals, tactic outcomes, errors──▶ extract.explore
                                                                                    │ AND-OR solve
          dataset.plan (dedup, splits) ◀────────────────────────────────────────────┘
                     │
                     ▼
     shards *.jsonl.gz ──shortest proofs──▶ certify (lean file + #print axioms) ──▶ certificate per theorem
                     │
                     ▼
     packed/*.npz (training)          search.Controller (evaluation, live Lean states)
```

Build: `python -m lcgrt.dataset build --name lcgrt-v1 --workers 3`. 26,800 theorems took 962.6 s on three CPU workers and 869,832 Lean calls, with 0 extraction failures; every theorem was certified ([manifest](results/data/lcgrt-v1_manifest.json), [statistics](results/data/lcgrt-v1_stats.json)). Shards live outside the repository at `D:\lean_cgrt\data\lcgrt-v1` (27 MB); the manifest records each shard's sha256.

## Lean proof-state server

[`lean/Main.lean`](lean/Main.lean) builds with `lake build` into `lgym.exe`, a JSON-lines server over `Lean.Meta`:

| op | Lean does |
|---|---|
| `init {stmt}` | parse and elaborate the statement, `introNP` the binders, return the goal (structured target, local context) |
| `expand {sid}` | try `apply h` for every hypothesis and `constructor`; per candidate, success with the resulting goal list, or failure with Lean's message (first *k* rendered) |
| `run {sid, tactic}` | elaborate and run tactic text; any logged error is a failure (no error-to-sorry recovery) |
| `focus {sid, goal}` | move a goal to the front (the `pick_goal` permutation) |
| `kernel {sid}` | instantiate the proof term, `addDecl` it (kernel check), report `collectAxioms` |

Contexts are sent once per context hash. A depth-12 theorem with 85 hypotheses costs 7.4 ms per `expand`.

## What one record contains

[Example record (chain-cycle, depth 3)](results/data/example_record.json). Fields:

1. **Theorem statement and hypotheses:** `statement` (exact text Lean elaborated), `stmt_sha256`, `context` (atoms, facts, rules as Lean reported them; cross-checked against the generator, mismatch = `extraction_mismatch`).
2. **Proof states before and after tactics:** `path` lists, for each step of the shortest proof, Lean's goal list before the tactic and the tactic. Lean produced the lists by running the tactic *text* (`run`).
3. **Candidates and outcomes:** `table[goal][tactic]` holds the goals each Lean-accepted tactic leaves. `rejected[goal]` counts Lean-rejected candidates; `diagnostics[goal]` keeps a sample of Lean's error messages (e.g. `apply r3: tactic 'apply' failed, failed to unify P5 with P2`).
4. **Proof dependencies:** `size[goal]` is the minimum number of tactics proving the goal (least fixpoint over `table`), or `null` if unprovable; `proof.script` is the shortest proof.
5. **Failed branches:** every explored goal with `size = null` (6.4 per training theorem, 14.0 per depth-OOD theorem), the tactics leading there, and their messages.
6. **Verified, replayable proof:** `proof.script` replays as tactic text. `proof.repl_kernel` holds the in-process kernel check (axioms `[]`); `certificate` holds the result of compiling a standalone file.

Provenance per record: `tid`, `family`, `depth`, `seed` (regenerates the statement exactly; the evaluator checks the sha256), `canonical_hash`, `lean.version`, `lean.calls`, `lean.seconds`, `certificate.source_sha256`, `certificate.lean_version`, plus `gen` (generator metadata, never a model input).

Each record also gets one **agreement check**: a mid-path state is re-reached by tactic text and re-expanded, and its children must equal the table entries. 26,800/26,800 passed.

## Synthetic theorems

[`lcgrt/fragment.py`](lcgrt/fragment.py) generates propositional Horn theorems for backward chaining. `apply r` replaces a goal by the rule's premises, `apply h` closes it with a fact, and `constructor` splits a conjunctive goal. v1's problems had one producer per atom; here goals have competing producers:

| element | effect on search |
|---|---|
| skeleton `chain` / `tree` / `alt` | depth-*d* derivation; `tree` adds side subgoals; `alt` adds a valid but longer second derivation |
| decoy `dead` | a producer whose premise has no derivation |
| decoy `deep` | a producer whose premise starts a 2–4 step chain that ends in a dead atom |
| decoy `cycle` | a producer whose premise is derivable only through an ancestor goal |
| distractors | rules among unrelated atoms |
| names | atoms alpha-renamed, hypothesis order shuffled |

## Splits and holdouts

Dedup comes first. Every planned theorem is hashed by a WL canonical hash over the rule hypergraph (invariant to names, rule order and fact order). Held-out splits are planned first, and a later split drops any theorem whose hash is already taken: 9 training theorems were dropped.

| split | families | depths | dependencies | theorems | states | mean / max shortest proof |
|---|---|---|---|---:|---:|---|
| train | 7 train families | 2–6 | ≤ 2 premises | 24,000 | 163,124 | 6.8 / 27 |
| val | train families | 2–6 | ≤ 2 | 400 | 2,730 | 6.8 / 18 |
| test_id | train families | 2–6 | ≤ 2 | 600 | 4,082 | 6.8 / 24 |
| **test_family** | `alt-cycle`, `tree-deep` (never trained) | 2–6 | ≤ 2 | 600 | 4,488 | 7.5 / 21 |
| **test_depth** | train families | **7–12** | ≤ 2 | 600 | 8,464 | 14.1 / 38 |
| **test_dep** | train families | 2–6 | **3-premise rules** (never trained) | 600 | 8,106 | 13.5 / 38 |

Held-out families recombine components seen in training (the `alt` skeleton with `cycle` decoys, and the `tree` skeleton with `deep` decoys). No variant of a held-out theorem is in training: families and depths are disjoint by construction, 3-premise rules never occur in training, and exact duplicates are removed by hash. Training uses `train` only. `val` is monitored but never used for selection.

## Certification contract

[`lcgrt/certify.py`](lcgrt/certify.py) writes the file itself:
- `theorem t_<tid> <binders from the dataset fields> : <target> := by` followed by the script, then `#print axioms` for each declaration;
- every script line must match `^(apply [hr][0-9]{1,4}|constructor)$`, so `sorry`, `admit`, `native_decide`, `set_option`, `#eval`, macros and axioms cannot be written;
- a theorem passes only if (a) its declaration's own `#print axioms` line says it depends on no axioms, (b) Lean reports no error inside its line range, and (c) the compile did not time out.

An incomplete replay leaves unsolved goals, which is an error, hence a rejection. A proof attached to a different statement fails to typecheck. Tests: `tests/test_lcgrt.py::CertifierTests`.

The same certifier runs on every proof the model finds at evaluation time; a proof counts only if it passes.

## Memory bounds

- Extraction streams one theorem at a time per worker, and the REPL drops its states after each theorem.
- Shards hold 500 theorems.
- Training reads `packed/train.npz`: 121 MB of flat uint8/int16 arrays, with features computed by the same `encode.state_features` the search uses, and batches assembled per step.
- Evaluation streams shards.

## Mathlib (not done in this campaign)

The server takes module names as arguments (`lgym.exe Mathlib`), `run` executes any tactic text, and non-propositional types fall back to `{"raw": <pretty-printed>}`. So the extraction seam exists. Mathlib was not installed, for two reasons:

1. While other agents' jobs ran, this machine had 0.05–3 GB of free RAM (31 GB total). Importing Mathlib keeps several GB resident per process.
2. This campaign's model consumes the propositional graph encoding only.

A Mathlib extraction needs a pinned `mathlib4` tag matching Lean 4.19.0, a tactic vocabulary beyond `apply`/`constructor`, and premise retrieval. None of that is built, and no Mathlib number is reported anywhere in this campaign.
