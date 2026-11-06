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

## Mathlib: real proofs, traced by Lean (added 2026-11-06, after the RAM upgrade)

**Install.** Mathlib tag `v4.19.0` (commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`) matches the pinned Lean 4.19.0. It lives at `D:\lean_cgrt\mathlib4-v4.19.0`, 4.9 GB with oleans.

GitHub git and codeload transfers were throttled to about 20 KB/s and cut mid-stream on this machine, so the install route is different from a plain clone:
- The file listing comes from a blobless clone (`git ls-tree`, [`results/data/mathlib-v4.19.0_lstree.txt`](results/data/mathlib-v4.19.0_lstree.txt)).
- [`lcgrt/fetch_mathlib.py`](lcgrt/fetch_mathlib.py) fetches every file from raw.githubusercontent.com and **verifies it against its git blob SHA-1**: 6,816 files, 76 MB, one pass.
- The pinned dependency revisions are fetched shallow, then `lake exe cache get` downloads 3,838 archives into 6,305 Mathlib oleans.

**Tracing.** Re-proving a Mathlib theorem in an environment that already contains it is meaningless (`exact?` finds the theorem). So the pipeline traces the **real** proofs instead.

[`lean/Trace.lean`](lean/Trace.lean), built as `lgym_trace.exe`, re-elaborates a source file with Lean's frontend and walks its info trees. For every source tactic it writes:
- the enclosing declaration, source position and tactic text;
- the goals before and after, as Lean pretty-prints them.

On every *k*-th step it also runs a candidate vocabulary on the same goal, in the environment as it stood at that point of the file, where the theorem being proved is not yet in scope. The vocabulary is `rfl`, `simp`, `simp_all`, `assumption`, `trivial`, `decide`, `omega`, `tauto`, `constructor`, `intro`, `exact?` and `aesop`. Each candidate gets a 5M-heartbeat budget, and its outcome or Lean error is recorded.

**Per declaration**, the checker status is `certified` when there is no error inside the declaration's span and its axioms are within {`propext`, `Classical.choice`, `Quot.sound`}. Lean's `addDecl` kernel-checks every declaration it elaborates. Anything with `sorryAx` or another axiom is `rejected`.

**Process and memory.** [`lcgrt/mathlib.py`](lcgrt/mathlib.py) runs one tracer process per file: the file's imports are loaded from oleans into RAM and freed when the process exits, at 0.7–1.4 GB peak per process. The steps are then deduplicated on a hash of the goal state, with files in sorted order. Held-out splits:
- `test_family`: all of `Mathlib/Algebra`;
- `test_file`: whole held-out files, whose lemmas never appear in training;
- `test_long`: declarations in the top 10% by proof length.

**Pilot** (15 files from Logic, Order, Data and Algebra; [manifest](results/data/mathlib-pilot_manifest.json)):

| metric | value |
|---|---|
| declarations | 799, all certified, 0 errors |
| tactic states (after dedup) | 3,049 traced, 603 duplicates dropped |
| splits | train 1,065, test_long 592, test_file 328, test_family 461 |
| candidate trials | 4,644 with 1,246 successes; `simp`, `exact?`, `aesop` often apply, `omega`, `decide`, `tauto` rarely in these files |
| time and memory | 111 s wall on 2 workers, ≤ 1.11 GB per process |

**Full Mathlib** (6,305 files, 3 workers, candidates on every 16th step) started 2026-11-06 19:30 as `mathlib-full`; log `D:\lean_cgrt\logs\mathlib_full.log`.
- The first 1,184 files went in alphabetical order (mostly `Algebra`). Then the trace restarted in **hashed order** (sha1 of the path), so the files finished at any moment are a uniform sample of the rest of Mathlib. Per-file summaries make the restart lossless.
- In those first 1,184 files: 84,451 rows, of which 16,824 are bracket or separator tokens (see below). The 12 automation candidates closed 2,246 of 4,855 sampled goals (46%), mostly with `aesop`, `exact?` and `simp_all`.

**Assembly** (`python -m lcgrt.mathlib assemble --name mathlib-full --out <snapshot> --full`) builds the splits from whatever per-file summaries exist, so a training snapshot can be cut while the trace runs. Compared with the pilot it:
- drops token rows: the `]` that closes a `rw [...]`, `;` and `<;>` carry their own TacticInfo but are not tactics a policy chooses;
- marks `terminal` rows, where the source tactic closed the main goal;
- holds out `NumberTheory` as the family for full Mathlib. Algebra is about a fifth of the library, too much to withhold from training. This was chosen before any model saw full-Mathlib data.

**Certified select mode.** For evaluation, `lgym_cert.exe` (the same server as a third binary, since a bulk trace holds `lgym_trace.exe`) re-elaborates a file and, at chosen source positions, runs given tactic texts on the main goal. For each tactic that closes the goal, it:
1. abstracts the hypotheses the statement and proof depend on (transitively);
2. adds the closed proof as a standalone theorem, which the kernel typechecks;
3. reports its axioms.

A proof containing `sorry`, or with metavariables left, is never certified. Tests: `tests/test_lcgrt.py::SelectCertTests` (`exact h.2`, `simp_all` and a universe-polymorphic `rfl` certified; `sorry` and a wrong `exact` rejected). On the pilot smoke run, the source tactic itself completed 12 of 12 sampled states this way.

The tactic model trained on these records is described in [HRM_CGRT_ARCHITECTURE.md](HRM_CGRT_ARCHITECTURE.md#mathlib-tactic-model-c3), and its evaluation in [LOCAL_SCALING_RESULTS.md](LOCAL_SCALING_RESULTS.md#c3-mathlib-registered-before-results).
