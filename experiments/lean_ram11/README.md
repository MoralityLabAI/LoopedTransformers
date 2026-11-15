# RAM-11 | Exact-prefix proof-path audit

This is a **retrospective integrity correction** to the RAM-09/09B training graph and RAM-10W historical witness counts. [The protocol](PREFIX_FIBER_AUDIT_PROTOCOL.md) was written after the issue was discovered. The frozen RAM-08 data, all prior research results, and the pending RAM-08 and RAM-10 held-out tests are unchanged.

## Finding

The original RAM-09 graph joined a verified action's successor to an *independently observed goal-equivalent state*, even when the compiled tactic prefix that produced that state was different. This is a mathematical quotient of the recorded Lean prefix tree, not necessarily a recorded executable concatenation of the actions. RAM-11 requires the child prefix `p + [action]` to be present as a genuine original collector prefix with the same before/after printed goal list and visible definitions.

| Observation | Goal-quotient graph | Exact-prefix graph | Quotient-only |
|---|---:|---:|---:|
| Original TRAIN, 640 states | 213 | 178 | 35 |
| Original DEV, 74 states | 35 | 25 | 10 |
| **Total, 714 states** | **248** | **203** | **45** |

The audit found **931 TRAIN** and **138 DEV** nonterminal observations that could be joined to an existing state through a matching goal hash, while their exact concatenated script-prefix was absent. These 1,069 observations were classified as unexecuted quotient splices. They are **not** 1,069 failed Lean proofs. Their actual full-source validity is unknown.

The original 20,930 Lean candidate effects were independently verified, with zero recorded disagreements. This audit examines *composition* of recorded effects. The quotient-only 45 states previously counted as witnessed paths have an abstract graph route. Their exact-prefix proof sequence was not replayed in the source dataset.

## Accepted tactic progress versus supported path

| Original TRAIN tactic | Compiler-accepted progress | Quotient witness actions | Exact-prefix witness actions |
|---|---:|---:|---:|
| `exact_typed` | 5 | 5 | 4 |
| `exact_local` | 3 | 3 | 2 |
| `rw_local` | 218 | 29 | 11 |
| `rcases_local` | 11 | 0 | 0 |
| `induction_list` | 290 | 9 | 9 |

All censored actions remain eligible for future genuine Lean continuation. The strict graph is conservative because unvisited prefixes could complete; missing data never turns an accepted action into a failed one.

## Mathematical trust boundary

Let `P` be the finite set of original recorded tactic prefixes, and let `g(p)` be the canonical printed Lean goals at `p`. The strict prefix graph includes `p -> p·a` only when `p·a in P`, the action was accepted, and its printed full state matches. The quotient graph adds edges whenever the action's after-goals match `g(q)` at *some* other recorded prefix `q`.

Every strict edge induces a quotient edge. Induction on path length gives

\[
R_{\mathrm{prefix}}\subseteq R_{\mathrm{quotient}}.
\]

Neither graph gives a verified full original theorem by itself. A complete candidate proof must still compile and replay against the unmodified theorem statement, satisfy the permitted axiom inventory, and obey the registered search deadline. We do not assume quotient splices are valid compiler actions.

## Posthoc model diagnostic: corrected labels

The old RAM-09 15-root internal holdback has already been inspected, and the old RAM-08 DEV selected checkpoints. We reused both solely as diagnostic data. We kept all original Looped32/64 and Flat64 trunks frozen, repeated RAM-09B's 200-step family-balanced bounded (`|delta score|<=0.35`) residual-head training, and replaced positive route labels with strict exact-prefix routes.

| Model | Original frozen on 28 prefix-reachable holdback states | Corrected residual |
|---|---:|---:|
| Looped32 | 20/28 | 20/28 |
| Looped64 | 14/28 | 14/28 |
| Flat64 | 18/28 | 17/28 |

A cleaner supervision set did not improve the known-reachable first-action score in this exploratory replay. No new test theorem was scored.

## Reproduce

Download the original SHA256-pinned [RAM-08 training artifact](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43689874169), preserving `ram08-frozen-audited-train-43689874169.zip`. From the repository root, with the same original RAM-09 and RAM-09B Python modules available:

```bash
python experiments/lean_ram11/prefix_audit.py ram08-frozen-audited-train-43689874169.zip --out ram11_prefix_audit.json --with-models
RAM11_ORIGINAL_ARCHIVE=ram08-frozen-audited-train-43689874169.zip python -m unittest discover -s experiments/lean_ram11 -p 'test_*.py' -v
python experiments/lean_ram11/strict_prefix_guard.py ram08-frozen-audited-train-43689874169.zip --out ram11_first_prefix_guard
```

Original archive SHA256: `a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`. Original transitions SHA256: `f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`. Six local regression tests including the complete pinned historical corpus pass. All outputs refuse to overwrite an existing first run.

[Machine-readable graph audit](receipts/prefix_fiber_audit_20261115.json) · [Corrected-label frozen-head report](receipts/strict_prefix_guard_20261115.json).

**Prospective status:** RAM-08 PEF source-family proof score 0/120 cases; RAM-10 new Lean compiler observations 0. GitHub-hosted runner assignment has failed before execution. This audit supplies no new complete original Lean theorem certificates.

## RAM-11F: source-only prefix-replay frontier (unexecuted)

To turn some of the ambiguous goal-quotient paths into actual compiler evidence, [`frontier_worklist.py`](frontier_worklist.py) examines only the original RAM-08 **TRAIN** source observations. It proposes concatenations in which an accepted tactic has the identical full printed goals and visible definitions as a separately recorded strict closing-suffix state, while its exact concrete successor prefix was never collected.

The deterministic source-only scan identified **274 full-state-matching alias candidates** among original TRAIN observations. It selected **76 complete candidate scripts from 22 original TRAIN theorem roots**, capped at four scripts per root and eight tactics per proposed proof. This selection includes 12 `rw_local` tactics, 16 `constructor` tactics, and one `exact_typed` tactic. All have a provenance link back to original compiler-observed constituent actions. Their *new concatenations have not been checked*, and their current proven full-theorem count is **zero**.

The worklist is archived in [`candidate_complete_source_scripts.jsonl`](receipts/candidate_complete_source_scripts.jsonl), with [SHA-pinned receipt](receipts/worklist_20261115.json). The optional [Lean 4.35 source verifier](replay_frontier_lean.py) checks the original old MathCan commit and exact script worklist, invokes the original persistent compiler, then requires two independent complete-source compiler passes and axiom checks before a newly assembled source receives proof credit. Process timeouts produce `unknown`.

```bash
python experiments/lean_ram11/frontier_worklist.py ram08-frozen-audited-train-43689874169.zip --out ram11_frontier_plan
python experiments/lean_ram11/replay_frontier_lean.py \
  --archive ram08-frozen-audited-train-43689874169.zip \
  --mathcan /src/MathCan-old-TRAIN \
  --worklist ram11_frontier_plan/candidate_complete_source_scripts.jsonl \
  --lean /path/to/lean-4.35.0/bin/lean \
  --out ram11_frontier_first_native_replay
```

Before invoking the second command, build the pinned original MathCan old-source dependencies, compile `experiments/lean_ram03/ProbeServer.lean`, and set `LEAN_PATH` to include its OLean directory and the old MathCan `lake env printenv LEAN_PATH`. The first command needs only Python. The second **requires a working Lean4.35 local runner**, unavailable in this sandbox. Its proposed scripts are source-only diagnostics; they do not enter RAM-08's held-out PEF scoring.
