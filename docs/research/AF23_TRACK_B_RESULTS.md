# AF-23 Track B: Branched Horn-rule control experiment

**Registered scientific verdict: PASS (6/8 eligible seeds passed the stated gate). Engineering and independently audited CI: PASS.**

**Scope warning:** This result concerns the preregistered 15-atom, 14-rule **synthetic positive-Horn gym**, with six legal root actions for provable cases. It establishes a performance gain over first-legal and five blind random-legal baselines under that gym's shortest+1 firing budget. It does **not** establish a neural advantage over inexpensive graph-aware symbolic methods, because those methods were omitted from this registered comparison. The generation templates are highly structured and the model observes the full public Horn incidence graph.

- Registered source: [AF-23 protocol](../../experiments/activation_funnel_af23/PROTOCOL.json) (committed first on 2026-11-16); protocol SHA-256 `d3dfa7c6eb8eea6d80ee78bacaef407c2522f9ca0b36905c83ba1edc75c67662`.
- **Source-pinned successful CI:** [run #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), at commit `1312d138b685d1054d527e56e16e1ea757a90b08`.
- Original aggregate: `af23-track-b-aggregate`, GitHub artifact ID `14202115048`, artifact ZIP SHA-256 `a0950289f2545eebd9f1cd0b287c392a7fdecc04adb7ef64faf22f38b7491acf`, internal `af23_track_b_results.json` SHA-256 `8e52cd0cd61866b35bd9f39ce7c42e62f80398a79b26b63d4360253cb6486d31`.
- Frozen graph manifest SHA-256 `96f387e2b855201b7d58ca344fb439ad80f59cf37cc6efe00da9812f2aacfda5`.
- Draft [PR #116](https://github.com/MoralityLabAI/LoopedTransformers/pull/116); keep unmerged while reviewing the stronger controls.

## 1. Experiment

The original **6,036-parameter** recurrent TypedLoop architecture is trained with 240 fixed updates for each of eight registered seeds: 17, 29, 43, 61, 73, 89, 103 and 131. All variants within a seed use the same final checkpoint.

The gym uses at most 15 atoms and 14 physical rules. On provable starting cases, six rules are legal, including four decoy choices and two productive choices. Held-out physical layouts (202–206) differ from train/development layouts (170–175). The held-depth-4 and held-depth-5 rosters each contain 192 cases per seed, comprising 80 provable and 112 unprovable cases. The shortest valid proof has the registered depth on positive cases, and each policy must finish within the shortest proof length plus one rule firing.

Ten scored arms include Direct, VI-1, VI-2, first-legal fixed, five independently seeded blind random-legal controls and an exact shortest-rule oracle. All physical Horn transitions are host checked; no shortest-distance oracle is provided to the neural inference policy. The independent checker rebuilds graph incidence and proof distances.

The preregistered gate requires each qualified seed's VI-2 policy to exceed fixed by at least 10 percentage points, exceed the five-random mean by at least 10 percentage points and achieve at least 50% absolute provable-goal success at **both** held-out depths. At least six qualified seeds must pass both depths, with zero false accepted proofs.

## 2. Official pooled scores

Each depth has **640 provable episodes** (80 per seed, eight seeds).

| Policy | Depth 4 solved | Depth 5 solved |
|---|---:|---:|
| Direct | **640/640 (100.0%)** | 400/640 (62.5%) |
| VI-1 | 582/640 (90.9%) | 400/640 (62.5%) |
| **VI-2** | **598/640 (93.4%)** | **460/640 (71.9%)** |
| Fixed first-legal | 32/640 (5.0%) | 16/640 (2.5%) |
| Random legal 1 | 8/640 | 0/640 |
| Random legal 2 | 24/640 | 0/640 |
| Random legal 3 | 16/640 | 16/640 |
| Random legal 4 | 24/640 | 0/640 |
| Random legal 5 | 0/640 | 0/640 |
| Exact shortest oracle | 640/640 (100.0%) | 640/640 (100.0%) |

Across all eight seeds, VI-2 adds **60** depth-5 proofs over Direct (460 versus 400) and **60** over VI-1. At depth 4, Direct is stronger than VI-2 by 42 proofs. A learned value backup is therefore not the best observed inference mode at every depth.

All neural arms select the correct initial STOP on all **896/896 unprovable held-out cases** at either depth; none prematurely STOP on provable cases in the saved runs. For this gym, the main model error has shifted from early STOP (AF-22) to choosing a less effective rule path under the short action budget.

## 3. Full per-seed registered VI-2 gate

| Seed | Eligible | Depth 4 VI-2 /80 | Depth 5 VI-2 /80 | Both depths pass |
|---:|:---:|---:|---:|:---:|
| 17 | Yes | 80 | 60 | Yes |
| 29 | Yes | 80 | 80 | Yes |
| 43 | Yes | 80 | 0 | No |
| 61 | Yes | 80 | 80 | Yes |
| 73 | Yes | 80 | 80 | Yes |
| 89 | Yes | 38 | 0 | No |
| 103 | Yes | 80 | 80 | Yes |
| 131 | Yes | 80 | 80 | Yes |

All eight models reached **100% train and development proof completion** under their Direct policies at the fixed final update. Six reach the preregistered two-depth gate with VI-2; the remaining seeds fail conspicuously at depth 5.

These results should be treated as eight individual seeded observations. The pooled score does not replace the reported 6/8 registered pass count.

## 4. Independent verification

All 11 CI jobs succeeded, including source lock, smoke test, eight training/evaluation/audit jobs and independent aggregation.

The published aggregate records **42,496 host-audited proof episodes** across all seeds, covering both source diagnostics and the ten-arm held-out rosters, as well as **264 sampled checkpoint-forward replays**. The original proof oracle, independent physical-incidence auditor and Torch/NumPy checkpoint consistency checks passed; there were **zero false accepted Horn proofs**.

A previous [Track B CI run #43812273263](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812273263) failed and remains visible. The successful run uses a later source lock, and the registered gates must be assessed against the exact successful source revision.

## 5. What this result establishes

AF-23 Track B supplies better evidence of the utility of neural action scores **against uninformed action ordering** than AF-22 did. Its positive cases demand choosing among six initially enabled Horn rules. Direct and Bellman-guided action selection generally outperform first-legal and blind random choices under the short proof budget, with pronounced between-seed variation.

The positive inference is bounded by the gym. The physical rule graph is fully visible, there are only 14 rules, and the generator keeps the core productive/decoy motifs across train, development and evaluation while changing layouts and tested proof depths. A cheap **goal-ancestor backward-chaining heuristic**, a forward-closure controller, or a symbolic shortest-rule search could use that same public graph. The original shortest oracle already achieves 100%; the missing experiment is a **non-oracular computationally matched symbolic heuristic** that can be implemented and measured from the public incidence alone.

The fact that Direct gets every depth-4 proof and all seeds achieve 100% on train/dev warrants a saturation warning. This benchmark may be testing recognition of a fixed rule grammar, even when physical slot permutations change. It does not yet test cross-grammar transfer or autonomous Lean theorem proving.

## 6. AF-24 decision gate

Before claiming a search advantage, register an AF-24 control-only campaign with no new model training:

1. **Source-only graph-aware controls.** Evaluate a deterministic goal-ancestor greedy policy, relevance-filtered random policy, monotone forward-closure controller and a bounded symbolic BFS/A* procedure. Give each controller the exact same public graph and action budget. Record node expansions, number of examined rules, CPU inference time and proof completions; make STOP treatment explicit.
2. **Structural adversarial splits.** Modify the grammar prospectively so productive and decoy rules can trade places, goal-ancestor branches can be expensive, and root action count, true-optimal fraction and proof depth vary across held-out templates. Break shortcuts that identify proof relevance purely by a fixed role or fixed set of root facts.
3. **Causal mechanism check.** Hold the checkpoint fixed while replacing model distance and reach heads separately with constant or permuted values; compare with Direct to isolate whether the Bellman backup contributes reliable information beyond logits and public structure.

A new capability claim should require an advantage over the **best non-oracular graph-aware control** under a matched inference compute or rule-expansion budget, plus transfer to unseen rule compositions. If a simple symbolic controller matches the neural results, report the corresponding null without changing AF-23's frozen registered PASS.

**Current scientific status:** AF-23 Track A registered **FAIL**; AF-23 Track B registered **PASS** against its specified weak baselines; broader neural search advantage **unestablished**.
