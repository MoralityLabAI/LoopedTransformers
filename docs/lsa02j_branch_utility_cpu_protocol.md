# LSA-02J: Learned branch viability and recovery-cost ranking for bounded Lean DFS

**Registered November 24, 2026.** CPU-only development study, following [LSA-02I](../reports/lsa02i_native_bounded_backtrack_cpu_20261124.md), where charged backtracking improved native full-proof completion from 21.88% greedy to 65.63% DFS but model-free uniform DFS already completed 65.0%. The remaining question is whether a learned branch ranker can improve on frozen Plan B and strong model-free search.

## Native labels and model input

On **training-only** theorem roots, native Lean 4.19 explores each public reference-path proof state and labels every visible candidate action using its actual AND/OR graph. A candidate is **finite viable** when its transition is accepted and all resulting and pending goals have finite native shortest proofs. Its recovery cost is one plus the sum of native minimum tactic counts for all successor and pending goals; unreachable or rejected actions receive a null cost. The label *bounded viable* means the native tactic cost is at most 16. These costs exclude possible future backtracks and therefore do not promise online search success.

The native source theorem reference proof is independently compiled before branch labeling. Exact public packet, candidate menu and root lineage must match the already frozen teacher. The labels are never provided to an online policy.

The frozen 57,475-parameter Plan B backbone processes the same 384-byte pp96 public typed-plus-pretty packet and two AB recursive passes. A small additional learned head reads its 32-dimensional candidate features plus eight transparent public-context features: premise count, directly known premises, missing direct premises, fact/rule counts, menu size, remaining budget and pending-goal count. The head predicts a finite-viability logit and normalized cost. Its training loss is binary cross-entropy plus 0.25 times smooth-L1 cost regression on finite labels. At inference a fixed, registered 5× viability-minus-half-cost score is added to the unchanged frozen Plan B tactic logit.

## Equal-step sham and model-free controls

Two heads, **viability** and **shuffled_sham**, use exactly the same frozen backbone, source frames, root/frame training draws, initial weights, MLP architecture, optimizer, learning rate and 480 update steps. For shuffled_sham, each native training frame's viability/cost pairs are deterministically permuted among its candidate tactics, preserving the positive count and within-frame cost distribution while breaking label-to-action identity. Singleton menus necessarily remain unchanged.

A separate **public_horn_dfs** control computes Horn proof costs by deterministic forward chaining over *public local hypothesis types alone*, without a trained model, hidden Lean transition graph or generator proof. It ranks the current candidate menu by minimum finite public proof cost and falls back to syntactic ordering if none is derivable. This unusually strong classical solver is essential for judging whether the generated Horn grammar is already solvable from public premises.

| Policy | Decision source | Search budget |
| --- | --- | ---: |
| frozen_dfs | Frozen Plan B logits and charged DFS | 16 |
| utility_dfs | Same Plan B plus correctly trained viability/cost head | 16 |
| sham_utility_dfs | Same Plan B plus shuffled-label head | 16 |
| public_horn_dfs | Model-free public Horn fixed-point costs | 16 |
| uniform_dfs | Model-free seeded uniform live menu, five streams | 16 |

Each attempted tactic and explicit reconsideration costs one search-work unit, with real Lean verdicts, independent terminal proof compilation, and model-free full journal replay.

## Fresh cohort and CPU constraints

Freeze new G03 theorem generator seed **20261203**, with 64 independent roots per train, calibration, ordinary-test and OOD shard. Compile **64 train** shortest reference proofs in native Lean and collect finite/dead action labels on those training states alone. Keep **32 disjoint calibration roots** for development evaluation. Ordinary/OOD proof outcomes remain sealed.

Model seeds are **183, 191**. Each source-only Plan B fit receives **720 FP32 AdamW updates** and is repeated exactly on one physical GitHub-hosted CPU worker. Then freeze the backbone and cache training-only candidate features. Each head receives **480 updates**, sampling theorem roots identically in each mode, with half the eligible draws targeting genuinely mixed viable/dead candidate menus. Duplicate each head fit on the same worker and require exact tensor, checkpoint and optimizer-trace hashes. Use Torch 2.5.1+cpu, one thread, deterministic operators and disabled MKLDNN.

On the same 32 calibration roots, run four policies for each model seed, plus five independently seeded uniform DFS episodes per root (17, 31, 47, 61 and 83). Native Lean 4.19 kernel verdicts, axiom-free standalone compilation and model-free replay validate every claimed full proof.

Only *after* these native proof results are frozen, independently explore and compile the 32 calibration shortest-reference scripts, then evaluate frozen viability/cost predictions on the resulting candidate states. Post-scoring metrics include root-weighted viable/dead discrimination, Brier score, finite-cost error, mixed-fork choice accuracy and public-Horn exact-cost agreement. None may update a model.

## Preregistered endpoints and gates

**Primary endpoint:** native certified full-proof completion under utility_dfs minus frozen_dfs, paired by theorem root and model seed. The utility arm must improve by **at least +5 percentage points over frozen DFS**, improve by **at least +5 points over shuffled sham**, and remain nonnegative versus DFS at both seeds. Report a descriptive 2,000-draw bootstrap over **32 unique theorem roots**. Repeated model seeds and random-stream episodes never multiply independent theorem N.

The separate **learned competence preflight** requires at least 32 independent roots, **at least 40%** whole-proof closure, and **at least +10 percentage points over both** the independently sampled uniform DFS policy and the strong model-free public Horn solver. A public Horn solver that saturates the generator reveals a benchmark limitation, even if the learned policy exceeds random search.

A development screen pass does not authorize a gym, Mathlib transfer or architecture promotion. Any positive result needs new sealed confirmation, FSI-01, independent source ancestry checks, and an adequately powered random baseline.

## Custody and CI

The [CPU workflow](../.github/workflows/lsa02j-branch-utility-cpu.yml) certifies source training proofs, generates native TRAIN-only viability labels, fits frozen backbones and two equal-budget heads with bitwise duplicates, performs uniform/model native proof evaluation, then executes the post-scoring root-level audit. It runs on GitHub CPU/RAM with at most two concurrent native-model jobs, no GPU and no paid inference. All native model, head, source and certificate manifests are SHA-256 bound with 14-day Actions artifact retention. A permanent numerical report is committed after green verification.
