# AF-25: Four-fact Horn transfer with structurally gated inexpensive controls

**Registered scientific verdict: NEURAL_ADVANTAGE_NOT_SHOWN (0/8 seeds pass).**  
**Engineering and independent verification: PASS (all 11 GitHub Actions jobs).**  
**Date:** November 16, 2026.

AF-25 tests whether frozen 6,036-parameter recurrent proof policies retain an inference-time advantage on two new positive-Horn graph families after **all four preregistered inexpensive public-graph controllers** have been independently shown to fail part of the held-out set. Every new case begins with four known facts, the original AF-23 *positive-training-root cardinality*. The source-only structural gate passed before model scoring; the subsequent eight-seed model experiment failed its preregistered two-depth criterion.

The more expensive, **64-expansion-per-decision symbolic BFS** still completes every positive case and correctly stops on all unprovable cases. AF-25 makes a feasible comparison against the cheap-control set, without demonstrating a general neural advantage over symbolic search or equal-latency inference.

## 1. Preregistration and provenance

- **Scientific registration:** [AF-25 PROTOCOL.json](../../experiments/activation_funnel_af25/PROTOCOL.json), first commit [`adc50ba`](https://github.com/MoralityLabAI/LoopedTransformers/commit/adc50ba55531e14c55539d8cd1ae4165eee45346). Two **pre-generator/pre-scoring** amendments [`6d6c628`](https://github.com/MoralityLabAI/LoopedTransformers/commit/6d6c6288bf6112d5f1633219f6c44abbbcc17991) and [`7f7ddc6`](https://github.com/MoralityLabAI/LoopedTransformers/commit/7f7ddc682cd1e9ffc136e0c4f1c2c278e13b40f4) froze the distinct successor count and hard-goal physical tie breaker. Final registration file SHA-256: `c05173d5848adfb55156b0a501598a118cc65548989f8eb438c2bdd8f9ae0133`.
- **Source-only structural acceptance:** [GitHub Actions #43828215369](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43828215369), PASS before any AF-25 neural checkpoint load. Independent manifest SHA-256: `636373237bb44371281e82c827b57f7703b1808e3d8836358dd821ed3b783507`. Source-only manifest original artifact ID `14206919341`.
- **Frozen original AF-23 checkpoints:** [AF-23 Track B run #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), committed scientific source `1312d138b685d1054d527e56e16e1ea757a90b08`. Eight fixed-final-update seed ZIP artifacts, their members and the Torch/NumPy state dictionaries were independently checked. **No training updates were performed for AF-25**.
- **Official AF-25 scored CI:** [run #43829157457](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43829157457), source-lock commit `13c5301a90590d98e0e0efa066247f0f66efdeed`. All 11 jobs passed, including source lock, source-bound frozen-checkpoint smoke, eight independent per-seed scoring/audit jobs, and independent aggregate.
- **Original official aggregate artifact:** `af25-frozen-transfer-aggregate`, GitHub Actions artifact ID **14208435623**, ZIP SHA-256 `5ca614b06cee3e2e3fdccb64bb842cda9533938d3dfb001fb3c4861effbc1f5c`. Its `af25_results.json` member SHA-256 is `5624d2bfe913c870dada6be00a722eceb8157be26824a875d36d0723eec99163`.
- **Engineering amendment:** a copied GitHub Actions branch filter was fixed at `d0d9976` before scoring and re-registered in the source lock request `13c5301`. Neither amendment changed any scientific threshold, graph case, source neural model, or inference algorithm.

These GitHub Actions artifacts and the full source allow the exact seed-level outcome to be replayed. The 0/8 scientific verdict was evaluated only on the independently audited results.

## 2. Frozen design

The original AF-23 network is TypedLoop with **6,036 parameters**, four native recurrent calls per visited state and a public ancestor readout. Eight seeds (17, 29, 43, 61, 73, 89, 103, 131) reuse their original final checkpoints, source-bound to Torch 2.5.1 CPU. Direct uses original legal-action logits; VI-1 scores one-step successors; VI-2 applies a second min-plus Bellman backup to the same weights.

Every candidate has the original 15-atom, 14-physical-rule-slot interface. The two held-out cohorts each have **192 cases**, of which **80 are provable** within an exact shortest proof of five or six firings and **112 are unprovable**. Budget = independent shortest length plus one firing for a provable goal, and exactly two decisions for an unprovable goal. The host exclusively validates legal Horn actions and terminal goals.

Each cohort contains:

- **48 hard positive cases:** a productive shared-prefix AND derivation competes with a longer decoy AND derivation. Multiple root decoys are short reverse-hop goal ancestors; the additive-cost heuristic counts shared prefix work twice, and the long competing goal rule wins the registered physical tie break.
- **32 balanced positive cases:** a second productive goal shortcut provides source-only control coverage and prevents an all-negative toy.
- **112 negative cases:** 56 per grammar family, created by replacing critical input atom 2 with one decoy while keeping exactly **four root facts**.

The source-only generator independently verified exact BFS costs with two implementations, closure truth, disjoint public feature hashes, and at least **four distinct enabled successor states** on every positive root. The new physical layouts are 410–413 and 420–423. Source-only adversarial acceptance required all four cheap methods to solve between 20 and 60 of 80 positive cases and fail at least 24/48 hard positives.

**Distribution scope:** the *cardinality* of positive roots matches the original AF-23 positive training pattern. AF-25's negative starts and graph motifs remain distribution-shifted: negative roots contain one pre-derived decoy atom, and the complete root-atom identity distribution does not match the original train distribution. This limits the causal interpretation of STOP generalization.

## 3. Registered control difficulty

The structural gate had no access to model outputs or checkpoint selection.

| Symbolic arm | Depth 5 solved /80 | Depth 6 solved /80 |
|---|---:|---:|
| Ancestor greedy | 32 | 32 |
| Additive-cost closure | 32 | 52 |
| BFS cap 4 | 32 | 32 |
| BFS cap 16 | 32 | 32 |
| **BFS cap 64 (higher-budget diagnostic)** | **80** | **80** |
| First-legal | 3 | 4 |
| Exact shortest oracle | 80 | 80 |

In the hard family, ancestor greedy and BFS4/16 complete **zero of 48** positive cases at each depth. Additive closure finishes **0/48** hard cases at depth 5 and **20/48** at depth 6, with some on-line replanning recovery. This establishes a genuine *low-budget algorithmic control failure* before any frozen model evaluation.

BFS cap 64 remains a strong source-aware solver on this finite task. Its performance is reported independently of the science gate because it has a substantially larger search expansion allowance per decision.

## 4. Registered primary outcome

The preregistered target was VI-2. For each seed, **both depths** had to satisfy all of the following: improve by **at least eight solved goals out of 80** against the best of ancestor greedy, additive closure, BFS4 and BFS16; achieve **at least 40/80 absolute** provable completion; correctly STOP immediately on at least **101/112 unprovable** cases; and produce zero false accepted Horn proofs. **Six of eight** original seeds had to meet these requirements.

All eight original checkpoints qualified by source/checkpoint/audit requirements. **Zero seeds met the full two-depth scientific gate.**

### Full pooled held-out proof completion

Each depth has **80 original graph cases × eight neural checkpoints = 640 positive proof episodes**. Deterministic symbolic results repeat the same 80 graphs and should be interpreted as 80 unique problem instances.

| Controller | Depth 5 solved /640 | Depth 6 solved /640 |
|---|---:|---:|
| **Direct** | **432 (67.50%)** | 128 (20.00%) |
| VI-1 | 176 (27.50%) | 96 (15.00%) |
| **VI-2, registered target** | **195 (30.47%)** | **146 (22.81%)** |
| Ancestor greedy | 256 (40.00%) | 256 (40.00%) |
| Additive-cost closure | 256 (40.00%) | **416 (65.00%)** |
| BFS cap 4 | 256 (40.00%) | 256 (40.00%) |
| BFS cap 16 | 256 (40.00%) | 256 (40.00%) |
| **BFS cap 64, diagnostic** | **640 (100%)** | **640 (100%)** |
| First-legal | 24 (3.75%) | 32 (5.00%) |
| Shortest-proof oracle | 640 (100%) | 640 (100%) |

Depth 5 is a notable inference-algorithm counterexample. **Direct** completes 432 proofs while the same weights under **VI-2** complete only 195, a reduction of **237** successes. At depth 6, VI-2 adds 18 successes over Direct (146 against 128). The second backup is neither a uniformly useful improvement nor a substitute for checking the direct policy.

### Complete seed-level outcomes

Each cell shows solved positive goals out of 80. The final column reports the *same registered VI-2* criterion at both depths.

| Seed | D5 Direct | D5 VI-2 | D6 Direct | D6 VI-2 | Registered both-depth gate |
|---:|---:|---:|---:|---:|:---:|
| 17 | 80 | 0 | 0 | 0 | FAIL |
| 29 | 80 | 37 | 48 | 20 | FAIL |
| 43 | 0 | 0 | 0 | 0 | FAIL |
| 61 | 80 | **80** | 32 | **58** | FAIL |
| 73 | 32 | 0 | 0 | 31 | FAIL |
| 89 | 80 | 0 | 48 | 0 | FAIL |
| 103 | 0 | 17 | 0 | 0 | FAIL |
| 131 | 80 | 61 | 0 | 37 | FAIL |

Seed 61 completes every depth-5 positive under VI-2 and 58 depth-6 positives. At depth 6, the best cheap controller has 52 completions, so seed 61 misses the required eight-proof improvement by **two** completions. Its correct immediate STOP rate on unprovable cases is **0/112 at each depth**, an independent failure. Seed 131 clears the depth-5 relative completion requirement with 61 proofs, then fails depth 6 (37 vs best-cheap 52) and the depth-5 negative-STOP criterion (56/112).

## 5. Family-specific behavior

Each hard family contains 48 positive cases per seed, hence **384** repeated positive evaluations over eight seeds. Each balanced family has 32 positive cases per seed, hence **256**.

| Family / method | D5 hard /384 | D5 balanced /256 | D6 hard /384 | D6 balanced /256 |
|---|---:|---:|---:|---:|
| Direct | **240** | **192** | 96 | 32 |
| VI-1 | 48 | 128 | 0 | 96 |
| VI-2 | 131 | 64 | 95 | 51 |
| Ancestor greedy | 0 | 256 | 0 | 256 |
| Additive closure | 0 | 256 | 160 | 256 |
| BFS cap 16 | 0 | 256 | 0 | 256 |
| BFS cap 64 | 384 | 256 | 384 | 256 |

The depth-5 Direct policy finds **240/384 (62.5%)** hard-family proofs on a source family where all four inexpensive graph-aware controls fail every case. This is a real empirical distinction within the fixed gym, and is stronger evidence of useful *learned rule ordering in certain checkpoint seeds* than the earlier AF-23 result against uninformed first-legal. The same result has substantial dependence on the seed and depth; VI-2 gives back much of the hard-family gain.

At depth 6, Direct completes 96/384 hard cases (25%). The best cheap control, additive closure, completes 160/384 hard cases. No general depth-stable neural advantage was established.

## 6. STOP calibration and failure location

The recovered action traces expose a further distinction from AF-24B's root-STOP-dominated failure. AF-25's VI-2 often begins by firing a rule and later stops prematurely on provable goals.

| VI-2 measure | Depth 5 | Depth 6 |
|---|---:|---:|
| First-action STOP on provable /640 | **0** | **112** |
| Any premature STOP before completing provable goal /640 | **160** | **379** |
| Correct first-action STOP on unprovable /896 | **560 (62.50%)** | **658 (73.44%)** |

The earlier AF-24B VI-2 policies stopped immediately on most provable roots while correctly STOPping on every negative root. AF-25 shifts the observed error pattern. Models frequently proceed on positive examples but later terminate incorrectly, while also failing to stop early on numerous new unprovable cases.

The model's STOP policy is sensitive to both public graph structure and the composition of already-proved facts. These receipts support a change in where failures occur under the source distribution shift, without establishing which specific model component causes them.

## 7. Compute and independent verification

All 11 hosted jobs passed on the immutable scored source commit. Original checkpoint member/ZIP SHA-256 checks and Torch/NumPy parity passed for all eight seeds. Every saved proof action was independently checked against model-visible physical rule incidence; independent bitset BFS recomputed the shortest proof budget, a separate numerical checker recomputed the logged Bellman/STOP values, and sampled real Torch forwards were replayed.

| Verification item | Official count |
|---|---:|
| Independent proof episodes | **30,720** |
| Reconstructed neural planner nodes | **282,045** |
| Sampled original checkpoint forward replays | **96** |
| False accepted Horn proofs | **0** |
| Additional model training updates | **0** |

### Native compute receipts

Totals over eight seed repetitions and all positive/negative cases at each depth. The methods use different primitive operations; the table is a workload description, not matched latency or FLOP accounting.

| Operation | Depth 5 | Depth 6 |
|---|---:|---:|
| VI-2 neural boundary evaluations | 53,097 | 38,444 |
| Closure controller public rule examinations | 216,576 | 253,632 |
| BFS cap 16 public state expansions | 55,256 | 61,584 |
| BFS cap 64 public state expansions | 123,080 | 124,152 |

BFS64's perfect result must remain visible when interpreting any algorithmic gain. The experiment does not establish comparable GPU FLOPs, equal CPU wall-clock across methods, or efficiency-adjusted model advantage. The low-budget structural gate was intentionally tuned to the specific public algorithms in the source-only stage, then frozen before any original model inference; its scores are conditional on that adversarial selection.

## 8. Interpretation and next discriminating experiment

AF-25 independently resolves the previous AF-24B design problem in a **narrow sense**: all four low-budget public-graph policies fail a meaningful portion of the positive task set, so the registered neural-comparison gate can theoretically pass. The frozen original VI-2 checkpoints nevertheless fail it in every seed. Direct logits show substantial seed-specific learned ordering on depth-5 hard cases, while the Bellman backup often removes that capability and depth-6 performance remains weak.

The candidate direction for a new *registered* intervention is to preserve Direct's rule rankings while independently correcting its STOP calibration and testing whether a value head supplies reliable information beyond the rule logits. Such a study needs per-seed fresh development and held-out grammatical families, norm/compute-matched controls, and explicitly sourced model selection. Another discriminating direction is **search-compute efficiency**, using a genuinely matched budget of rule examinations or host-state expansions so that BFS64's perfect completion can be evaluated against a neural-guided controller at the same budget.

Any new neural training must be preregistered separately from this frozen evaluation. AF-25's registered science is **NEURAL_ADVANTAGE_NOT_SHOWN**; original AF-22 FAIL, AF-23 weak-baseline PASS, AF-24 SYMBOLIC_CONTROL_MATCH, and AF-24B NEURAL_TRANSFER_FAIL all remain unchanged. The AF-25 [draft PR #129](https://github.com/MoralityLabAI/LoopedTransformers/pull/129) stays unmerged pending independent review.
