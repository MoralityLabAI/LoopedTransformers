# AF-27: VI-2 STOP-gate transfer fails on two unseen Horn grammars

**Registered scientific verdict: STOP_TRANSFER_FAIL (0/8 fixed seeds).**  
**Hosted engineering / independent proof audit: PASS (all 11 jobs).**  
**Date:** November 16, 2026.

AF-27 tests whether the learned VI-2 STOP gate improves recognition of unprovable goals **when the legal-rule ranking is held fixed to the original Direct head**. It reuses the eight exact 6,036-parameter AF-23 checkpoints, with no further training, on 384 entirely new source-registered positive-Horn graphs. Two held-out proof depths (5 and 6) and two new rule-grammar families replace the previously scored AF-25 Horn construction.

The result is **STOP_TRANSFER_FAIL**. At each depth VD (VI-2 STOP + Direct rules) completes **40 more seeded provable proof episodes** than DD (Direct STOP + Direct rules). Correct immediate STOP on unprovable inputs, however, **falls by 126/896 at depth five** and improves by only 42/896 at depth six. The preregistered two-depth gate requires a positive termination gain at *both* depths in at least six of eight fixed seeds, and none pass.

A separate posthoc, independently audited diagnostic shows substantial learned-cost rule-rank error: at the positive initial states, Direct's top legal rule belongs to the exact optimal-action set in **480/640** depth-five and **520/640** depth-six seeded cases, compared with **192/640** and **280/640** under VI-2. This is a *source-scored root action diagnostic* and should not be equated with completed proofs or a model capability claim. The unchanged public closure controller also solves **100%** of both new graph cohorts and correctly stops on every unprovable goal, substantially limiting any neural proof-search interpretation.

## 1. Registration and immutable experimental source

- **Predata protocol:** [AF-27 PROTOCOL.json](../../experiments/activation_funnel_af27/PROTOCOL.json), first committed [`db8d52b`](https://github.com/MoralityLabAI/LoopedTransformers/commit/db8d52bc0a5f38157397fda3e12c8bd433f8a21d) before creating the new grammar or any score-generating source. SHA-256 `ef75aedeb294582f53312f85c36acff720f4ec5ac31334d61daa3a5257fe1495`.
- **New source-only structural gate:** [CI #43891716564](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43891716564) passed, after a pre-neural source-only repair to the length of one depth-six alternative path. The independently checked manifest SHA-256 is `b1d592ee70efb679bc0aba92d155135395973b153f385b7006ccfc3d9a9ad896`. Original model checkpoint loading is **absent** from this preflight.
- **Official frozen-model run:** [GitHub Actions #43894259975](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43894259975), successful source-locked head `0a511665fbd827cdbfcf319513b7e437c3e2456e`. Source lock, original-checkpoint smoke, eight per-seed evaluation/audit jobs and eight-seed aggregate all passed.
- **Original checkpoint provenance:** [AF-23 Track B run #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), final 6,036-weight TypedLoop / ReadoutPolicy(ancestor, 4) checkpoint per seed, with original GitHub ZIP/member SHA-256 and Torch/NumPy state-dict parity checked before inference. **Zero AF-27 optimizer updates**.
- **Official aggregate artifact:** `af27-stop-transfer-aggregate`, GitHub artifact ID **14238612773**, ZIP SHA-256 `0bdf0886480549b8c8264cbe9a458cffc3a70daede132d3c82baf03f2c6b8c4c`. Its internal `af27_results.json` member SHA-256 is `9ddf06206f4ed55a83cb21cd8112ece94b70a854c7dbeeac93b3d0e88d2cc88f`.
- **Evidence digest archive:** [AF27_EVIDENCE_MANIFEST.json](AF27_EVIDENCE_MANIFEST.json) records the eight original independently audited seed ZIP artifact IDs/SHA-256, the source-locked run identity, the smoke artifact, the aggregate member hash and the verification totals. GitHub Actions ZIPs remain subject to their configured retention period.
- **Earlier blocked attempts:** [#43893195508](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43893195508) found an audit-only smoke roster count mismatch; [#43893698825](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43893698825) generated full original scores but the independent auditor incorrectly tried to measure legal-rule top-two margins with zero legal rules. Both issues were repaired with dated source-lock amendments **without changing the model, inference algorithms, generated cases or preregistered science thresholds**. The blocked attempts remain publicly inspectable.

## 2. Design and primary gate

Each held-out depth has **192 new physical Horn cases**: 80 provable and 112 unprovable. The two families, `forked_and` and `or_gate`, each contribute 40 provable and 56 unprovable examples at each depth. Every case begins with **four known facts**. One critical fact is replaced with a decoy in negative roots while preserving the fact count. Each family has two alternate goal-concluding rules and several distinct legal initial successor states. The source-only independent bitset/host-BFS audit verifies exact positive proof length 5 or 6 and negative closure, with public hashes disjoint from the AF-25/AF-24B cohorts.

The frozen checkpoint supplies the same native Direct and VI-2 complete action scores at every visited state. The registered policies are:

| Arm | STOP gate | Rule ranking |
|---|---|---|
| **DD** | Direct native STOP decision | Direct |
| **VD (primary)** | VI-2 native STOP decision | **Direct** |
| **VV** | VI-2 native STOP decision | VI-2 |
| **ND** | Continue while any legal rule exists | Direct |
| **BFS node-matched** | Public bounded BFS | Cap at original two-step neural distinct state count *per decision* |
| **BFS64** | Public bounded BFS, higher-cost diagnostic | 64 expansions per decision |
| **Closure** | Original public Horn additive-cost closure | No neural weights |
| **Shortest** | Exact host BFS oracle upper control | Evaluation only |

All policies execute their own full source-bound host-verified trajectories, under **shortest proof length +1 action** for provable goals, and **two decision steps** on unprovable goals.

**Preregistered gate:** the same original seed must show VD correctly initial-STOPping on at least **7 more of the 112 unprovable cases** than DD, with **at most four fewer completed proofs out of 80 provable cases**, at *both* depths, and zero false accepted Horn proofs. At least **6 of 8 seeds** must meet every condition at both depths.

**Official result: 0/8 seeds pass.** The full source-checked experiment is registered **STOP_TRANSFER_FAIL**.

## 3. Complete pooled proof completion

Denominator **640 provable episodes** per depth: 80 unique physical cases × eight original model seeds. The deterministic symbolic policies repeat the same 80 graphs across seeds.

| Policy | Depth 5 solved /640 | Depth 6 solved /640 |
|---|---:|---:|
| **DD** | 200 (31.25%) | 40 (6.25%) |
| **VD** | **240 (37.50%)** | **80 (12.50%)** |
| **VV** | 172 (26.88%) | 46 (7.19%) |
| **ND** | 360 (56.25%) | 160 (25.00%) |
| BFS node-matched | **640 (100%)** | **640 (100%)** |
| BFS64 | **640 (100%)** | **640 (100%)** |
| Public additive closure | **640 (100%)** | **640 (100%)** |
| Exact shortest oracle | **640 (100%)** | **640 (100%)** |

VD improves completion by **40/640 (6.25 percentage points)** relative to DD at *each* depth. Its complete-success advantage is concentrated in the `forked_and` family. Both neural DD and VD complete zero `or_gate` depth-six positive cases. Forced continuation with Direct rules increases completion substantially yet cannot correctly stop immediately on unprovable states while a rule remains enabled.

### Positive cases by grammar family

Each family contains 40 unique provable graphs per depth, repeated across eight source seeds, for denominator **320** per cell.

| Policy | D5 forked /320 | D5 OR-gate /320 | D6 forked /320 | D6 OR-gate /320 |
|---|---:|---:|---:|---:|
| DD | 200 | 0 | 40 | 0 |
| **VD** | **240** | 0 | **80** | 0 |
| VV | 92 | 80 | 46 | 0 |
| ND | 320 | 40 | 160 | 0 |
| Public closure | 320 | 320 | 320 | 320 |
| Node-matched BFS | 320 | 320 | 320 | 320 |

The new grammars are OOD to the original AF-23 models and retain substantial seed dependence. The public symbolic controllers saturate at 100% proof completion, so this benchmark does **not** support a neural search-performance advantage over public Horn inference.

## 4. STOP calibration: the primary failed requirement

Correct immediate STOP on a truly **unprovable** goal has denominator **896** per depth, comprising 112 negative cases × eight fixed seeds.

| Policy | D5 correct STOP /896 | D6 correct STOP /896 |
|---|---:|---:|
| **DD** | 588 (65.63%) | 672 (75.00%) |
| **VD** | **462 (51.56%)** | **714 (79.69%)** |
| VV | 462 (51.56%) | 714 (79.69%) |
| ND, forced continuation | 0 | 0 |
| Node-matched BFS | 224 (25.00%) | 0 |
| BFS64 | 896 (100%) | 896 (100%) |
| Public closure | 896 (100%) | 896 (100%) |
| Shortest oracle | 896 (100%) | 896 (100%) |

The VD–DD negative STOP difference is **−126 at depth five** and **+42 at depth six**. At depth five, VD loses correct initial STOP on 126 paired cases, rescuing none. At depth six, VD rescues 42 cases without paired losses. The same primary gate requires a gain at both depths, and its failure is robust to the positive-proof-completion improvement.

On provable goals, immediate STOP/early premature STOP also matter:

| Arm | D5 initial STOP /640 | D5 premature STOP /640 | D6 initial STOP /640 | D6 premature STOP /640 |
|---|---:|---:|---:|---:|
| DD | 360 | 400 | 360 | 560 |
| VD | 320 | 360 | 440 | 480 |
| VV | 320 | 400 | 440 | 594 |
| ND | 0 | 0 | 0 | 0 |

VD improves some *positive* outcomes by delaying or changing STOP decisions after Direct-ranked rule firings, while its source gate incorrectly rejects more depth-five unprovable roots. The two classes have different effects and require separate scoring.

### Every original seed, registered VD versus DD

For each held depth the table shows completed provable goals and correct immediate STOP on unprovable goals, from 80 and 112 cases respectively.

| Seed | D5 DD solved | D5 VD solved | D5 DD neg STOP | D5 VD neg STOP | D6 DD solved | D6 VD solved | D6 DD neg STOP | D6 VD neg STOP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 40 | 40 | 14 | 14 | 0 | 0 | 42 | 42 |
| 29 | 40 | 40 | 84 | 70 | 0 | 40 | 112 | 112 |
| 43 | 0 | 40 | 70 | 56 | 0 | 0 | 70 | 112 |
| 61 | 0 | 0 | 56 | 14 | 0 | 0 | 56 | 56 |
| 73 | 40 | 40 | 112 | 98 | 0 | 0 | 112 | 112 |
| 89 | 40 | 40 | 56 | 56 | 40 | 40 | 56 | 56 |
| 103 | 0 | 0 | 112 | 112 | 0 | 0 | 112 | 112 |
| 131 | 40 | 40 | 84 | 42 | 0 | 0 | 112 | 112 |

Only seed 43 passes the *single depth-six* registered condition. **No seed passes both depths.** The positive DD versus VD paired comparison adds 40 successes and loses none at each depth, while the depth-five negative-STOP paired comparison loses 126 and gains zero.

## 5. Source-locked legal-rule gap diagnostic

This is a **posthoc action ranking** analysis on initial **provable** states. The two native sources are evaluated over the exact same host-enabled rule slots and source model; the independent true optimal rule set is attached *only after inference*. Each depth has 80 unique positive graphs × eight neural seeds = 640 scored root states.

| Root-score diagnostic | Depth 5 /640 | Depth 6 /640 |
|---|---:|---:|
| Direct top-ranked legal rule is exact-optimal | **480 (75.0%)** | **520 (81.25%)** |
| VI-2 top-ranked legal rule is exact-optimal | 192 (30.0%) | 280 (43.75%) |
| Direct optimal, VI-2 nonoptimal | **328** | **320** |
| VI-2 optimal, Direct nonoptimal | 40 | 80 |
| Direct and VI-2 rank different legal rules | **488 (76.25%)** | 400 (62.5%) |
| VI-2 legal top-two margin below \(10^{-5}\) | 160 | 0 |
| Direct legal top-two margin below \(10^{-5}\) | 0 | 0 |
| VI-2 successor interior STOP binding events | 1,200 | 880 |

The root-level VI-2 interior STOP boundary binds on **1,200/1,920 legal successors (62.5%)** at depth five and **880/1,280 (68.75%)** at depth six, counting every seeded positive root's registered legal rule options. This is a **branch-binding trace count**, not an independently identified cause of wrong action selection. The action-ranking failures nevertheless corroborate AF-26's fixed-policy observation that VI-2's learned-cost rule ordering can be less useful than Direct logits in these synthetic graphs.

The native legal rule top-two median margins are:

| Policy-native legal-score margin | Depth 5 median | Depth 6 median |
|---|---:|---:|
| Direct | 0.23793 | 0.14859 |
| VI-2 | 0.14876 | 1.21187 |

The depth-six VI-2 rank errors are often associated with **substantial native score margins**. Numerical tie sensitivity alone cannot explain them. The scores remain different-scale signals, so comparing Direct and VI-2 **margin magnitudes to each other** does not quantify which source is more calibrated.

See [AF27_RULE_GAP_MATH.md](AF27_RULE_GAP_MATH.md) for the exact formula, piecewise STOP-reachability sign reversal, misranking witness and sufficient action-gap stability condition. None of those mathematics relies on the AF-27 empirical outcome.

## 6. Independent proof audit and computation

Official source-pinned [run #43894259975](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43894259975) passed all eleven jobs. It decoded the physical graph from actual model-visible tensors, used an independent bitset BFS to derive truth and shortest costs, replayed every host rule firing, re-derived source Direct and VI-2 logits and min-plus values, checked the registered native STOP/rank factor, recomputed all root gap metrics, and sampled genuine original checkpoint forwards.

| Verified receipt | Count |
|---|---:|
| Original immutable checkpoints | 8 |
| Independently audited new-grammar proof episodes | **24,576** |
| Reconstructed neural planner-node evaluations | **193,839** |
| Sampled genuine frozen Torch forward replays | **128** |
| Sampled near-tie action replays | 0 |
| False accepted physical Horn proofs | **0** |
| New optimizer/training updates | **0** |

The original score and distance truth remain strictly separated at inference. The per-seed archive includes complete model step/node ledgers, oracle-free action decisions and posthoc host-optimal root rule labels. The independent evaluator checks the same experiment seed, public graph source hashes, archived model ZIP signatures, fixed protocol SHA and full roster identity.

### Budget and workload comparison

Totals below cover **1,536 episodes per depth** (640 positive + 896 negative), including the eight repeated original neural seeds.

| Controller and work unit | Depth 5 | Depth 6 |
|---|---:|---:|
| DD neural model-node forwards | 24,402 | 14,872 |
| VD neural model-node forwards | 26,178 | 15,198 |
| VV neural model-node forwards | 25,056 | 11,311 |
| ND neural model-node forwards | 42,430 | 34,392 |
| Node-matched BFS public state expansions | 36,144 | 28,664 |
| BFS64 public state expansions | 69,232 | 56,728 |
| Public closure physical rule examinations | 232,512 | 283,104 |

BFS node-matching **caps per-decision expanded host states at the size of the actual two-step neural public neighborhood** at the same state. It does not match Torch arithmetic, GPU FLOPs, number of examined physical Horn clauses, or host latency. The public closure algorithm completes all unique graph cases correctly without trained parameters. Consequently neither DD nor VD establishes a useful capability advantage over symbolic inference on the AF-27 public grammar.

## 7. Scientific interpretation and next experiment

AF-27 supports a narrow, source-verified finding: **VI-2's native STOP gate does not transfer a consistent advantage in distinguishing provable and unprovable goals** when paired with fixed Direct legal-rule ranking on these new graph structures. It can improve proof completion, and it can occasionally improve negative STOP, but it also gives up 126 correct negative STOP decisions at depth five across the eight seeded evaluations.

The original AF-26 RULE_ONLY finding was about its **specific hard depth-five AF-25 grammar**, and remains unchanged. AF-27's independently recorded native legal-rule ranking differences provide further evidence of a mechanism-level weakness in VI-2's learned value scorer. A successful subsequent model should preserve or improve reliable Direct rule ranking and separately calibrate STOP against **genuinely fresh, structurally varied positive/negative development grammars**, with frozen evaluation cases and a strong public closure baseline. No present result warrants a general neural proof-search or autonomous Lean-theorem-proving claim.

The public data are **one fixed finite 15-atom/14-rule benchmark** with repeated physical graph sets across model seeds. A learned advantage against node-matched public-state counts would still require careful compute profiling, and BFS64/closure remain decisive high-quality symbolic controls on these instances.

**Registered outcome: AF-27 STOP_TRANSFER_FAIL.** All AF-22–AF-26 original science results are preserved. Keep [draft PR #135](https://github.com/MoralityLabAI/LoopedTransformers/pull/135) unmerged pending independent review.
