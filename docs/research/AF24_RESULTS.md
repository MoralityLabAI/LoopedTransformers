# AF-24: Public-graph symbolic controls match or exceed the AF-23 neural proof results

**Registered result: SYMBOLIC_CONTROL_MATCH (8/8 paired seed comparisons).**  
**Source-pinned hosted engineering and independent audit: PASS (all ten jobs).**  
**AF-23 Track B's preregistered PASS remains unchanged.**

This result establishes a specific benchmark weakness: the visible, finite 15-atom/14-rule Horn grammar admits a constructive **nonlearned, additive-cost forward-closure planner** that completes all provable goals under the original shortest+1 proof budget and immediately stops on every unprovable goal. AF-23's 6,036-parameter neural policies provide no measurable proof-success advantage over that control on the saved test cases. The symbolic method requires dynamic graph traversal and is **not established to have equal wall-clock time, GPU FLOPs or latency** to any neural method.

## Source and registration

- **Scientific preregistration:** [AF-24 PROTOCOL.json](../../experiments/activation_funnel_af24/PROTOCOL.json), committed at [`25ba1a8`](https://github.com/MoralityLabAI/LoopedTransformers/commit/25ba1a8005776d08e0944e9347b6c3786be15eff) **before any AF-24 scored evaluation**. Protocol SHA-256 `d80214d8dcb12ae550f3f24504e05a1033cff046a382dbece4cb7c31bd3ae47c`.
- **Source lock and official hosted run:** [GitHub Actions #43818115192](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43818115192), source-run commit [`379ba975`](https://github.com/MoralityLabAI/LoopedTransformers/commit/379ba97567822363090ea674b6af55e024302f1a). Source lock, eight seed-score/audit jobs and independent aggregate all succeeded.
- **Exact original AF-23 neural source:** [AF-23 Track B CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), original successful scientific source commit `1312d138b685d1054d527e56e16e1ea757a90b08`. Original aggregate `af23_track_b_results.json` SHA-256 `8e52cd0cd61866b35bd9f39ce7c42e62f80398a79b26b63d4360253cb6486d31`, retrieved from GitHub artifact ID `14202115048` whose ZIP SHA-256 is `a0950289f2545eebd9f1cd0b287c392a7fdecc04adb7ef64faf22f38b7491acf`.
- **Official AF-24 aggregate:** artifact `af24-symbolic-aggregate`, ID **14202609013**, ZIP SHA-256 `42e816f9df87bd0cb796a3a8d66b0dafd88af2a5408e56191318ea15a9aa6fe8`; internal `af24_results.json` SHA-256 `78fc33625bd186c759d9284e26c85fb10e013f7c0030063b9a7476c6e5b7000d`. The archive also preserves the exact original AF-23 aggregate file used for pairing.
- **No model training or tuning.** The AF-24 challenge is an entirely source-only controller comparison. AF-23's science verdict and trained weights were neither rewritten nor reinterpreted as a preregistered failure.

## Question and design

AF-23's branch-rich gym contains six initially legal Horn actions on provable goals, but the **entire physical Horn rule incidence and the goal marker are public to the controller**. AF-24 asks whether the neural proof-search advantage inferred from comparison to first-legal and blind random legal-rule choices persists against inexpensive public-graph algorithms.

The original held-out graph family remains fixed: 192 cases on each of two depths, with **80 provable goals and 112 unprovable goals** per depth. Every policy gets the **original shortest proof length plus one firing** to finish provable tasks. Eight original neural seeds are preserved as separate paired comparison units, although all source-only deterministic controls see the same 80 unique positive graphs at a given depth and thus repeat their result across seeds.

Registered controls use only public physical incidence, current known facts, and the marked goal:

| Controller | Online inference |
|---|---|
| **Ancestor greedy** | Backpropagate which atoms can contribute to the goal; choose the enabled rule whose head has the shortest *relaxed reverse hop count* to the goal |
| **Ancestor random, five streams** | Sample independently among equal-relevance enabled goal-ancestor rules with fixed seed/case/arm PRNG |
| **BFS cap 4 / cap 16** | Explore up to four or sixteen legal host states at each decision and follow a found goal path; if the cap is reached, fall back to ancestor greedy |
| **BFS cap 64** | Larger-cost diagnostic, kept separate from the primary registered matching claim |
| **Closure direct** | Compute the monotone positive-Horn least fixed point with **additive rule-cost witnesses**, recursively identify a supporting goal derivation, then fire a currently enabled required rule |
| **First legal** | Fire the first physically enabled rule without stopping before exhaustion |
| **Shortest oracle** | Evaluation-only exact shortest proof controller, upper bound |

The **closure direct** controller is deterministic, uses zero learned weights, and receives no precomputed truth labels or distances. It can compute exact goal reachability by traversing the complete visible Horn graph. Its additive derivation cost is a relaxation that can overcount shared subproofs; it is **not a universal theorem of globally shortest Horn proofs**. It happened to choose an admissible shortest+1 proof for every positive case in this specific grammar.

All controllers record host-state expansions, examinations of public rules, terminal decisions, and same-runtime symbolic wall time. Direct CPU latency or equal-FLOP claims against original neural inference are excluded because those neural runs used separate Torch environments.

## Official results: proof completion

The counts below aggregate **80 provable cases × eight original seeds = 640 proof episodes** per depth. Original neural results are source-verified fields from AF-23's immutable scored aggregate, and symbolic results come from AF-24's independently replayed episodes.

| Policy | Held depth 4 solved /640 | Held depth 5 solved /640 |
|---|---:|---:|
| **AF-23 Direct (frozen)** | **640 (100%)** | 400 (62.50%) |
| **AF-23 VI-2 (frozen)** | 598 (93.44%) | 460 (71.88%) |
| AF-24 ancestor greedy | **640 (100%)** | 112 (17.50%) |
| AF-24 ancestor random 1 | 640 (100%) | 123 (19.22%) |
| AF-24 ancestor random 2 | 640 (100%) | 130 (20.31%) |
| AF-24 ancestor random 3 | 640 (100%) | 129 (20.16%) |
| AF-24 ancestor random 4 | 640 (100%) | 153 (23.91%) |
| AF-24 ancestor random 5 | 640 (100%) | 141 (22.03%) |
| AF-24 bounded BFS cap 4 | 640 (100%) | 112 (17.50%) |
| AF-24 bounded BFS cap 16 | 640 (100%) | 112 (17.50%) |
| AF-24 bounded BFS cap 64 (diagnostic) | 640 (100%) | 296 (46.25%) |
| **AF-24 closure direct** | **640 (100%)** | **640 (100%)** |
| Original first-legal | 32 (5.00%) | 16 (2.50%) |
| Exact shortest oracle | 640 (100%) | 640 (100%) |

The closure controller recovers **240 additional depth-5 completions relative to AF-23 Direct** and **180 relative to AF-23 VI-2**, while matching Direct and the oracle at depth 4. The same control selects immediate STOP correctly on **896/896 unprovable episodes at each depth** and has **zero premature STOP on the 640 provable episodes** at each depth.

The **simpler goal-ancestor heuristic**, which ignores AND-premise derivation costs, solves every depth-4 case but only 14/80 unique depth-5 cases. Limited BFS with a four- or sixteen-node cap does not improve depth-5 completion over ancestor greedy, while 64 expanded nodes per decision improve it to 37/80. These comparisons show how much the rule-conjunction structure and search budget matter.

### Registered paired gate

A single named primary symbolic policy must solve at least **76/80 depth-4 goals** in a seed (within five points of original Direct's 80/80) and finish at least **original seed-specific VI-2 depth-5 completions minus four**. It must clear both depths in **at least six of eight seeds**, with zero false accepted Horn proofs. Each named arm was registered before evaluation.

| Named primary control | Paired seeds matched /8 |
|---|---:|
| Ancestor greedy | 2 |
| Each ancestor-random stream | 2 |
| BFS cap 4 | 2 |
| BFS cap 16 | 2 |
| **Closure direct** | **8** |

The two matching seeds for ancestor-based policies are the original neural seeds whose depth-5 VI-2 proof completion was zero; this **does not imply substantial absolute problem solving**. Closure direct meets the gate by solving all 80 cases at both depths regardless of the neural seed.

**Registered AF-24 outcome: SYMBOLIC_CONTROL_MATCH, 8/8 paired seeds.**

## Compute, STOP, and verification

Closure direct costs are measured in the same source-only Python runtime as the other symbolic controls. Across **1,536 episodes per depth** (640 provable + 896 unprovable), including all eight repeated seed comparisons:

| Metric | Depth 4 | Depth 5 |
|---|---:|---:|
| Closure proof successes | 640/640 | 640/640 |
| Correct initial STOP on unprovable | 896/896 | 896/896 |
| Premature STOP on provable | 0 | 0 |
| Explicit BFS node expansions | 0 | 0 |
| Public Horn rule examinations | **179,648** | **220,416** |
| Mean rule examinations per episode | **116.96** | **143.50** |
| False accepted proofs | **0** | **0** |

The algorithm still traverses a fixed point and costs O(iterations × rules); zero **BFS** expansions means it never branches over saved proof-state subsets in an explicit BFS queue. It is not constant-time.

Independent verification was run on **36,864 AF-24 per-case, per-controller, per-seed proof episodes**. Every applied symbolic action was checked against the original physical Horn host. An independent BFS recomputed the proof target and short budget. The auditor replayed controller selections, checked numerical work receipts and detected forged final proof states. All eight auditor jobs passed with zero false accepted proofs.

The source-locked GitHub Actions aggregate restored the exact original AF-23 results only after verifying their immutable archive and internal JSON checksums. All source-only results retain the full per-case action traces in the eight signed CI artifacts.

## Scientific interpretation

The AF-23 result is **valid under its registered comparison**: compact recurrent neural policies beat first-legal and blind random selection on a tiny branching Horn grammar, and six of eight initializations met the registered two-depth gate. AF-24 demonstrates that this task's **public structure permits perfect deterministic goal derivation**, with straightforward additive-cost planning and appropriate STOP classification. A statement that neural values provide independent search capability beyond that public structure would therefore exceed the available evidence.

The distinction between *reachability* and *shortest rule-firing budget* matters. Positive Horn reachability can be found by forward closure in polynomial time, while globally minimal proof length is a separate combinatorial objective on arbitrary overlapping Horn derivations. AF-24's particular backward-reconstructed additive witnesses satisfy the original shortest+1 budget on every saved graph. We do not infer that they will do so for arbitrary Horn programs.

A meaningful next evaluation needs frozen neural checkpoints on **source-registered nonisomorphic grammars** featuring misleading goal-relevant detours, expensive ancestor subproofs, varied goal conjunctions, and shortcut inversions. Both symbolic and neural arms must receive identical public incidence and host-known facts. Source-level structural properties and metric thresholds must be registered before scoring. Equal-cost claims require observed compute, not just a common action budget.

**Status:** AF-24 registered **SYMBOLIC_CONTROL_MATCH**, independent engineering **PASS**. AF-23 Track B remains preregistered **PASS against its original fixed/random controls**. Broader neural proof-search capability remains **unestablished**.

Keep the AF-24 PR **draft and unmerged**, retaining this null result and its original negative/positive control evidence.
