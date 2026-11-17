# AF-31: producer-supported cyclic Horn dependencies under matched local observations

AF-30 ended **RULE_SELECTION_TRANSFER_FAIL (0/8)**: the new learned linear ranker completed all source positive proofs, but the equal-certificate hand-coded PATH control also completed all of them, using fewer public computations than symbolic closure would justify for a neural advantage. The AF-30 misleading paths depended on atom 10 with **no producer**, an elementary public reverse-path dead-end.

AF-31 changes the data-generation family so every noninitial atom has a producing rule. Unreachable atoms 8/10 form a mutually dependent **2-cycle** in TRAIN/DEV; held families balance 2-cycles against previously untrained **3-cycles 8→10→13→8**. Cycles have no seed input from the four known root atoms. The root locally matched reachable/unreachable pair differs by exactly two physical middle premise incidences, retaining the AF-29/AF-30 identity-preserving seven-anchor radius-two signature, complete per-node degree sequence, same initial legal rule slots and exact physical maps.

## Prior frozen registration and controls

See [PROTOCOL.json](PROTOCOL.json), committed first at **b62ee7b** before AF-31 source generator, train fit or checkpoint evaluation. The primary ranker uses AF-30's exact eight-feature public reverse-graph feature architecture and fixed train-only 500-step softmax optimizer, with eight independent seeds. Every trained legal-action controller uses only the full public graph and current physical known bits.

Two separate science questions must remain separate:

- **RANK_CERT transfer:** an eight-feature learned rule ranker shares an exact public Horn negative STOP certificate with a norm/feature-matched *untrained* ranker, PATH, ancestry, fixed-rule and random controls. The same public closure and bounded BFS16/BFS64 are independent strong baselines. A six-of-eight PASS requires at each held depth >=60/96 completed reachable proofs, >=80/96 correctly certified immediate negative STOP, at least **10/96** completed proofs beyond the strongest same-certificate baseline, and zero false accepted Horn proofs.
- **LRANK_STOP learned-only:** an additional ten-feature public-graph summary STOP classifier is independently fitted on TRAIN-only reachable/unreachable states with balanced logistic BCE over 600 updates. Its threshold is selected on DEV-only roots. It uses neither public least-fixed-point closure nor BFS at inference, and must earn its own heldout verdict for paired positive completion and negative STOP. It shares the learned rule ranker for continuation. Report its paired score ranking separately from absolute threshold decisions.

The three original **frozen AF-23 6,036-weight** controls, Native Direct, forced-continue Direct and exact-public-negative-certificate + Direct legal rule ranker, run against **the same** AF-31 graph cases in every seed. Their Torch checkpoint zip/member SHA-256 evidence is inherited unchanged from the exact original AF-23 source run, with zero optimizer steps. The model encoder sets all legacy truth targets to zero and retains the public physical atom/rule/role encoding.

The shortest-positive-proof+1 firing budget and two-decision negative budget remain scorer-only. All chosen rules receive independent physical Horn host replay. The main public-policy work unit is the count of each rule examination including the once-at-root symbolic certificate and repeatedly computed reverse features, while the original neural policies record actual frozen model-node forwards. Runtime counters are supplementary, with no claim of equal FLOPs.

## Source splits

| Source split | Positive/negative matched pairs | Positive shortest path | Swap positions | Cyclic trap |
|---|---:|---|---|---|
| TRAIN | 96 | 5 or 6 firings | 1,2 | 2-cycle |
| DEV | 96 | 5 or 6 firings | 1,2 | 2-cycle |
| HELD depth 5 | 96 | 5 firings | 1,2,3 (32 each) | 48 2-cycle; 48 3-cycle |
| HELD depth 6 | 96 | 6 firings | 1,2,3 (32 each) | 48 2-cycle; 48 3-cycle |

Exact independent public physical-bitset BFS, an independent logical host BFS, and the original goal closure confirm every source pair's answer. The complete source-only manifest also checks each blocked SCC has producers but no seed, all split public hashes are disjoint, and AF-29/AF-30 public graph hashes do not recur.

The [cyclic trapping lemma](../../docs/research/AF31_SUPPORTED_CYCLE_THEOREM.md) explains why all local graph vertices can have a producer while the cycle remains unreachable. The held three-cycle shift removes AF-30's producerless-atom shortcut without invoking any private oracle.

## Reproduction

The source-only and deterministic linear model files use Python 3.12 standard library:

\`\`\`bash
python -m unittest discover -s experiments/activation_funnel_af31/tests -p 'test_*.py' -v
python -m experiments.activation_funnel_af31.gym > af31_source_manifest.json
\`\`\`

For one immutable learned seed run:

\`\`\`bash
python -m experiments.activation_funnel_af31.evaluate --seed 17 --out out/af31-seed-17
python -m experiments.activation_funnel_af31.audit --seed 17 --out out/af31-seed-17
\`\`\`

The genuine 6,036-weight original Direct source comparison requires original AF-23 seed ZIP archives from [AF-23 Track B](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), and CPU Torch 2.5.1 and NumPy as used by AF-29:

\`\`\`bash
python -m experiments.activation_funnel_af31.direct --seed 17 --original-zip-dir originals --out out/af31-direct-seed-17
python -m experiments.activation_funnel_af31.direct_audit --seed 17 --original-zip-dir originals --out out/af31-direct-seed-17
\`\`\`

For all eight source seeds, run the [source-locked workflow](../../.github/workflows/activation-funnel-af31.yml) and aggregate both artifact families:

\`\`\`bash
python -m experiments.activation_funnel_af31.aggregate --root receipts --out af31_results.json
\`\`\`

The first scored result should be preserved as its original PASS/FAIL once all independently verified seed receipts are available. Source-only engineering amendments, their failed checks, and original immutable score inputs remain visible. Earlier AF-22 through AF-30 verdicts are unaffected.
