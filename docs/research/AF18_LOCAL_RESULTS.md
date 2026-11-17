# AF-18 local scored result | 15 November 2026

**Registered scientific verdict: FAIL.** **Independent local engineering verdict: PASS.** Four seeds produced 20 final-step frozen native TypedLoop checkpoints and passed the independent source/BFS/host and sampled numerical audit. The evaluation used Python 3.13 / Torch 2.10 CPU with two threads. The registered GitHub Actions Python 3.12 / Torch 2.5.1 and Lean 4.19.0 checks are outstanding because hosted preflight failed before runner execution.

- Frozen scientific source: [`909b426f`](https://github.com/MoralityLabAI/LoopedTransformers/commit/909b426f0ff4d6b5c6bd4613e16d18e6858be96d).
- Registered protocol SHA-256: `7b607eb2bf6d63f4332b07102063a25c64fa5f5007be6d1821cf8b602ee59eaf`.
- Committed manifest SHA-256: `50b782ec57370b616d6cad0c3b53ceb0811f207629e83e44698933629301432d`.
- [AF-18 draft PR #93](https://github.com/MoralityLabAI/LoopedTransformers/pull/93), based on `main` after the AF-14 through AF-17 merges.
- Original hosted [AF-18 workflow #43716898298](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43716898298), preflight failure before runner steps; no claimed hosted verification.

## Registered study

The full 5,268-parameter AF-17 BellmanTypedPolicy and original hash-pinned TypedLoop cell are shared. Only recurrent call count (2/4/6/8) and the two-loop cap ablation (10 versus 16) vary. Per seed, all five models start from byte-identical initial parameter tensors and share the 240-step sampler. More recurrent calls incur more forward and backward FLOPs, so this is an input/step-matched **depth** experiment, not a compute-matched architectural superiority claim.

Train/dev are the original AF-17 928/352 cases and train-only source states. The new 416 heldouts use layouts 57–61 with the same AF-17 logical Horn templates, partitioned into 32 fresh-layout, 192 depth4 and 192 depth5 cases. There are no new logical operators. The host alone enforces legal Horn rule firings; the student receives public incidence, current proved atoms and legal-action mask. Its Bellman training labels and goal reachability oracle never enter the neural features.

## Development eligibility and heldout success

Under the preregistration, `l2_cap16`, `l6_cap16`, and `l8_cap16` must each reach 65% proof success separately in train and development. Eligible-seed success then needs at least a 5-point gain from **both** 6 and 8 loops over 2 loops at **each** deep heldout family, at least 70% absolute, no added early-STOP failures beyond 2 points and zero false accepted proofs. Any eligible required failure yields an aggregate **FAIL**.

| Seed | Eligible? | Dev 2 / 6 / 8 loops (cap16) | Held depth4 2 / 4 / 6 / 8 loops | Held depth5 2 / 4 / 6 / 8 loops | Seed verdict |
|---:|---|---|---|---|---|
| 17 | No | 0.673 / 0.480 / 0.813 | 0.000 / 0.514 / 0.595 / 0.703 | 0.000 / 0.324 / 0.000 / 0.162 | **INCONCLUSIVE** |
| 29 | Yes | 0.920 / 0.907 / 0.780 | 0.527 / 0.297 / 0.162 / 0.000 | 0.270 / 0.270 / 0.081 / 0.000 | **FAIL** |
| 43 | Yes | 0.867 / 0.947 / 0.900 | 0.743 / 0.000 / 0.784 / 0.649 | 0.676 / 0.000 / 0.784 / 0.405 | **FAIL** |
| 61 | Yes | 0.773 / 1.000 / 1.000 | 0.351 / 0.622 / 0.541 / 0.784 | 0.351 / 0.405 / 0.541 / 0.676 | **FAIL** |

**Seed 17** is ineligible because its 6-loop development proof-success rate is 48.0%. **Seeds 29, 43 and 61** are eligible but fail required deep-family gains or absolute-success/early-STOP gates. Seed61 shows a pronounced descriptive benefit from eight loops (depth4: 35.14%→78.38%; depth5: 35.14%→67.57%). Seed29 experiences the opposite behavior: two-loop cap16 succeeds on 52.70% of depth4 goals, while eight-loop cap16 solves none. This heterogeneity drives the failed cross-seed hypothesis.

### Two-loop Bellman-cap ablation

With all other architecture and sampling choices held fixed, moving from cost cap10 to cap16 changes the learned objective scale. The cap16 target guarantees a positive numerical ranking gap between any finite shortest proof and unreachable/dead-end action, because a valid proof adds at most 15 distinct atoms. This **label property does not guarantee learned success**.

| Seed | Depth4 cap10 → cap16 | Depth5 cap10 → cap16 |
|---:|---:|---:|
| 17 | 0.446 → 0.000 | 0.405 → 0.000 |
| 29 | 0.135 → 0.527 | 0.081 → 0.270 |
| 43 | 0.703 → 0.743 | 0.649 → 0.676 |
| 61 | 0.514 → 0.351 | 0.541 → 0.351 |

Seed29 gains from cap16, seed17 loses heavily, and seed61 also deteriorates. Target separation is mechanistically necessary for a uniform cost-ranking guarantee, yet representation learning and optimization remain limiting.

## Recurrent causal-cone measurement

The goal-only STOP, reachability and distance heads depend on its goal-atom state. Each native cell mixes atom→rule→atom along incidence-masked edges, so after L calls the goal readout can depend only on initial atom features within undirected incidence radius 2L. Three fixed-graph pairs toggle a fact that changes the true goal reachability; their graph distances are exactly 8, 10 and 12 edges for depths 3,4,5.

| Dependency depth | Incidence distance | First possible recurrent calls |
|---:|---:|---:|
| 3 | 8 | 4 |
| 4 | 10 | 5 |
| 5 | 12 | 6 |

All **60** saved two-input probe records were replayed independently from trained weights. Every observation outside the radius had identical goal-only heads to tolerance. When within the radius, STOP readout differences were often minute; in seed61, the depth5 difference was 0.000144 at six calls and 0.000733 at eight calls, while L=2 gave exactly zero. The graph allows influence, and a learned network can still attenuate it strongly.

**This isolates a causal-transport bottleneck.** Extra recurrence changes the computational graph’s receptive field, while correct long-path influence remains training-dependent. Because individual rule logits read other graph locations, exact GOAL STOP equality does not imply that the full legal action-vector is equal.

## Independently checked evidence

| Quantity | Checked result |
|---|---:|
| Original host episodes | 13,312 |
| Trained forward replays | 297 |
| Independent BFS Bellman label checks | 656 |
| Paired causal-cone replays | 60 |
| Host false accepted proofs | **0** |

The independent checker reconstructs physical Horn edges from the actual public input matrices, computes full BFS shortest proof distances, validates every executed rule’s premises/head, verifies original goal/budget and all saved per-action logits, loads each checkpoint against an independent NumPy named-tensor export, and replays saved steps. Training cost receipts are tested at both registered caps. Both regular and optimized Python modes passed eight tests each. A mutation test altered a saved enabled-rule list and recomputed the file hash; the semantic checker rejected it.

An abstract Bellman cap lemma and the numeric outside-radius corollary are included in `experiments/activation_funnel_af18/CausalCone.lean`. **Lean compilation is pending** because the hosted runner failed before execution; the graph-locality argument itself follows from the verified incidence-masked cell source and the independent numerical probe.

## Follow-up

An AF-19 study could test a readout that aggregates rule-state summaries, rather than relying entirely on the goal atom, with a parameter/FLOP-matched control. Another useful intervention is training directly on paired goal-reachability contrasts at varying graph distances, measuring whether distance-12 influence can become numerically robust without breaking the host soundness invariant. These changes require a separate frozen protocol.

**Keep PR #93 draft and unmerged** until review and hosted Lean/Torch verification. The earlier AF-01–17 reports and their negative results are unchanged.
