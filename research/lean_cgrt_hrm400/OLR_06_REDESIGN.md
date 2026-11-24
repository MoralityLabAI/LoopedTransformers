# OLR-06 redesign: exact receipt ledger, learned verification allocation

**Date:** 2026-11-24  
**Decision:** REDESIGN. Freeze OLR-METTA-05 as a negative architecture-selection result.  
**Status:** exact-reference preflight completed; proposed next scientific gate, pending review of the referee's full exported specification.  
**Parent:** [OLR yellow paper v0.1](OLR_YELLOW_PAPER.md), [05 registration](OLR_METTA_05_REGISTRATION.md), [04D audit](OLR_METTA_04D_CARRY_RESULTS.md).  
**Referee:** Fable review supplied by the investigator, with six accompanying scripts/log files. Its full Claude artifact and three outstanding questions were not accessible at this review snapshot. This document is our synthesis and proposed work order, not a reproduction of that unseen report.

## 1. Native evidence closes the 05 question

The completed [05 native workflow 45281855683](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45281855683) is pinned to bfdacac96882ea81b9efe11eb85dec46fd68f817. Its [artifact 14869777603](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45281855683/artifacts/14869777603) was downloaded and all 12 manifest-listed files independently hash-checked. Native toolchain, source allocation, fitted model identities, frozen decisions, native episodes and the standalone source remain in that artifact.

- 64 TRAIN worlds; 24 DEV and 24 five-obligation transfer worlds.
- 24 complementary evaluation pairs; 1,104 policy rows; 281 distinct native policy executions and independent replays; 48 independent grading replays.
- 48 independent axiom-free evaluation declarations compiled.
- A repeated model seed on the same paired world is not an independent theorem sample.

| Model | Parameters | DEV proof closures / 72 | Transfer closures / 72 | DEV correct addresses / 216 | Transfer correct addresses / 360 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Addressed recurrence | 5930 | 36 | 34 | 163 | 270 |
| Dense recurrence | 5930 | 8 | 2 | 109 | 187 |
| Explicit-map MLP | 746 | 72 | 72 | 216 | 360 |
| Erased addressed memory | 5930 | 7 | 4 | 108 | 180 |

The exact receipt map closed 24/24 distinct worlds in each panel. Addressed recurrence beat the poor dense control at matched registered computation, but the explicit-map MLP solved every query. The original architectural decision cannot rest on addressed versus dense alone.

The native scores match the referee's neural-only replay at the registered three seeds for addressed, dense and explicit-map models. The referee's additional variants used synthetic generator receipts, with no Lean in those runs:

| Arm, 160 updates | Parameters | Seeds exactly fitting TRAIN / 12 | DEV whole-vector accuracy | Transfer whole-vector accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original addressed | 5930 | 1 | .462 | .319 |
| Dense | 5930 | 0 | .122 | .056 |
| Learned dense gate | 6003 | 0 | .097 | .049 |
| Local event/slot attention only | 5930 | 12 | 1.000 | 1.000 |
| Plain neural KV table | 1538 | 12 | 1.000 | 1.000 |
| Explicit-map MLP | 746 | 12 | 1.000 | 1.000 |

At 2000 updates the original addressed arm fits all 12 source seeds, with .941 transfer whole-vector accuracy. The table already solves the task at 160. Completely zeroing attention in the original topology prevents event information from entering the slots, so that ablation is not informative. Restricting attention to self and the event token preserves the input path and removes the unnecessary cross-slot interference. These controls identify an unsuitable inductive bias for independent receipt storage, not a universal negative result about attention.

### 1.1 Correct the causal interpretation

Let c be the indicator that the unedited binary focal decision is correct, and f the indicator that an input edit flips that decision. Then the edited correctness c' satisfies

$$c-c'=f(2c-1).$$

For a local decoder that always flips the queried answer when its receipt bit is flipped, the expected accuracy drop is 2p-1, where p is its initial focal accuracy. The original task hard-codes which address and receipt are relevant; the target edit changes the implied answer while a nonfocal edit leaves that answer fixed. Native proof closure with oracle-completed nonfocal choices is a check of this input dependence. It cannot establish a new semantic propagation mechanism.

Native focal natural/target/sham counts were 52/20/51 of 72 in DEV and 57/14/59 of 72 in transfer. The DEV target drop is exactly 52-20=32, equal to 2*52-72. The transfer target drop is 43, close to the always-flip calculation 42. The algebraic identity is conditional on actual prediction flips; it is not an equality that every arbitrary model must satisfy with f=1.

The yellow paper's no-overwrite, observation-alias, sufficient receipt and conditional entropy propositions remain valid under their stated assumptions. They motivate the observation contract and storage separation. They do not select the learned attention architecture.

## 2. Fable's proposed two-direction feasibility pilot

The attached ledger_pilot.py defines a forest of latent signs w_i in {-1,+1}. A recorded relative receipt satisfies r_i=w_i*w_parent*t_i; t_i and r_i are public when the receipt exists. An absolute receipt fixes w_i. A component without an anchor is UNKNOWN. The pilot uses 20,000 TRAIN, 3,000 DEV, and 3,000 deep evaluation graphs.

The structured 33,123-parameter model embeds public rows, gathers parent messages, scatters child messages, then applies a tied residual MLP and layer norm. It uses the public parent relation, including edges without an admitted relative receipt, so the cell must learn to filter such messages. This is a bidirectional graph-message model, not yet the full dynamic native Lean architecture. The 29,763-parameter tied dense comparator and 194,739-parameter fixed-depth untied comparator use public parent/address indicators as input. Parameters and actual active compute differ.

Three-seed mean active-slot accuracy, computed from the recorded rounded per-seed logs:

| Arm | Deep T=8 | Deep T=16 | Deep T=32 |
| --- | ---: | ---: | ---: |
| Bidirectional incidence | .9957 | .9917 | .9900 |
| Deep-supervised bidirectional | .9957 | .9900 | .9867 |
| Dense tied | .8057 | .7413 | .6813 |
| Dense untied, fixed eight physical blocks | .7517 | .7517 | .7517 |

The training generator limits tree depth to 3, with observed receipt-to-anchor distance up to 5 in TRAIN. The deep panel has tree depth up to 9 and observed anchor distance up to 9. These are meaningful graph-size/depth diagnostics, not threefold extrapolation of every relevant path length. The bidirectional feasibility result is promising relative to the neural controls. More steps are not uniformly helpful: one bidirectional seed degrades on unknown components at T=16 and T=32. Deep supervision did not uniformly remove this issue. A recurrence stability and known/unknown calibration analysis is necessary.

The pilot is synthetic, with no native proof check. Receipt signs and ground-truth labels were constructed from a known consistent hidden assignment. The referee's "exact" ceiling used graph reachability and hidden signs in the label generator. A separate solver using only the exposed features is therefore an essential independent check.

## 3. Exact algebraic reference implemented and validated

[ppt_gym/olr06_exact_ledger.py](ppt_gym/olr06_exact_ledger.py), initial commit 1e6415874ef8c90ad3aa97b4af5d8dbe3add73a8.

For every admitted relative edge (u,v,s), s=t*r, require w_v=s*w_u. Within a connected component C choose a representative o, and assign relative signs q_o=1, q_v=s*q_u along known edges. Every cycle must have product +1. An absolute anchor a at vertex i imposes w_o=a*q_i. All anchor-implied representative values must agree.

- A consistent anchored component has exactly one satisfying signed assignment.
- A consistent unanchored component has exactly two assignments related by global sign reversal. Each individual node remains UNKNOWN.
- An inconsistent cycle or incompatible absolute anchors is CONFLICT. Such a component is quarantined as a ledger consistency failure; no native proof status is minted.

These are algebraic consequences of this restricted representation, not a general Lean implication system. The traversal is linear in vertices, edges and anchors apart from canonical sorting. Native proof authority remains outside the solver.

The independently implemented pilot adapter takes only seven-field public rows and parent pointers. It exactly reproduced all source classes (0,1,UNKNOWN,padding) on **26,000/26,000 graphs** from the attached pilot:

| Split | Whole graphs exact | Active slots | Max observed anchor hops |
| --- | ---: | ---: | ---: |
| TRAIN | 20,000/20,000 | 89,606 | 5 |
| DEV | 3,000/3,000 | 13,521 | 4 |
| Deep | 3,000/3,000 | 28,579 | 9 |

The committed implementation's measured local execution (including input conversion) was approximately .385+.053+.076 seconds for these three splits. This is an illustrative host-side runtime, not a matched latency benchmark.

[test_olr06_exact_ledger.py](ppt_gym/tests/test_olr06_exact_ledger.py), commit c533a7f1f3f924438fc03dd9f73f6eda859daa8d: all five local tests passed, including 400 randomized graphs compared with exhaustive assignments, input-order invariance, unknown/conflict handling, malformed parent input and 150 randomized budget plans compared with exhaustive action subsets. Local source and test Git blob hashes matched the committed files exactly (22a1377dc8fb8608438d264737e6043199c4c1f1 and bccf79173e0f3aabb777bab9e6dac80610302d49).

### 3.1 Static paid anchor planning is another exact baseline

If every offered query has a known positive integral cost and guarantees a consistent absolute anchor in its chosen component, one query resolves all nodes in that unanchored component. With no new edges or revisions, repeated queries in the same component bring zero additional coverage.

For component C, reduce all offers to its cheapest cost c_C. Let g_C=|C|, the number of newly resolved addresses. The optimal maximum-coverage policy for budget B is the knapsack

$$\max_{z_C\in\{0,1\}}\sum_C g_Cz_C,\qquad\sum_C c_Cz_C\le B.$$

The committed dynamic program computes this exact reference. Under equal costs, taking the largest unknown components suffices. Different utility functions, failed queries, correlated outcomes, edge discovery and revision changes invalidate the simple reduction. This reference prevents another synthetic "learned probe allocation" result from merely relearning known component sizes.

### 3.2 Cost algebra correction

For a fully observed positive-cost acyclic AND-OR derivation model, h_add counts derivation trees and can overcharge shared proofs. h_max is a lower bound for additive cost of a feasible proof DAG under the same edge-cost accounting; h_add is an upper bound on the optimum DAG cost if the selected proof tree itself supplies a feasible DAG. Neither claim automatically transfers to arbitrary Lean tactic cost. The shared-premise example gives h_max=3, optimal DAG=4, h_add=5.

Softmin over finitely many complete derivation trees has the free-energy expression

$$F_\tau=-\tau\log\sum_T e^{-C(T)/\tau},\qquad
0\le C_{\min}-F_\tau\le \tau\log|\mathcal T|.$$

This bound applies to the specified finite tree partition function. A cyclic fixed-point iteration, repeated counting via retain-old alternatives, or an ill-defined proof-DAG cost has different semantics. Keep any soft learned scores as estimates for action ranking; use exact verification and appropriate sound bounds for pruning.

## 4. Revised architecture proposal: verifier-owned ledger, learned incidence policy

Retire the neural memory write path for known facts. Persist exact scoped receipts in a versioned host ledger. A native receipt stores state identity, action and returned proof/diagnostic status, and its scope and dependencies. New observation states distinguish verified facts, rejected tactics, untried or censored checks, derived algebraic consequences, and conflicts. No model logits can promote a proof fact or revise a native state.

At a policy step, solve the exact known fragment. Collapse consistent known-edge components if helpful. Construct a typed obligation/rule/query graph with public candidate semantics, remaining budget, scope, available checks and exact component summaries. Keep the actual tool events distinct from internal recurrent visits. The controller has no hidden generator truth.

Let e_i^k be evidence/applicability features and d_i^k be downstream obligation demand at internal refinement step k. Let r denote a candidate rule or native check, with a typed edge token x_r, premise incidence P(r), and head h(r). A proposed shared cell computes messages

$$
u_r^k=\phi_\theta(x_r,\{e_i^k:i\in P(r)\},d_{h(r)}^k,b),\quad
v_r^k=\psi_\theta(x_r,e_{h(r)}^k,d_{h(r)}^k,u_r^k,b).
$$

The forward aggregator uses the correct premise incidence. A backward aggregator sends demand and candidate-specific uncertainty to unresolved prerequisites. A tied residual/gated cell updates the latent estimates with immutable exact ledger features reinjected each visit. Attention or permutation-invariant typed pooling may implement these operations; compare them at equal information and compute. Distinguish AND-premise compatibility from OR-rule competition. For the first graph-learning gate, keep the graph exact and public. Dynamic semantic binding is a subsequent experiment.

Known exact constraint closure runs deterministically. Learned outputs predict expected marginal verified utility, applicability likelihood, or useful information from unresolved checks. They only rank legal query, restore or proof actions. The host decides whether a proposed action is affordable and valid, records actual outcomes, updates the ledger and recomputes the exact fragment. A STOP that claims completion must still pass the native kernel.

This changes the scientific question from "can a neural network store a Boolean receipt?" to "can learned semantic scores and bidirectional refinement allocate costly verification better than direct scoring and exact structural search under the same information and budgets?"

## 5. Provisional OLR-06 work order

### Gate A: complete the referee reproduction and known-relation reference

- Preserve Fable's original script, logs, source hashes and exact neural seeds as a separately labeled external feasibility record. The independent parity solver and static anchor planner are now available.
- Audit consistent versus contradictory graphs, unknown component coverage, node permutation, known-edge deletion and false-anchor injection. Native-certificate identity must remain separate from algebraic status.
- Reproduce Fable's seven learned/control arms on the supplied generator before changing it; register any further training schedule. Record batch indices, hardware, actual forward/backward work, inference visits, wall time and per-seed known/unknown accuracy. Compare matched one-direction, two-direction, masked-edge and dense arms. Add typed edge messages with explicit known-relation masks. Pure parity accuracy is an implementation gate; an exact solver is expected to remain superior.

### Gate B: costly queries with genuine incomplete information

- Build a small independently enumerable problem class where a query may reveal an anchor, discover a signed relation, or return an inconclusive receipt. Some offered queries connect known and unknown components; others only join unknown components and become useful in combination with later evidence. A revision can invalidate a previous relation. Give public candidate metadata and expression features to all nonprivileged policies. The actual query outcome must remain inaccessible until the charged action.
- Freeze exact public-state transition/cost semantics, including a positive verified fact versus a failed tactic versus timeout. Ensure a failed query supplies no logical negation without a registered exhaustive two-branch argument. Record all attempted actions and censoring.
- Independently enumerate all reachable outcomes for small roots and compute a same-information finite-horizon expected-utility reference where the prior is known. A hindsight planner given hidden outcomes is a separate upper bound, never a deployable baseline. The deterministic anchor-only solver is the exact comparator on its restricted subfamily.
- Compare the learned incidence cell to (i) exact closure with a greedy expected marginal-gain reader, (ii) exact closure plus bounded lookahead, (iii) direct same-input semantic query scores, (iv) an explicit-history reader, (v) matched tied dense and masked one-direction cells, and (vi) the privileged upper bound. All arms use the same exact ledger updates and verifier interface.
- First evaluate on synthetic independent roots. Freeze training distribution, candidate expressions, graph topology and public observation signatures before any fresh score. Split by normalized relation expression, graph structure, evidence/anchor layout and revision regime; different random numeric identifiers alone are insufficient.

### Gate C: native Lean semantics and budgets

- Implement a small dependent-witness family in which actual native tactics can establish proposed relations between scoped witnesses or determine that a particular attempted relation is unsupported. Positive receipts require independently replayable Lean proofs of the relation. A verified equality/disequality or a complete enumerated two-branch relation may establish a signed relation in a precisely registered fragment. Other tactic failures are censored action outcomes.
- Expose all legitimate public goal/type information to the same-information controls. Include symbolic normalization, unification, exact graph closure and bounded Lean search. If an observation is intentionally masked to enforce a controlled task, label that restriction. A family that collapses to direct normalization or finite enumeration is an engineering preflight.
- Charge Lean attempts and actual solver work. Count all proof closure, focus/restore and native semantic probe operations separately. No extra private verifier access for the neural model. Freeze the source fit and policy before evaluating new native roots. Complete successful proofs independently compile and pass the registered axiom audit.
- Use cases with alternate branches, shared premises, revisions and ambiguous binding where correct probe allocation has meaningful headroom over the strongest nonlearned same-information policy. Keep a static-parity subfamily as a fail-safe calibration control.

### Gate D: prospective promotion decision

Primary endpoint: root-level certified proof completion under a fixed native-call cap. Secondary: completion minus registered costs, false STOP/unsafe proposals, calibration of unknown status and actual end-to-end compute. A separate synthetic task can use resolved obligation coverage, but cannot substitute for native proof completion.

Before score, register a minimum effect size, power/sample allocation (FSI-01), seed roster, paired-root uncertainty, multiple-comparison correction, and actual resource matching. A suggested development threshold is a five-percentage-point improvement over the strongest applicable same-information baseline on a fresh native cohort, with a positive root-clustered lower confidence bound, consistent direction across independently fitted seeds and no increase in invalid commits. Do not treat this suggestion as a finalized FSI-01 design. If no useful headroom exists after exact structural methods and direct semantic scoring, close the family rather than weaken the comparator.

For mechanism attribution, intervene on **edge incidence/direction or semantically relevant messages**, keeping native graph truth and legal menus fixed. Compare target edge effects with matched unrelated-edge and action-margin controls; evaluate both immediate predictions and native proof outcomes. A known-edge symbolic solver must remain correct under its original unedited inputs. Hard routing alone is an inductive bias; an input flip whose correct answer is known to flip does not establish a new learned mechanism.

No 5M model or new architecture claim until this gate demonstrates a reason to use learning on a nontrivial verifier-budget endpoint. For DataGoblin, a lightweight exact ledger is already useful independently of whether the neural mechanism succeeds.

## 6. Evidence custody

Native 05 archive SHA-256: 374fa6f217d3256b6ecd8ef839b65cabd667d5755be6c5a8451ec0ac03df3930.

| Archive manifest | SHA-256 |
| --- | --- |
| Bank | 69d5c66fa9da511d4b5297a627ff5754743628929671664bd6f878780ae80178 |
| Fit | de796b7ef7f9abd46c19d2cb6a472ee1052c03916e7614ca10f7e1910f8639ff |
| Evaluation | 9ec1faf3e2ad985e060a723527cebc2d31d8f021585ec3cc1679451ff1dfaf1e |

Referee-supplied files, independently hashed on receipt; external reports and scripts are not claimed to be native CI artifacts:

| File | SHA-256 |
| --- | --- |
| review_logs.txt | 98e8573ed4e9ad7ee0925f356984d1703afc49523e04845b5df4da69a881fe10 |
| replicate05.py | 779c60611e7447c1ee08acbca707209665d5e9e5991fdf91369140d503187dfe |
| variants05.py | 77230156952b3b471c8706158a34bbfee1bffe47b18599f7455bc9e91e81ab9f |
| variants05b.py | d3141b6b6b44dff74b8a6d15edbfa7c097156ee861e28fd1d68e83c727d5044b |
| softmin_check.py | 6cff00163fc4576ee2d0c6c84c5dc1850b15fcba1622b8e1e54fe02ba1e6637c |
| ledger_pilot.py | b25157ace05b48a81ce5627e3bdf959d90f690f689eb964fdae38cfc5868366f |

The full referee artifact could not be opened through the available reader. Obtain the Markdown export before asserting that the provisional work order above reproduces Fable's OLR-06 registration or answering its three unresolved questions.
