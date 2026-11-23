# OLR-01 investigation: novelty audit, alias witness, and executable structural probe

**Date:** 2026-11-23  
**Status:** Investigative findings and deterministic correctness tests; no OLR neural training or Lean-certified OLR performance claim  
**Companion:** [OLR_01_OBLIGATION_LEDGER_RECURRENCE.md](OLR_01_OBLIGATION_LEDGER_RECURRENCE.md)

## 1. Re-read of the reported reward advantage

The published 1217 directed BPD repair counted 192 residual/budget scenarios from 32 roots. The native tactic-tree ceiling is 32 completions. The reward-trained two-slot arm completed 20, the matching imitation two-slot arm also completed 20, and full-history TRM completed five.

The utility difference between reward and imitation is **exactly accounted for** by action spending at the registered costs:

| Quantity | Two-slot reward | Two-slot imitation | Difference (reward - imitation) |
| --- | ---: | ---: | ---: |
| Completed scenarios | 20 | 20 | 0 |
| Target tactic calls | 192 | 170 | +22 |
| Charged sibling probes | 0 | 137 | -137 |
| Root utility | 0.079167 | 0.046354 | +0.032813 |

With each target call charged 0.025 and sibling probe charged 0.05, the derived utility difference is:

    [(137 × 0.05) - (22 × 0.025)] / 192 = 0.0328125.

The gain is a budget-management result for **the same successful proofs**. The *architecture* comparison with TRM remains substantial (20 vs 5 completions on this one frozen test), but the reward-versus-imitation comparison does not show additional proof-recovery capability. The candidate also reaches 20 of the 32 scenarios accessible to the registered native reference. The rest is headroom whose source needs a future error decomposition.

This changes the experiment priority: OLR-01 must include a *fixed-call, forced-action diagnostic* separate from the STOP/probe/allocation policy. Otherwise a future efficiency win could again be mistaken for better propagation.

## 2. Formal mean-pooling collision

Take four primitive symbols A, B, C, D with evidence A=true, B=true, C=false, D=false and two binary Horn rules producing target T.

**World W1:**

    A & B -> T
    C & D -> T

**World W2:**

    A & C -> T
    B & D -> T

Both worlds contain the same source facts, the same number of rules, equal arities, and the same multiset of primitive evidence entering T: [1,1,0,0]. An aggregator that collapses the incoming premises to their unstructured mean sees 0.5 in either world. A function whose sole relevant input is this pooled statistic cannot distinguish their targets. However W1 has a satisfied rule, while W2 has none. The AND-OR recurrence obtains T=reachable in W1 and T=unreachable in W2.

This identifies an information-loss mechanism **for that restricted pooling interface**, and motivates retaining premise-to-rule incidence. It does not prove that a full graph transformer with edge information must fail, or that adding a neural recurrent block yields a gain over exact symbolic evaluation.

## 3. Executable oracle and local verification

The branch adds:

- [ppt_gym/obligation_ledger.py](ppt_gym/obligation_ledger.py): positive-cost AND-OR hypergraph, synchronous min-plus fixed point, immutable source-fact bits, derived reachability, convergence trace and backward possible-obligation frontier.
- [ppt_gym/tests/test_obligation_ledger.py](ppt_gym/tests/test_obligation_ledger.py): ten deterministic tests of serial depth, conjunctive and disjunctive rules, cyclic support, known-inapplicable rules, shared-dependency counting, edge order, remaining-step limits and the pooling collision.
- [ppt_gym/tests/test_obligation_ledger_random.py](ppt_gym/tests/test_obligation_ledger_random.py): 400 seeded acyclic graphs compared with a separately implemented topological tree-cost evaluator.

**Local probe:** 11 test methods passed, including the 400-case randomized oracle comparison, using Python's unittest on the model workspace. The test run measures a deterministic structural algorithm, with no model weights, Lean theorem calls, independent certificates or performance comparison to existing trained architectures. Run inside the repository's research/lean_cgrt_hrm400 directory using:

    python -m unittest discover -s ppt_gym/tests -p "test_obligation_ledger*.py" -v

Additional existing PPT regressions and hosted CI should be checked before the branch is merged.

An important semantics restriction is explicit in the implementation: graph-derived facts represent reachability *under the supplied Horn fragment*, and do not become Lean kernel facts. A rejected tactic excludes or downgrades that action in the current proof state; it does not refute the logical proposition that action attempted to prove.

## 4. Novelty audit against prior art

The following ideas have existing literature and must be cited as antecedents:

| Prior work | Relevant result | What it means for OLR |
| --- | --- | --- |
| Zhu et al., **Neural Bellman-Ford Networks**, NeurIPS 2021, https://proceedings.nips.cc/paper_files/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html | Generalized Bellman-Ford with learnable indicator/message/aggregation operators on paths | Learnable semiring-like propagation on graphs is established, so OLR cannot claim its min-plus recurrence alone as new |
| Fang et al., **MC-HNN**, ICML 2026, https://proceedings.mlr.press/v306/fang26g.html | Multi-channel hypergraph learning and rank-collapse/semantic-dependency analysis | Hyperedge identity and multi-channel messages are also established architectural ingredients |
| Movahedi et al., **Fixed-Point Reasoners**, 2026, https://arxiv.org/abs/2606.18206 | Stable recurrent updates and convergence-based compute allocation | Halting at a fixed point is prior work; OLR should evaluate a semantics-based residual as an additional control |
| Liang et al., **Beyond the Training Horizon**, Sep 2026, https://arxiv.org/abs/2609.33144 | Recurrent latent content has a computational status (fresh/aged or live/consumed), and longer runs can lose usable state | Motivates explicitly measuring active/pending/consumed obligation channels, with causal interventions |
| Labovich, **Stability and Generalization in Looped Transformers**, latest Nov 2026, https://arxiv.org/abs/2604.15259 | Recall and normalization conditions for input-dependent fixed points | Inspires explicit re-injection of the immutable public proof context |
| Popescu et al., **Adaptive Depth in Looped Transformers**, 2026, https://arxiv.org/abs/2607.20519 | Halting gate and trained trajectory are intertwined; frozen-trajectory readouts can be competitive | Fit stopping after architecture training and retain the same saved recurrent trajectories for every stopping control |

The candidate research contribution is therefore **a task-facing coupling**: an AND-OR obligation ledger, an independently marked native-evidence ledger, a weight-tied semantic updater, and a strict accounting controller, all on live Lean proof transitions. These are components with antecedents whose usefulness as a whole still needs a direct experiment.

## 5. Critical limitation: symbolic Horn planning is already strong

In [PPT_HYBRID_RL_PENDING_RECOVERY.md](PPT_HYBRID_RL_PENDING_RECOVERY.md), the public Horn structural solver chose the certified branch on all 128 worlds. In [PPT_COMPUTATIONAL_GUARDS.md](PPT_COMPUTATIONAL_GUARDS.md), the exact arithmetic fold oracle certified all 80 worlds, while the learned Bellman two-slot actor completed 38. The former shows that a fully exposed graph can make learning unnecessary; the latter shows that symbolic reachability alone misses computational applicability.

Thus a benchmark built solely out of fully parsed, deterministic Horn graphs would mostly test whether the neural model can imitate the exact oracle. Keep these as **correctness controls** and train the neural extension on *partially specified and computationally conditional* proof transitions.

Three questions should decide OLR-01B:

1. Can the learned semantic channel recognize candidate tactics whose syntactic prerequisites look identical but whose delayed computational or dependent-type guards differ?
2. Does the obligation channel preserve the identity of pending and shared subgoals across a real, externally verified tactic step and bounded rollback?
3. At matched calls and FLOPs, does removing the obligation channel reduce **new certified completions**, and does an equal-dose matched margin sham leave the claimed direction-specific effect intact?

A small sidecar can operate on the public graph and legal menu while leaving existing Lean 4.19 action execution and proof certificates untouched. A learned applicability head needs a third status for unobserved/unknown actions, keeping timeouts and model failures censored.

## 6. Refined OLR-01B contract

Keep the experiment narrow:

**Inputs.** Public Lean goal(s), local hypotheses and binder scope, legal tactic candidates, their visible premise hyperedges when parseable, failed/native-accepted result history, remaining calls, and saved proof-tree state. Preserve exact source strings for unsupported arithmetic or dependent fragments and mark unparsed structure as unknown.

**Two recurrent channels.** A frozen, exact structural ledger for known Horn subgraphs and a trainable semantic/status ledger over the same edge identities. The learned model ranks tactics and predicts successor obligations. It cannot mutate native facts, source budgets or certificates.

**Dynamic computation.** Seed hidden-state recurrence from public input every pass, apply shared weights and small residual scaling, and report per-pass change in the structural and learned value states. Stop fitting separate compute-readout gates until the action trajectory is frozen. A stalled structural ledger does not imply the learned semantic layer is converged.

**Strong controls.** Source two-slot reward, newly fitted two-slot reward/imitation, structural oracle, direct learned guard reader, frozen-body OLR, shuffled hyperedges, head-only/linear calibration, plain tied loop, and a matched FLOP variant. Every model receives the same legally visible cues.

**Preflight and prospective cohorts.** Freeze enough new roots with independently witnessed nontrivial candidate outcomes before fitting or scoring. Stratify directed propositional joins, pending multi-goal proofs, opaque computational guards, and shared-witness dependent rewrites. Avoid saturated Horn/equality sets as the primary target. Account at the root level and by independent seeds.

**Next implementation gate.** Construct a native-facing adapter and a single paired forced-action test from newly generated 2-premise cases. Verify that an edge swap changes the oracle value, that the same candidate menu stays legal, that both worlds independently compile in Lean, and that all arms are scored at the same calls. Proceed to trainable OLR only after this gate.

## 7. Outcome of the investigation

The hard AND-OR structural component is easy to execute and test; its representation preserves an important distinction lost by unstructured incoming means. This supplies an algorithmic reference, not evidence of a superior looped transformer. The first scientific opportunity is in **uncertain applicability with delayed native feedback** and in **matching the cost of keeping multiple obligations live**, where exact Horn closure and ordinary candidate calibration leave genuine headroom.
