# AF-17: exact Bellman proof costs and calibrated STOP, November 15, 2026

**Registered scientific verdict: INCONCLUSIVE.** Every preregistered seed fails the comparison's joint train/dev source-qualification requirement. The local source-bound numerical and host-proof audit **PASSED** for every seed. This is Python 3.13 / Torch 2.10 CPU evidence. GitHub Actions' fixed Python 3.12 / Torch 2.5.1 CPU and Lean 4.19.0 preflight failed before executing runner steps and remains unverified.

- Frozen scientific source commit: [`89adc61cbef49b8879efbef8796e6e74208666b0`](https://github.com/MoralityLabAI/LoopedTransformers/commit/89adc61cbef49b8879efbef8796e6e74208666b0).
- Registered protocol SHA-256: `578f8e8664bf7930d66b5de940d85b0409553f846f3b29636afc055694e5f4aa`.
- Independent local aggregate JSON SHA-256: `01b98652acc3a7be6bbf5b46baa0d794e77ce1c2b203121644991b917aa1ebe2`.
- Draft [AF-17 PR #89](https://github.com/MoralityLabAI/LoopedTransformers/pull/89), stacked on AF-16 PR #82. Preserve the source and every scientific outcome, including failures and ineligibility.

## Registered comparison

Four arms share an unchanged 5,268-parameter native TypedLoop(16), identical source weights per seed and action inference, 16 original AF-15 states plus 16 new AF-16 composition states per minibatch, and a common initial 240-sample schedule. `imitation` uses 240 updates; `stop` adds explicit STOP-versus-legal classification and state-reachability supervision for 240 updates; `bellman` adds source exact capped action-cost and distance-to-go Huber regression for 240 updates; `imitation_extra` has **360** identical-style updates. Objective-specific backward arithmetic differs; the extra imitation gets **50% more** model forwards/optimizer updates.

The model observes public Horn graph incidence, initial facts, current proved atoms and the host's enabled-rule mask. The exact shortest path and Bellman costs are source-only targets. After model inference, the host alone can accept a rule with all premises already proved and a fresh head. All registered scores use the minimum legal rule-fire budget plus one action. Every accepted inference preserves the host's finite Horn closure.

The original AF-16 train/dev cases (928/352) are reused; train-only detour states expand supervisory coverage to **3,051 original** and **992 new-composition** `(case,state)` examples. Heldout includes **416** fresh physical graph/fact inputs, with 192 cases in each depth-4/depth-5 family, using new logical compositions and layouts 52–56. The input-hash manifest rejects AF-16 train/dev/previous-scored heldout overlaps.

## Eligibility

The frozen gate requires both `stop` and `bellman` to achieve at least **0.75 training** and **0.65 development** provable-goal success. Every seed is ineligible for the registered causal comparison.

| Seed | Imitation dev | STOP-only dev | Bellman dev | Extra imitation dev | Reason ineligible |
|---:|---:|---:|---:|---:|---|
| 17 | 48.00% | 48.00% | **80.00%** | 48.00% | STOP-only below 65% |
| 29 | 48.00% | 48.00% | **77.33%** | 84.67% | STOP-only below 65% |
| 43 | 48.00% | 48.00% | **85.33%** | 53.33% | STOP-only below 65% |
| 61 | 85.33% | **84.67%** | 53.33% | 95.33% | Bellman below 65% |

## Heldout proof success

Numerators are solved **provable** Horn goals under the registered budget. Each of the two deep families has **74 provable of 192 total** cases. Reported cells show depth4 / depth5.

| Seed | Imitation | STOP-only | Bellman | Extra imitation | Fixed rule order |
|---:|---|---|---|---|---|
| 17 | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 67.57% / 89.19% |
| 29 | 0% / 0% | 0% / 0% | **18.92% / 16.22%** | 10.81% / 13.51% | 67.57% / 89.19% |
| 43 | 0% / 0% | 0% / 0% | **24.32% / 24.32%** | 10.81% / 2.70% | 67.57% / 89.19% |
| 61 | 45.95% / 54.05% | 51.35% / 54.05% | 0% / 0% | **54.05% / 56.76%** | 67.57% / 89.19% |

The registered successful Bellman result would require, in **every eligible seed**, a 5-point gain over STOP-only, a 2-point gain over extra imitation at each deeper depth, absolute success ≥75%, no more than 5% early STOP on provable goals, no >3-point shortfall versus fixed order, and zero falsely accepted proofs. Those conditions are unsatisfied, while all seeds fail eligibility. The scientific aggregate remains **INCONCLUSIVE**, with no positive claim inferred from descriptive seed gains.

### Early-STOP behavior

The fraction of provable depth4/5 cases whose **first** model decision is STOP is identical across these two deep families, since each graph pair has the same immediate goal neighborhood. In seed29, Bellman stops immediately on 70.27% of provable cases and succeeds on 18.92% / 16.22%; the extra-imitation control stops immediately on 83.78%. Seed43 Bellman stops immediately on 72.97%. Seed17's Bellman policy stops immediately on every provable case, despite 80% development success. Seed61's Bellman policy also stops immediately on all provable heldouts, while its STOP-only arm succeeds on 51.35% / 54.05%.

Seed29's recovered depth4 proofs come from 14 of 32 provable `contingent_goal` cases. Seed43 solves 4/24 `serial_shortcut`, 2/18 `dual_fallback`, and 12/32 `contingent_goal` cases. The model remains substantially below fixed rule order, demonstrating limited transfer of rule costs across composition and proof depth.

## Independent evidence

The per-seed independent verifier recomputed public Horn rule incidence, legal/illegal transitions, shortest budgets, BFS reachability, source Bellman costs and recorded action/logit consistency. It separately loaded the actual Torch checkpoints, checked each tensor against named NumPy exports, then replayed sampled forward predictions. Complete aggregate receipts:

- **16 frozen models**: 4 seeds × 4 training arms, plus 16 named-array exports.
- **11,648** original-host scored episodes across all four seeds and 7 scored policies/baselines.
- **159** sampled checkpoint forward replays.
- **328** deterministic sampled independent Bellman-BFS cost-label verifications.
- **Zero false accepted proofs**, with every individual rule firing checked against public incidence.
- Nine source-level tests passed in standard and optimized local Python modes, including a full smoke train/evaluate/independent replay and detection of a rehashed altered host episode.

Local results are archive-bound by `AF17_local_evidence_manifest.json`, including each original file SHA-256 and a final ZIP-level SHA-256. GitHub Actions [run #43712445248](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43712445248) failed preflight before any steps in its original run and retry; its pinned runtime and formal Lean compile remain outstanding. The source and preregistered gate were frozen before the full local scientific evaluation.

## Mechanism follow-up

The post-hoc [AF-17 causal receptive-field analysis](AF17_RECEPTIVE_FIELD.md) establishes an architectural information limit of the **two** source-hash-pinned TypedLoop calls at the goal. Paired Horn inputs with different ground-truth goal reachability but the same neighborhood around the goal produce *identical* STOP/reachability/distance readouts. In **48 frozen-checkpoint paired replays**, the signed difference of each readout is exactly zero.

The action-cost target also clips distances to 10. Eight provable depth5 inputs require **11** rule firings; their optimal source-rule costs are clipped to the same value as unreachable/dead-end rule costs. This erases cost-margin supervision precisely in those cases. A new, prospectively registered study should vary the native loop count and the target cost cap, while holding the original host soundness gate, corpus splits, training budget and checkpoint qualification fixed.

**AF-17 remains draft and unmerged.** The local negative/inconclusive evidence does not imply the hosted source run passed.
