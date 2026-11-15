# AF-15: host-sound Horn rule policy, local scored results

**Scientific verdict: FAIL under the preregistered all-seed gate.** **Local engineering and independent audit: PASS.** This is a local Python 3.13 / Torch 2.10 CPU result. The registered hosted Torch 2.5.1 / Lean 4.19.0 run remains unexecuted because GitHub Actions failed before assigning a preflight runner on its original attempt and two retries.

- **Frozen scientific source:** [eedb938b7d8025801a7b107693ad7fb63b723331](https://github.com/MoralityLabAI/LoopedTransformers/commit/eedb938b7d8025801a7b107693ad7fb63b723331).
- **Registration:** [AF-15 PROTOCOL.json](../../experiments/activation_funnel_af15/PROTOCOL.json), SHA-256 `4b01129a6afc0c1a96975a939210128c421bd659525c3a5764c403a1f3f0b90f`.
- **Exact heldout manifest:** `experiments/activation_funnel_af15/fresh_manifest.json`, SHA-256 `3dc3f201ae6ea17b152e5fb982f99697740ab3c2292dc498e45788cbe70ba0ca`.
- **Independent local aggregate:** `results.json` from the checked local archive. Its registered verdict is `fail`, with `engineering_pass=true`, 6,560 verified episode rows and 91 sampled forward replays.
- **Original draft:** [PR #77](https://github.com/MoralityLabAI/LoopedTransformers/pull/77), based on `main`. Preserve the frozen AF-15 scientific source, all prior experimental outcomes, and the independent local evidence.

## Experiment

A 5,234-parameter native TypedLoop (width 16, two recurrent calls) ranks legal Horn rule firings. The **host** exclusively controls the proved-atom set and enforces a precise rule-eligibility mask: a rule can add a previously unproved head only when all its premises have already been proved. Every accepted inference preserves (S_0\subseteq S_t\subseteq S_{t+1}\subseteq S_\infty). Structural soundness is shared by every neural and symbolic baseline.

The `imitation` model maximizes log-probability on the set of exact shortest-goal legal actions. The `frontier` model uses the same target plus a pairwise legal-rule ranking margin. For each seed, these arms start from identical weights and see the same sampled training states and number of native-cell forwards. Each receives 240 AdamW updates. The ranking penalty adds backward arithmetic, so the runs are forward-call matched rather than total-FLOP matched.

Train/dev use the fixed AF-05 Horn templates (672 and 224 cases, with 1,644 training decision states). Prospective heldout uses 656 cases on six unseen physical graph layouts 32–37, including 256 depth-4 and 256 depth-5 inputs. These are new graph layouts of existing AND/OR Horn logic, rather than unseen logical operators. The host allows an exact minimum number of rule fires plus one extra action for provable cases.

## Development qualification

Training requires >=0.80 provable-goal success and development >=0.75 for both neural objectives separately. Every arm qualifies.

| Seed | Arm | Train proof success | Dev proof success |
|---:|---|---:|---:|
| 17 | Imitation | 85.45% | 85.45% |
| 17 | Frontier | 85.45% | 85.45% |
| 29 | Imitation | 85.45% | 85.45% |
| 29 | Frontier | 88.79% | 89.09% |

## Prospective scores

All success fractions are over provable goals under the registered action budget.

| Seed | Family | Imitation | Frontier | Gain | Fixed-rule order | Uniform legal action | Exact shortest oracle |
|---:|---|---:|---:|---:|---:|---:|---:|
| 17 | Depth 4 | 0.718750 | 0.718750 | 0.000000 | 0.664063 | 0.578125 | 1.000000 |
| 17 | Depth 5 | 0.718750 | 0.718750 | 0.000000 | 0.726563 | 0.531250 | 1.000000 |
| 29 | Depth 4 | 0.718750 | **0.828125** | **+0.109375** | 0.664063 | 0.617188 | 1.000000 |
| 29 | Depth 5 | 0.718750 | **0.812500** | **+0.093750** | 0.726563 | 0.617188 | 1.000000 |

The primary gate requires the frontier policy to exceed imitation by >=0.05 on both depths for **every eligible seed**, achieve >=0.80 absolute success, retain at least fixed-rule-order success minus 0.03, and earn zero false accepted proofs. Seed 29 satisfies the per-seed conditions; seed 17 fails the comparative gain and absolute success conditions. The cross-seed registered verdict is therefore **FAIL**.

On the small heldout OR/OR-to-AND composition slice, seed 29 frontier proves 6 of 9 provable cases, compared with 0 of 9 under imitation and 5 of 9 with fixed order. Seed 17 frontier proves 0 of 9. These nine-case secondary results are descriptive only.

## Independent source and proof audit

The local audit verified **6,560 host episodes** across the two seeds, their complete action traces and raw model logits, the full original premise/head incidence, the legal actions and post-action host state, checkpoint/source SHA-256 receipts, and matching Torch `.pt` / NumPy `.npz` exports for all four models. **91 sampled neural forward predictions** reproduced within the registered tolerance. **Zero false accepted proofs** were observed. A rehashed mutation of a proof-action record was rejected. All nine local unit/integration tests passed in standard Python mode, with a separately recorded optimized-mode nine-test pass.

The [abstract soundness file](../../experiments/activation_funnel_af15/RuleSoundness.lean) expresses host soundness/inflationarity. Lean 4.19.0 compilation remains outstanding because the [hosted workflow #43701257848](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43701257848) repeatedly failed before any runner executed a step. Green local engineering checks cannot replace that separate kernel or runtime verification.

## Research direction

AF-15 establishes reliable proof **validity** with a host-owned transition rule and measures the independent problem of learned proof-selection efficiency. The positive frontier gain in seed 29 is substantial within the given templates and budgets, while its absence in seed 17 demonstrates initialization sensitivity. A separately preregistered AF-16 should evaluate rule-priority transfer on new Horn rule compositions using more initializations and a compute-matched status/shortest-proof calibration. It should continue to hold the host transition gate fixed.

Keep this PR **draft and unmerged** until the source-pinned hosted environment can execute. Preserve every original proof trace and failed or ineligible scientific outcome.
