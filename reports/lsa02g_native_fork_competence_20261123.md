# LSA-02G: native Lean fork-choice competence and shuffled-loss sham

**Date:** November 23, 2026. **Disposition:** **development screen FAIL**, native competence **BLOCKED_INCOMPETENT_CHECKPOINT**. **Code:** [PR #328](https://github.com/MoralityLabAI/LoopedTransformers/pull/328), merged as `5893889576185d75657789a8e8bd20b3bddef8b9`. **Primary completed CPU/native CI:** [GitHub Actions run 45201791177](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45201791177). **Compatibility-corrected second run:** [45202097050](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45202097050). All locked ordinary-test and OOD proof outcomes remain unopened.

## 1. Research objective and new supervision

The earlier [LSA-02F dosage result](lsa02f_pretty_dose_uniform_cpu_20261123.md) produced a best 96-byte-pretty Plan B completion of 7/32 on new Lean roots, versus 23/32 independently attainable and a 15.625% five-view study's three-stream random baseline of 15/96. This exposed an action-selection and competence shortfall. LSA-02G investigated whether **training-time emphasis on genuinely discriminating shortest-continuation tactic choices** improves later verification outcomes.

The native Lean 4.19 `lgym` source used a new theorem-generator seed `20261129`, freezing 64 independent training lineages and 32 disjoint calibration lineages. Real Lean exploration and a second standalone axiom-free compiler certified all **64 training reference proofs**, yielding **345 shortest-continuation training frames**. **57/64** shortest reference proofs fit the eight-tactic check budget. Of 345 train frames, **143** had a public action menu containing both shortest and nonshortest continuations. A label of zero means a tactic was not on a minimum-length continuation; it can still be a tactic Lean accepts.

The frozen teacher manifest SHA-256 was `ed37c2edb5e48b1fc85a359f2b2448ea239c299ee471c5b9b4a771754acc84f2`, associated with suite manifest `fdedbca60ffe6942827b2846734551b2fb5dc208b0150bfd2bf31cd046daefa8`.

## 2. Strict same-worker matched CPU experiment

Three conditions used *the exact same* Plan B architecture, **57,475 parameters**, 32-wide hidden states, two AB tied recurrent passes, byte encoder 384 and `pp96` public state. The only intervention was the train-only loss-weight map:

| Training arm | Source loss emphasis |
| --- | --- |
| `plain` | Existing uniform per-example loss, all source frames weight 1 |
| `fork4` | Weight 4 on the **143 verified train-choice forks**, weight 1 elsewhere |
| `sham4` | Weight 4 on **143 independently shuffled train frame identifiers**, weight 1 elsewhere |

The weighted and sham conditions divided loss by the common source-wide normalizer \(1+3(143/345)\). Their *aggregate source* loss mass matches by construction. The source-sham selection overlapped genuine forks in 52/143 boosted frames at seed 149 and 66/143 at seed 151, providing substantial nonfork weighting. A root-uniform, then within-root-uniform sampler was used for **720 optimizer updates per model**. Importantly, each model seed's exact sampled theorem roots and frames, initial tensors, architecture, trained AB order, public input, and optimizer budget were identical across all three arms.

Seeds `149` and `151` were trained on one GitHub-hosted CPU worker per campaign. Every arm was independently fitted **twice on the same worker** with pinned CPU Torch 2.5.1, deterministic algorithms, one thread and disabled MKLDNN: **12 fits per campaign**, with exact bitwise model-state, checkpoint-byte and optimizer-step loss-trace agreement for all six duplicates.

**Control limitation:** matching the boosted-frame count and *global* train loss mass does not guarantee equal numbers of boosted **optimizer draws** under the root-uniform sampler when roots have different frame counts. The archive records boosted optimizer draws for each cell; future sham designs should root-stratify the boost schedule or match effective per-root expected weight as well.

## 3. Independently sampled native random-menu proof baseline

Five prespecified public-only uniform-menu seeds, `17`, `31`, `47`, `61`, `83`, were applied to the **same 32 development theorem roots** under eight native Lean tactic checks each. The policy used no neural weights, generator hint, model output, or proof labels to select tactics. Every tactic verdict came from real Lean; completed terminal scripts passed standalone compilation and every model-free action transcript replayed.

| Native random seed | Standalone-certified whole proofs |
| --- | ---: |
| 17 | 2/32 |
| 31 | 10/32 |
| 47 | 4/32 |
| 61 | 3/32 |
| 83 | 5/32 |
| **Five-stream mean** | **24/160 = 15.00%** |

Both completed CI campaigns reproduced the same five-stream random baseline and original source hashes. These are **32 independent theorem roots with five repeated stochastic-policy draws**, rather than 160 independent theorems.

## 4. Real Lean full-proof completion

All six trained checkpoints were evaluated on **exactly the same 32 calibration roots**, with native kernel verdicts, independently compiled terminal proofs and separate model-free replay. The native, independently verified shortest-reference attainable ceiling was **26/32**, and the public-order `auto8` control closed **7/32** on this new cohort. The table uses the trained **AB** order throughout.

**First completed campaign, run 45201791177:**

| Seed | `plain` | `fork4` | `sham4` | Independently attainable |
| --- | ---: | ---: | ---: | ---: |
| 149 | 8/32 | 8/32 | 7/32 | 26/32 |
| 151 | 2/32 | 5/32 | 6/32 | 26/32 |
| Root-paired seed mean | **15.625%** | **20.3125%** | **20.3125%** | **81.25%** |

**Compatibility-corrected repeat, run 45202097050:**

| Seed | `plain` | `fork4` | `sham4` | Independently attainable |
| --- | ---: | ---: | ---: | ---: |
| 149 | 5/32 | 7/32 | 7/32 | 26/32 |
| 151 | 5/32 | 4/32 | 5/32 | 26/32 |
| Root-paired seed mean | **15.625%** | **17.1875%** | **18.75%** | **81.25%** |

The sole program change between the two CI campaign heads restored legacy unweighted training support for **singleton-only synthetic fixtures** in other PPT regression workflows. This change does not alter actual train examples containing forks, the register's loss formulas, or calibration data. The earlier and corrected runs used identical frozen teacher and generator hashes and seeded starts, but some final trained tensors differed across GitHub workers, consistent with the documented LSA-02C/D cross-worker CPU numerical sensitivity. **Both campaigns are technical repeats on the same 32 theorem roots, not new independent confirmation samples.**

## 5. Post-scoring native reference-fork audit

Only *after* all checkpoint proof evaluations completed, a separate job explored the 32 calibration roots in native Lean 4.19 and independently compiled their shortest reference scripts. **25 roots contained eligible discriminating reference-path forks under the eight-tactic reference ceiling, totaling 70 source states.** All frozen checkpoints then ranked the same public `pp96` candidates, with **zero optimizer updates**. Correctness meant choosing a Lean-verified minimum-length continuation. This is a *reference-trajectory* metric; it does not cover all model-visited, off-reference states.

Each theorem root's fork accuracy was averaged over its reference-path forks, matched across model seeds, and compared by a **descriptive 2,000-draw bootstrap over the 25 independent fork-bearing theorem roots**. Whole-proof contrasts use all 32 theorem roots.

**First campaign:**

| Paired fork-weighted contrast | Estimated difference | Descriptive 95% root-bootstrap interval |
| --- | ---: | ---: |
| Fork-action correctness, `fork4 - plain` (25 roots) | **−3.133 points** | −12.967 to +7.267 |
| Fork-action correctness, `fork4 - sham4` (25 roots) | **−4.633 points** | −12.500 to +3.400 |
| Full-proof completion, `fork4 - plain` (32 roots) | +4.688 points | −4.688 to +14.063 |
| Full-proof completion, `fork4 - sham4` (32 roots) | 0.000 points | −7.813 to +9.375 |

**Compatibility-corrected repeat:**

| Paired fork-weighted contrast | Estimated difference | Descriptive 95% root-bootstrap interval |
| --- | ---: | ---: |
| Fork-action correctness, `fork4 - plain` (25 roots) | **+0.700 points** | −8.600 to +9.200 |
| Fork-action correctness, `fork4 - sham4` (25 roots) | **−2.200 points** | −10.500 to +7.300 |
| Full-proof completion, `fork4 - plain` (32 roots) | +1.563 points | −3.125 to +6.250 |
| Full-proof completion, `fork4 - sham4` (32 roots) | −1.563 points | −10.938 to +6.250 |

The prespecified developmental screen required **at least +5 absolute points of independently verified fork accuracy over both plain and the matched source sham**, plus nonnegative full-proof effects. Its result was **FAIL in both campaigns**. All fork-accuracy intervals include zero and both runs find `fork4` below the shuffled sham in the primary fork-action endpoint. None of these findings supports a specific fork-weighting causal improvement.

## 6. Competence and scientific boundary

The best full-proof closure observed in the first campaign was `fork4 = 13/64 = 20.3125%`. The corrected repeat's best arm was `sham4 = 12/64 = 18.75%`. Both are below the registered **40% native completion competence floor**. Against the independently sampled five-stream uniform-menu reference of **15%**, their maximum observed margins were **+5.3125** and **+3.75 points**, below the **+10-point** required margin.

Thus `BLOCKED_INCOMPETENT_CHECKPOINT` remains the scientific disposition. The full Gym and model-scale promotion are gated. This negative result also remains restricted to generated Lean Horn problems; external proof corpus ancestry, Mathlib transfer and general mechanism claims have not been established. The separate `auto8` heuristic cannot substitute for the independent uniform-menu control.

## 7. Next falsifiable CPU research direction

The present supervision reweights **states already on a shortest reference trajectory**. It cannot teach the model what to do after its own earlier deviation, even though native proof episodes encounter such states. A prospective next CPU study can test **verified off-reference state coverage**:

Freeze a new train-only set of reachable Lean states visited under a previously frozen policy or controlled public-menu exploration. Ask Lean to verify the local tactic successor graph and shortest-continuation set for each state, and independently check complete proof scripts where possible. Train a matched, equal-compute learner against a source-only reference-path control and a **shuffled state-selection sham**, with distinct theorem root lineages across train and calibration. Score new native full proofs and reference-fork action correctness, preserving all independent source/precision/algorithm receipts. Include states whose distance to the goal exceeds the remaining tactic budget to prevent false solvability claims.

This is a prospective hypothesis, not a result of LSA-02G. Keep GPU work, sealed ordinary/OOD evaluations and architecture promotion paused until native competence improves.

## 8. CI evidence, files and artifact retention

The code and both completed GitHub Actions campaigns are available in:

- [Merged PR #328](https://github.com/MoralityLabAI/LoopedTransformers/pull/328)
- [LSA-02G preregistered source-and-sham protocol](../docs/lsa02g_fork_competence_protocol.md)
- [Source-only fork weighting implementation](../research/lean_cgrt_hrm400/ppt_gym/native_train.py)
- [Same-worker deterministic matched CPU trainer](../research/lean_cgrt_hrm400/ppt_gym/fork_train.py)
- [Post-proof native Lean reference-fork and root-paired auditor](../research/lean_cgrt_hrm400/ppt_gym/fork_eval.py)
- [First complete CI run 45201791177](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45201791177)
- [Compatibility-corrected complete CI run 45202097050](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45202097050)
- [Corrected-run native train corpus](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45202097050/artifacts/14832956828)
- [Corrected-run exact-fit checkpoints](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45202097050/artifacts/14831633705)
- [Corrected-run uniform-menu control](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45202097050/artifacts/14831234842)
- [Corrected-run native reference-fork and paired-result receipt](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45202097050/artifacts/14834170430)

Every artifact has a SHA-256 custody manifest and the workflow registers **14-day retention**. This Markdown report preserves the denominator, scientific decision and exact source identities after the ephemeral artifacts expire. Both full CI campaigns and the relevant existing PPT Gym CPU/Lean regressions completed green on the corrected PR head.
