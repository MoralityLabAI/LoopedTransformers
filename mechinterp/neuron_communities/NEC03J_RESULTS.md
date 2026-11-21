# NEC-03J | Native GELU Jacobian directions and twelve-visit nonlinear transport

**Executed November 21, 2026. Evidence class: read-only, posthoc mechanistic diagnosis on the same eight previously opened NEC-03H-v2 Boolean teacher worlds.** The first PEF-07 research stack was merged to `main` via [PR #67](https://github.com/MoralityLabAI/LoopedTransformers/pull/67) after its security audit was repaired and all six final checks passed. NEC-03J builds on the integrated source without retraining any native or SAE model.

## Provenance and execution

The [source-locked research protocol](NEC03J_PROTOCOL.md) and [exact source/checkpoint policy lock](NEC03J_LOCK.json) specify the 64-D native GELU and 32-D post-MLP-down differentiation contract, clean-argmax fixed margin orientation, source-only finite differences, hubs-only lesion baseline, four previously frozen active-feature sham controls and full twelve-visit memory drifts.

The exact historical source is the eight development-world native/SAE checkpoint artifacts from [NEC-03H-v2 run 44878411694](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44878411694). Original hash-scrambled, disjoint *active* TopK16 control outputs are from [NEC-03I-B run 44885003105](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44885003105). All original source manifest, checkpoint, 16-state truth-table, SAE dictionary, locked ranking and raw donor output hashes were verified before new gradient analysis.

**[Final successful CPU run 44890958899](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44890958899)** passed the source-only Jacobian unit suite, all eight root-level nonlinear/gradient replays, all eight independent raw audits and the pure-stdlib eight-world aggregation. It archived **1,536 original-model variant/context records × 16 focal masks = 24,576 correlated focal-case records**, including the full twelve-visit 32-D memory-state drift sequences and all local 64-D/32-D directional gradient claims.

The final aggregate artifact is **`nec03j-eight-world-direction-jacobian-transport-audit`**, ID **14702046830**, ZIP SHA256 **c7591889a8c91a0b8b30d677e438252519b8243d4a38e7ef981b04d589998a7b**. Eight source-locked raw transport artifacts are archived on the same run, named `nec03j-transport-root-8` through `root-15`.

### Build repair history

The initial source-only differential tests passed, but [run 44890415751](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44890415751) stopped at the first development context because the replay module lacked an import of the clean GELU activation collector. This produced no new scored outcome and was fixed with a source-regression test and an updated source lock. [Run 44890604722](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44890604722) completed the actual nonlinear/gradient forward traces but failed the independent auditor because it expected a source SAE checkpoint hash in the derived report instead of the original frozen source manifest. The auditor was corrected to read that authoritative manifest, with a further source lock revision.

Both failed runs are preserved. The physical measurement definition, clean-chosen margin, original source models, four donor controls, finite-difference tolerance and all original primary/ITT outcomes remained unchanged across these engineering repairs.

## Jacobian and nonlinear finite-distance measurements

The local derivative target is the *clean predicted-class margin*, with class orientation held fixed across clean, hubs-only, source and control trajectories. It is label-blind as a derivative-selection procedure. We measure the native GELU memory-token gradient in **64 coordinates**, and the post-`mlp_down` gradient in **32 coordinates**. The identity

\[
\nabla_{a_t}m\cdot\delta_t
=
\nabla_{y_t}m\cdot W_{down}\delta_t
\]

passed on all audited selected source sites within the precommitted **0.001** maximum discrepancy. Centered ±0.003 native GELU finite differences on training-only source histories had a maximum absolute error **0.0011253**, below the registered **0.02** tolerance.

The hubs-only Jacobian's first-order directional prediction agreed in sign with the **exact, full nonlinear source-versus-control final margin difference in 21,980/24,576 focal comparisons (89.44%)**. The clean-model Jacobian's prediction agreed in **17,266/24,576 (70.26%)**. The results are across four correlated controls in each of eight source worlds; there are **eight independent teacher/world roots**, and the case-level fractions do not provide independent statistical significance.

The improvement in sign prediction on the **hubs-only reference trajectory** is mechanistically interpretable: the nonlinear forward intervention takes place in a hubs-only, locally suppressed recurrent path. An intact clean-state gradient is a different tangent point and is less representative of that damaged-system intervention. At finite donor magnitude, both tangent approximations still miss nonlinear interactions.

## World-by-world donor directions and recurrent memory drift

Each value below is the descriptive mean across **four correlated source-ranked-versus-hash-active-control variants** within the same original teacher world. “Signed gain” is additional correctly classified cases out of 768 for source-ranked active TopK16 compared with the disjoint hash active-feature donor at equal direct MLP-down output L2 dose. It is **not** a new world-level confirmatory endpoint.

| World | Site | Mean signed gain /768 | Mean local 32-D cosine | Hubs Jacobian sign agreement | Clean Jacobian sign agreement | Mean final memory L2 drift |
|---|---|---:|---:|---:|---:|---:|
| dev-008 | Graph | −1.00 | +0.040 | 83.9% | 68.8% | 2.180 |
| dev-009 | Graph | **+23.50** | **+0.114** | 84.9% | 59.6% | 0.985 |
| dev-010 | Fallback | −0.25 | −0.006 | 99.9% | 98.9% | 0.052 |
| dev-011 | Graph | +2.75 | +0.054 | 76.4% | 65.1% | 1.163 |
| dev-012 | Graph | +0.75 | +0.015 | 93.9% | 72.2% | 0.383 |
| dev-013 | Fallback | **−18.75** | **−0.100** | 89.7% | 75.7% | 0.959 |
| dev-014 | Graph | −8.75 | +0.104 | 91.1% | 56.0% | 0.951 |
| dev-015 | Graph | +0.25 | +0.008 | 95.7% | 65.9% | 0.375 |

The reference site and final memory state can differ substantially among task families, even at matched immediate local output norm. The average source-versus-control final memory drift is on the scale of 0.05 in dev-010 but exceeds 2 in dev-008.

## Focus comparison: dev-009 and dev-013

The source-ranked active TopK donor performed better than all four historical SHA-ranked active controls on the **graph-selected dev-009** world, with signed correctness gains **+30,+41,+15,+8**. Its mean physically matched local MLP-down donor-direction cosine was **+0.114**, with an average final source-versus-control memory-state distance **0.985**. Across variants, the hubs-only Jacobian predicted the sign of the nonlinear final margin difference on **74.7% to 89.5%** of focal cases.

For **fallback-selected dev-013**, source-ranked TopK lost **−6,−16,−38,−15** correct predictions relative to the same four controls. Its mean local directional cosine was **−0.100**, while final memory-state distance was **0.959**. The hubs-only Jacobian's prediction agreed in sign on **88.0% to 93.0%** of focal cases. The negative sign persists in the physically matched, substantially nonzero donor setting, and the local directional measurement is aligned with the observed margin displacement. This supports a mechanism hypothesis based on *which output direction the selected active features drive*.

The signed effects share a recurrent trajectory and the reference gradient only approximates the finite perturbation. The matched immediate 32-D L2 magnitude does not control nonlinear downstream memory transport.

## Where local gradient direction is insufficient

The **exact margin-change magnitude** often differs substantially from the sum of first-order local visit terms. Across four controls, the average absolute nonlinear remainder was **4.22 logit-margin units in dev-008** and **3.45 in dev-011**, versus **0.80 in dev-009** and **0.70 in dev-013**. The first-order hubs-gradient sign can be right while its magnitude is a poor approximation to the full multi-visit intervention.

The source donor's mean 32-D cosine also does not determine actual *classification correctness*: dev-014 had an average cosine of **+0.104**, yet source ranking lost about **8.75** correct predictions relative to its hash controls, averaged over four correlated variants. Classification flips depend on the initial decision margin, donor-specific nonlinear transport and interactions with the task's Boolean function and focal positions. Dev-010's near-perfect gradient sign agreement occurs with extremely small memory drift and a clean-model competency failure, another limitation of using a sign fraction as a success metric.

Across just these eight world means, the descriptive Pearson correlation between mean local cosine and signed correct-count gain is about **0.68**, a posthoc exploratory statistic with **no independent-world inference**.

## Next prospective mechanism program

The data suggest a *candidate direction-alignment mechanism*: source-ranked features are privileged in dev-009 partly because their post-MLP local donor displacement is more aligned with the damaged-system downstream margin gradient, while dev-013's source-ranked displacement points against the margin gradient. The robust posthoc sign agreement is worth carrying into a precommitted new-world test, with the following requirements.

A stronger null must match the **per-site direct MLP-down output L2 dose and local gradient-projection distribution** on a source-training population, alongside the number and activation of SAE features. It should compare source attribution ranking against direction-conditioned shams while preserving the original hubs-only injury and a complete independent source family. The design must report sign prediction and actual correctness effects separately, with nonlinear remainder and final memory drift gates to detect uninformative tangent approximations.

The experiment used **only the original eight already opened development roots**. All 64 SHA256-committed confirmation worlds and the 48 reserved contexts per world remain untouched. Neither 24,576 correlated focal cases nor the four hash variants provide the **59+ independent matched-sham teacher worlds** needed for the EvalMath 95/95 upper tolerance floor. The original NEC-03H-v2 intention-to-treat mean **+0.423 percentage points** remains below its fixed 2pp meaningful-effect threshold. **FSI-01 verdict: REDESIGN_OR_PILOT_ONLY.**
