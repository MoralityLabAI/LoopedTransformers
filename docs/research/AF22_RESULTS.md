# AF-22: Host-gated Bellman value iteration for small Horn proof policies

**Official scientific verdict: FAIL.** **Engineering and independent replay: PASS.** This report records the original preregistered eight-seed, three-policy evaluation, including the complete negative and positive seed-level results. All 10 jobs in [source-pinned GitHub Actions run #43788715688](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43788715688) passed on November 16, 2026.

**Frozen experimental source:** [`5c16b342323977b3045f34c9be4eb24113dc691e`](https://github.com/MoralityLabAI/LoopedTransformers/commit/5c16b342323977b3045f34c9be4eb24113dc691e). All training, scoring, numerical backup and independent audit code belongs to this unchanged source commit.

- Registered [`PROTOCOL.json`](../../experiments/activation_funnel_af22/PROTOCOL.json): SHA-256 `df8f6cec2a7c79931278e4cd0c01d8bba38204432b1e4988a83b425e4bb47d5f`.
- Held-out [`fresh_manifest.json`](../../experiments/activation_funnel_af22/fresh_manifest.json): SHA-256 `18b97f3351dd4addad38762aa95b9a3532acf66641edcbbc41111f95b1f28c33`.
- Independently aggregated **official `results.json`**: SHA-256 `9c37b107f0bf8f693da731b880e2f3b3830e2a727b95abd0b26cfa64fa77325b`, from GitHub Actions artifact `af22-results-43788715688` (artifact ID `14191511528`; artifact ZIP SHA-256 `06240ba5c084695a3609022f867366926008d2c12a8bc4f907831cad9b803804`).
- Draft [AF-22 PR #107](https://github.com/MoralityLabAI/LoopedTransformers/pull/107) is stacked on AF-21. This documentation commit does not change the registered experiment or any earlier AF-01 through AF-21 scientific verdict.

## Question and design

AF-21 found that training the actual greedy STOP margin and rule ranking did not consistently generalize to previously unseen Horn compositions. AF-22 asks whether **inference-time Bellman lookahead over host-verified successor states** improves proof completion using the same trained value model. Every accepted inference is checked by the host against the original Horn premises and proved-atom set.

Each of **eight independently trained seeds** (17, 29, 43, 61, 73, 89, 103, 131) produces exactly **one frozen 6,036-parameter TypedLoop checkpoint**, after 240 fixed AdamW updates. The model has four native recurrent calls per state, with the original graph-ancestor readout. All three policy modes reuse the same weights:

| Policy | Inference-time decision |
|---|---|
| **Direct** | Choose the highest-scoring legal rule/STOP from the neural policy's original logits |
| **VI-1** | Compare host STOP cost with one-step successor costs estimated by the shared neural value model |
| **VI-2** | Compare STOP with a two-step min-plus Bellman backup through host-legal successor states |

For a state `s`, let `p(s)=sigmoid(reach(s))` and `d(s)=clamp(softplus(distance(s)),0,16)`. The preregistered value boundary is

\[
V_0(s)=p(s)d(s)+(1-p(s))16,\qquad C_{\mathrm{STOP}}(s)=16p(s).
\]

The host overrides both quantities to zero for already-proved goals. For legal rule transitions `T(s,a)`,

\[
V_1(s)=\min\Big(C_{\mathrm{STOP}}(s),\;1+\min_{a\in A_{\mathrm{legal}}(s)}V_0(T(s,a))\Big).
\]

VI-1 ranks successors with `1+V0`; VI-2 ranks them with `1+V1`. Stable physical-slot tie breaking is frozen in source. The planner receives only the public Horn graph, current host-proved atoms and legal transitions, plus neural outputs. Exact BFS proof costs and goal reachability labels are used for training and independent scoring, never as inference features.

Training uses 928 existing AF-21 cases and 352 development cases. The **416 unseen evaluation cases** use fresh physical layouts 82–86: 32 novel-layout cases, 192 held-depth-4 cases and 192 held-depth-5 cases. Each deeper family has **80 provable goals**; percentages in the primary table are proof completions among those 80 under the exact shortest-proof budget plus one rule firing.

This comparison holds model weights fixed across inference policies. The larger successor graph explored by VI-2 gives it additional effective inference compute and access to a wider public-graph horizon. The source records planner-node and neural evaluation work, so the result is an algorithmic lookahead comparison rather than an equal-latency inference claim.

## Preregistered pass criteria

A seed qualifies when the **shared direct policy** achieves at least 65% provable-goal proof success on both original training and development inputs, with finite checkpoint weights and successful independent replay. **All eight seeds qualified**.

For **each** qualifying seed and **each** held-out depth 4 and 5, the preregistered primary gate requires:

1. VI-2 proof success exceeds Direct by at least **5 percentage points**.
2. VI-2 achieves at least **70% absolute** proof success.
3. VI-2 increases premature STOP on provable cases by at most **2 percentage points** versus Direct.
4. VI-2's proof success falls no more than **3 percentage points** below VI-1.
5. The host audit finds **zero false accepted Horn proofs**.

Any eligible failure makes the registered aggregate **FAIL**. No seed passed the full two-depth gate.

## Full official seed-level results

The percentages below come from the original source-bound `results.json` in successful hosted CI; they are **not** the earlier local Torch 2.10 diagnostics.

| Seed | Qualified? | Depth 4 Direct | Depth 4 VI-1 | Depth 4 VI-2 | Depth 5 Direct | Depth 5 VI-1 | Depth 5 VI-2 | Primary verdict |
|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| 17 | Yes | 72.50% | 72.50% | 73.75% | 70.00% | 45.00% | 68.75% | FAIL |
| 29 | Yes | 0.00% | 2.50% | 17.50% | 0.00% | 0.00% | 0.00% | FAIL |
| 43 | Yes | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | FAIL |
| 61 | Yes | 67.50% | 62.50% | 71.25% | 0.00% | 0.00% | 0.00% | FAIL |
| 73 | Yes | 67.50% | 50.00% | 66.25% | 50.00% | 32.50% | 40.00% | FAIL |
| 89 | Yes | 30.00% | 75.00% | **77.50%** | 7.50% | 25.00% | **40.00%** | FAIL |
| 103 | Yes | 15.00% | 0.00% | 25.00% | 7.50% | 0.00% | 0.00% | FAIL |
| 131 | Yes | 0.00% | 45.00% | 52.50% | 0.00% | 25.00% | 35.00% | FAIL |

**Seed 89** has the largest robustly observed relative gains: **+47.5 percentage points at depth 4** (24 to 62 solved of 80) and **+32.5 at depth 5** (6 to 32 solved). It meets the depth-4 registered gate, including absolute success, yet fails the required 70% absolute success on depth 5. Seed 131 improves from zero direct proofs to 42/80 and 28/80 with VI-2, also remaining below absolute success requirements.

**Seed 17** begins with the strongest direct generalization across both depths. VI-2 changes depth-4 success from 58/80 to 59/80 and depth-5 success from 56/80 to 55/80. Its depth-5 premature STOP rate rises from 15% (Direct) to 30% (VI-2), independently failing the STOP stability gate. **Seed 73** regresses from 54/80 to 53/80 at depth 4 and from 40/80 to 32/80 at depth 5. Seeds 29, 43, and 61 all retain zero VI-2 proof completion at depth 5.

### Aggregate descriptive effect, without hypothesis credit

Across eight seeds, each with 80 provable goals per depth, the **unweighted pooled counts** are:

| Policy | Depth 4 solved / 640 | Depth 5 solved / 640 |
|---|---:|---:|
| Direct | 202 (31.56%) | 108 (16.88%) |
| VI-1 | 246 (38.44%) | 102 (15.94%) |
| VI-2 | 307 (47.97%) | 147 (22.97%) |
| **VI-2 minus Direct** | **+105 (+16.41 pp)** | **+39 (+6.09 pp)** |

These are descriptive pooled results with large between-seed variation. The preregistered all-seed/per-depth criterion remains **FAIL**. Pooled improvement cannot substitute for depth-stable qualification and absolute performance.

### Premature STOP diagnostics

The most conspicuous failure mode is selecting STOP while a valid proof remains available.

| Seed | Depth | Direct premature STOP | VI-1 premature STOP | VI-2 premature STOP |
|---:|---|---:|---:|---:|
| 17 | 5 | 15.0% | 45.0% | **30.0%** |
| 29 | 5 | 100.0% | 100.0% | 100.0% |
| 43 | 5 | 100.0% | 100.0% | 100.0% |
| 61 | 5 | 100.0% | 100.0% | 100.0% |
| 73 | 5 | 40.0% | 67.5% | **60.0%** |
| 89 | 5 | 92.5% | 70.0% | **60.0%** |
| 103 | 5 | 77.5% | 100.0% | 100.0% |
| 131 | 5 | 100.0% | 75.0% | **65.0%** |

Some models obtain material gains from Bellman lookahead, including reduced premature stopping for seeds 89 and 131. Others are harmed when errors in successor-state value estimates enter the backups. The latter is a plausible mechanism suggested by the scores, rather than a separately proven causal diagnosis.

## Independent verification, exact scope

The hosted workflow completed **10/10 jobs**: source/manifest/Lean preflight, eight parallel per-seed training and evaluations, and the independent aggregate.

| Audited item | Official count |
|---|---:|
| Trained neural checkpoints | **8** |
| Host-checked proof episodes | **16,640** |
| Independently reconstructed planner-node records | **56,814** |
| Sampled exact Torch forward replays | **606** |
| False accepted Horn proofs | **0** |

The independent verifier reconstructs legal Horn transitions from the physical incidence graph; checks every scored source transition, STOP choice, terminal state and proof budget; recomputes the saved min-plus Bellman backups; matches Torch checkpoints to named NumPy exports; and replays sampled original model forwards. The aggregate rechecks per-seed source hashes, training receipt hashes and independent evaluation digests. A passed audit certifies the recorded finite Horn transitions and numerical ledger, while the neural value and policy generalization remain empirical.

Lean 4.19.0 compiled [`BellmanIteration.lean`](../../experiments/activation_funnel_af22/BellmanIteration.lean). The formal result states a **conditional** property: if an exact natural-number cost decreases by one at every nonterminal step, the process reaches cost zero in at most its starting cost. It neither certifies trained model predictions nor establishes autonomous Lean theorem proving.

### Original immutable GitHub Actions artifacts

All IDs and digest values below refer to the source-pinned [run #43788715688](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43788715688). Digests are GitHub's SHA-256 of each original artifact ZIP, distinct from the internal `results.json` SHA above.

| Artifact | ID | Original ZIP SHA-256 |
|---|---:|---|
| seed 17 | 14191616945 | `caf8e248d01692cc6823750081552fcd7037f37364295f79fd43b1845c4d2ec8` |
| seed 29 | 14190474655 | `4c1740864f85c9fc30ff5798cd840e2ae998c5c779ab3ddc3fa77e167db480bc` |
| seed 43 | 14191597065 | `dc37af9fee00df29b583fdd49f7dd3e58277141e81155b0443e4b04c92be0c2a` |
| seed 61 | 14190968541 | `ee6078f5ea8c4807b46757513c153f4ea8bfd3dcd1080a34fd9205f5170b43e7` |
| seed 73 | 14191247370 | `91cafb136324f9524282cd2fcb7c567d9eb63fac0a27c8a4100d1fb985edd6a3` |
| seed 89 | 14191421676 | `3ee4db5943e482e90ce005a900fab14351097f81284a694d049a7e29ee130b36` |
| seed 103 | 14191547246 | `377970d392d57f8aa5d218cfcfc8dffa03305ed09c757027621b4a97a9ab1ac6` |
| seed 131 | 14191965183 | `58765ba877c3da7d8689887196f6f05393295a487aac3b45fbfa29e0a4acdb35` |
| aggregate | 14191511528 | `06240ba5c084695a3609022f867366926008d2c12a8bc4f907831cad9b803804` |

The original CI artifacts are available through the workflow's artifact page subject to GitHub retention. A complete separate local source-bound TAR was also assembled from the original hosted receipts, without changing the official aggregate.

## Interpretation and next falsifiable test

AF-22 isolates inference-time planning from checkpoint differences: one source-trained value model produces direct, VI-1 and VI-2 decisions. The larger pooled VI-2 success and seed-89/131 improvements establish that legal successor-state lookahead can produce useful additional proof actions in this compact model. The depth-5 failures and heightened premature STOP for some seeds demonstrate that longer planning horizons do not ensure sufficiently accurate successor values.

A next experiment should **calibrate value estimates on the exact host-generated successor-state distribution**, separately score value error by backup depth and Bellman gap, then compare a learned confidence threshold for choosing between direct logits and VI-2 against a fixed, preregistered gate. Train and test the calibration on disjoint graph layouts, keeping the same host-owned Horn proof validation. This is a proposed follow-up, not a post-hoc modification to AF-22.

**Research status:** scientific **FAIL**; source-pinned hosted engineering **PASS**; all eight eligible seeds and their failed controls retained. Leave [PR #107](https://github.com/MoralityLabAI/LoopedTransformers/pull/107) **draft and unmerged** until reviewed.

## Permanent source-pinned evidence release | November 16, 2026

The [verified GitHub prerelease](https://github.com/MoralityLabAI/LoopedTransformers/releases/tag/af22-ci-evidence-5c16b34-20261116) preserves all **nine original successful CI artifact ZIPs** byte-for-byte. [Publication run #43790720456](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43790720456) passed independent checksum and member audits before uploading. GitHub's uploaded TAR digest matches the archive builder: **3,973,120 bytes**, SHA-256 `9221513b43ce48a151990e06e8392738fc474fbe0fb065adea817e1de5aec07c`. The release tag targets the immutable original scientific commit `5c16b342323977b3045f34c9be4eb24113dc691e`.

The [original-artifact manifest](AF22_EVIDENCE_MANIFEST.json) lists every ZIP/member SHA-256; the independent [Bellman value certificate](AF22_BELLMAN_CERTIFICATE.md), [retrospective held-out numerical audit](AF22_POSTHOC_VALUE_DIAGNOSTIC.json) and [replay script](af22_posthoc_value_diagnostic.py) were added after scoring. The retrospective calculation shows large successor/STOP boundary errors on several seeds; its certificate is a sufficient condition for correct first-action selection, not a retrained or preregistered result. The original registered **FAIL** verdict is unchanged. The GitHub release contains the previously committed scored report snapshot.
