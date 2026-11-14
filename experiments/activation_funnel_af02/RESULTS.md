# AF-02 scored results and CI evidence

**Experiment:** addressed continuation preservation in the native TypedLoop cell  
**Registered:** 2026-11-14, before scored evaluation  
**Frozen code:** [`94bb2be`](https://github.com/MoralityLabAI/LoopedTransformers/commit/94bb2be03be7534ddf768b130a02f5e3edc07605)  
**Parent:** [AF-01 PR #34](https://github.com/MoralityLabAI/LoopedTransformers/pull/34)  
**Registered protocol:** [PROTOCOL.json](PROTOCOL.json), SHA-256 `f6c3a939a09183c265466d506de985ec1a92c76d2d57f50cdd5f3f984e224c41`.

## Status

The registered primary experiments and independent audit have completed. The CI reproduction has two successful jobs, including fresh training, held-out evaluation, Lean checking, inverse diagnostics, and independent numerical replay:

[GitHub Actions run 43556164814](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43556164814)

Scientific judgments differ from engineering integrity: the independent audit **passed**; the protected-benefit hypothesis fails on eligible native seed 17 because the task-only codec already has 100% depth-context exact audit accuracy. Preserve all seed eligibility and negative results.

### Eligibility and held-out clean accuracy

Eligibility is fixed using both receipt orders and branch-audit accuracy on training and development data, separately for each teacher seed, before held-out scoring. All 18 teacher/task/joint model states were trained and frozen.

| Architecture | Seed | Source eligible | Depth task-codec exact audit | Depth joint-codec exact audit | Protected-benefit judgment | Delayed-preservation judgment |
|---|---:|---|---:|---:|---|---|
| GRU | 17 | No | 0.3241 | 0.9167 | Inconclusive | Inconclusive |
| GRU | 29 | **Yes** | 0.3148 | **1.0000** | **Pass** | Inconclusive |
| GRU | 43 | No | 0.3519 | 1.0000 | Inconclusive | Inconclusive |
| Native | 17 | **Yes** | **1.0000** | **1.0000** | **Fail** | **Pass** |
| Native | 29 | No | 0.5556 | 1.0000 | Inconclusive | Inconclusive |
| Native | 43 | No | 0.5926 | 1.0000 | Inconclusive | Inconclusive |

On eligible **native seed 17**, teacher and joint-codec task and exact-status accuracy are **1.0000** across depth-context, two-wait, and four-wait held-out families. Native addressed-pair causal preservation **passes** with source and joint selectivity both 1.0000; source and joint first-branch/other-branch preservation gaps are zero. Native task-only audit accuracy on four-wait inputs is 0.4444, demonstrating a delayed-status loss recovered by the joint objective on that seed. This is a registered delay-preservation observation, while the protected-benefit gate uses depth-context where the task codec had already saturated.

The GRU seed 29 meets the depth-context protected-benefit gate (+0.6852 exact-status improvement, without extra tainted attempts). Its delay4 teacher reaches 0.8889 task and exact-audit accuracy; joint exact audit is 0.7778, leaving the full delayed-preservation aggregate inconclusive under the preregistered rules.

**Aggregate gates reported by the scorer:** GRU protected benefit inconclusive, GRU delayed preservation inconclusive; native protected benefit fail, native delayed preservation inconclusive, and native addressed-causal preservation inconclusive. Per-seed positives must not be promoted to architecture-wide claims where the required development or informative-source gates fail.

### Exact evidence and audit counts

- 18 models, with matching Torch and named-array exports; 2,592 train/development prediction records.
- 10,800 original-host outcomes and 29,160 patched counterfactual rows.
- 4,570 kernel-certified submission rows through 240 distinct Lean proofs, under Lean 4.19.0; all 1,598 rejected submission rows remain in the receipt.
- 108 retained native Jacobians; 5,400 independently replayed canonical traces and 19,440 graph-layout runs.
- Independent NumPy absolute-error maximum: `4.641711711883545e-05`, inside its registered tolerance. Separate finite-difference controls cover all 80 ambient coordinates.
- 200/360 successful trials from the independent fixed-order native Lean baseline.

These are original-host proofs and counterfactual mathematical diagnostics in separate ledgers. A patched history receives no host-proof or attack credit against the original theorem. The local inverse reconstruction measures predecessor error and forward-return error separately; low forward-return error alone cannot establish reversibility.

### Artifact provenance

Exact GitHub Actions artifacts were recovered and their SHA-256 digests checked:

| CI artifact | Files in ZIP | SHA-256 | GitHub artifact ID |
|---|---:|---|---:|
| `af02-training-43556164814-1` | 50 | `a2520c013f478632b821c43c0ee971f44ba7667bfdaa1936e1f6ae7000279ca4` | 14097739321 |
| `af02-evaluation-43556164814-1` | 1,180 | `ca98d8b7d6981714d29e3138bf27a8f28ef58d40c8cc1d3ae321dc3c91a37c2b` | 14098702940 |

The evaluation artifact includes an embedded copy of the training tree. The two CI ZIPs are distinct from the 1,202-file **local primary-run archive** in the original handoff. Both are reproducible at frozen commit `94bb2be`; the CI manifest verifies that all 18 development and 54 held-out clean score records match the primary run.

The original GitHub Actions artifacts have a scheduled expiry of **2026-12-14**. Their exact ZIP bytes are now also preserved in the GitHub evidence prerelease linked below.

### Durable GitHub evidence release

The complete, original CI ZIP bytes are preserved within the private-repository [AF-02 evidence prerelease](https://github.com/MoralityLabAI/LoopedTransformers/releases/tag/af02-ci-evidence-94bb2be-20261114). The published TAR is **174,663,680 bytes**, SHA-256 `acbf2a382428cbe46526975f4fb77e2c6a232afbb4408db623e4d478e330fb01` as independently reported by the publication job and GitHub's uploaded-asset digest. It includes both ZIPs, each with its pinned SHA-256 and per-member digest roster, and this scored-results report. The release tag points to immutable scientific source commit `94bb2be`; the publication workflow ran successfully at [run 43562442454](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43562442454).

## Reproduction

See [README.md](README.md) for the pinned Torch/Lean runtime, the per-split reproducer, and the independent replay commands. The score source is `evaluation/evaluation.json` in the archived evaluation artifact; the gate and evidence integrity source is `independent_audit.json`. Original AF-01 work and its controls remain unchanged.
