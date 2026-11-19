# Deuterium: hosted replay audit and next scientific gate

**Status (19 November 2026):** Engineering reproduction passed; scientific verdict remains `NO_REGISTERED_ADVANTAGE`. The recovered native model's task-level steering remains `BLOCKED_ARCHITECTURE_PROVENANCE`. This audit is retrospective to the exposed test seeds.

## Evidence and exact provenance

- Review branch: `research/deuterium-port-actions-20261119`, PR [#211](https://github.com/MoralityLabAI/LoopedTransformers/pull/211).
- Frozen local registration commit: `f1e3e9ba663ea906a6468d76426bb7381a6e23ad`. Source protocol SHA-256: `c6bd49b7423091f02fd55812a51a24aae8f4f07332b3178e006a64a015f908a9`.
- First hosted archive audit: [run 44448038368](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44448038368), passed.
- First hosted steering replay: [run 44448038362](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44448038362), failed at the independently executing verifier because its original `git show` commands addressed the prior repository. Training, checkpoint preflight and held-out scoring had completed. This failure is retained.
- Corrected port head at inspection: `57fb49314a12387dbd181428a1b77718eafde714`, which adds the guarded relocation verifier while preserving the original verifier's bytes.
- Verified hosted steering replay: [run 44449171565](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44449171565), success. Artifact ID `14493919867`, `deuterium-steering-replay-*`.
- Verified hosted historical audit: [run 44449171571](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44449171571), success. Artifact ID `14494951200`, `deuterium-archive-audit-*`.

The Actions artifacts were downloaded and checked against the committed pre-port records. All **38** comparable regenerated historical/audit/paper output files were byte-identical to the source records. The hosted steering verifier independently checked **6,144 held-out rows** and passed **265/265 arithmetic and receipt checks**. The frozen source and upstream model/decomposition implementation are authenticated by the relocation wrappers; the underlying scientific functions remain byte-identical to the registered code.

## Hosted numerical comparison

Primary effect: selected rank-one, equal-dose, calibration-matched random steering *minus* selected learned-VPD steering on target-veto NLL. Positive means lower NLL under VPD. All rows were scored once per independent source seed on the previously disclosed evaluation sets.

| Seed | Clean held-out accuracy | Random - VPD target NLL (nats) | VPD collateral accuracy change | Scoring-region CPU time |
|---:|---:|---:|---:|---:|
| 103 | 0.977051 | +0.001377650 | +0.000710726 | 4.03 s |
| 211 | 0.971191 | +0.000628515 | -0.000703752 | 3.31 s |
| 307 | 0.967285 | +0.000663796 | -0.001450300 | 3.31 s |

Across-seed mean: **+0.0008899869669 nats**. Student-t 95% interval over the three independent model/decomposition seeds: **[-0.0001600500971, +0.0019400240309]**. This interval includes zero. The registered positive decision rule fails. The task-gradient baseline achieves lower target NLL than VPD in all three seeds as a descriptive secondary observation; its source training differs from the VPD calibration-search regime.

The earlier local mean was +0.0008904018517 nats; the hosted mean differs by -0.0000004148848 nats. Both runs satisfy the clean performance, rank, edit-dose and faithful replay prerequisites and both return `NO_REGISTERED_ADVANTAGE`. Timing excludes environment installation and recovery download. GitHub Actions green status means reproducible execution and independently checked arithmetic, **not method superiority**.

## Historical evidence interpreted at its proper level

- The six-seed replication contains **16 accepted grafts**, with per-seed counts 3/4/1/0/8/0. The older 12-entry summary was truncated. Every accepted graft tied its selected guarded random comparator on recorded task deltas; a final logit-movement tie breaker accepted it. This acceptance rule demonstrates neither strict task improvement over random nor a causal mechanism.
- The earlier, distinct full-arena summary's 23 accepts are report-only because its row-level sweep is absent.
- The four nominal ARC folds have substantial sample overlap; the held-out cards overlap their complement by 27, 28, 30 and 26 of 32 rows. The recorded cached rule gain from 44/62 to 48/62 must not be represented as fold-independent generalization.
- Wordle's 24/72 to 42/72 positive result is a constrained score-policy result with incomplete upstream puzzle lineage. It remains useful as an example of a *verified rule-level decision change*, not a parameter-steering validation.
- The historical governance cleanup reduced collateral accuracy to 0.0100 after reaching target accuracy 1.00. It is a strong negative control result for any proposed controller.

## Experimental critique and follow-up readiness

The 1,763-parameter recurrent gate study probes an actual recurrent model and a real parameter decomposition optimizer, but it intervenes in a small head matrix on an IID synthetic threshold task. Its three seeds give low across-seed precision, with the proposed dose of 1% of the head's Frobenius norm producing effects under 0.0014 nats. The comparison tests causal steering at that exact scale and selection rule. It supplies limited evidence about internal recurrent mechanisms, deep state transfer or task-general control. No new scientific result is created by the hosted rerun.

The 11,191,328-parameter recovered classifier passes the 91/91 frozen-weight equality check. Its original forward class, dataset/tokenizer implementation, and independent reference logits remain missing, so genuine clean task inference and parameter editing are blocked. Reconstruction from guessed architecture would create a *new* model, requiring a new validity argument and external reference outputs before it could support a historical claim.

For the next native recurrent study, use a **separately registered** protocol after selecting a public source checkpoint with executable native forward inference and an authenticated source task. The existing LOOPSCOPE-08/LGA-05 frozen 44,001-parameter four-pass TinyRecursiveModel and its source receipts are a candidate, subject to checkpoint and task-access preflight. Register independent new data seeds before opening any evaluation rows. First establish exact source checkpoint hashes, clean accuracy, forward-logit parity, intervention identity, and complete four-pass replay. Then compare VPD rank-one and rank-matched random/task-gradient parameter edits at equal realized Frobenius dose and equal calibration search budgets. Keep the recurring-state hook controls distinct from parameter edits; report depth-specific target effects, off-target degradation, false commits, dose response, runtime and matched-token/source effects. Use an across-model-seed confidence interval and preserve failed replay gates as explicit stop outcomes.

Source hash gates, clean inference and intervention parity take precedence over steering comparisons. The immediate missing input for the recovered large model remains original code and reference logits. Do not substitute the synthetic study, cached policy rule, or a green CI job for native-model evidence.
