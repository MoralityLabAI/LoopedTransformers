# NEC-03S: independently audited source-only affine-offset × sparse-code-budget factorial

**Completed 21 November 2026. Source-training in-sample mechanistic experiment.** Study code, eight original five-input model roots, two teacher arms, original source training groups 16–23 and the single primary endpoint were frozen at [cb91921](https://github.com/MoralityLabAI/LoopedTransformers/commit/cb9192133c55e1a49120e2bb277533e0ec58017a) before any new model-scoring. The first [preflight run 44911659634](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44911659634) failed **before scoring any roots**, because one test referred to an outdated source-lock field name. A versioned [test-only amendment 3806f55](https://github.com/MoralityLabAI/LoopedTransformers/commit/3806f5551cbeb2ab3c60a2a8963040dfc96fb948) preserved the original frozen scorer, sham policy, physical dose anchor and full ITT denominator.

[Successful hosted Actions run 44911823750](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44911823750) passed the corrected seven-test preflight, **8/8 independent source-root neural replay/audit jobs**, and the pure-standard-library eight-world ITT aggregate. Aggregate artifact ID **14709623021**, named nec03s-eight-root-source-offset-budget-audited-aggregate. Eight individual source root archives retain the exact two model arms, all 26 full recurrent-logit variants, raw 256 source truth labels per arm, active-code and angular-sham choices, four-way dose receipts, and independent audits.

## One preregistered primary contrast

The experiment uses only eight complete original source-training context groups 16–23 times 32 five-bit focal masks, **256 already-seen source inputs per original model arm**. Source/native checkpoints are original frozen NEC03O A/B reset1800 models and fresh checkpoint-specific 128-feature TopK16 source dictionaries with 40/40 source fit/rank split. No native or sparse autoencoder weights were optimized in NEC03S.

For the original native source-site n-code feature budget, one **exhaustively selected disjoint active-code angular sham** is chosen before any task label using the smallest maximum 32D physical gradient-cosine gap under both offset conditions, threshold ≤0.25. Source and that same sham, with and without the shared physical SAE affine offset, are **four donor vectors attenuated to the same single minimum of their physical output L2 norms at every input/visit**, without amplification. Unsupported or invalid matched controls at any recurrent visit yield identical hubs-only source and sham trajectories across offset conditions, with every input retained.

The primary per-world effect is the composed-minus-additive difference of the source-minus-sham **offset interaction** divided by 256, averaged equally over eight original world roots. Previously source-ineligible roots 3 and 5 contribute original preregistered ITT zeros. Minimum meaningful positive effect **+2 percentage points**. Wide-code-budget, full16 reconstruction, offset-only and hash-active random sham variants remain secondary descriptive conditions.

| Source world | Primary original-n interaction, pp | Case-level ITT numerator |
|---|---:|---:|
| dev5-000 | −1.171875 | −3/256 |
| dev5-001 | −0.781250 | −2/256 |
| dev5-002 | +1.562500 | +4/256 |
| dev5-003 | 0 | frozen source-ineligible zero |
| dev5-004 | +0.781250 | +2/256 |
| dev5-005 | 0 | frozen source-ineligible zero |
| dev5-006 | +0.390625 | +1/256 |
| dev5-007 | −1.171875 | −3/256 |
| **Eight-root ITT macro** | **−0.048828 pp** | **−1/(256×8)** |

**Exploratory two-sided exact root sign-flip p=1.0000**, with six nonzero root effects, conditional on independent-root sign symmetry. This is a negative in-sample source-specific-offset contrast, below the preregistered +2pp scientific floor.

Within the six source-eligible roots, the matched original-n physical-dosed correct counts were:

| Native teacher arm, six roots (1536 source cases) | Source no offset | Sham no offset | Source plus offset | Same sham plus offset |
|---|---:|---:|---:|---:|
| Composed | 1250 | 1249 | 1231 | 1229 |
| Additive | 1512 | 1510 | 1503 | 1499 |

Composed model source advantage increased by **+1 case** when the offset was introduced, while additive control source advantage increased by **+2 cases**. The registered composed-minus-additive interaction is therefore **−1 case** across the six admitted roots, divided by 256 within each root and averaged over eight roots. Original-n four-way dose and sham availability covered **1240/1536 (80.73%) composed** and **1458/1536 (94.92%) additive** source cases. All other cases remain in the ITT denominators with zero signed source-sham contrast.

The secondary wider budget min(2n,8) showed an average **+5/(256×8) = +0.244141 percentage points**, below 2pp, with effects of both signs across world roots. Random hash-active controls were preserved with their separately matched pairwise doses; those controls were never substituted for the preregistered angular sham.

## Full SAE reconstruction and physical offset interaction

Composed model aggregate correctness across **8 original source roots ×256 = 2048 already-trained input histories**:

| Intervention | Correct /2048 | Signed recovery of clean-minus-hubs gap |
|---|---:|---:|
| Original native clean | **1976** | clean reference |
| Hubs-only lesion | **1649** | zero reference |
| Frozen native nonhub clean restoration | **1976** | 327/327 |
| All 16 TopK feature decoder contributions WITHOUT source affine offset | **1808** | 159/327 = 48.62% |
| All 16 contributions WITH source affine offset | **1976** | 327/327 = 100% |
| Affine source offset ONLY | **1565** | −84/327 |
| Source original-n code donor, natural dose, no offset | **1716** | 67/327 |
| Source original-n code donor, natural dose, with offset | **1667** | 18/327 |
| Source wider code-budget donor, natural dose, no offset | **1760** | 111/327 |
| Source wider code-budget donor, natural dose, with offset | **1743** | 94/327 |

On additive models the original clean, hubs-only, native nonhub, all16-no-offset, all16-with-offset and offset-only counts were **2048, 1890, 2048, 1820, 2048 and 1754**, respectively. With all 16 active SAE codes and the source affine normalization/decoder-bias offset, source-trained composed and additive models both recovered the original clean source correctness on every scored training-source case. This full16 ceiling uses the donors' **natural physical dose** and is distinct from the physically matched, attenuated four-way sham tests.

The data establish a pronounced **offset × full sparse-reconstruction interaction**: all16 learned contributions plus the affine mean/bias reproduce the original native correctness, whereas the same contributions without that offset recover a smaller share of the clean-hubs gap. The affine offset alone may lower correctness relative to the hubs-only lesion. Smaller n-of-16 feature donors with matched physical dose did not exhibit a reliable feature-specific offset advantage over their equal-budget angular shams. The two observations refer to different dose conditions and cannot be interpreted as a single causal-mediation percentage.

The independent auditor recomputed every original source label, original frozen native/SAE checksum, physical source nonhub mask, feature-rank subset, exhaustive joint-minimax disjoint active sham, deterministic hash-selected random active control, all four matched donor output norms and attenuation gains, and **all 26 full 12-visit nonlinear recurrent logit trajectories** per complete source group. **All 8/8 root audits passed.**

## Inference and protected populations

This is **in-sample development source-training evidence**, with only eight independent world roots and correlated repeated masks, visits and model arms. The root sign-flip is exploratory. Neither source reconstruction nor a favorable descriptive budget comparison confirms population mediation or an independent-world sham tail.

Original [NEC03R source-only null](https://github.com/MoralityLabAI/LoopedTransformers/blob/main/mechinterp/neuron_communities/NEC03R_SOURCE_RESULTS.md) remains unchanged: original matched-sham macro **0pp**, exact exploratory p=1.0. Original NEC03Q causal-development macro remains **+0.016276pp**, exact exploratory p=1.0. The initial NEC03S failed preflight and later test-only amendment remain visible in CI history.

**Stop before population-scale sham production.** The 69 paired-source-eligible prospective world roots previously recruited under NEC03P have **zero newly measured causal-sham outcomes**. The 95th-percentile/95%-confidence sham tolerance requirement still needs at least 59 eligible independent world-root **causal sham outcomes** under a separately registered, scientifically resolvable intervention. All development-evaluation, reserved, newer recruited-world, 32 five-input embargoed and original 64 four-input confirmation datasets remained sealed.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
