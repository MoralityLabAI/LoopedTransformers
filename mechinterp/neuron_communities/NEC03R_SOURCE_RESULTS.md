# NEC-03R: source-only clean-gap and sparse-donor recoverability

**Executed 21 November 2026. Evidence class: in-sample source-training-only mechanism diagnostic.** Original eight model roots, source context groups 8–15, 10 donor/control variants and the one-primary comparison were frozen at [8eebda1](https://github.com/MoralityLabAI/LoopedTransformers/commit/8eebda14828834d02dc44f13cf930984f2b02fb4) before scoring.

The initial [preflight 44910031420](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44910031420) was blocked **before any root experiment** by a missing local test import alias (4 tests passed, 2 test-only NameError failures). A new source-lock was committed at [bf35720](https://github.com/MoralityLabAI/LoopedTransformers/commit/bf357206e2e2f55d7445e8e3f206ae05c47563d0) to correct that import; all original native/SAE intervention code, fixed roots and registered thresholds were unchanged.

[Successful hosted run 44910172341](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44910172341) independently audited **all 8/8** original root source-only jobs and the final pure-standard-library 8-root ITT aggregate. The aggregate artifact is **14709203491**, named **nec03r-eight-root-source-clean-gap-recovery-aggregate**. Raw source-group and independent model/SAE auditor receipts are attached as individual artifacts nec03r-source-root-0 through -7. Exactly 8 original training-source groups ×32 complete five-bit focal counterfactual masks = **256 previously trained input histories per arm**, two native arms paired per world. All ten source intervention variants' original raw logits and correctness were retained and independently replayed; no model or SAE training occurred.

## Composed-model source correctness

| Original source root | Clean | Hubs-only | Native nonhub restoration | Full affine TopK16 SAE | 16 code contributions only | Natural sparse ranked n | Matched sparse n | Angular active sham |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev5-000 | 233 | 194 | 233 | 234 | 220 | 208 | 197 | 197 |
| dev5-001 | 256 | 238 | 256 | 256 | 256 | 247 | 243 | 244 |
| dev5-002 | 252 | 204 | 252 | 252 | 251 | 217 | 207 | 207 |
| dev5-003 | 241 | 222 | 241 | 238 | 229 | 222 | 222 | 221 |
| dev5-004 | 256 | 179 | 256 | 256 | 249 | 198 | 194 | 193 |
| dev5-005 | 223 | 208 | 223 | 224 | 226 | 212 | 209 | 207 |
| dev5-006 | 256 | 191 | 256 | 256 | 168 | 189 | 189 | 190 |
| dev5-007 | 256 | 214 | 256 | 256 | 221 | 222 | 215 | 215 |
| **Sum / 2048** | **1973** | **1650** | **1973** | **1972** | **1820** | **1715** | **1676** | **1674** |

The signed composed-model clean-minus-hubs gap is **323 correct cases**. Original clean nonhub restoration returns **323/323** (100%) of that signed gap. Full affine TopK16 SAE recovery returns **322/323 (99.69%)**. Sixteen positive active decoder-code contributions alone return **170/323 (52.63%)**, natural ranked n-code donor **65/323 (20.12%)**, angle-dose-matched sparse source donor **26/323 (8.05%)**, and the physically matched angular sham **24/323 (7.43%)**.

The additive teacher aggregate is a separate paired control: clean/hubs/native/full-affine totals **2047/1856/2047/2047** out of 2048, with signed gap **191**, fully restored by native and affine SAE. The additive 16-code contribution-only donor scored **1819**, natural sparse ranked n-code donor **2016**, angle-matched ranked donor **1958**, and angular sham **1953**. Its original dev5-003 root had no source candidate and remains an explicitly ineligible original source instrument. The dev5-005 composed arm remains a previous source-native competence failure and is retained in the full diagnostic table.

### Physical interpretation

The full affine SAE reconstructs physical native GELU state as:

\[
a_{\text{affine}}(x)=\operatorname{decoder}(z(x))\odot \sigma+\mu.
\]

Here the decoder includes its learned bias and the source-fitted normalization uses mean \(\mu\) and scale \(\sigma\). The contribution-only TopK16 donor includes only the sparse decoder feature columns:

\[
a_{\text{features}}(x)=z(x)\,[W_{\text{decoder}}^{T}\operatorname{diag}(\sigma)].
\]

The omitted physical offset is \(\mu+b_{\text{decoder}}\odot\sigma\). The source-ranked n-code donor keeps only n of 16 active learned components at each original source-selected recurrent visit. Low overall TopK16 normalized reconstruction MSE therefore fails to guarantee that the isolated n-code feature contribution reproduces the clean recurrent GELU state or its 32D MLP-down effect.

The independent auditor reconstructed original 64D GELU/32D MLP-down affine and contribution-only errors, native nonhub and sparse natural L2 doses, shared attenuated source/sham dose, all exhaustive disjoint active-code shams and deterministic hash-selected random shams. It independently replayed complete 12-visit nonlinear logits and correct task labels for every source input. Full SAE and native donors use **natural doses**, while original ranked-source versus angular-sham controls use **equalized attenuated physical output dose**. The recovered fractions are descriptive signed correctness differences and should not be interpreted as isolated component mediation fractions or population effects.

## Single preregistered source-only primary

The single registered root-level effect on **original paired-source-admitted roots 0,1,2,4,6,7** subtracts additive ranked-source-versus-angular-sham correctness from the same composed-model contrast, dividing by 256 original source cases per native arm. Historical source-ineligible roots 3 and 5 are **fixed zeros** in the eight-root ITT macro denominator.

| Root | Primary ITT effect in percentage points |
|---|---:|
| dev5-000 | 0.0000 |
| dev5-001 | −0.3906 |
| dev5-002 | +0.3906 |
| dev5-003 | 0.0000, source ITT miss |
| dev5-004 | +0.3906 |
| dev5-005 | 0.0000, source ITT miss |
| dev5-006 | −0.3906 |
| dev5-007 | 0.0000 |

The **eight-root source-only primary macro effect is exactly 0.0000 percentage points**, compared with the prespecified **+2.0-percentage-point meaningful floor**. Its two-sided exploratory exact root sign-flip **p=1.00**, with four nonzero root effects, conditional on independent-root sign symmetry. The strong direction-matched sham is specifically designed to track a similar physical gradient angle and matched output dose, making the contrast a stringent diagnostic.

This is a **negative in-sample source-only discrimination result**. The recovered clean state from full affine SAE is a useful engineering ceiling that identifies two separately testable design limitations: omission of the source normalization/decoder-bias offset and restriction to n of the 16 active code contributions, followed by source/sham attenuation. The current experiment cannot apportion the causal loss between those effects at a single matched dose. Any new offset/budget factorial needs its own registered source-only condition.

## Causal evidence boundary

All 8 source roots and 16 attempted native model arms remained in the denominator; independent audits passed all 8 roots. No original five-input development-evaluation or reserved context, no non-source context of the 88 recruited NEC03P worlds, none of the 32 five-input embargoed roots and none of the original 64 four-input confirmation worlds were scored. The previous NEC03Q developmental causal outcome of **+0.016276 percentage points, exploratory p=1.0** and its original first-auditor failure are unchanged.

The independent NEC03P recruitment identified **69/88 paired-source-eligible worlds**, yet none has a new causal sham root outcome. Source availability and full SAE reconstruction cannot establish a 95/95 behavioral sham-tail bound, which still needs at least 59 eligible independent causal-sham world outcomes under a separately frozen, scientifically resolvable contrast.

**FSI-01: REDESIGN_OR_PILOT_ONLY.**
