# NEC-03U: verified retrospective NEC03T teacher-signed logit-margin audit

**Completed 22 November 2026. Evidence class: retrospective, in-sample source-training archive analysis.** All eleven model intervention modes were already evaluated in [NEC03T original Actions run 44913605626](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44913605626). NEC03U changes none of those native models, SAE weights, intervention sites, source feature ranks, target labels, original readout or raw archives. The margin method and fixed full-affine-margin bins were byte-frozen at [NEC03U commit 2b1e517](https://github.com/MoralityLabAI/LoopedTransformers/commit/2b1e5177a80637daf339ba26387b4f8fb6d4fb5d), before this new retrospective margin computation.

The **[successful hosted archival audit 44934901309](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44934901309)** passed its standalone standard-library method tests and the complete archived-data audit. Its verified public output is GitHub Actions artifact **14718948612**, named **nec03u-readonly-nec03t-eight-root-margin-audited**, containing the complete original-source analysis aggregate and all **4096** recalculated case-arm margin records, with a new canonical JSONL SHA256.

## Original source custody and sanity checks

Only the eight original NEC03T root artifacts from that prior run were read, alongside its independently audited eight-root ITT aggregate artifact **14711051887**. Each original root scorer JSON, original independent audit JSON, composed raw case JSONL, additive raw case JSONL and the old ITT summary passed the independently pinned **SHA256** checks. Native model/SAE checkpoint hashes are preserved and indirectly supported by the original independent neural audits; **NEC03U never loads those weights**.

Within each of **eight** world roots and **two** native model teacher arms, source-context groups **24–31** and their complete 32 focal masks contributed exactly **256 already-trained cases per arm**. Every original 11-mode binary-logit pair was finite, and re-derived class predictions matched its saved prediction and original source scorer/audit correct counts. All original physical direct-output dose summaries, original disjoint active code IDs, sham candidate cardinalities, nonamplifying gain receipts and invalid case zero-contrast masks were checked against the intact archive. Root 3/5 original paired-source-ineligible ITT zeros were preserved.

The original pre-registered NEC03T **correctness** result was recovered unchanged: **+1/(256×8) = +0.00048828125 correctness fraction = +0.048828125 percentage points**, two-sided *historical* exploratory root sign-flip p=1.0. These historical numbers were used solely as sanity gates. There was **no new model forward, retraining, or held-out readout**.

## Posthoc teacher-signed margins

Each original case truth label y was applied to the frozen original raw binary logits to produce a *teacher-signed margin*, **M = logit[y] − logit[1−y]**, in uncalibrated native logit units. Its source-feature-necessity margin change is **M(after physically matched sham removal) − M(after source-ranked removal)**, computed per case in the *original n-code* ablation family. The original paired composed-minus-additive difference is averaged within each independent world root (256 cases), then equally across the eight roots. The original source-ineligible roots 3 and 5 contribute **zero** to the reported eight-world ITT-like retrospective margin mean, although their raw source margins remain in the archived descriptive records.

| World | Composed sham−source mean margin | Additive sham−source mean margin | Root composed−additive margin mean | Root paired-case median margin difference |
|---|---:|---:|---:|---:|
| dev5-000 | +0.030646 | −0.000746 | **+0.031392** | +0.037985 |
| dev5-001 | +0.006622 | −0.004722 | **+0.011344** | +0.004057 |
| dev5-002 | +0.005444 | +0.004203 | **+0.001242** | +0.001388 |
| dev5-003 | −0.004304 | 0.000000 | **0 (original zero root)** | 0 |
| dev5-004 | −0.049390 | +0.039242 | **−0.088633** | −0.043733 |
| dev5-005 | +0.009902 | +0.052329 | **0 (original zero root)** | 0 |
| dev5-006 | +0.045918 | −0.004347 | **+0.050266** | +0.014208 |
| dev5-007 | −0.156164 | +0.033814 | **−0.189977** | +0.042477 |
| **Eight-root descriptive mean** | | | **−0.02304586 logit units** | |
| **Median of eight root means** | | | **+0.00062076 logit units** | |

Four original roots have positive retrospective root means, two have negative means, and the two previously source-ineligible roots remain fixed zero. The unweighted mean is **−0.02305 logit units** and the median root mean is **+0.00062**. These outcomes are heterogeneous, with the original composed-model margin often changing little even as individual masked cases show stronger perturbations.

The largest negative root-mean, **dev5-007 = −0.18998**, has a **positive within-root casewise median of +0.04248**. The three most negative cases, all from previously trained source group 25, have paired composed-minus-additive margin differences approximately **−11.63**, **−11.63** and **−11.19** logit units, summing to roughly −34.45 of the root's −48.63 aggregate signed margin units. The heavy tail makes the original root7 *mean* highly sensitive to a few repeated source-history masks. These masks belong to the already-opened original training-source set and were **not selected to change the fixed ITT result**.

Raw teacher logit units may have different scales across independently seeded composed/additive models and worlds. The equal-root cross-model margin mean is therefore a **descriptive arithmetic contrast under an unverified logit-scale comparability assumption**, rather than a calibrated population effect. Neither its sign nor its magnitude may be treated as a new confirmed feature-necessity claim.

## Accuracy saturation and predeclared margin strata

The original full-affine baseline margin was divided into frozen, exhaustive bins: negative, [0,2), [2,5), [5,10) and [10,+∞), all in raw teacher-signed logit units. Within the original **2048 composed** source case-arm records, **1202** full-affine baseline margins were at least 10; within the **2048 additive** source cases, **1877** were at least 10. Those large true-class margins explain why modest relative logit movements frequently leave the binary class unchanged, especially in the additive teacher. Individual bin counts and valid matched-sham coverage for every original root and teacher arm are available in the audited result JSON.

| Original full-affine margin bin | Composed source cases | Additive source cases |
|---|---:|---:|
| Negative | 79 | 0 |
| [0, 2) | 138 | 0 |
| [2, 5) | 166 | 0 |
| [5, 10) | 463 | 171 |
| [10, +∞) | **1202** | **1877** |
| **Total** | **2048** | **2048** |

The auditor independently reports source/sham per-case margin loss from the original full-affine background, the sign/absolute difference distributions, all valid/null cases, and whole-root median distributions. After computing all these quantities, **no new statistical p-value or confidence bound** was assigned to them; original NEC03T correctness p=1.0 remains a historical result only. This analysis is retrospective source-trained diagnostic material. It does not establish equivalence between source and disjoint active-code removal, nor justify an out-of-sample causal confirmation claim.

## Decision and protected cohorts

The original-n **code-necessity hypothesis remains unsupported as a scientifically meaningful, independent-root-robust correctness mediator** in this family. Its registered NEC03T correctness effect was small (+0.0488pp versus +2pp meaningful floor). The newly observed signed-margin difference has mixed root signs, a near-zero median root effect and a negative mean driven substantially by a few source-case outliers. Source-only margin sensitivity can be documented as a future diagnostic endpoint if a new physical dose/variation floor and root-level power plan are pre-registered; these posthoc margins do not qualify.

This audit ran **no native/SAE model forwards**. It opened no new development-evaluation, reserved, newly source-recruited, 32 five-input embargoed or 64 old four-input confirmation-world contexts. It leaves the independently recruited **69/88 paired-source-eligible worlds** protected. Their **59 eligible independent causal-sham-root outcomes for a 95/95 tolerance calibration remain unmeasured**.

**Scientific gate: FSI-01 REDESIGN_OR_PILOT_ONLY.**
