# NEC-03T: independently audited full-affine SAE necessity ablation

**Completed 21 November 2026. Source-training in-sample mechanism diagnostic.** [Preregistered scorer/independent auditor commit c2c4920](https://github.com/MoralityLabAI/LoopedTransformers/commit/c2c492021293a1d01b46bd38910501ff94a3a780) froze all source groups, original root and arm identities, matched ablation doses, one primary comparison and the eight-root ITT before any NEC03T model scoring.

[**Successful hosted CPU campaign 44913605626**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44913605626): source ancestry and **7/7** local tests passed, **8/8** independent source-root full neural/dose/gradient/logit replays passed, and the final pure-stdlib eight-world ITT aggregate passed. Final aggregate artifact ID **14711051887**, *nec03t-eight-root-source-necessity-independently-audited*. All original root archives, each containing complete 256-case raw JSONL source/sham paths and the separate independent audit, remain attached to the hosted Actions run.

## Scientific question and intervention

Earlier NEC03R/S source-only results showed that a 128-feature TopK16 SAE's **full affine decoder** almost perfectly recovers clean native task correctness when added to hubs-only nonhub-lesioned recurrent sites, while sparse n-code source donors at matched physical 32D output L2 dose failed to improve over disjoint active controls.

NEC03T directly tests **necessity rather than sufficiency**. For each source-selected recurrent site, begin with **all 16 active code contributions plus the frozen source-normalized mean and learned decoder bias**, masked to the original native nonhub 64D GELU coordinates. This is the full-affine native SAE donor background with native hub channels remaining live. Calculate the hubs-restored clean-chosen margin gradient in the *full-affine* 12-visit recurrent trajectory. Select the original n top source-ranked **positive active** code contributions, as fixed by the historical 40/40 source fit/rank split, and exhaustively choose one **equal-n disjoint active-code sham** minimizing the absolute 32D cosine gap to the source-removal direction relative to that full-affine gradient.

Source and sham removal vectors both use the same original decoder/physical nonhub mask. Reduce their individual projected output-32D L2 norms to the shared **minimum of the two**, never amplify, with a 1% physical dose-matching gate. An insufficient TopK16 active set, weak gradient, angle gap above 0.25 or physically weak/unmatched dose at **any** selected visit makes both source and sham removals **zero across every selected visit** for the entire input. These invalid cases still contribute to the original 256-case denominator. Ground-truth task labels enter only after routing and dose are frozen.

The two model arms remain the previously trained composed and equal-prior additive teachers. All original eight Boolean world roots dev5-000..007 are retained. Source groups **24–31** provide exactly **256 already-seen training-source histories per model arm**, each containing all 32 focal masks. Native and sparse autoencoder optimizer updates = 0. Root indices 3 and 5 remain the earlier paired-source-ineligible **fixed-zero primary ITT roots** while their native arms are still scored descriptively.

## One preregistered primary necessity contrast

A positive source-code necessity effect means removing source-ranked codes causes more correctness loss than removing the equal-dose disjoint sham. The single predeclared root effect is:

\[
D_r=\frac{
(C^{comp}_{sham\ removed}-C^{comp}_{source\ removed})
-(C^{add}_{sham\ removed}-C^{add}_{source\ removed})}{256}.
\]

The equal-root macro is the mean across **eight** registered world roots. Prespecified positive meaningful floor: **+2 percentage points**. Root sign-flip is exploratory, conditional on independent-root sign exchangeability.

| Original root | Primary necessity effect | Root effect, percentage points | Source admission |
|---|---:|---:|---|
| dev5-000 | +1/256 | +0.390625 | paired source eligible |
| dev5-001 | 0/256 | 0 | paired source eligible |
| dev5-002 | 0/256 | 0 | paired source eligible |
| dev5-003 | 0, fixed | 0 | earlier source ITT failure |
| dev5-004 | 0/256 | 0 | paired source eligible |
| dev5-005 | 0, fixed | 0 | earlier source ITT failure |
| dev5-006 | 0/256 | 0 | paired source eligible |
| dev5-007 | 0/256 | 0 | paired source eligible |
| **Eight-root primary macro** | **+1/(256×8)** | **+0.048828125** | **8/8 independent audits** |

The exact exploratory two-sided sign-flip **p=1.0**, with **one nonzero world-root effect**. The observed magnitude is approximately **41 times smaller** than the registered 2-point meaningful floor. This is a **negative, power-limited source-only feature-necessity result** for the current selected native sites, physical donor budget and disjoint active null. It neither proves a mechanistic equivalence nor excludes distributed multi-feature necessity outside these n-code ablations.

## Original-source aggregate correctness and recovery

Each composed and additive teacher was tested on the original eight-world ×256 source-training histories = **2,048 in-sample cases per arm**. Source correctness is reported as a diagnostic only; repeated cases, masks and visits within a world are not independent statistical units.

| Intervention | Composed correct /2048 | Additive correct /2048 |
|---|---:|---:|
| Unmodified native clean | **1970** | **2048** |
| Hubs-only lesion | **1592** | **1911** |
| Exact native nonhub restoration | **1970** | **2048** |
| Full-affine SAE nonhub restoration | **1969** | **2048** |
| All16 decoder contributions without offset | **1802** | **1859** |
| Full-affine restored minus original-n source codes | **1966** | **2048** |
| Full-affine restored minus equal-dose disjoint angular sham codes | **1967** | **2048** |
| Full-affine minus widened source budget | **1956** | **2048** |
| Full-affine minus widened disjoint angular sham | **1957** | **2048** |
| Full-affine minus hash-selected source removal | **1965** | **2048** |
| Full-affine minus hash-selected disjoint removal | **1967** | **2048** |

The composed native clean-minus-hubs signed correctness gap is **378** cases. Full-affine SAE reconstruction returns **377/378 (99.74%)** of that signed gap; all16 contribution-only reconstruction returns **210/378 (55.56%)**. The additive clean-minus-hubs signed gap is **137** cases and is fully recovered by the full affine reconstruction. The original feature removal causes only **3 fewer** composed correct cases than the full-affine baseline, and its matched disjoint sham causes **2 fewer**, leaving just **1** additional source-specific lost case across the full 8-root descriptive cohort. The source-eligible ITT primary agrees with that single-case numerator. No matched original-n ablation changes correctness on the original additive teachers.

The widened-budget secondary contrast remains descriptive: eligible-root signed numerator +3/(256×8), approximately **+0.146484 percentage points**, also below +2pp and heterogeneous across worlds; the original ineligible root3 was retained as zero despite source-scoreable widened effects. Source/full restoration ceilings use natural doses, whereas removed source and sham contributions use their **matched output-L2 removal dose**.

The independent auditor reconstructed all full SAE affine sources and historical native/SAE SHA receipts, checked the original selected sites, reenumerated every k-disjoint active-code control, verified source and sham physical output-32D doses and attenuation gains, reproduced the fixed hash control and invalid-case zero-effect masks, independently scored every five-bit Boolean source truth label and **all eleven 12-visit recurrent nonlinear logit trajectories**. Every original root passed.

## Interpretation, stop gate and protected world families

The engineering bottleneck remains **causal specificity at the original sparse feature budget**. A full SAE-affine representation can be an accurate source-trained reconstructive coordinate system while individual source-ranked code subsets have little *additional* measurable necessity compared with equally physical disjoint active code subsets. Shared representational redundancy, strong control matching, local removal scale and nonlinear recurrence are possible explanations requiring a new source-only, explicitly dose-controlled design. These source data do not decide among them.

The former NEC03Q development-evaluation causal null (+0.016276 percentage points, exploratory p=1.0), NEC03R source-only sufficiency null (0pp), and NEC03S source-only offset-factorial null (−0.048828pp) remain unchanged. The NEC03P cohort has **69/88 paired-source-eligible** recruited independent roots but **zero causal-sham outcome measurements**. No new development-evaluation, reserved, recruited-world non-source, 32 five-input embargoed or 64 old four-input confirmation contexts were opened. The 95/95 sham tolerance campaign needs ≥59 independently eligible **causal-sham world-root outcomes** under a separate preregistered and physically resolvable mechanism.

**Stop before population-scale evaluation. Scientific gate: FSI-01 REDESIGN_OR_PILOT_ONLY.**
