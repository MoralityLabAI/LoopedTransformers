# Sourcing and consistency audit

This note records the resolution of the external reviewer comments, 20 November 2026. It is kept **outside** the scientific manuscript to avoid turning the article into an editorial change log.

| Review concern | Disposition |
| --- | --- |
| Which VPD implementation? | Corrected: `2df0092` belongs to the MoralityLabAI fork. Its `run_param_decomp.py` and `models/component_model.py` are byte-identical to Goodfire upstream `1f531b3` (6 May 2026); the fork supplies experiment wrappers and toy model. Study-level negative conclusion is confined to those small models, fitting schedules, edits, and matched controls. |
| Cache attribution | The cache behavior is dated to Goodfire upstream commit `1f531b3` (6 May 2026), inherited unchanged in the fork, *under weight-tied four-pass reuse*. No assertion is made about Goodfire's original model's execution behavior. |
| DT04 tiny effect | Retains all 59 attempted roots, 56 audit-clean, 16 fully eligible. Corrects the implicit one-board claim: there were 16 scored boards/root and 8 sham interventions/root, so the mean contrast is fractional. |
| Concentration of routing gap | Gives 5 non-competent roots: dense 68/320, online 27/320, fixed 0/320; all 8 fixed-route solves occur in the 3 competent roots. |
| Registered criteria | Defines the toy positive gate explicitly and links the frozen protocol and machine-readable registration; local commit is labeled as local source freeze, not externally timestamped preregistration. Native and DT04 protocols are linked as well. |
| Title provenance | Added primary sources: LA-602 (1946), Livermore's December 23, 1975 Gilbert correspondence and Weaver & Wood (1979, DOI 10.1103/PhysRevA.20.316). Atmosphere and ocean calculations are distinguished. |
| BPD terminology | Expanded from the BPD repository's explicit definition, Behavioral Parameter Decomposition. Named the different DT05 transport and routing pilots. |
| Missing explanatory glossary | Defines Sudoku4/9, task-Gramian, signed-Jacobian shams, organelle and specific recurrent/transformer architecture boundaries in prose. |
| Wordle and ARC | Adds 18/72 cached Wordle observation count and the exact four ARC Challenge held-out card counts (5, 4, 2, 6); separates ARC Challenge from ARC-AGI and cites Clark et al. (2018). |
| Controlled toy and native protocols | Gives 1% Frobenius dose, candidate bank sizes, signed search counts, 200 warmup plus 300 main updates with two adversarial mask steps for the synthetic gate, target endpoints, and task-gradient information baseline. Distinguishes first local result from hosted replay. |
| Uncertainty and coverage | Clarifies two-degree-of-freedom across-root intervals, native previously viewed teachers and shared test cohorts. |
| Actual transformer in reported experiments | Explicitly states that the 1,763 and 44,001 parameter studies contain no attention-based transformer stack, and that looped transformers are the prospective target. |
| 6.83M checkpoint provenance | Credits the two Hugging Face model-card owners; names the local checkpoint aliases and full SHA-256s plus the solver-verified case-set digest, while preserving the absence of a documented exact download revision. |
| 94.90% denominator | Clarifies percentage of all evaluated Sudoku4 boards on 19 selected roots, with privileged teacher mask information. |
| Missing intellectual lineage | Adds APD, SPD, Saunshi et al., Kaissis et al. and Angsten's separate adversarial critique. |
| Build issues | Replaces raw LaTeX table with a Pandoc table, resolves the `3*` notation, uses TeX math notation for exponents, checks that bibliography entries resolve and that all PDF pages render. |
| Figure reproducibility | Includes integer board counts, `routing_root_values.json`, `verify_routing.py` and `make_figure.py`, with assertions for all competent-subset sums and both zero-teacher roots. The optional verifier mode compares against the full pinned source receipt. |

**Remaining boundary:** The article and figure can be rebuilt from the supplied source. Historical model training still requires its pinned checkpoint/data dependency and protocol-specific execution environment. Fresh independent scientific evidence requires a separate root and sealed evaluation cohort.

## Follow-up review, 20 November 2026

- **Historical search:** classified separately as SVD/head-row rather than trained VPD. Nine accepted SVD and seven head-row proposals.
- **DT04:** added SD = 0.0118 alongside its 0.000265 across-root mean, retained 59 attempted versus 56 audit-clean roots.
- **DT05 transport:** decomposed the -1/128 overall effect by root, called the EvalMath verdict a planning preflight, and disclosed `git_dirty: true` with a file-hash manifest.
- **DT05 rank:** reported per-root Gramian participation ratios (5.81→10.57, 1.20→1.75, 1.85→1.80) as well as the mean.
- **DT05 routing:** noted the two zero-competence dense roots and the online 8/64 vs teacher-informed 7/64 exception; removed any ceiling claim.
- **Presentation:** corrected captions, first-use Sudoku4 expansion, specific ARC row counts, and added checked competent-subset counts.
- **Bibliographic boundaries:** Verified the 1975 report's author as Roger E. Batzel (signed, OSTI 443240), its exact title and 20,000 factor; dated Angsten's October 20 LessWrong publication.

## Final reviewer insertions, 20 November 2026

- **Denominator:** 1,305,171 logged candidate-evaluation rows overall; 131,376 `vdp_or_class_row` candidate rows, 110,731 stable, sixteen accepted (nine `head_svd`, seven `head_row`). Sixteen accepted *rows* become fourteen distinct source-target-component combinations when scales are collapsed. The exploratory method did not fit learned VPD components.
- **Controlled fits:** toy 200/300 warmup/main updates and two adversarial mask steps; native 32/64 updates, batch eight, two adversarial mask steps. The native arm searches sixteen rank-one factors in both signs, producing 32 signed candidates per arm. Native teacher 211 has a paired-string interval [−0.000268, −0.000022], favoring the selected random arm within that teacher.
- **Mask diagnostic:** first/final difference was observed on eight old training strings per fitted checkpoint. Adapter trajectory parity used one excluded untrained teacher with the 32/64 update schedule, and the scientific reserved teachers were never opened.
- **DT04:** the competent nineteen-root group contains the three failed float32 gauge-audit outliers (`2604008`, `2604026`, `2604050`); their inclusion limits conclusions from that postselected subset.
- **DT05 teacher-definition ambiguity:** five per-root `raw_depth8_exact_board` values differ from the explicit dense teacher's depth-eight accuracy. At root `2605904` the raw value is 33/64, compared with 22/64 from `dense_teacher.by_depth["8"]`. The published table and three-root subset consistently use the latter. The receipt lacks a demonstrated generation contract for the raw field; changing to raw would yield four retrospectively competent roots. This remains unresolved and is independently flagged by the enhanced verifier.
- **ARC:** selected cached rule altered correctness on zero unseen examples, whose counts were 5, 4, 2 and 6 over the four cards.
- **Pinned sources:** BPD DT04 result link pinned to commit `c24c676` and protocol pinned to `8e02d09`. The bibliography and main text now cite the verified Batzel report, Goodfire's official article with arXiv identifier, Angsten's dated LessWrong post and Kaissis's PMLR page range.

**Data boundary:** these are editorial changes and checks of previously exposed receipts. The PDF and source ZIP are rebuilt from this same manuscript. No new model evaluation or training was conducted.
