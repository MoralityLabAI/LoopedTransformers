# Sourcing and consistency audit

This note records the resolution of the external reviewer comments, 20 November 2026. It is kept **outside** the scientific manuscript to avoid turning the article into an editorial change log.

| Review concern | Disposition |
| --- | --- |
| Which VPD implementation? | Corrected: `2df0092` is the MoralityLabAI fork commit. The core optimizer and component model are unchanged from Goodfire `1f531b3` (6 May 2026); the fork carries experiment wrappers and toy-model code. Study-level negative conclusion is confined to those small models, fitting schedules, edits, and matched controls. |
| Cache attribution | The cache behavior is attributed to the Goodfire code *as of 6 May 2026*, under four-pass weight-tied reuse. No assertion is made about Goodfire's original model's execution behavior. |
| DT04 tiny effect | Retains all 59 attempted roots, 56 audit-clean, 16 fully eligible. Corrects the implicit one-board claim: there were 16 scored boards/root and 8 sham interventions/root, so the mean contrast is fractional. |
| Concentration of routing gap | Gives 5 non-competent roots: dense 68/320, online 27/320, fixed 0/320; all 8 fixed-route solves occur in the 3 competent roots. |
| Registered criteria | Defines the toy positive gate explicitly and links the frozen protocol and machine-readable registration; local commit is labeled as local source freeze, not externally timestamped preregistration. Native and DT04 protocols are linked as well. |
| Title provenance | Added primary sources: LA-602 (1946), Livermore's December 23, 1975 Gilbert correspondence and Weaver & Wood (1979, DOI 10.1103/PhysRevA.20.316). Atmosphere and ocean calculations are distinguished. |
| BPD terminology | Expanded from the BPD repository's explicit definition, Behavioral Parameter Decomposition. Named the different DT05 transport and routing pilots. |
| Missing explanatory glossary | Defines Sudoku4/9, task-Gramian, signed-Jacobian shams, organelle and specific recurrent/transformer architecture boundaries in prose. |
| Wordle and ARC | Adds 18/72 cached Wordle observation count and the exact ARC Challenge unseen counts 5, 4, 2 and 6; separates ARC Challenge from ARC-AGI and cites Clark et al. (2018). |
| Controlled toy and native protocols | Gives 1% Frobenius dose, candidate bank sizes, signed search counts, 200 warmup and 300 main optimizer updates with two adversarial mask steps in the gate, and distinguishes initial local scoring from hosted replay. |
| Uncertainty and coverage | Clarifies two-degree-of-freedom across-root intervals, native previously viewed teachers and shared test cohorts. |
| Actual transformer in reported experiments | Explicitly states that the 1,763 and 44,001 parameter studies contain no attention-based transformer stack, and that looped transformers are the prospective target. |
| 6.83M checkpoint provenance | Names both third-party checkpoint authors, local SHA-256s and the verified case-set digest, without assuming an unrecorded upstream download revision. |
| 94.90% denominator | Clarifies percentage of all evaluated Sudoku4 boards on 19 selected roots, with privileged teacher mask information. |
| Missing intellectual lineage | Adds APD, SPD, Saunshi et al., Kaissis et al. and Angsten's separate adversarial critique. |
| Build issues | Replaces raw LaTeX table with a Pandoc table, resolves the `3*` notation, uses TeX math notation for exponents, checks that bibliography entries resolve and that all PDF pages render. |
| Figure reproducibility | Includes integer board counts, `routing_root_values.json`, `verify_routing.py` and `make_figure.py`; the verifier now asserts the competent-subset counts and two zero-teacher roots. The optional verifier mode compares against the full pinned source receipt. |

**Remaining boundary:** The article and figure can be rebuilt from the supplied source. Historical model training still requires its pinned checkpoint/data dependency and protocol-specific execution environment. Fresh independent scientific evidence requires a separate root and sealed evaluation cohort.

## Blocking review correction, 20 November 2026

The historical SVD/head-row search is separated from the subsequent fitted VPD interventions. DT04's +0.000265 effect is paired with its across-root SD 0.0118. DT05 transport reports the -1/128 board-equivalent outcome, dirty worktree receipt and planning preflight. The causal Gramian participation-ratio mean was driven by the root moving 5.81 to 10.57, with other roots 1.20 to 1.75 and 1.85 to 1.80. Two routing roots were unsolved even by the teacher; root 2605904 had online 8/64 versus privileged 7/64. Table caption duplication, citation metadata, and the branch-mutable DT04 protocol link were corrected.
