# NEC-03G | Fresh Boolean-world development protocol and confirmation lock

**Status:** code and statistical procedure registered on a new PEF-07 research branch on November 21, 2026, before development-world training. This document governs the first eight development worlds and the preflight. The 64-world confirmation cohort remains outcome-sealed and has no authorized training/evaluation workflow.

Parent lineage: [PEF-07 PR #67](https://github.com/MoralityLabAI/LoopedTransformers/pull/67); native-neuron NEC-01/02/02B; dense SAE NEC-03D; attenuation NEC-03E; genuine source TopK16 NEC-03F. The retrospective NEC-03F result was 19/27 dose-and-signal-feasible matched pairs, up from 8/27 under the earlier fixed-feature policy. Every PEF-07 source root had been exposed before this fresh-world program.

## Research question and experimental unit

Does a source-trained, input-conditioned TopK16 SAE select GELU-feature donors that recover more correct outputs after a recurrent visit-local MLP lesion than equally large, lower-ranked **active** SAE-feature donors, at an identical and adequately strong direct physical MLP-down perturbation dose?

The independent *analysis unit* is one distinct four-input Boolean function, one distinct set of four chronological positions in a twelve-bit recurrent sequence, plus its independently seeded PEF-style checkpoint **and its matched additive teacher control**. Outcomes from the 16 focal counterfactuals within a context, the 48 evaluation contexts, the three donor comparisons, and the additive teacher are repeated measurements on the same root.

## World registry and novelty boundary

The registered generator is in [nec03g_worlds.py](nec03g_worlds.py). It enumerates four-variable Boolean programs built from AND, OR, XOR, NAND, NOR and XNOR, with balanced and nested structural templates plus input-level literal inversion. Its acceptance rule requires all four inputs to be causally essential and between 5 and 11 positive assignments among the 16 possible focal inputs. Programs are deduplicated by the **complete 16-bit truth function**, then deterministically shuffled by SHA256. Focal-position quartets are also deduplicated, require early and late input positions, and are deterministically shuffled.

**128 distinct truth functions and 128 distinct position quartets** are assigned fixed training seeds; the first 64 worlds are the development population, and the remaining 64 form a SHA256-committed confirmation population. Both populations use the same grammar, so this experiment is *within-grammar* generalization across fresh Boolean compositions and focal positions. Different worlds still share the twelve-bit input alphabet, many underlying source strings, the model architecture and the generator grammar. The registered claim is rooted in independently trained models and distinct teacher functions, with these limitations explicit.

The registry published by GitHub Actions contains full development-world specs and **only SHA256 world-spec commitments** for the 64 confirmation roots, together with a digest of their complete internally regenerable specification. There are no confirmation-model checkpoints, generated outcomes or confirmation eval workflows. Confirmation source data are not materialized by the development runner. A new, separately reviewed CI workflow is required for any future confirmation release.

Each world maps all 256 possible nonfocal eight-bit contexts through a world-specific SHA256 shuffle. Context groups remain complete with all 16 focal counterfactuals. Ownership is **160 parameter-training contexts**, **48 development-evaluation contexts**, and **48 reserved contexts**. Parameter-training contexts are divided by prefix into the first 80 SAE-fit/native-graph contexts and the second 80 feature-ranking contexts. The reserved 48 groups remain unused throughout NEC-03G. Any claim about *completely unseen twelve-bit input words* would be false, because the old PEF-07 finite alphabet already covered those strings.

## Trained teachers and source-only circuit selection

For each newly sampled development world, train the original 32-hidden-unit/12-tied-visit PEF-07 `Looper` for **450 CPU AdamW updates** on its own 2,560 source-training histories. Train a *second, separately initialized checkpoint* on an **additive prethreshold control** using binary weights 1,2,4,8 on the four focal bits and a threshold chosen to give **exactly the same number of positive labels** as the composed teacher. Retain both models, training step/seed receipts, original source assignment, checkpoint SHA256 and final clean accuracy. There is no optimization by heldout feedback.

On each frozen model, use the first **80 parameter-training contexts** to identify the top-two native memory-token GELU occurrences per visit by mean absolute activation × gradient. Build the context-level positive phi co-attribution graph, remove features appearing in at least 90% of source contexts, and choose only the **first** seeded-Louvain community satisfying size 2–12 and at most eight occurrences at any given visit. If no community qualifies, the root is included as an intention-to-treat **zero**. Candidate selection is frozen before entering development-evaluation data.

Fit an independent **TopK=16/128** SAE on native 64-D GELU memory-token activations from the first 80 training contexts, with **480 CPU steps** and the same TopK training family used in NEC-03F. Freeze its decoder, coordinate normalization and training receipts. On the next 80 source-training contexts, fit a 12-visit ranking by code activation × downstream margin gradient projected through the physical SAE decoder. This ranking is fixed before evaluation.

At each evaluation input, the label-blind policy selects the highest-ranked **currently active** TopK feature codes matching the native community's component count at each selected visit. The comparison control selects equally many lowest-ranked **currently active** feature codes. Native-neuron shams are source-frequency-matched, on the same visits, and made disjoint from hubs and native community occurrences.

## Primary causal endpoint and root-level failure policy

Every donor pair is injected into the same live hubs-only recurrent GELU path, with the selected donors obtained from the **original clean input's recurrent activations**. Pairwise direct MLP-down output L2 doses are equalized to the **smaller attainable** original dose per input and visit. Both donor gains are at most 1; the policy forbids amplification and outcome-based scaling.

The predeclared three comparisons are: (a) native community vs same-site frequency-matched native sham; (b) native community vs source-ranked active TopK features; and (c) source-ranked active TopK vs lower-ranked active TopK. Every comparison reports clean/hubs/left/right 16-state logits, exact correctness and per-site dose. The mechanical gate requires at least 99% of site/examples within 1% direct physical dose, at least 80% sites with positive shared dose, and median retained left-donor direct dose of at least 25%.

**Primary root-level effect:** correctly classified probability on the composed teacher after source-ranked active TopK donation **minus** the same-dose lower-ranked active TopK feature control, averaged over that root's 768 development-evaluation histories. Its unit is an accuracy fraction in [-1,1]. Additive-control teacher's same endpoint and the native-vs-native-sham effects of both teachers provide predeclared development null/heterogeneity comparisons. Difference-in-differences between composed and additive arms is retained as a descriptive secondary.

To avoid selecting easy roots after observing performance, the primary effect is assigned **zero in intention-to-treat** if either newly trained teacher model scores below **90% native clean evaluation accuracy**, either arm lacks a source-selected community, or any of the three pair families in either arm fails the physical feasibility gate. Reasons and underlying raw outcomes are retained for every root. The proposed registered family must pass a later development-powered review; a large mean from a few surviving roots cannot silently erase the failed ones.

The smallest scientifically meaningful root-mean correctness gain for further planning is provisionally **0.02** absolute, with target effect **0.04**. These are prospective *planning assumptions*, not outcomes and not declarations of a successful mechanism. Set the familywise alpha to 0.05 with **three planned comparison families**. One supervised Boolean task root and its additive control count together as **one** independently generated root. Under the EvalMath guidance, an independent 95% sham-tail upper tolerance bound with 95% confidence needs at least **59 independent sham worlds even using their maximum**. Fewer roots make that floor unresolved.

## Development schedule and hard stop

The first GitHub Actions workflow runs:
1. A source integrity test and tiny two-step engineering smoke. A short-run result is permanently labeled engineering-only.
2. **Eight** of the 64 development worlds with the full 450-step native training and 480-step source-only SAE, each with its separately trained additive teacher control. Every world gets a separate immutable artifact.
3. A root-level independent auditor checks source group assignments, candidate freeze, input-conditioned active-route provenance, model/SAE hashes, the exact paired labels and class outcomes, and local physical doses.
4. The aggregator derives one signed effect and one predeclared sham-effect set per **independent world**.
5. The exact EvalMath mechanistic-power preflight script is run from pinned commit `9708f1edaf1cb25904b88b07c3f23c1899df7cac`. It computes resolution, discrete exact-sign limits, the available sham tolerance floor and power planning for 64 confirmation roots.

Eight development worlds are **insufficient** to calibrate a 95/95 sham-tail bound. A `REDESIGN_OR_PILOT_ONLY` preflight is therefore expected and retained as a valid result. The remainder of the 64 development worlds are available for further source-only calibration following review of cost, task-learning competence and effect sizes, before any confirmation outcome is opened.

**Confirmation hard stop:** No confirmation teacher is trained, no confirmation model is initialized for scoring, no reserved input context is evaluated, and no confirmation CI job exists in this branch. A statistical `READY_FOR_CONFIRMATION` verdict, if ever achieved, remains advisory; the confirmation release must receive a separately committed preregistration, locked alpha/endpoint/sham family and explicit reviewed authorization. Any negative result or uncalibrated sham floor is preserved without retroactively relabeling opened development roots as confirmation data.

## Evidence scope

NEC-03G is a fresh *world/task root pilot* rather than a repeat of the original PEF-07 source contexts under the same teacher. It remains within one small Boolean-program grammar and one PEF-style recurrent model family. Strong evidence of reusable native or SAE circuits in natural-language transformers requires further architecture and data transfer studies, with the same physical dose and causal provenance controls.

Source: [EvalMath mechanistic-power skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power), [FSI-01 inference gate](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md). This protocol is a development preregistration and does not claim a measured prospective effect.
