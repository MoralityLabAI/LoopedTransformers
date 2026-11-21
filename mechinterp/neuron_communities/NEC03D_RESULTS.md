# NEC-03D: source-only native versus SAE matched-dose pilot results

**Date:** 2026-11-21. **Evidence class:** retrospective development, mechanistic dose-comparability failure. The PEF-07 local registered decision and previous NEC-01/02B outcomes are unchanged.

## Execution and provenance

- Source code in [NEC-03D PR #240](https://github.com/MoralityLabAI/LoopedTransformers/pull/240), based on the PEF-07 research branch.
- [Successful public GitHub CPU Actions run 44860214701](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44860214701); eight new unit tests passed and full source-only feature training, heldout replay and independent raw/dose audit passed.
- Artifact: `nec03d-source-sae-matched-dose-pilot`, ID **14689443261**, GitHub ZIP digest **5316f57debf1e7706366788597813bcdf5d5cb1cf2d1d5dc965b8d034564a7a4**. It contains two learned SAE state dictionaries and frozen train/dev context manifests, two source feature rosters, 432 candidate-by-context rows with 16 paired clean/intervention logits per row, sample-level local-dose receipts, and `nec03d_audit.json`.
- Original model source: the six frozen PEF-07 CPU checkpoints and prior community rosters in NEC-02B artifact ID 14655419843. Only seed **103 coalition and threshold** were analyzed here, using the original 48 already-exposed PEF heldout context quartets.

## SAE fit quality and sparsity

| Metric | Coalition | Threshold |
|---|---:|---:|
| Native GELU basis dimension | 64 | 64 |
| SAE features | 128 | 128 |
| Fit contexts / selection contexts | 80 / 80 | 80 / 80 |
| Optimization steps | 240 | 240 |
| Normalized training reconstruction MSE | 0.0071219 | 0.0065891 |
| Mean active SAE features (L0) | 62.95 | 63.33 |
| Source heldout contexts used for SAE training | 0 | 0 |

This model reconstructs training activations well but is comparatively **dense**, with roughly half the SAE features active. The rank-ordered feature probes are usable development diagnostics, not a source of demonstrated causal disentanglement.

## Conditional restoration and failed dose matching

The clean-graft benchmark compares restoration of a community's original native GELU activations against a same-site, same-cardinality frequency-matched native sham and two SAE-feature bases. All control donors are locally scaled toward the native target's frozen `mlp_down` output L2 norm, subject to a maximum gain of eight. Controls whose gains are capped retain their realized dose mismatch in the receipt.

**None of the nine discovered candidate communities passed the preregistered 90%-of-site-samples-within-5%-dose gate across all controls.** The source-only SAE-feature controls frequently require a gain larger than eight, and random alternate features have zero ReLU activation on substantial subsets of examples.

The original intact trained PEF models scored all 768 contexts correctly per task, while the hubs-only reduced baseline was damaged only for selected candidates. The following are exploratory *unmatched-dose* counts, retained as diagnostics:

| Frozen candidate | Task | Hubs-only errors | Native graft errors | Native sham errors | SAE graft errors | SAE alternate errors |
|---|---|---:|---:|---:|---:|---:|
| `nec-68cd70ab69a8` | Coalition | 10 | 0 | 3 | 0 | 7 |
| `nec-b30c4edbbb42` | Threshold | 59 | 25 | 46 | 23 | 54 |
| `nec-0cf33bd0b573` | Threshold | 5 | 5 | 0 | 4 | 5 |

The first coalition group's SAE restoration achieved a nonzero classification rescue, but its within-5%-dose match fraction was **60.0%**; native sham achieved **88.9%**. The strongest threshold group's SAE restoration fraction was **71.5%**, with SAE alternate at **40.3%** and native sham at **75.8%**. At this level of physical mismatch, differences in error counts cannot establish greater causal specificity of one basis.

Every result, including communities with no class errors, is preserved in the artifact. The experiment ran on public CPU and did not mutate or retrain the PEF-07 trunk.

## Scientific interpretation

The negative result is the **inability to match sparse-feature and native-donor dose under the eightfold gain ceiling**, even though SAE reconstruction MSE is small. Sparse feature availability varies with the input and recurrent visit; a near-zero decoder contribution demands very large scaling to compete with a native neuron.

The next experimental step is a strictly common-achievable-dose study: for each frozen input/visit and a specified pair of candidate/control donors, define their shared dose as the lesser of the native target dose and the maximum admissible dose of the weaker donor. Scale **both** interventions to the same shared dose, retain the achieved fraction of native dose, and predeclare a physical dose-retention floor. Separate pairwise native-vs-native sham, native-vs-SAE and SAE-vs-alternate SAE comparisons, because enforcing one shared dose over all five arms would usually collapse the dose to zero.

An all-conditions matched-dose comparison with a negligible surviving physical dose is a low-signal engineering result, not proof of equivalent causal mechanisms. Source-only feature choices stay sealed, and every newly obtained outcome from already opened PEF contexts remains in the **posthoc development** evidence class.

**EvalMath FSI-01 verdict: REDESIGN_OR_PILOT_ONLY.** Confirmation requires new model/task roots, an independent source/test split and prospectively frozen meaningful correctness and multiplicity thresholds. See [NEC-03 research issue #239](https://github.com/MoralityLabAI/LoopedTransformers/issues/239).
