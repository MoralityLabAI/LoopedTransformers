# NEC-02 | Hosted trained recurrent neuron community pilot: observed results

**2026-11-21. Evidence class:** `exploratory-causal-pilot-retrained-CPU`. Follow-up to merged [NEC-01 PR #236](https://github.com/MoralityLabAI/LoopedTransformers/pull/236) and original PEF-07 [PR #67](https://github.com/MoralityLabAI/LoopedTransformers/pull/67).

- [Successful six-checkpoint hosted CPU training and first causal analysis](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44787384510); artifact `nec02-hosted-six-model-frozen-and-two-causal-arms`, ID `14654612211`.
- [Successful immutable-roster read-only raw-logit replay](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44788623483); artifact `nec02-raw-context-replay-checkpoints-and-causal-effects`, ID `14655785897`. Full ZIP artifact SHA256 reported by GitHub: `b7b76707ccd873ea5c3070a6fb32682d5938391f60a73fa7af3c879251047842`.
- The frozen discovery graphs, 6 hosted checkpoint bytes, full PEF-07 score receipts, 432 independently audited candidate-by-context rows and 16-state logits for each of 5 intervention modes are in the second artifact.

## Hosted task and source behavior

All six **newly trained** seed/task checkpoints scored **768/768** correct on the native h12 task head and **768/768** on the separately fitted h14 readout. Both PEF-07 decision rules were met in this separate execution. The 17 combined PEF-07/NEC tests passed before analysis. The earlier November 15 local experiment had different checkpoint byte hashes; **all six hosted checkpoint SHA256 values differ from the historical local byte hashes**.

The hosted canonical source-manifest SHA256 is `de4e0263864ee70c2bf94ca95008ace10fce1323d1a51b4c8829ec26e40bbd9b`. The November 15 local report recorded `849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1`. NEC-02 independently verifies the committed source's preregistered SHA256 sort of all 256 context assignments and all 16 focal configurations per context. The original local source-manifest bytes are unavailable for a direct diff. Both digests remain in `nec02_audit.json` and the difference is also flagged on PEF-07 PR #67.

## Discovered communities

Only source **training contexts** were used to construct the graph and select candidates. For each trained seed-103 model, the 160 development contexts have 16 closely related focal histories each. Graph nodes are (loop visit, native GELU neuron), with 12 visits and 64 MLP units per visit. Co-attribution uses the mean absolute activation × gradient on the memory token. Pearson-phi-positive edges and seeded Louvain define the candidates. The roster was frozen and checked before all 48 heldout context quartets were scored.

| Model | Source graph edges | Hub occurrences | Discovered communities | Heldout context groups |
|---|---:|---:|---:|---:|
| Coalition, seed 103 | 24 | 15 | 4 | 48 |
| Threshold, seed 103 | 36 | 15 | 5 | 48 |

**All nine communities could be ablated from otherwise intact models without changing any of the 768 heldout class predictions per candidate.** Their mean KL changes were small, with the largest observed positive *ablate minus frequency-matched sham* KL around `4.14e-6` for the threshold `nec-b30c4edbbb42` candidate. The unadjusted 95% context-bootstrap percentile interval was approximately `[2.78e-6,5.73e-6]`. This remains a tiny continuous effect and is neither a preregistered behavioral mechanism threshold nor a multiplicity-corrected discovery.

## Conditional recovery under MLP sparsification

The following candidates were identified using the source-only graph. At only the visits in their community, an intervention zeros all GELU memory-token MLP neurons except always-on hubs, or except hubs plus the candidate community. Attention, residual paths, input injection, other visits and final readout remain active. Counts are errors relative to the clean model's 768/768 correct predictions.

| Task and candidate | Native occurrences | Selective ablation | Blank GELU on candidate visits | Hubs only | Hubs + community |
|---|---:|---:|---:|---:|---:|
| Coalition `nec-68cd70ab69a8` | 5 | 0 | 80 | 10 | 1 |
| Threshold `nec-b30c4edbbb42` | 7 | 0 | 92 | 59 | 34 |
| Threshold `nec-0cf33bd0b573` | 4 | 0 | 18 | 5 | 12 |

The first coalition community recovered **9/768** predictions when added to the hubs-only system. The threshold seven-occurrence community recovered **25/768**, while a separate four-occurrence threshold community harmed performance under the same operation, illustrating nonmonotone contextual effects. The strongest threshold conditional KL gain (hubs-only minus retained-community) was **0.15686**, with a descriptive context-bootstrap interval **[0.09676,0.22566]**. The coalition five-occurrence candidate's corresponding KL gain was **0.02108**, interval **[-0.00012,0.04897]**.

These conditional effects are evidence of **dependency under severe sparsification** in a small recurrent architecture, with results specific to the chosen visits, model and heldout family. A positive conditional rescue does not identify a unique AND/XOR circuit or demonstrate community sufficiency for the whole network. The original clean model's zero class losses under targeted ablation are a major constraint on claims of necessity.

## Full raw evidence and verification

The second hosted workflow downloaded the original successful NEC-02 artifact by exact run ID, read the six frozen checkpoint files, and verified hashes **before** evaluating the unchanged community roster. It performed no training and did not rerun discovery. It then wrote `raw_paired_context_traces.jsonl` separately for coalition and threshold:

- 192 records for four coalition candidates × 48 contexts.
- 240 records for five threshold candidates × 48 contexts.
- Each record stores the eight-bit context, all 16 focal-state masks, task labels, clean two-class logits, logits for `ablate`, `sham`, `retain`, `hubs` and `blank`, and measured condition-level effects.

The independent Python standard-library audit validates source ownership, candidate rosters, raw trace SHA256, all expected candidate/context combinations, full 16-state logits, finite values and correctness deltas recomputed from raw logits. The audit results and chain-of-custody hashes are stored in `nec02_audit.json` and `raw_replay_receipt.json`. There are 432 candidate-context records; each context's 16 bit combinations remain **correlated observations** and share the same source-unit identity across candidates.

## Inference and next experiment

**FSI-01 verdict: `REDESIGN_OR_PILOT_ONLY`.** We have 48 heldout context groups per fixed model, 1 trained seed per task examined with neuron interventions, 9 graph-selected candidates, and 5 conditions per candidate. The same context appears across many candidate comparisons. Confidence intervals are unadjusted descriptive percentile bootstrap intervals. No sham variation floor or prospective minimum meaningful behavioral effect was registered, and a frequency-matched neuron set is not an activation-/outcome-matched causal null.

A meaningful NEC-03 preregistration should use independent newly seeded model/task roots, freeze intervention candidates on development data, set a smallest meaningful **correctness** effect and full Holm family in advance, and use sham controls matched for activation magnitude and `mlp_down` output displacement. Two further targeted ideas are (1) test whether lesion effects emerge when removing one community *alongside* particular alternate reader circuits and (2) train an SAE on the **same 64-dimensional GELU coordinate system**, source-only, then compare SAE-feature and native-neuron interventions at matched downstream dose. Maintain separate evidence classes for the synthetic PEF task, mechanistic circuits and speculative self-model behavior.
