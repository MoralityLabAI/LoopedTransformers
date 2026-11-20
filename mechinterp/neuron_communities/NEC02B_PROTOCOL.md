# NEC-02B | Read-only, six-model recurrent community stability

This is a **retrospective development extension** of [NEC-02](NEC02_RESULTS.md). The November 15, 2026 PEF-07 preregistration, original checkpoints, and all heldout decisions are left unchanged.

NEC-02 found four candidate communities in seed-103 coalition and five in seed-103 threshold; targeted ablations left intact-model classification unchanged. Some candidates recovered a meaningful number of class predictions when reintroduced alongside always-on hubs after visit-specific MLP suppression. The immediate question is whether this conditional behavior recurs across the other trained seeds.

## Fixed data and models

Use the already audited original NEC-02 hosted archive from [GitHub Actions 44788623483](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44788623483), artifact ID 14655785897. It contains the **six PEF-07 models** trained on the same registered 160/48/48 context split. All checkpoint SHA256 values are verified before any new discovery. No GPU time, model optimization, readout fitting or checkpoint modification is permitted.

The seed-103 coalition and threshold raw paired trace results are reused unchanged. For remaining four models, seeds 131 and 167 × coalition and threshold, run source-only NEC-01 discovery with the exact same hyperparameters: top two native GELU neurons by mean absolute activation × gradient for each of 12 visits, positive phi co-attribution graph with min joint 3 contexts and phi >=0.18, >=90% hubs removed, seeded Louvain and up to eight communities. Freeze each community roster and frequency-matched sham before entering the original 48 heldout context quartets. Run all five interventions and archive the full 16-state logits for every candidate/context.

## Descriptive endpoints

- Candidate count and graph density across six frozen models, including zero-candidate outcomes.
- Number of new class errors after selective native-neuron ablation compared with intact network. The prior seed-103 result was zero across all its nine candidates.
- Conditional rescue count: errors with hubs-only minus errors with hubs + selected community, allowing negative values.
- Exact source units are 48 heldout eight-bit contexts *per fixed model*; their 16 focal variants are paired measurements. The six models share the same heldout contexts and task family. Grouping by model and source context is necessary before statistical inference.
- Save all candidate/context raw paired logits, causal conditions, exact source and checkpoint SHA256, and the strongest and weakest conditional rescue selected from heldout results, explicitly labeled posthoc.

This is neither a new independently sampled task family nor a prospective hypothesis test. The EvalMath FSI-01 gate remains REDESIGN_OR_PILOT_ONLY. The next confirmatory experiment must register a physically matched intervention-dose null, a minimum meaningful behavioral effect and a corrected hypothesis family on independent new model/task roots.

## Execution

[GitHub Actions workflow](../../.github/workflows/nec02b-six-seed-frozen-causal.yml) downloads the sealed artifact, verifies six checkpoints, runs the four missing trained-model arms, then audits all six model/task pairs without retraining. Generated JSON reports and the entire raw paired evidence set are uploaded to Actions, with only compact documentation committed to Git.
