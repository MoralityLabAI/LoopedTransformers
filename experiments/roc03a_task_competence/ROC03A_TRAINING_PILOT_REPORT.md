# ROC-03A: longer training is insufficient for delayed-rule competence

**Date:** 17 November 2026. **Status:** registered single-seed pilot **FAIL**. Preregistered in `configs/roc03a_source_locked_training_v1.json` at commit `39ef4b657edca1b77ada9be745cc71f57e0776b2`, before the 1,920-update run. Source-family 6/7 sealed files were not generated, opened, inspected, or evaluated.

## Purpose and frozen protocol

ROC-02E #147 established a negative rank-7 rule-steering causal comparison on twelve short-trained, low-accuracy recurrent checkpoints. ROC-03A tests whether substantially greater training exposure alone makes the original task learnable under identical architecture and source partition.

This first pilot uses one new independent seed `2026111701`, depth 8, 128 hidden units, original `TiedRuleModel` architecture, and source families 0–3 for training. Source families 4–5 provide the already exposed 256-case calibration set. The batch size is 32, learning rate 0.001, optimizer AdamW with weight decay zero, gradient clip 5, and *fixed final checkpoint* at 1,920 updates. Training draws with replacement from the original 1,024 train examples. The original ROC-02 setup used 240 updates with batch size 8, so this pilot increases the update count eightfold and the number of example presentations 32-fold (61,440 versus 1,920).

All train and calibration source SHA-256 values agree with the original merged ROC-02 manifest. Milestones 240, 480, 960 and 1,920 were fixed in advance, with the final checkpoint selected by training step, rather than calibration accuracy. The preregistered success gate was at least 90% train and 70% calibration accuracy at step 1,920.

## Results

| AdamW update | Train correct /1024 | Calibration correct /256 | Train CE | Calibration CE |
|---:|---:|---:|---:|---:|
| 240 | 296 | 54 | 1.39670 | 1.43565 |
| 480 | 338 | 66 | 1.36582 | 1.40506 |
| 960 | 330 | 58 | 1.36118 | 1.40464 |
| **1,920** | **362** | **72** | **1.34496** | **1.39429** |

The final accuracy is **35.35% train and 28.125% calibration**, failing both fixed thresholds. This single model improves descriptive dev accuracy over the mean of the original six depth-8 short-trained models (21.55%), but these are different seeds and budgets, so this is not a matched causal or generalization comparison. The original ROC-02 eight-way chance reference is 12.5%; the exact symbolic source interpreter remains 100% by construction.

The training accuracy remains low on the original data itself. This points to a **model/optimization task-learning bottleneck**, upstream of any proposed leakage-regularized observer transfer. More examples of the same four training rule families, at this fixed recipe, failed to establish reliable retrieval of the early A binding.

## Verified provenance and constraints

- Training input SHA-256: `843de4a066b456129ff32df8906d076e051cd7352f7fd73a07dbb78beb0f4054`.
- Calibration input SHA-256: `a9a8190aee2b4c2111afd8ceaf59422b55b483f7e54182795d986cc8739b818c`.
- Original 1,920-update CPU checkpoint SHA-256: `bac422572c6bac3dbeca11f3f49feed4a25d61742855a6e8690186f0c088be3d`.
- A separately loaded checkpoint reproduced the exact final train and calibration cross-entropies and accuracies.
- Full intermediate and final metrics are recorded in `roc03a_R8_S2026111701_metrics.json`. The binary training checkpoint remains a separate reproducibility attachment, with its SHA and step/seed identity registered in that JSON.
- Only **one** new model initialization was tested. No inference over model-seed variability, observer mechanism types, or new-family task performance is available.

**Next step:** an independently registered query-conditioned binding retrieval architecture, with a matched-compute original-architecture control and explicit held-out source-family boundary. Any later causal observer test must require demonstrably competent source-rule predictions, including a true legal counterfactual. The ROC-01/02 sealed family 6/7 gate remains closed.
