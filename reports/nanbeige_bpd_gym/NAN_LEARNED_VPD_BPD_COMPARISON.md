# Learned BF16 VPD/BPD pilot: results and specificity limits

Both final learned assets are real pretrained rank-8 layer-21 down_proj fits, each with eight stock optimizer steps and independent checkpoint/controller/diagnostic audits. Each completed all 28 source/loop SAE/SVD/Haar probe groups and six geometry cells. The original failed T2 attempt and NF4 failed competence gate remain intact.

| Asset | Fit depth | Factor update L2 | Controller update L2 | Saved-tensor CPU peak MiB |
|---|---:|---:|---:|---:|
| VPD_T1 | 1 | 0.145263 | 0.191444 | 69.49 |
| BPD_T2 | 2 | 0.117528 | 0.182308 | 536.03 |

| Asset | DEV loop | Mode | Correct | Full-vocabulary KL mean |
|---|---:|---|---:|---:|
| VPD_T1 | 1 | all_one | 2/8 | 0.000000 |
| VPD_T1 | 1 | clean_informed | 3/8 | 0.089136 |
| VPD_T1 | 1 | component_only | 2/8 | 0.078037 |
| VPD_T1 | 1 | reached_state | 3/8 | 0.089136 |
| VPD_T1 | 1 | static_CAL_mean | 2/8 | 0.084169 |
| BPD_T2 | 1 | all_one | 2/8 | 0.000000 |
| BPD_T2 | 2 | all_one | 7/8 | 0.000000 |
| BPD_T2 | 1 | clean_informed | 2/8 | 0.004754 |
| BPD_T2 | 2 | clean_informed | 7/8 | 0.538796 |
| BPD_T2 | 1 | component_only | 2/8 | 0.004809 |
| BPD_T2 | 2 | component_only | 7/8 | 0.610177 |
| BPD_T2 | 1 | reached_state | 2/8 | 0.004754 |
| BPD_T2 | 2 | reached_state | 7/8 | 0.538629 |
| BPD_T2 | 1 | static_CAL_mean | 2/8 | 0.004760 |
| BPD_T2 | 2 | static_CAL_mean | 7/8 | 0.563963 |

BPD loop-2 all-one, component-only, static-CAL-mean, clean-informed and reached-state routes each retain 7/8 DEV correctness. Static replacement also retaining correctness means this pilot does not demonstrate behavioral specificity of learned components. Reached-state masks differ from clean-input masks at loop 2, but that alone is not mediation. The full aggregate preserves the measured write/state doses, mask differences and KL shifts; high label retention is not a substitute for a matched causal contrast.

The two learned physical component operators have relative Frobenius difference 0.018662 and cosine 0.999827. Output-subspace principal cosines are in the JSON. These use physical V@U and subspaces, not same-column component matching. No backend SAE-index subtraction.

T1 and T2 use different recurrent fit depths and registered ablation/adversary rules; the second condition also uses a separately registered memory implementation. Their difference is not an isolated causal BPD advantage. Each dataset has only eight CAL/eight DEV families, with reused readouts and one checkpoint. No formal comparison, equivalence, mechanism confirmation or donor transfer is claimed. A useful next causal pilot must discriminate learned donors from matched sham/static controls at fixed physical doses, check future clean competence at the target depth, and calibrate a meaningful effect floor before any sealed confirmation.

The original 168-case test remains sealed. Raw predictions, states and per-record derivatives remain local.

[Complete grouped evidence](nan_learned_VPD_BPD_comparison.json) · [Original fit/failure](NAN_FIT07_PROGRESS.md) · [Completed BPD memory condition](NAN_RECOMP09_PROGRESS.md)
