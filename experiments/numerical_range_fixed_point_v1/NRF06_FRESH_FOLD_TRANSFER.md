# NRF-06: Fresh-input transfer of the task/leakage depth tradeoff

Date: 2026-11-15. The frozen protocol was committed in `configs/nrf06_regional_observation_transfer_v1.json` before producing new data. It selected `lambda=1` **after** NRF-04, so this study tests transport of an exploratory prediction to different input samples at the same trained checkpoints.

## Experimental setup

- Twelve original tied ToyLoop checkpoints, R16 and R32 at state-visit exposures 2048 and 4096, seeds 101,103,107; unchanged weights and previously SHA-256-pinned model identities.
- Eight-dimensional target-supervised decoder trained on the unchanged input fold 1 with 160 examples; eight matched-rank random projections plus first-eight-coordinate and label-shifted controls.
- Held-out task examples from previously unused generator fold 5 (96 examples); local Jacobian-fiber leakage measured on fold 6 (12 separate inputs).
- Fixed total objective for each decoder arm: (J=\mathrm{MSE}+\mathrm{Leakage}^2). Random comparator is the **median of eight complete random-arm objective scores**. The observed difference is paired by checkpoint.

## Prespecified transfer comparison

| Model | Final supervised wins (lower objective) | Final random wins |
|---|---:|---:|
| R16 | 3/3 | 0/3 |
| R32 | 0/3 | 3/3 |

The selected prediction holds across all six final cells. The mean final objective at R16 is 0.4253 for supervised and 0.5731 for random controls. At R32, supervised is 1.3459 and random 1.0498. Every matched supervised objective also decreases from 2048 to 4096 state-visit exposures.

The per-checkpoint arms, task MSE, leakage and objective are preserved in `nrf06_r16_fresh_fold.json`, `nrf06_r32_early_fresh_fold.json`, and `nrf06_r32_final_fresh_fold.json`. A Python PyTorch implementation and an independent deterministic JavaScript float64 tensor replay are committed. The latter was used to produce the receipts through the GitHub connection, with preserved source Git blob identities.

## Interpretation boundary

This is a new **input split** with unchanged trained checkpoints and the same model-training seeds. The objective weight was selected using prior evaluation data. The replication supports a depth-dependent task/leakage tradeoff on these models. It remains exploratory, lacks independent model-seed replication, and does not establish an exact or uniform fiber quotient. The exact global linear quotient is obstructed by NRF-05's modular no-go result, motivating *regional* approximate representations.

The private repository's GitHub Actions runs continue to fail before worker steps, so native PyTorch cloud verification is outstanding. Both implementations are available for local cross-validation.
