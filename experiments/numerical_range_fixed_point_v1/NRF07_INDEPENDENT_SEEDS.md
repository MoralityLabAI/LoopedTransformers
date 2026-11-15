# NRF-07: independent-seed replication of the depth-dependent observable tradeoff

Date: 2026-11-15. The protocol was **committed before any new seed was trained** in configs/nrf07_independent_seeds_v1.json. New seeds 109, 113, 127 were selected for both R16 and R32 tied ToyLoop models. Each model was trained from initialization with the original signed-permutation loss, CPU PyTorch 2.10.0+cpu, AdamW learning rate 0.001, residual alpha=1 and beta=0.25, batch size 8, 4096 state-visit exposures. Snapshots were evaluated at 2048 and 4096 exposures.

The held-out decoder experiment followed NRF-06 exactly: rank-8 train-only supervised covariance subspace, eight equal-rank random projection controls, one shifted-label null and one coordinate baseline, ridge=1, 160 train examples on fold 1, 96 evaluation inputs on fold 5 and 12 shadow Jacobian inputs on fold 6. The objective is J = heldout_MSE + mean_shadow_leakage^2, with lambda fixed to 1 after prior NRF-04 and **not selected from these outcomes**. The random comparator is the median of eight complete random-arm J scores.

## Results

| New seed cohort | Supervised mean J | Random median mean J | Supervised wins |
|---|---:|---:|---:|
| R16 at 2048 exposures | 0.468331 | 0.604103 | 3/3 |
| R16 at 4096 exposures | 0.406400 | 0.571176 | 3/3 |
| R32 at 2048 exposures | 1.550201 | 1.137233 | 0/3 |
| R32 at 4096 exposures | 1.296228 | 1.059480 | 0/3 |

The **precommitted joint gate passes** on all six final independently trained models. Supervised J improves between the two exposures on all six paired models.

## Reproducibility and limits

- Reproduction code: research_gym/analysis/nrf07_independent_seed_train.py.
- Deterministic source training construction: lsa/kappa_probe.py ToyLoop, and research_gym/scripts/bench_loop_schedule_algebra_v0_1.py signed_permutation_batch.
- Compact row-level measurements and state_dict SHA-256 values: nrf07_independent_seed_summary.json.
- The conversation's downloadable nrf07_reproduction.zip includes twelve .pt snapshot files, the full eleven-arm JSON records for six new model-seed/depth combinations, and the standalone CPU trainer. Full model checkpoint bytes were kept out of this GitHub commit because they can be regenerated from seeds.

As a source-recipe parity check, rerunning old seed 101 at R16 on CPU gave W1[0,0]=-0.0599963814 and W2[0,0]=0.0425123386 at exposure 2048. The original CUDA checkpoint entries were -0.0599963889 and 0.0425123349, respectively, with comparable recorded training losses. CPU and CUDA exhibit small arithmetic differences, and the new-seed checkpoint bytes belong to the new CPU protocol rather than the old CUDA campaign.

**Interpretation:** This replicates the *direction* of the depth-dependent tradeoff with freshly initialized model weights and new task seeds inside one synthetic family. It remains an exploratory follow-up because lambda=1 was selected after studying NRF-04. The six final cells share the same architecture and protocol, and the three seeds at a given depth are the independent units. Neither cross-architecture transfer nor neural capability improvement has been established.
