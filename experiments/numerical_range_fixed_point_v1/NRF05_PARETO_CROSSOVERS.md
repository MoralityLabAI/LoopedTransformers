# NRF-05: Post-hoc objective crossover and depth effect

The twelve NRF-04 observations expose a tradeoff between held-out task error and local leakage. For a frozen decoder arm (a), define (J_a(\lambda)=E_a+\lambda L_a^2\), where (E) is held-out task MSE and (L) is average local dark-fiber sensitivity. The reference at each lambda is the **median of the eight complete random-arm objectives** (J_{r,i}(\lambda)). Computing the medians of (E_i) and (L_i) separately creates a fictitious projection and is explicitly rejected.

The post-hoc crossing for each checkpoint is the smallest nonnegative lambda at which supervised (J_s(\lambda)) reaches the random-objective median. A standard-library exact-rational program `research_gym/analysis/nrf05_pareto_crossings.py` partitions at all pairwise random-line intersections and solves each resulting affine segment.

At 4,096 exposures, the three-seed median crossing is **12.49 for R16** and **0.270 for R32**. At lambda=1, all six R16 checkpoints favor the supervised decoder and all six R32 checkpoints favor the random-control median. The observation is depth-dependent because the R32 leakage penalty is much larger relative to its task advantage.

This analysis is post hoc, after the original evaluation data were exposed, and has no prospective claim. A new-data transfer study may freeze lambda=1 and evaluate fresh data folds without reselecting based on those folds. The outcome would be a within-checkpoint distributional replication, not independent replication across model-training seeds.

Exact values, provenance, and caveats are stored in [nrf05_pareto_crossovers.json](nrf05_pareto_crossovers.json).
