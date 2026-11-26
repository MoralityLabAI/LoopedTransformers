# KITE-01: exact regret certificates and recurrent strategic memory

Development pilot inspired by [Cogentic's MaximinKiteNE](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/bimatrix_nash_equilibrium). This is an independently derived certificate and heuristic benchmark, not an implementation of the paper's full polynomial-time algorithm.

Run from this directory with Node 18+:

    node run.mjs > replay.json

All payoff entries are BigInt numerators on a common positive denominator D, and strategy vectors are nonnegative integer weights with positive sum. Rational inputs can be mapped to this representation by clearing denominators. No floating-point arithmetic enters the regret test or repair. The only floating-point values in receipts are descriptive aggregate means and pseudorandom integer generation.

For regret numerators a,b over common denominator d, certify 5a <= d and 5b <= d. The kite test is 4a+b <= d and a+4b <= d. If 5a>d, move the row player's strategy toward an exact best response with mixing weight 1-d/(5a). The row regret becomes 1/5 and the column regret is at most (1-theta)b/d+theta <= 1/5 by 4a+b<=d. The column case is symmetric. Both regrets cannot exceed 1/5 in the kite. Outside the kite the repair returns null (some outside profiles may still be repairable by other methods).

Payoff perturbation: if each matrix entry changes by at most eta, a best-response payoff and a profile payoff each move by at most eta, so each regret and their maximum move by at most 2 eta. A compressed-game certificate at 1/5-2 eta transfers to the original game. Exact score on the original game is the final authority.

## Evidence

- Exhaustive finite audit: every binary 2x2 bimatrix game (256), every pair of denominator-four mixed strategies (25 per game), direct deviation oracle independently compared with the optimized formula. Both repair directions checked. This is exact finite-domain evidence, not a Lean formalization.
- Perturbation audit: 64 generated game pairs at eta=0.03; 192 exact inequality checks, sampled checks rather than a finite exhaustive proof.
- Development pilot: 128 generated roots, 4 families, sizes 4 and 8, 16 seeds per family/size. One root is one game; steps and arms share roots. Fictitious play uses uniform-prior action counts; multiscale uses equal mixture of the full-history strategy and recent 4/16/64-action distributions. Both simultaneously add one best response per player per step. Deterministic lowest-index tie break. Each arm receives 2*(depth+1) solver best-response queries; repair certification is separately counted (2 extra queries per successful kite check). The memory arm uses more arithmetic and storage, so equal oracle budget does not establish equal wall-clock cost. Raw and repaired profiles and exact regrets retained at every checkpoint.

Readout is development-exposed. No trained checkpoints, sham, sealed test, causal claim, FSI significance test, wall-time match, or polynomial convergence claim. No estimate of power until development effect/sham calibration. Comparisons: five paired depths, four exploratory family strata. Primary future contrast: certified pass-rate at depth 64, proposed meaningful floor 5 percentage points above the matched baseline, pending calibrated sham and resolution. Physical dose: fixed four-component history mixture; prospective learned intervention dose TBD. Consult EvalMath mechanistic-power/FSI-01 before any confirmatory experiment.

## Next experiment

Train a shared-update TRM with a strategy head and optional memory router, using games separate from development roots. External verifier alone decides success. Compare task-only, fixed window, learned window routing, rank/parameter-matched random routing, and LP-dual feedback with matched forward FLOPs, memory traffic, verifier calls and wall time. Include support enumeration for small games, zero-sum LP, independent multistart and practical approximate-Nash solvers as strong baselines. Register a fresh sealed family split and independent model seeds. Quantify both direct 1/5 success and repair-assisted success. Test action permutation equivariance, rectangular games, degenerate games, scaling, and rational input bit length.

For BPD, use donor and matched-sham replacements of specific memory components and measure paired future regret and certificate flips; a component probe alone is observational. For TheySing, use this only for explicitly frozen two-player subgames; pairwise certificates do not imply a multiplayer Nash equilibrium. For DataGoblin, emit root IDs, payoff matrices or exact generator seeds, action weights, regrets and certificate verdicts. For Compactification-Theory, certify approximation errors in regret/termination witnesses instead of assuming that compressed memory preserves the Cogentic proof.
