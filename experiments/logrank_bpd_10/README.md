# LOGRANK-BPD-10 / 10B: noisy parity identifiability and targeted verifier calls

Development only. 24 independently generated worlds (six per k=16,32 and rank=8,12), 72 noise conditions; conditions share worlds. Known coordinate correspondence, independent uniform hidden signs, even nonzero masks, no singleton labels. Training has r basis observations and 3r redundant observations (repeats allowed). In-span test masks, out-of-span test masks and probe pool disjoint. Each noise condition shares nested noise uniforms across rates and probe methods. Bayesian decoder knows the binary symmetric flip probability. One unit of observation is a noisy parity equation, not a token or an independent task family.

## Reproduce
```sh
cd experiments/logrank_bpd_10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py
python audit.py > audit.json
OPENBLAS_NUM_THREADS=1 python acquisition.py
```
NumPy only. Full receipts and audit locally archived; this branch records aggregate and all root-level outcomes plus digests. Audit recomputes posterior probabilities with exact integer likelihood weights, checks saved hard-sign predictions and chosen entropy probes, and constructs 768 explicit nullspace witnesses. 20,736 probabilities and 288 entropy choices passed. It does not replay optimizer trajectories. 10B checks the chosen probe's expected Brier gain against both hypothetical Bayes updates (288 checks), using the same exposed roots. Source Git blobs match local code.

## Results (768 in-span queries per noise rate)
| Noise | Basis only | Hard sign | Posterior | Entropy +4 | Random +4 | Target +4 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 768 | 379 | 768 | 768 | 768 | 768 |
| .15 | 452 | 387 | 740 | 744 | 754 | 741 |
| .35 | 399 | 372 | 441 | 444 | 458 | 463 |

Four-restart straight-through hard sign optimization, 200 full-batch Adam steps, selected by minimum final training error, is a limited gradient proxy, not native BPD or TRM training. Clean-data failure is evidence about this optimizer and fixture, not a general neural impossibility. No native model or new CI execution. At .15 posterior 90%-confidence policy answers 683/768 and all 683 are correct in this sample; at .35 answers zero. At all rates the out-of-span posterior is exactly uniform by nullspace symmetry; hard-sign guesses are near 50%. Theoretical 50% ceiling is in expectation under the uniform hidden-sign prior, not a finite observed ceiling.

Entropy chooses most uncertain available in-span probe. 10B uses downstream query masks (public, not their answers) for expected mean Brier-risk reduction. For posterior Walsh moments mu_q=E[(-1)^(q dot H)], rho=1-2p, the expected Brier gain for target q and probe a is rho^2 (mu_(q xor a)-mu_q mu_a)^2/[4(1-rho^2 mu_a^2)], zero when the observation is deterministic. This is law of total variance and regression on a binary observation; mean over target masks, maximize over pool, compute moments via fast Walsh-Hadamard transform. Direct conditional-update checks validate the chosen gains. 10B is adaptive after original results, not prospective. At .35 it gains only five correct queries over random, while at .15 it loses thirteen; no superiority claim. Brier losses at .35: initial .242871, entropy .241173, random .241568, target .240656. No inferential pooling of 768 correlated query rows.

## Capacity
For uniform r-bit quotient state S and m binary symmetric observations (even adaptively selected from past responses with independent noise), I(S;Y^m) <= m(1-H_2(p)) by the conditional entropy chain rule. Thus H(S|Y^m)>=r-mC, C=1-H_2(p). Fano implies full quotient-state block error >=1-(mC+1)/r (clip at zero). At .15, C=.39016 and the capacity lower bound for asymptotically reliable full-state recovery is m/r>=2.563; at .35 C=.065932 and m/r>=15.167. These are asymptotic information limits, not sufficient finite-sample budgets or a bound on every individual query. Initial m/r=4 is above the former threshold and below the latter. All in-span calls preserve exact out-of-span symmetry even under noise. Expanding rank with informative out-of-span probes requires a separately costed protocol.

## Next
LOGRANK-BPD-11: prospective redundancy/rank sweep, noisy-channel thresholds, target-aware rank-expansion probes, GF(2) solver with uncertainty and structured decoding baselines, and a learned acquisition policy evaluated on fresh worlds. Explicitly distinguish identifiability, confidence, cost and realized error. No protected cohort touched.
