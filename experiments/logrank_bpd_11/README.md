# LOGRANK-BPD-11: sparse decoding, noise budgets and rank expansion

Development only. 32 fresh generator worlds: 24 small worlds (six per k=16/32, r=8/12), eight scale-only worlds (four each k=64, r=24/48). Known coordinate correspondence, independent uniform hidden signs, r+3 independent even physical parity masks. All observed physical queries have even weight; sparsity refers to coefficient masks in the supplied quotient basis, not the physical coordinates or a learned BPD gauge. Every schedule starts with r basis observations, then repeats basis equations or samples sparse (2/3 coefficient bits) or dense (any nonzero coefficient mask) rows. 32 novel test masks excluded from observation schedules. Known independent binary symmetric noise, common uniforms by observation step, nested budgets. No model training or native BPD intervention. Independent unit is the world, not each repeated budget, query or condition.

## Reproduction

```sh
cd experiments/logrank_bpd_11
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py > audit.json
```

NumPy only. Source blob hashes match executed local files. The full receipt and all root-level rows are retained in the experimental bundle outside Git; summary_receipt.json records aggregate outcomes and raw/source digests. Timing is descriptive and host-specific.

## Small-world high-noise results (p=.35; 768 query decisions per arm and budget)

| m/r | Repeat exact/BP | Sparse exact/BP | Dense exact/BP |
|---:|---:|---:|---:|
| 2 | 387/387 | 388/366 | 359/364 |
| 4 | 397/397 | 406/391 | 436/356 |
| 8 | 423/423 | 498/471 | 503/371 |
| 16 | 477/477 | 638/578 | 630/419 |
| 32 | 647/647 | 740/709 | 750/538 |

At 32r the exact MAP state is correct on 14/22/23 of 24 roots (repeat/sparse/dense). BP bitwise state is correct on 14/20/10. Sparse BP covers 690/768 queries at a nominal 90% confidence cutoff, with 672 correct; dense BP covers 320, all correct in this sample. Approximate beliefs are not certified posterior confidence. At .15 noise, sparse BP and exact decode all 24 states at 8r; dense exact also recovers all states, but dense BP only 21. At clean noise all small-world conditions are exact.

Repeated observations on a mask are combined by summing factor LLRs. BP uses synchronous damped sum-product (0.5 damping, at most 60 iterations, finite clipping) and products of approximate single-bit marginals for parity predictions. Dense random parity factors are difficult for this message rule: long products can suppress messages, and loops can induce correlated/overconfident estimates. We did not isolate the cause of every failure. No assertion that sparsity always outperforms dense observations, other BP schedules or modern error-correcting codes. Channel capacity is a necessary asymptotic full-state bound, not a finite threshold.

## Large-world scale diagnostic

At rank 24/48, .15 noise, 8r sparse BP recovered 8/8 blocks and all 256 query decisions. At .35, 32r, rank 24 sparse BP 128/128, 4/4 blocks; repeat 75/128, 1/4 blocks. Rank 48 sparse BP 111/128, 3/4 blocks; repeat 72/128, 1/4 blocks. Large ranks have no exact posterior control. Four worlds per rank, no inferential claim. Median local sparse BP solve times at .35 and 32r: approximately 4.1 ms (r=24), 11.1 ms (r=48), excluding world construction and scoring; exact posterior enumeration is exponential.

## Three rank-expanding calls versus within-span calls

Starting at dense 4r, allocate three extra calls. Three previously unobserved independent basis directions each define eight public target masks. Expansion probes use one direction plus an offset drawn from 48 candidates excluding the direct targets. Within-span baseline uses three new in-span masks. The initial posterior on S is unchanged by a single noisy observation involving an independent uniform new bit U. For target Z_q=(-1)^(U+q dot S) and probe Y=(-1)^(U+a dot S+E), the posterior mean is E[Z_q|Y]=rho Y mu_(q xor a), where rho=1-2p and mu denotes the original posterior Walsh moment. This follows because Y is uniform conditional on S and E. Expected Brier improvement from the original uniform target is rho^2 mu_(q xor a)^2/4. Choose a maximizing average squared moment over the eight target offsets. This is exact one-step Brier gain under the uniform new-direction prior, not a realized accuracy guarantee.

| Noise | Within span | Random expansion | Target expansion |
|---|---:|---:|---:|
| 0 | 297/576 | 576/576 | 576/576 |
| .15 | 297/576 | 509/576 | 509/576 |
| .35 | 297/576 | 315/576 | 336/576 |

The within-span outputs are deterministic tie guesses on genuinely uniform posteriors; the 297 correct is sample fluctuation. Expansion increases observation rank by three; it leaves the original in-span posterior untouched. At .35 within-span probes improve original target correctness from 436 to 456/768, whereas expansion leaves 436. This is the rank-versus-redundancy tradeoff. At .15 or .35 all expansion policies abstain on new queries at the 90% confidence cutoff: one noisy call per new direction cannot exceed 1-p posterior confidence. Target selection improves expected Brier by construction, but observed gains versus random are small; at .35 Brier .24721 vs .24802, at .15 .10290 vs .10325. No claim of statistical superiority.

## Independent audit

PASS: 34,560 exact posterior query probabilities recomputed using integer likelihood weights, 39,680 BP query probabilities, 16,560 BP marginal values recomputed by a separate edge-list implementation, 12,096 expansion probabilities by explicit enlarged-state posterior enumeration, and 216 target choices independently checked. Maximum BP marginal difference 1.1879386363489175e-14; zero iteration-count/numeric mismatches. Verified rank, even-mask property, test exclusion, hidden-state consistency, observation noise, block correctness and score counters. No native model or optimizer replay. Raw bundle includes complete receipts.

## Next
Prospective rank-adaptive query policy with a unified downstream loss: allocate calls between increasing rank and reducing uncertainty, compare random, exact small-r Bayes risk, sparse BP and hybrid decoder. Add uncertain/noisy learned basis maps only after the supplied-basis controls. Track graph degree, edge count, iterations, memory, calibration and confidence-gated verifier repair. No protected data, native causal claim or confirmation release.
