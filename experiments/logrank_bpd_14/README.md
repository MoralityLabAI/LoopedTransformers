# LOGRANK-BPD-14: multi-frame BP and bounded likelihood repair

Development only. 36 fresh worlds: 12 calibration (four per d=11/24/48) and 24 separate evaluation worlds (eight per width). Paired .15/.35 noise conditions. Identical fixed-budget induced-BP acquisition trace from 13 for each decoder: 24 extra cost units, no early stopping; initial 4r observations excluded. No planted frame used by new decoders, native TRM/BPD training, causal intervention, protected data or confirmation.

## Reproduce

NumPy >=2.0 (bitwise_count).

```sh
cd experiments/logrank_bpd_14
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot14.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit14.py
```

Public frames retain independent observed rows first, ordered chronologically, by absolute repeated-observation vote or by a seed-determined shuffle. Public cost-ordered candidate rows complete the frame. The first R basis rows span observations. BP yields approximate marginal means. Ensembles average query predictions, using exact span gating. Repair enumerates a subcube by fixing all but u selected bits to BP modal values. Map each state to a common observable quotient by evaluating the canonical basis rows in the proposed frame, deduplicate, score with full observed likelihood. Normalizing over visited states gives a truncated posterior, not a certified full posterior. Outside-span targets return .5. Candidate counts are capped; RAM is not explicitly capped. Timings exclude acquisition and include common construction of eight frames even for single-frame arms. Scratch counts selected numerical arrays, not peak RSS.

## Main evaluation

Weighted Brier (lower is better), eight independent worlds per width/noise. All arms retained.

| Decoder | d11 .15 | d24 .15 | d48 .15 | d11 .35 | d24 .35 | d48 .35 |
|---|---:|---:|---:|---:|---:|---:|
| fixed_induced | .01025 | .04062 | .09613 | .24891 | .24977 | .25000 |
| observed | .05106 | .00176 | .07431 | .25001 | .24982 | .25000 |
| ensemble4 | .02468 | .02463 | .10748 | .24944 | .24997 | .25000 |
| ensemble8 | .03561 | .05194 | .12387 | .24892 | .24998 | .25000 |
| repair1 | .01944 | .00081 | .04077 | .23812 | .26443 | .25367 |
| repair4 | .01942 | .00074 | .04017 | .21832 | .26150 | .29347 |
| repair8 | .01940 | .00082 | .04290 | .22432 | .24883 | .25808 |
| random_repair8 | .01940 | .00053 | .02337 | .22406 | .24194 | .29095 |
| exact | .01940 | n/a | n/a | .22567 | n/a | n/a |

At d48/.15 random-coordinate repair reaches .97917 weighted accuracy vs .87760 baseline, 75.7% lower realized Brier, but improves only 5/8 paired worlds (two near numerical zero and one materially worse). Single-frame repair improves 6/8, .94661 accuracy at about 20.3ms mean decode time vs 6.8ms baseline and 95.9ms random eight-frame repair. At d24/.15 observed-first nearly solves the task. Probability averaging can be harmful. More states and uncertainty-selected neighborhoods do not reliably improve performance. At d48/.35 every uncalibrated repair worsens Brier vs uniform .25; random repair reaches .29095. Eight roots per stratum, best arm identified after evaluation; descriptive only.

## Retained posterior mass

For visited set C and P the true posterior, Q=P(.|C), alpha=P(C)>0, TV(P,Q)=1-alpha. The sum of absolute differences inside C equals 1-alpha, and outside C equals 1-alpha. For every Boolean event the absolute probability error is at most 1-alpha; its true probability is at least alpha times its truncated probability. No global confidence certificate follows from a small residual or agreement unless a lower bound on alpha is available. Outside-span targets are uniform by nullspace symmetry.

At d11 exhaustive enumeration measures alpha directly. At .15 repair8 mean .999994, min .999985; at .35 mean .935077, min .833785. Maximum probability errors .00001411 and .03472, respectively. One-frame repair at .35 retains mean .43823 with max probability error .48504. No exact retained mass at larger widths. Written proof, not Lean formalized.

Temperature grid {1,2,4,8} chosen by calibration-only Brier, pooling widths: T=2 at .15, T=8 at .35. Applies to repair8, not random repair. At d48/.15 calibrated coverage .33594 vs uncalibrated .66536, all covered correct in these roots; unanimous eight-frame guard drops coverage to zero. d24/.15 guard .04688; d11/.15 .40625. At .35 calibrated policy abstains everywhere. At d48/.15 temperature worsens Brier .04290 to .06609; .35 improves .25808 to .24876. Guard is too conservative; not a deployment certificate.

## Audit

PASS: 72 trace conditions, 1,089 budget/noise/rank steps, 18,432 independently replayed frame moments, 2,304 baseline probabilities, 6,912 ensemble probabilities, 9,984 integer-likelihood probabilities, 225,365 distinct candidate-state likelihoods within condition unions, 23,808 target score records and 288 same-source candidate-generation replays. Maximum probability/moment discrepancy 7.36e-13. Calibration grid replayed on calibration roots; exact quotient at d11 verifies retained mass. No independent nonrandom acquisition-choice replay, no CI. Full raw receipts, candidate IDs, full-precision aggregate summary and all source files are in the downloadable experiment bundle outside git. The repository summary is a compact rounded transcription of aggregate results and exact source/raw digest records.

## Next

Costed MMS gate: compare one-frame repair, random multi-frame repair and likelihood-residual candidate expansion on fresh diverse observation families. Match candidate evaluations and measure actual wall time and RSS. Bound omitted likelihood with disjoint branch-and-bound subtrees and factor likelihood maxima; use a lower bound on retained mass for confidence or explicit abstention. Keep high-noise uniform baseline and exact small-quotient control. No protected cohort or native causal claim.
