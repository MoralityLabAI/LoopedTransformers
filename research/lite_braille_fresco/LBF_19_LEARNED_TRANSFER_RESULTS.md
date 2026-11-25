# LBF-19 Track R Results — Can a small recurrent cell learn the guarded support transfer?

Development-only track; registration: [`LBF_19_LEARNED_TRANSFER_REGISTRATION.md`](LBF_19_LEARNED_TRANSFER_REGISTRATION.md) (commit `ab09c870`, pushed before any test-set scoring). Code: [`lbf19_learned_transfer.py`](lbf19_learned_transfer.py), tests: [`tests/test_lbf19_learned_transfer.py`](tests/test_lbf19_learned_transfer.py). Data: [`results/lbf19/learned_transfer/learned_transfer_summary.json`](results/lbf19/learned_transfer/learned_transfer_summary.json). Figures: [K](results/lbf19/learned_transfer/fig_extrapolation_K.svg), [L at K=3](results/lbf19/learned_transfer/fig_extrapolation_L_K3.svg), [L at K=8](results/lbf19/learned_transfer/fig_extrapolation_L_K8.svg).

> **Status of claims.** Boolean support stays the proof-relevant object. Every learned score here is a *ranking/approximation signal with no soundness guarantee*. The only sound Boolean output is the hard-sound hull `U_k` (exact Boolean pos1 rewrite hull; learned scores only rank cells inside it). Test setup was exactly the registered one; nothing was tuned on test. Deviations: none in design; execution was sharded one process per training seed (1 torch thread each, because the machine was saturated by other tracks).

## 1. Setup (as registered)
Target `Y_k` = symbol support of `symbolic_prov_adj2_guarded(fable_len_guard, fp=4)` after `k` passes (∪ start word). Rule systems: 400 train / 80 val / 100 test (**roots**), disjoint by canonical lineage (asserted; unit-tested). Train: `L∈8..12`, `k≤3`, 3,200 words. Test: 100 roots × 2 words × `L∈{8,10,12,16,20,24}`, `k=1..8` (1,200 words). Arms: recurrent cell with/without zero-preserving gate (29,079 params), `nonrec_param` (29,279), `nonrec_compute` (77,133; 3 untied blocks ≈ FLOPs of 3 recurrent steps), hard-sound in-loop-masked variants, post-hoc masked variants, hull baseline, oracle. 5 training seeds × 6 trained arms, 1,500 steps. `τ` = largest threshold with pooled val recall ≥ 0.99, per arm (applied unchanged to all test cells).
Hull soundness check: `Y_k ⊄ U_k` violations on test = **0** for all `L` (all words, `k≤8`).

**Reading the numbers.** Most cells in `Y_k` are the start word's own symbols (trivially reachable), so recall/precision are high for everything, including the **arm-independent hull baseline (precision 0.95/0.92/0.88 at K≤3/K=3/K=8; recall 1.0)**. The informative quantities are therefore: *novel-cell recall* (cells not in the start word), word-level false-negative rate, precision *relative to the hull*, and in-hull AP. Means are over 5 seeds (± sd of seed means), root-level means over 100 roots.

## 2. Main results (test; `rec_prior` unless noted)

| subset | arm | recall@τ | prec@τ | novel recall | word-FN rate | AP | in-hull AP |
|---|---|---|---|---|---|---|---|
| K=3, L≤12 | `rec_prior` | 0.983±0.003 | 0.959±0.004 | 0.954 | 0.139 | 0.998 | 0.998 |
| | `rec_noprior` | 0.991±0.003 | 0.274±0.012 | 0.972 | 0.119 | 0.888 | 0.987 |
| | `nonrec_param` | 0.990±0.001 | 0.273±0.008 | 0.970 | 0.118 | 0.881 | 0.987 |
| | `nonrec_compute` | 0.989±0.002 | 0.307±0.015 | 0.967 | 0.132 | 0.926 | 0.990 |
| | hull (sound) | 1.000 | 0.919 | 1.000 | 0 | 0.925 | 0.925 |
| K=8, L≤12 | `rec_prior` | 0.959±0.003 | 0.965±0.003 | **0.907** | **0.187** | 0.997 | 0.997 |
| | `rec_noprior` | 0.987±0.004 | 0.286±0.011 | 0.964 | 0.157 | 0.879 | 0.980 |
| | `nonrec_param` | 0.995±0.002 | 0.273±0.004 | 0.988 | 0.063 | 0.873 | 0.979 |
| | `nonrec_compute` | 0.996±0.002 | 0.286±0.010 | 0.989 | 0.060 | 0.918 | 0.984 |
| | hull | 1.000 | 0.882 | 1.000 | 0 | 0.894 | 0.894 |
| L=24, K=3 | `rec_prior` | 0.983±0.003 | 0.964±0.003 | 0.955 | 0.228 | 0.998 | 0.998 |
| | hull | 1.000 | 0.928 | 1.000 | 0 | 0.928 | 0.928 |
| L=24, K=8 | `rec_prior` | 0.960±0.003 | 0.969±0.003 | 0.911 | **0.321** | 0.997 | 0.997 |
| | `rec_noprior` | 0.989±0.004 | 0.276±0.009 | 0.968 | 0.266 | 0.881 | 0.980 |
| | hull | 1.000 | 0.884 | 1.000 | 0 | 0.884 | 0.884 |

Full tables for all 11 evaluated arms and 7 subsets (including masked and `_hs` variants, strict-`>0` counts, size ratios) are in the JSON and reproducible via `--stage merge`. Strict `y>0` false negatives: **0 for every arm and subset** — but this is nearly vacuous: sigmoid outputs are never exactly 0, and the gated prior's strict support equals the Boolean hull by construction (`test_models_shapes_params_and_zero_preservation`). Strict-`>0` precision equals the hull's for the prior arm.

Curves (figures): recall of `rec_prior` decays slowly with `K` (0.992 → 0.959 from pooled K≤3 to K=8) and with `L` at K=8 it stays at ≈0.96; unmasked non-prior arms *gain* recall with `K` only because the fixed low `τ` (0.02–0.03) predicts ~3.6× too many cells (size ratio 3.6–3.9, precision ≈ 0.27–0.29).

## 3. Hypothesis outcomes (registered criteria)

| H | verdict | evidence |
|---|---|---|
| **H1** learnable in-dist (K=3, L≤12: recall ≥0.98 & prec ≥0.90) | **Pass, marginal** | recall 0.983, prec 0.959. Novel-cell recall only 0.954; 13.9% of words miss ≥1 cell. |
| **H2** K-extrapolation (K=8: recall ≥0.95 & prec ≥0.80) | **Pass on the letter, weak** | recall 0.959, prec 0.965; **novel recall 0.907, word-FN 18.7%** — no collapse (<0.90), but visibly degrading, and no longer sound-looking at word level. |
| **H3** L-extrapolation (L=24, K=3) | **Pass** | recall 0.983, prec 0.964; word-FN 22.8% (it grows with L because words contain more cells). Joint L=24,K=8: recall 0.960, word-FN 32.1%. |
| **H4** prior helps (AP, root-level 95% CI) | **Prior helps** | `rec_prior − rec_noprior` AP: K=8,L≤12 **+0.118 [0.110, 0.125]**; L=24,K=3 **+0.107 [0.099, 0.115]** (seed-level t-CI also > 0: [+0.095,+0.140]/[+0.084,+0.131]). With hard-sound masking on both sides the effect shrinks but persists: +0.091 [0.072, 0.111] / +0.060 [0.048, 0.072]. Much of the unmasked gain is that the prior structurally removes out-of-hull false positives, which a free readout does not learn at this budget (precision 0.27 vs 0.96). Cost of the prior: **lower recall at fixed val-calibrated τ and higher word-FN rate** (K=8: 0.187 vs 0.157 for `rec_noprior`, and vs 0.06 for nonrec arms). |
| **H5** recurrence helps (no-prior arms, AP) | **No** | `rec_noprior − nonrec_param`: +0.006 [0.004, 0.008] at root level but seed-level CI [−0.034, +0.046] (negligible/not reliable). `rec_noprior − nonrec_compute`: **−0.038 [−0.041, −0.035]** (root), seed-level [−0.075, −0.001]: the compute-matched stacked net with 2.6× the parameters is *better*. After post-hoc hull masking all three are equal within ±0.001 AP. Same pattern at K=8, L=24 and joint extrapolation; the non-recurrent nets given `k/8` as a scalar **extrapolate in K as well as the recurrent cell** in this setup. Confound (pre-registered): arm (a) has no non-recurrent counterpart with the prior. |
| **H6** hard-sound readout | **Sound by construction; learning adds ranking only** | (i) `Y_k ⊆ U_k` on 100% of test words, k≤8; (ii) all `_hs`/masked arms strict FN = 0; (iii) in-hull AP lift over the hull's own AP, root-level: `rec_prior` **+0.073 [0.060, 0.087]** (K=3), **+0.103 [0.083, 0.125]** (K=8), +0.113 at L=24,K=8; `rec_noprior_hs` only +0.009–0.014 (trained in-loop masked it is *worse* than post-hoc masking of the unmasked nets: AP 0.906 vs 0.980 at K=8). The sound Boolean output is exactly `U_k`: precision 0.95/0.92/0.88 (K≤3/K=3/K=8), i.e. the learned model cannot shrink a sound output; it only ranks. |
| **H7** cost | descriptive | below |

## 4. Cost (CPU; **timings are noisy: the machine was saturated by concurrent tracks**)
Params: 29,079 (recurrent), 29,279 (`nonrec_param`), 77,133 (`nonrec_compute`). Training (1,500 steps, 1 thread, contended): 120–200 s for 29K-param arms, 322 s for `nonrec_compute` (uncontended single-run smoke ≈ 17 ms/step ≈ 26 s). Exact oracle (pure Python guarded transfer, K=8): 17.4 / 24.2 / 34.7 / 47.7 / 68.6 / 94.1 ms per word at L = 8 / 10 / 12 / 16 / 20 / 24 (grows with L). Learned arms, batched 200 words per L, K=1..8: 2.0–3.2 ms/word; single-word 3.4–13 ms (unmasked); hull (batched Boolean torch) 1.4 ms/word. So the learned nets are roughly 3–30× cheaper than the Python oracle at long L, but (a) the repo's production support path is a `u64` bitset/Rust implementation far faster than this Python reference, which was not benchmarked here, and (b) the non-sound learned output is not a substitute for it. No claim of speedup over the production bitset support.

## 5. Honest conclusions
1. **A tiny gated recurrent cell learns the one-step rewrite structure well enough to rank**, and ranks inside the hull far better than the hull alone (in-hull AP 0.997 vs 0.89 at K=8), also at `L=24` and `K=8` never seen in training. But as a *Boolean* approximation it is **not sound**: at the val-calibrated τ, 4–5% of true cells (9% of *novel* cells) are missed at K=8 and 19–32% of words lose at least one cell. Recall degrades slowly with K and L; no collapse below 0.90 overall, but novel-cell recall falls to 0.907.
2. **The zero-preserving, support-gated prior helps clearly in precision/AP and extrapolation of ranking, but costs recall at fixed τ.** A free recurrent cell (no prior) does *not* learn to stay in the hull (precision 0.27, 3.7× over-approximation) and recovers high recall only by predicting far too much.
3. **Recurrence per se did not help**: matched-parameter feed-forward is equal, the compute-matched stack is better (no-prior comparison). The earlier suspicion (referee §9) that recurrent computation could be replaced by lookahead/direct readout is not contradicted: here a direct `k`-conditioned net does as well on K-extrapolation.
4. **What is genuinely useful is the structural part**: the exact Boolean hull as mask (sound) + learned ranking inside it. Learning does not buy a tighter *sound* set.
5. Limitations: single alphabet (6), equal-length rules, lhs length 2–3, ≤10 oriented rules; targets are symbol-level supports (not bigram supports); the hull is empirically (not formally) verified to contain `Y_k` (0/1,200 violations, plus 8-system unit test; also holds on val); training budget fixed at 1,500 steps and not tuned; `τ` from in-distribution val applied under extrapolation (threshold-free AP reported alongside); non-recurrent arms lack the prior; timings contended; word-level FN rates are per-word over all `L×6` cells and so increase mechanically with `L`; the 5 training seeds give seed-level t-CIs with 4 d.f. (wide).
6. **Not claimed:** soundness of any learned arm; any change to LBF-18 theorems or results.

Reproduce: `python lbf19_learned_transfer.py --stage shard --seed S --threads 1 --shards <dir>` for S=0..4, then `--stage merge --shards <dir>` (or `--stage all`).
